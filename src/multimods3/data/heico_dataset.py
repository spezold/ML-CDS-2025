"""
Provide a `Dataset` that samples clips from the in-house dataset and from the HeiCo dataset
(Maier-Hein et al. 2021, doi:10.1038/s41597-021-00882-2).
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import json
from logging import getLogger
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm

from multimods3.data.pyav_reader import Reader as VideoReader
from multimods3.data.folder_reader import Reader as FolderReader
from multimods3.utils.video.transforms import VideoTransform
from multimods3.utils.weighted_sampler import DistributedWeightedSampler, DistributedWeightedSamplerArgs
from multimods3.utils.misc import SensorTransform, TiledRangeMapping


logger = getLogger()


@dataclass(frozen=True)
class Folder:
    """
    An abstraction for the in-house data; should be instantiated via ``from_dict()``, under the following assumptions:

    * The video for each surgery is represented by a folder that contains successive ``*.mp4`` files of video parts,
      with the file names in alphabetical order corresponding to their temporal order.
    * All video frames are valid, as opposed to masked (all-blue) ranges in the HeiCo data.
    * There is one label per surgery, as opposed to HeiCo's per-frame labels.
    * The label is available through a metadata dictionary: if the video's folder is called ``surgery00``, then the
      dictionary has a corresponding entry ``"surgery00": {"num_frames": …, "complications": …, "length_of_stay": …}``,
      with "num_frames" the total number of video frames in the folder, "complications" the label for CCI, and
      "length_of_stay" the label for LoS.
    * For each video, corresponding vital data samples are available as a ``*.csv`` file:

      * If the video's folder is called ``./videos/surgery00``, then the corresponding ``*.csv`` file can be found at
        ``./resampled-tables/vital-surgery00.csv``.
      * The ``*.csv`` file contains the columns "video", "frame", "time", "Herzfrequenz" (heart rate),
        "arterielle Blutdruckmessung - Diastolisch" (diastolic pressure), "arterielle Blutdruckmessung - Mean" (mean
        pressure), "arterielle Blutdruckmessung - Systolisch" (systolic pressure), where "video" is the 0-based index of
        the video part, "frame" is the 0-based index of the frame within the video part, "time" is the frame's
        wall-clock presentation timestamp, and the remaining columns contain the actual vital data measurements.
    """
    path: Path
    num_total_frames: int
    label: int
    sensors: None | pd.DataFrame = field(hash=False)

    @classmethod
    def from_dict(cls, path: Path, dct: dict[str, Any], load_sensors: bool, label: str):
        """
        Create instance from the given folder path and metadata dictionary (see class docstring for details).

        :param path: folder path containing video parts of one surgery
        :param dct: metadata dictionary (only the entry for this surgery, i.e. a subdictionary)
        :param load_sensors: whether to load vital data from ``*.csv`` file (True) or not (False)
        :param label: which outcome label to load ("complications" or "length_of_stay")
        :return: new instance
        """
        num_frames = dct["num_frames"]
        label = dct[label]
        if load_sensors:
            sensors = _vital_data_from(path.parents[1] / "resampled-tables" / f"vital-{path.stem}.csv")
            assert len(sensors) == num_frames
        else:
            sensors = None
        return cls(path=path, num_total_frames=num_frames, label=label, sensors=sensors)

@dataclass(frozen=True)
class Video:
    """
    An abstraction for the HeiCo data, as used in EndoVis2017 and as provided via the corresponding Synapse project
    (https://doi.org/10.7303/syn21903917); should be instantiated via ``from_dict()``, under the following assumptions:

    * The video for each surgery is represented by a single ``*.avi`` file.
    * Frame-wise labels and masks for out-of-body sequences are provided as metadata dictionaries (see
      ``resources/HeiCo/video-metadata.json`` for the actual EndoVis2017 values, converted to the expected format).
    * For each video, corresponding sensor data samples are available as a ``*.csv`` file:

      * If the video is called ``./Surgery_0.avi``, then the corresponding ``*.csv`` file is called
        ``./Surgery_0_Device.csv``.
      * The ``*.csv`` file contains the device data measurements, as provided via the Synapse project.
    """
    path: Path | None  # None for debugging
    num_total_frames: int
    masks: TiledRangeMapping = field(hash=False)  # True: valid (unmasked), False: invalid (masked)
    labels: TiledRangeMapping = field(hash=False)
    sensors: None | pd.DataFrame = field(hash=False)  # None for video-only analyses

    @classmethod
    def for_debugging(cls, num_total_frames: int, labels: dict[range, int]):
        masks = TiledRangeMapping({range(num_total_frames): True})
        labels = TiledRangeMapping(labels)
        return cls(path=None, num_total_frames=num_total_frames, masks=masks, labels=labels, sensors=None)

    @classmethod
    def from_dict(cls, path: Path, dct: dict[str, Any], load_sensors: bool):
        """
        Create instance from the given file path and metadata dictionary (see class docstring for details).

        :param path: file path of video of one surgery
        :param dct: metadata dictionary (only the entry for this surgery, i.e. a subdictionary)
        :param load_sensors: whether to load vital data from ``*.csv`` file (True) or not (False)
        :return: new instance
        """
        range_from_str = lambda s: range(*(int(v) for v in s.split(":")))
        num_total_frames = dct["num_frames"]
        masks = [range_from_str(rng) for rng in dct["masked_frames"]]
        masks = cls._mapping_masks_from(masks, num_total_frames)
        labels = {int(k): [range_from_str(rng) for rng in v] for k, v in dct["phases"].items()}
        labels = cls._mapping_labels_from(labels)
        if load_sensors:
            sensors = _sensor_data_from(path.parent / f"{path.stem}_Device.csv")
            assert len(sensors.index) >= num_total_frames  # Sanity check: enough sensor data present?
        else:
            sensors = None
        return cls(path, num_total_frames=num_total_frames, masks=masks, labels=labels, sensors=sensors)

    @staticmethod
    def _mapping_masks_from(mask_rngs: list[range], num_frames: int) -> TiledRangeMapping:
        num = num_frames
        invalid = sorted(mask_rngs, key=lambda rng: rng.start)
        if invalid:
            # Make valid (unmasked) ranges between invalid (masked) ranges explicit
            valid = [range(r0.stop, r1.start) for r0, r1 in zip(invalid[:-1], invalid[1:]) if r0.stop != r1.start]
            # Create initial valid range if necessary (i.e. first mask does not start at 0)
            valid = ([range(0, current_start)] if (current_start := invalid[0].start) != 0 else []) + valid
            # Create final valid range if necessary (i.e. last mask does not stop at `num_frames`)
            valid = valid + ([range(current_stop, num)] if (current_stop := invalid[-1].stop) != num else [])
        else:
            # Create complete valid range if necessary (i.e. no mask was given)
            valid = [range(0, num)]
        valid_by_rng = {**{k: True for k in valid}, **{k: False for k in invalid}}
        return TiledRangeMapping(valid_by_rng)

    @staticmethod
    def _mapping_labels_from(rngs_by_label: dict[int, list[range]]) -> TiledRangeMapping:
        label_by_rng = {rng: label for label, rngs in rngs_by_label.items() for rng in rngs}
        return TiledRangeMapping(label_by_rng)

    def is_valid(self, frame_index: int | slice) -> bool | list[bool]:
        """
        Check if the given ``frame_index`` is valid, i.e. is not in a masked range.

        :param frame_index: to be checked
        :return: True if valid (not masked), False otherwise
        """
        return self.masks[frame_index]

    def label_at(self, frame_index: int | slice) -> int | list[int]:
        """
        Return the surgical phase of the given ``frame_index``.

        :param frame_index: index of interest
        :return: int label of phase
        """
        return self.labels[frame_index]


class Sample(NamedTuple):
    """
    Representation of a single sample (i.e. a video segment that is further subdivided into several clips), providing
    the video it belongs to, its start frame, and its class label.
    """
    video: Video | Folder
    start_frame: int
    label: int


@dataclass(frozen=True, kw_only=True)
class HeicoDatasetArgs:
    """
    Wrap arguments for ``HeicoDataset``.

    The number of resulting samples is implicitly determined by (1) the ranges of valid frames in the given videos and
    (2) the number of frames to be spanned by each sample, as determined by

    * the number of clips that subdivide each sample, ``num_clips``,
    * the number of frames to be extracted per clip, ``num_frames_per_clip``,
    * the gap between extracted frames, ``frame_step``,
    * the offset between the start frames of consecutive samples, ``stride``.
    """
    num_clips: int
    """number of clips to be extracted per sample"""
    num_frames_per_clip: int
    """number of frames to be extracted per clip"""
    frame_step: int
    """number of frames to step between sampled frames (i.e. dilation)"""
    stride: str | int
    """"sliding" (for overlapping sliding window patches) or "patched" (for adjacent patches) or actual int"""
    relative_frame_indices: bool
    """return relative frame indices, i.e. starting with 0 for each sample (True) or the actual frame indices from the
    respective video (False) (also see ``HeicoDataset.__getitem__()``)"""
    weight_by_class: bool
    """determine ``HeicoDataset.sample_weights`` as inversely proportional to the number of samples in each class (True)
    or provide a constant value (False)"""
    skip_classes: list[int] | None
    """if given, skip samples of the corresponding classes completely"""


class HeicoDataset(Dataset):

    def __init__(
            self,
            rank: int,
            videos: list[Video] | list[Folder],
            video_transform: VideoTransform | None,
            sensor_transform: SensorTransform | None,
            args: HeicoDatasetArgs
    ):
        """
        A ``Dataset`` that samples clips from the in-house dataset and from the HeiCo dataset.

        :param rank: current rank in a distributed environment (for logging purposes)
        :param videos: videos or video folders from which to be sampled
        :param video_transform: transform to be applied to sampled frames
        :param sensor_transform: transform to be applied to sensor streams
        :param args: see ``HeicoDatasetArgs``
        """
        skip_classes = [] if args.skip_classes is None else args.skip_classes
        self._videos = videos
        self._v_transform = video_transform
        self._s_transform = sensor_transform
        self._args = args
        self._num_frames_per_segment = fps = args.num_clips * args.num_frames_per_clip

        samples, num_samples_by_label = self._init_samples_from(videos, fps, args.frame_step, args.stride, skip_classes)
        self._samples = samples
        self._num_samples_by_label = num_samples_by_label
        self._sample_weights = self._init_sample_weights_from(samples, num_samples_by_label, args.weight_by_class)

        # To be completed in ``_post_init`` by ``worker_init_fn``
        self._id: tuple[int, int | None] = (rank, None)
        self._reader_by_video_path: dict[Path, VideoReader | FolderReader | None]  # None for debugging

    def _post_init(self, worker_id: int, num_workers: int):
        """Intended to be called by ``worker_init_fn``"""
        if self._id[-1] is not None:
            raise RuntimeError("Dataset has been post-initialized already.")
        self._id = (self._id[0], worker_id)
        self._reader_by_video_path = {}
        for video in self._videos:
            if isinstance(video, Video):
                reader = None if video.path is None else VideoReader(video.path, max_threads_per_process=16,
                                                                     num_processes=num_workers, verbose=False)
            else:
                reader = FolderReader(video.path, max_threads_per_process=16, num_processes=num_workers, verbose=False)
            self._reader_by_video_path[video.path] = reader
            logger.info(f"Video reader initialized for video '{video.path}' in {self._id}")

    @classmethod
    def worker_init_fn(cls, *args):
        info = get_worker_info()
        ds = info.dataset
        if not isinstance(ds, cls):
            raise RuntimeError(f"Expected {cls.__qualname__}, got {ds.__class__.__qualname__} instead.")
        ds._post_init(worker_id=info.id, num_workers=info.num_workers)

    @staticmethod
    def _init_samples_from(videos: list[Video] | list[Folder], num_frames_per_segment, frame_step, stride,
                           skip_classes) -> tuple[list[Sample], dict[int, int]]:
        # Frames *per* segment: number of sampled frames; frames *in* segment: number of spanned frames, including steps
        num_frames_in_segment = 1 + (num_frames_per_segment - 1) * frame_step
        if stride == "sliding":
            sample_step = 1
        elif stride == "patched":
            sample_step = num_frames_in_segment
        elif isinstance(stride, int):
            sample_step = stride
        else:
            raise ValueError(f"Unknown stride value {repr(stride)} (should be 'sliding' or 'patched' or int)")
        samples, num_samples_by_label = [], {}
        skip_classes = set(skip_classes)
        for video in tqdm(videos, desc=f"Extract samples from {len(videos)} videos", file=sys.stdout):
            for start_idx in range(0, video.num_total_frames - num_frames_in_segment + 1, sample_step):
                if isinstance(video, Video):
                    sample_slice = slice(start_idx, start_idx + num_frames_in_segment, frame_step)
                    # As we want to predict, last frame determines label
                    label = video.label_at(sample_slice.stop - 1) if all(video.is_valid(sample_slice)) else None
                elif isinstance(video, Folder):
                    label = video.label
                else:
                    raise ValueError(f"Unsupported video type {type(video)}")
                if label is not None and label not in skip_classes:
                    samples.append(Sample(video, start_idx, label))
                    num_samples_by_label[label] = num_samples_by_label.get(label, 0) + 1
        logger.info(f"{len(samples):,} samples extracted from {len(videos)} videos" +
                    (f", skipping class(es): {', '.join(str(c) for c in skip_classes)}" if skip_classes else ""))
        return samples, num_samples_by_label

    @staticmethod
    def _init_sample_weights_from(samples: list[Sample], num_samples_by_label: dict[int, int], weight_by_class)\
            -> tuple[float, ...] | None:
        if weight_by_class:
            assert all(num_samples != 0 for num_samples in num_samples_by_label.values())
            weights_by_label = {label: 1 / num_samples for label, num_samples in num_samples_by_label.items()}
            sample_weights = tuple(weights_by_label[sample.label] for sample in samples)
        else:
            sample_weights = None
        return sample_weights

    @property
    def sample_weights(self) -> tuple[float, ...] | None:
        return self._sample_weights

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item: int) -> tuple[list, int, list, int, dict]:
        """
        For each sample index, return ``(frames, label, frame indices, sample index, sensor data)``.

        Let ``n := num_clips`` and ``fpc := num_frames_per_clip``. Then:

        - ``frames`` are the sampled and potentially transformed video frames (type: see below);
        - ``label`` is the surgical phase of the current sample (scalar);
        - ``frame indices`` are the (absolute or relative) indices of the sampled frames (``n``-element list of
          ``fpc``-shaped Numpy arrays);
        - ``sample index`` is the unique index of the returned sample within the current dataset;
        - ``sensor data`` is the corresponding excerpt of the ``Video.sensors`` table, if given; the
          dataset's ``frame_step`` is not respected here, so an ``n``-element list of ``m×14``-shaped tensors is
          returned for HeiCo, where ``m := 1 + (fpc - 1) * frame_step``; if given, the list is returned wrapped inside a
          dictionary with key "sensor_data"; if not given, an empty dictionary is returned

        The type and shape of ``frames`` depends on whether ``video_transform`` is None or given, following the V-JEPA
        codebase:

        - ``video_transform`` is None: ``frames`` is an ``n``-element list of ``fpc×H×W×3``-shaped, not yet normalized
          Torch tensors with ``H``, ``W`` being the actual height and width of the corresponding video;
        - ``video_transform`` is given: ``frames`` is an ``n``-element list of 1-element lists of ``3×fpc×S×S``-shaped,
          normalized Torch tensors with ``S`` being the crop size specified in the given ``transform`` instance.

        :param item: sample index
        :return: sample (see above)
        """
        def get_frames(video, slc, n_clips, n_fpc):  # N-element list: fpc×H×W×3 (tensors or ndarrays)
            if video.path is None:  # Debugging → create random frames
                frms = np.random.default_rng().integers(256, size=(n_clips, n_fpc, 600, 800, 3), dtype=np.uint8)
            else:
                frms = self._reader_by_video_path[video.path][slc]
            return list(frms.reshape(n_clips, n_fpc, *frms.shape[-3:]))

        # Frames *per* X: number of sampled frames; frames *in* X: number of spanned frames, including steps
        num_frames_in_segment = 1 + (self._num_frames_per_segment - 1) * self._args.frame_step
        num_frames_in_clip = 1 + (self._args.num_frames_per_clip - 1) * self._args.frame_step

        sample = self._samples[item]
        frame_slice = slice(sample.start_frame, sample.start_frame + num_frames_in_segment, self._args.frame_step)
        frame_idxs = np.arange(frame_slice.start, frame_slice.stop, frame_slice.step)
        frame_idxs = frame_idxs.reshape(self._args.num_clips, self._args.num_frames_per_clip)
        start_idxs = frame_idxs[:, 0].tolist()

        frames = get_frames(sample.video, frame_slice, *frame_idxs.shape)
        label = sample.label
        frame_idxs = list(frame_idxs - sample.start_frame if self._args.relative_frame_indices else frame_idxs)

        if self._v_transform is None:
            # Need copy: negative stride from BGR → RGB
            frames = [(torch.from_numpy(clip.copy()) if isinstance(clip, np.ndarray) else clip) for clip in frames]
        else:
            # Need copy: negative stride from BGR → RGB
            frames = [self._v_transform(c.copy() if isinstance(c, np.ndarray) else c.cpu().numpy()) for c in frames]
        suppl = {}  # Supplementary (read: optional) data
        # Subtract 1 because ranges in ``loc[]`` are upper inclusive
        if sample.video.sensors is not None:
            sensor_data = [sample.video.sensors.loc[i:i+num_frames_in_clip - 1].to_numpy() for i in start_idxs]
            if self._s_transform is None:
                sensor_data = [torch.from_numpy(item) for item in sensor_data]
            else:
                sensor_data = [self._s_transform(torch.from_numpy(item)) for item in sensor_data]
            suppl["sensors_encoder"] = sensor_data

        return frames, label, frame_idxs, item, suppl

    def sample_at(self, index: int) -> Sample:
        """
        Return meta information for the sample at the given index.

        :param index: index of interest
        :return: corresponding meta information
        """
        return self._samples[index]


@dataclass(frozen=True, kw_only=True)
class DataLoaderArgs:
    batch_size: int
    num_workers: int
    drop_last: bool = False
    collate_fn: Callable | None = None
    persistent_workers: bool = False
    pin_memory: bool = False
    prefetch_factor: int = 1
    worker_init_fn: Callable = HeicoDataset.worker_init_fn


@dataclass(frozen=True, kw_only=True)
class MetadataLocationArgs:
    split_name: str
    split_file: Path
    metadata_file: Path
    base_dir: Path | None = None


@dataclass(frozen=True, kw_only=True)
class MetadataArgs:
    """
    Wrap arguments to create ``HeicoDataset`` from the metadata and split info in the given JSON documents.

    The ``split_name`` may contain information on the nesting level and on combining certain splits; in particular:

    - a *comma* in the name implies a new level in the JSON document, with surrounding spaces ignored, e.g.
      "training" implies {"training": <video directories>, …}, while "testing, stage 2" is interpreted as
      {"testing": {"stage 2": <video directories>, …}, …};
    - a *plus* implies that the given splits are to be combined, with surrounding spaces ignored, e.g. "testing,
      stage 1 + testing, stage 2" causes the combination of the two splits "testing, stage 1" and "testing, stage 2"
      within the resulting dataset (with the commas interpreted as above).

    :param split_name: name of the split to be returned; see above for the special meaning of commas and plus signs
    :param split_dict: JSON document that contains split information
    :param metadata_dict: JSON document that contains video metadata
    :param base_dir: use as base directory path for video files if ``split_dict`` contains relative paths
        (default: None)
    """
    split_name: str
    split_dict: dict[str, Any]
    metadata_dict: dict[str, Any]
    base_dir: Path | None


def _metadata_args_from(args: MetadataLocationArgs) -> MetadataArgs:
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    return MetadataArgs(
        split_name=args.split_name,
        split_dict=load(args.split_file),
        metadata_dict=load(args.metadata_file),
        base_dir=args.base_dir
    )


def _videos_for_debugging() -> tuple[list[Video], int, str]:
    videos = [Video.for_debugging(num_total_frames=300_000, labels={range(300_000): 0})]
    return videos, 1, "debug"


def _videos_from_metadata(args: MetadataArgs, load_sensors: bool, label: str | None = None) -> tuple[list[Video] | list[Folder], int, str]:
    video_dirs = []
    subset_keys = [s.strip() for s in args.split_name.split("+")]
    for subset_key in subset_keys:  # Split components to be combined ("+")
        subset = args.split_dict
        for key in (s.strip() for s in subset_key.split(",")):  # Split and descend into nesting levels (",")
            subset = subset[key]
        video_dirs += subset
    video_dirs = [args.base_dir / d for d in video_dirs] if args.base_dir is not None else [Path(d) for d in video_dirs]
    # If metadata keys are files, we need to produce `Video`s, otherwise `Folder`s
    if all(k.endswith(".avi") for k in args.metadata_dict.keys()):
        create_folders = False
    elif not any(k.endswith(".avi") for k in args.metadata_dict.keys()):
        create_folders = True
    else:
        raise ValueError(f"Don't know whether to create videos or folders")
    if not create_folders:
        video_paths = sorted(v for d in video_dirs for v in d.glob("**/*.avi"))
        videos = [Video.from_dict(p, args.metadata_dict[p.name], load_sensors=load_sensors) for p in video_paths]
    else:
        assert label is not None
        videos = [Folder.from_dict(d, args.metadata_dict[d.stem], load_sensors=load_sensors, label=label)
                  for d in video_dirs]

    return videos, len(subset_keys), args.split_name


def videos_from(args: MetadataLocationArgs | MetadataArgs | str, load_sensors: bool, label: str | None = None) -> list[Video]:
    """
    Create videos from the metadata and split info in the JSON documents given in ``args``.

    :param args: if "debug", provide dummy video data (without sensor data)
    :param load_sensors: if True, provide tabulary sensor data in the videos' ``sensors`` field
    :param label: for video folders with one common label: the name of the label to load
    """
    if args == "debug":
        videos, n, split_name = _videos_for_debugging()
    else:
        args = _metadata_args_from(args) if isinstance(args, MetadataLocationArgs) else args
        videos, n, split_name = _videos_from_metadata(args, load_sensors=load_sensors, label=label)
    logger.info(f"Loaded {len(videos)} video{'s' if len(videos) != 1 else ''} from {n} split{'s' if n != 1 else ''} "
                f"'{split_name}' ({sum(v.num_total_frames for v in videos):,} frames total)")
    return videos


def _sensor_data_from(csv_path: Path) -> pd.DataFrame:
    header = ["frame", "thermo-flow-cur", "thermo-flow-tgt", "thermo-press-cur", "thermo-press-tgt",
              "thermo-vol-used", "thermo-press-supply", "thermo-is-on",
              "orlight-is-off", "orlight-int-1", "orlight-int-2", "endolight-int",
              "endo-balance", "endo-gains", "endo-exposure"]
    df = pd.read_csv(csv_path, header=None, names=header)
    df.set_index("frame", inplace=True)
    return df


def _vital_data_from(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=False)
    df.drop(labels=["video", "frame", "time"], axis=1, inplace=True)
    df = df.astype("float32")
    return df


def loader_from(dataset: HeicoDataset, loader_args: DataLoaderArgs, sampler_args: DistributedWeightedSamplerArgs) \
        -> DataLoader:
    sampler = DistributedWeightedSampler(weights=dataset.sample_weights, num_samples=len(dataset), args=sampler_args)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, **asdict(loader_args))
    return dataloader
