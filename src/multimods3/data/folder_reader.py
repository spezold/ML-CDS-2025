"""
Provide a video folder loader that solely relies on `torchcodec`.
"""
from bisect import bisect_right
from collections.abc import Sequence
from multiprocessing import cpu_count
from pathlib import Path

from torchcodec.decoders import VideoDecoder
import numpy as np
import torch


def num_frames_for(file_path: str, *, max_threads_per_process: int = 4) -> int:
    """:return: number of frames for the video at ``file_path``"""
    return len(VideoDecoder(file_path, num_ffmpeg_threads=max_threads_per_process))


class Reader:

    def __init__(
            self,
            folder_path: str | Path,
            *,
            max_threads_per_process: int = 4,  # Empirical value
            num_processes: int = 1,
            verbose: bool = True,
    ):
        """
        An MP4 video folder reader that relies on `torchcodec`.

        :param folder_path: path to folder with videos (assuming alphabetical order is temporal order)
        :param max_threads_per_process: *upper limit* for the number of threads used for decoding in each process
            (also see ``num_processes`` below)
        :param num_processes: how many instances are currently run in parallel in different processes; this number has
            implications on the *default choices* for the numbers of threads used for decoding.
        :param verbose: if True (default), show information on the standard output
        """
        self._folder_path = str(folder_path)
        self._max_cores_per_process = max(1, cpu_count() // num_processes)
        self._num_threads = min(max_threads_per_process, self._max_cores_per_process)
        if verbose:
            print(f"Using {self._num_threads} threads per process")
        self._verbose = verbose

        self._files = sorted(Path(folder_path).glob("*.mp4"))
        if self._verbose:
            print(f"Read info from {Path(self._folder_path).name} ({len(self._files)} videos) ...")
        self._decoders = [VideoDecoder(f, num_ffmpeg_threads=self._num_threads, dimension_order="NHWC") for f in
                          self._files]
        self._num_frames = [len(d) for d in self._decoders]
        self._frame_offsets = np.cumsum(np.r_[0, self._num_frames[:-1]])
        self._total_num_frames = sum(self._num_frames)
        self._shape = self._decoders[0].metadata.height, self._decoders[0].metadata.width
        if self._verbose:
            print(f"Read info from {Path(self._folder_path).name} ({len(self._files)} videos) ... done")

    @staticmethod
    def _find_le(a, x):
        # https://docs.python.org/3/library/bisect.html (20250618)
        """Find rightmost value less than or equal to x, return (index, value, x - value)"""
        i = bisect_right(a, x)
        if i:
            actual_i = i - 1
            value = a[actual_i]
            return actual_i, value, x - value
        raise ValueError

    def __len__(self):
        return self._total_num_frames

    def __getitem__(self, item: int | slice | Sequence[int] | np.ndarray, /) -> torch.Tensor:
        """
        Return the RGB pixel data of the given frame(s) in the video as ``uint8`` torch tensor.

        :param item: frame index (int) or indices (slice, sequence, 1D int array); supporting positive steps only
        :return: RGB pixel data: H×W×3 (``item`` is int) or N×H×W×3 (``item`` is multiple ints), where N is the number
            of frames, H is the frame height, W is the frame width, and the 3 color channels are in RGB order
        """
        if isinstance(item, slice):
            is_multiple = True
            start, stop, step = item.indices(self._total_num_frames)
            if step <= 0:
                raise ValueError(f"Need step > 0; got '{item=}'.")
            frames = range(start, stop, step)
        else:
            is_multiple = hasattr(item, "__len__")
            item = np.asarray(item if is_multiple else [item], dtype=int)
            if np.any(item[1:] - item[:-1] <= 0):
                raise ValueError(f"Need strictly positively increasing sequence, got '{item=}'.")
            frames = item.tolist()
        frame_data = []
        for frame in frames:
            video_idx_for_frame, _, local_frame = self._find_le(self._frame_offsets, frame)
            frame_data.append(self._decoders[video_idx_for_frame][local_frame])
        return torch.stack(frame_data, dim=0) if is_multiple else frame_data[0]
