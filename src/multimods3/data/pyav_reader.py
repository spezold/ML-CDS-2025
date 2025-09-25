"""
Provide a video loader that solely relies on PyAV.
"""
from collections.abc import Sequence
from multiprocessing import cpu_count
from pathlib import Path

import av
import numpy as np
import torch


def _ceil(numerator: int, denominator: int) -> int:
    """:return: robust (i.e. numerical error–free) ceiling for given non-negative ``numerator`` and ``denominator``"""
    # Following https://stackoverflow.com/a/14878734/7395592 (20250204)
    return numerator // denominator + (numerator % denominator != 0)


def num_frames_for(file_path: str) -> int:
    """:return: number of frames for the video at ``file_path``"""
    with av.open(file_path, timeout=None) as container:
        frames = container.streams.video[0].frames
    return frames


class Reader:

    def __init__(
        self,
        file_path: str | Path,
        *,
        max_threads_per_process: int = 4,  # Empirical value
        num_processes: int = 1,
        verbose: bool = True,
    ):
        """
        A video reader that relies on PyAV.

        :param file_path: video's file path
        :param max_threads_per_process: *upper limit* for the number of threads used for decoding in each process
            (also see ``num_processes`` below)
        :param num_processes: how many instances are currently run in parallel in different processes; this number has
            implications on the *default choices* for the numbers of threads used for decoding.
        :param verbose: if True (default), show information on the standard output
        """
        self._video_path = str(file_path)
        self._max_cores_per_process = max(1, cpu_count() // num_processes)
        self._num_threads = min(max_threads_per_process, self._max_cores_per_process)
        if verbose:
            print(f"Using {self._num_threads} threads per process")
        self._verbose = verbose

        self._container = av.open(self._video_path, timeout=None)
        self._stream = self._container.streams.video[0]
        self._stream.thread_count = self._num_threads
        self._stream.thread_type = "AUTO"  # https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#threading (20250128)

        if self._verbose:
            print(f"Read info from {Path(self._video_path).name} ...")
        self._num_frames = self._stream.frames
        self._shape = self._stream.height, self._stream.width
        self._start_time = self._stream.start_time
        if not (duration_per_frame := self._stream.duration / self._stream.frames).is_integer():
            raise RuntimeError(f"Need integer duration per frame, got {duration_per_frame}.")
        self._duration_per_frame = int(duration_per_frame)
        if self._verbose:
            print(f"Read info from {Path(self._video_path).name} ... done")


    def __getitem__(self, item: int | slice | Sequence[int] | np.ndarray, /) -> torch.Tensor | np.ndarray:
        """
        Return the RGB pixel data of the given frame(s) in the video as ``uint8`` Numpy array.

        :param item: frame index (int) or indices (slice, sequence, 1D int array); supporting positive steps only
        :return: RGB pixel data: H×W×3 (``item`` is int) or N×H×W×3 (``item`` is multiple ints), where N is the number
            of frames, H is the frame height, W is the frame width, and the 3 color channels are in RGB order
        """
        t_start, t_frame = self._start_time, self._duration_per_frame
        if isinstance(item, slice):
            is_multiple = True
            start, stop, step = item.indices(self._num_frames)
            if step <= 0:
                raise ValueError(f"Need step > 0; got '{item=}'.")
            times = range(t_start + start * t_frame, t_start + stop * t_frame, step * t_frame)
        else:
            is_multiple = hasattr(item, "__len__")
            item = np.asarray(item if is_multiple else [item], dtype=int)
            if np.any(item[1:] - item[:-1] <= 0):
                raise ValueError(f"Need strictly positively increasing sequence, got '{item=}'.")
            times = (t_start + item * t_frame).tolist()
        # Get time of the key frame right at or below the initial frame; then extract frames at given times from there
        self._container.seek(times[0], stream=self._stream)
        frame_data = []
        for frame in self._container.decode(self._stream):
            if frame.pts in times:
                frame_data.append(frame.to_ndarray(format="rgb24"))
                if len(frame_data) == len(times):
                    break
        # Sanity-check result: did we exhaust the stream before getting the necessary number of frames?
        if len(frame_data) < len(times):
            raise RuntimeError(f"Could not extract all frames for '{item=}': {len(frame_data)} of {len(times)} found.")
        return np.stack(frame_data, axis=0) if is_multiple else frame_data[0]

    def __del__(self):
        (container := getattr(self, "_container", None)) is None or container.close()
