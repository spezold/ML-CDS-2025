"""
Miscellaneous own utility functions
"""
from bisect import bisect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import fields, asdict
from datetime import datetime
from enum import Enum
from functools import partial
from math import nan, isnan
import multiprocessing
import numbers
import os
from pathlib import Path
from pprint import pformat
import re
import subprocess
from threading import Thread, Event
from time import sleep
from typing import Any, TypeVar

import torch
from torch.nn import Module
from torch.utils.data import Dataset
import yaml

D = TypeVar("D", bound=Dataset)
M = TypeVar("M", bound=Module)
T = TypeVar("T")


class Split(Enum):
    TRAIN = "dataset_train"
    VAL = "dataset_val"

    def is_train(self) -> bool:
        return self == self.TRAIN


class _Invalid:  # To be used as a marker
    pass


class CpuMemory(Thread):
    """
    A simple estimator of the current memory consumption under the current parent process on the CPU side (in GiB).
    """
    # Following https://unix.stackexchange.com/questions/21836/ (20240614)
    _mem_cmd = ["bash", "-c", "ps -p {} --no-headers -o rss | awk '{{sum+=$1}} END {{print sum}}'"]
    _tree_cmd = ["pstree", "-p"]
    _tree_pattern = r"\((\d+)\)"

    def __init__(self, update_every_n_seconds: float = 1.0):
        # Following Doug Fort: "Terminating a Thread"
        # (https://www.oreilly.com/library/view/python-cookbook/0596001673/ch06s03.html; 20240614)
        self._stop_request = Event()
        self._update_every_n_seconds = update_every_n_seconds
        self._current = nan
        self._max = nan
        self._current_num = 0
        self._max_num = 0
        self._main_pid = None
        super().__init__(daemon=True)
        self.start()

    @property
    def main_pid(self) -> str:
        """
        :return: the process id (PID) of the current program's main process
        """
        if self._main_pid is None:
            parent_process = multiprocessing.parent_process()
            if parent_process is None:
                self._main_pid = str(os.getpid())  # We are the main process → get PID
            else:
                # Is the parent process the main process? Cf. https://stackoverflow.com/a/30808219/7395592 (20240618)
                if type(parent_process) is multiprocessing.Process:
                    raise RuntimeError("Neither the current process nor its parent appears to be the main process → "
                                       "a CpuMemory instance cannot be used from here.")
                self._main_pid = str(os.getppid())  # The parent process indeed is the main process → get parent's PID
        return self._main_pid

    def _current_memory_gb_and_num(self) -> tuple[float, int]:
        try:
            # (1) Get process tree under the main PID, parse PIDs from tree (TODO: Use check=True here?)
            tree = subprocess.run(self._tree_cmd + [self.main_pid], capture_output=True, text=True).stdout
            pids = list(set(re.findall(self._tree_pattern, tree)))  # Avoid duplicates (although there shouldn't be any)
            # (3) accumulate memory for PIDs (in junks, to avoid "argument list too long")
            chunk_size = 100
            current = 0
            for pids_chunk in (pids[i:i + chunk_size] for i in range(0, len(pids), chunk_size)):
                mem_cmd = self._mem_cmd[:2] + [self._mem_cmd[2].format(",".join(pids_chunk))]
                current_str = subprocess.run(mem_cmd, capture_output=True, text=True).stdout.strip()  # TODO: check=True
                current += 0 if not current_str else float(current_str) / 1024 ** 2  # KiB→GiB
            num = len(pids)
        except subprocess.CalledProcessError:
            current = nan
            num = 0
        return current, num

    def run(self):
        while not self._stop_request.is_set():
            current, current_num = self._current_memory_gb_and_num()
            self._current = current
            self._current_num = current_num
            self._max = current if isnan(max_ := self._max) else max(max_, current)
            self._max_num = max(self._max_num, current_num)
            sleep(self._update_every_n_seconds)

    def join(self, *args, **kwargs):
        self._stop_request.set()
        super().join(*args, **kwargs)

    def evaluate(self) -> tuple[float, float, int, int]:
        """
        :return: current (GiB), max (GiB), current number of processes, max number of processes
        """
        self.join()
        return self._current, self._max, self._current_num, self._max_num


def config_from(yaml_file: str | Path, *, drop_private: bool = True) -> dict[str, Any]:
    """
    Load and return configuration from the given file

    :param yaml_file: to be loaded
    :param drop_private: if True (default), drop keys that start with "__"  # FIXME: This should be obsolete
    :return: resulting configuration
    """
    def drop(data):
        match data:
            case Mapping(): data = {k: drop(v) for k, v in data.items() if not k.startswith("__")}
            case str(): pass  # Also matches `Sequence()` otherwise
            case Sequence(): data = [drop(item) for item in data]
        return data

    with Path(yaml_file).open("r", encoding="utf-8") as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    return drop(args_dict) if drop_private else args_dict


def now(message: str = "") -> datetime:
    """:return: the current time, after having enforced a CUDA synchronization and shows given ``message``"""
    torch.cuda.synchronize()
    if message:
        print(message)
    return datetime.now()


def to_path(p: str | Path | None, /) -> Path | None:
    """
    Convert given path to ``Path`` instance, expand user ``~``, resolve relative paths; pipe through None.
    :param p: path of interest
    :return: given path ``p``, with user ``~`` expanded, relative paths resolved, and converted to ``Path`` instance
    """
    return Path(p).expanduser().resolve() if p is not None else p


def prettified(dct: dict) -> str:
    """
    :return: a pretty string of the given (potentially multi-level) ``dct``
    """
    return pformat(dct, indent=2, width=120)


def sanitized(s: str, /) -> str:
    """
    Sanitize string by (1) replacing spaces with underscores and (2) dropping all characters that are not alphanumeric
    (i.e. characters ``c`` where ``c.isalnum()`` is False).

    :param s: string to be sanitized
    :return: sanitation result
    """
    return "".join(char for char in s.replace(" ", "_") if char.isalnum() or char == "_")


def frozen(model: M, set_eval: bool = True) -> M:
    """
    Freeze the given model *in-place* and return it for convenience.

    :param model: model whose parameters should be frozen
    :param set_eval: if True (default) switch the model to evaluation mode before freezing for convenience
    :return: the given model
    """
    if set_eval:
        model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def item(s: Sequence[T], /) -> T:  # FIXME: Do we still need this?
    """
    :return: single representative item of sequence ``s`` if possible, raise ValueError otherwise
    """
    if len(s) < 1:
        raise ValueError("Need at least one item")
    if not all(el == s[0] for el in s[1:]):
        raise ValueError("All items must be equal")
    return s[0]


def get_nested(container: Mapping | Sequence, key: str, sep: str = ".") -> Any:
    """
    :return: element from nested ``container``, with ``key`` components separated by ``sep``; e.g.
        ``get_nested(c, "a.b.1")`` returns ``c["a"]["b"]["1"]`` or ``c["a"]["b"][1]`` (on TypeError for "1")
    """
    for subkey in key.split(sep):
        try:
            container = container[subkey]  # Try string version ob subkey
        except TypeError:
            container = container[int(subkey)]  # Try integer version of subkey
    return container


def nested_dict_from(flat: Mapping[str, Any], sep: str = ".") -> dict[str, Any]:
    """:return: flat mapping with ``sep``-separated keys converted into a nested dictionary."""
    result = {}
    for flat_key, value in flat.items():
        keys = flat_key.split(sep)
        d = result
        for key in keys[:-1]:
            d[key] = {} if key not in d or not isinstance(d[key], dict) else d[key]  # Create next level if necessary
            d = d[key]
        d[keys[-1]] = value
    return result


def is_frozen(model: M, check_eval: bool = True) -> bool:
    """
    :return: True if none of the ``model``'s parameters require gradient computation, False otherwise
    """
    return not ((check_eval and model.training) or any(p.requires_grad for p in model.parameters()))


def map_args(src, dst, **kwargs):
    """
    Map arguments from the given ``src`` to the given ``dst``, converting names as specified in the ``kwargs``.

    Proceed as follows: (1) Arguments that are found under the same name in ``src`` and in ``dst`` will have their value
    transferred. (2) Arguments that are found under different names in ``src`` and in ``dst``, but have a name mapping
    specified via ``kwargs``, will have their value transferred. (3) Arguments that are found in ``src`` but not in
    ``dst`` (neither by same name nor by name mapping via ``kwargs``) will be ignored. (4) Arguments that are found in
    ``dst`` but not in ``src`` (neither by same name nor by name mapping via ``kwargs``) will (4a) use the default value
    if specified in ``dst``, or (4b) cause an error if no default value is specified.

    :param src: an *instance* of the source dataclass or a dictionary of name–value pairs
    :param dst: the *type* of the destination dataclass
    :param kwargs: {argument name in ``dst``: argument name in ``src``}
    :return: new instance of the destination dataclass
    """
    src = src if isinstance(src, Mapping) else asdict(src)
    dst_names = {f.name for f in fields(dst)}
    return dst(**{k: src[k] for k in dst_names & src.keys()} | {dk: src[sk] for dk, sk in kwargs.items()})


def save_with_backup(dct: dict, path: str | Path):
    """
    Before saving the given dictionary at the given path, make a backup of the preceding file if exists.

    :param dct: to be saved
    :param path: file path for saving
    """
    backup_suffix = ".bak"
    path = Path(path)
    backup_path = path.with_suffix(backup_suffix)
    if path == backup_path:
        raise ValueError(f"Given path '{str(path)}' should not end with '{backup_suffix}'.")
    if path.exists():
        path.replace(backup_path)  # Currently saved instance will be new backup instance (and replace existing backup)
    torch.save(dct, path)  # Given instance will be new saved instance


class TiledRangeMapping:

    @staticmethod
    def _check_on_init(sorted_rngs: list[range], values_sorted_by_rng: list[T]):
        if not len(sorted_rngs):
            raise ValueError("Cannot create empty instance.")
        if not all(rng.step == 1 for rng in sorted_rngs):
            raise ValueError("All given ranges must be dense (step == 1).")
        if not all(rng.stop > rng.start for rng in sorted_rngs):
            raise ValueError("All given ranges must be full and increasing (stop > start).")
        if not all(r0.stop == r1.start for r0, r1 in zip(sorted_rngs[:-1], sorted_rngs[1:])):
            raise ValueError("Ranges must tile the complete range (r0.stop == r1.start for consecutive ranges r0, r1)")
        if not all(v0 != v1 for v0, v1 in zip(values_sorted_by_rng[:-1], values_sorted_by_rng[1:])):
            raise ValueError("Ranges must not be subdivided (i.e. consecutive ranges must map to different values)")

    def __init__(self, value_by_rng: dict[range, T], /):
        """
        Represent a key-value mapping, where each value is valid for a range of integer keys.

        The mapping is tiled (or tessellated) in the mathematical sense; that is, ranges of keys must be (a)
        non-overlapping and (b) without gaps. In other words: each integer between and including the smallest key and
        the largest key maps to exactly one value. Internally, values are accessed using ``bisect.bisect()``, ensuring
        an optimal running time.

        CAUTION: (1) Negative indices in slices have a different meaning than usual here, in that they have to be
        understood as actual indices "from the left" rather than indices "from the right" (given that the tiled range
        may contain negative values). (2) Instances are also subscriptable with non-integer values and produce the
        intuitively correct result, e.g. ``TiledRangeMapping({range(0, 1): 42})[0.5] == 42``. Accessing non-integer
        values may or may not be meaningful in a given context, so indices should be checked beforehand, as no
        corresponding checks are provided in :meth:`__getitem__` for performance reasons.

        :param value_by_rng: for each range, the value that it should map to; other than being tiled, ``value_by_rng``
            must fulfill the following properties: (1) at least one range is given, (2) range are not unnecessarily
            subdivided, i.e. adjacent ranges have different values
        """
        sorted_rngs = sorted(value_by_rng.keys(), key=lambda rng: rng.start)
        values_sorted_by_rng = [value_by_rng[rng] for rng in sorted_rngs]
        self._check_on_init(sorted_rngs, values_sorted_by_rng)
        self._start = sorted_rngs[0].start
        self._stop = sorted_rngs[-1].stop

        # The lowest lower bound is implicit, so `len(breakpoints) == len(values) - 1`. To recognize leaving the valid
        # range, surround it with `_Invalid` marker values
        self._breakpoints = [rng.start for rng in sorted_rngs] + [self._stop]
        self._values = [_Invalid] + values_sorted_by_rng + [_Invalid]

    def __getitem__(self, key: int | slice) -> T | list[T]:
        # Cf. https://docs.python.org/3/reference/datamodel.html#object.__getitem__
        # Follow the `grade()` example from the `bisect` doc
        if isinstance(key, slice):
            # Follow https://stackoverflow.com/questions/13855288/ (20240607)
            rng_start = key.start if key.start is not None else self._start
            rng_stop = key.stop if key.stop is not None else self._stop
            rng_step = key.step if key.step is not None else 1
            if rng_step < 0:
                raise ValueError(f"Slices with negative step are currently not supported (got {key}).")
            keys = range(rng_start, rng_stop, rng_step)
            result = [self._values[bisect(self._breakpoints, k)] for k in keys]
            result = _Invalid if any(r is _Invalid for r in result) else result
        else:
            result = self._values[bisect(self._breakpoints, key)]
        if result is _Invalid:
            raise ValueError(f"Value of key must be {self._start} <= key < {self._stop} (got {key}).")
        return result

    def __len__(self) -> int:
        # Cf. https://docs.python.org/3/reference/datamodel.html#object.__len__
        return self._stop - self._start

    def __iter__(self) -> Iterator[int]:
        # Cf. https://docs.python.org/3/reference/datamodel.html#object.__iter__
        yield from range(self._start, self._stop)

    def __contains__(self, item: int) -> bool:
        # https://docs.python.org/3/reference/datamodel.html#object.__contains__
        return isinstance(item, numbers.Real) and self._start <= item < self._stop

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop


SensorTransform = Callable[[torch.Tensor], torch.Tensor]


def _transform(t: torch.Tensor, *, m_mean, r_std) -> torch.Tensor:
    return t.add(m_mean.to(t.device)).multiply_(r_std.to(t.device))


def sensor_transforms_from(stats: Mapping[str, tuple[float, ...]]) -> SensorTransform:
    m_mean = -torch.as_tensor(stats["mean"])
    r_std = 1 / torch.as_tensor(stats["std"])
    t = partial(_transform, m_mean=m_mean, r_std=r_std)
    return t
