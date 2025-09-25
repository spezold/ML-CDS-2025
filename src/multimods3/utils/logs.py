# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Any

import torch

from multimods3.utils.misc import to_path


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


def post_init_logger_from(args: dict, rank: int):
    """
    If the given arguments provide a "checkpointing_and_logging" or "logging" or "workflow" key, attach a
    ``RotatingFileHandler`` to the default logger, which writes to the folder ``<log_dir>`` into files named ``f"{<prefix>}-r{rank}.log"``.

    - For "checkpointing_and_logging", ``<log_dir>`` is ``a["folder"] / a["type"]``, ``<prefix>`` is ``a["tag"]``.
    - For "logging", ``<log_dir>`` is ``a["dir"]``, ``<prefix>`` is ``a["prefix"]``.
    - For "workflow", ``<log_dir>`` is ``a["logging"]["dir"]``, ``<prefix>`` is ``a["logging"]["prefix"]``.

    Here, ``a`` is the value at "checkpointing_and_logging", "logging", or "workflow", respectively.

    :param args: argument dictionary, usually resulting from loading a YAML config file
    :param rank: the current process number
    """
    log_dir_and_prefix = None
    if "checkpointing_and_logging" in args:
        args = args["checkpointing_and_logging"]
        log_dir_and_prefix = to_path(args["folder"]) / args["type"], args["tag"]
    elif "logging" in args:
        args = args["logging"]
        log_dir_and_prefix = to_path(args["dir"]), args["prefix"]
    elif "workflow" in args:
        args = args["workflow"]["logging"]
        log_dir_and_prefix = to_path(args["dir"]), args["prefix"]
    if log_dir_and_prefix is not None:
        log_dir, prefix = log_dir_and_prefix
        log_dir.mkdir(exist_ok=True, parents=True)
        handler = RotatingFileHandler(log_dir / f"{prefix}-r{rank:02d}.log", maxBytes=1_000_000, backupCount=5)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)


class CSVLogger:

    @staticmethod
    def _write_header_if_necessary(path, header_elements):
        path.touch(exist_ok=True)  # Ensure that file exists
        with path.open("r+") as file:
            h_given = file.readline()
            h_new = ",".join(header_elements) + "\n"
            if h_given:
                assert h_given == h_new, f"Header mismatch: is present as {repr(h_given)}, should be {repr(h_new)}"
            else:
                file.write(h_new)


    def __init__(self, path: Path, *, do_log: bool = True, **kwargs: str | None):
        """
        Write CSV log line by line at the given path, using the given formats.

        :param path: file to be appended to
        :param do_log: if True (default), actually write log entries; if False, don't write log entries
        :param kwargs: {column name: format string for column's entries (or None)}
        """
        self._path = path
        self._do_log = do_log
        self._format_by_col = {k: ("" if v is None else v) for k, v in kwargs.items()}
        if do_log:
            self._write_header_if_necessary(path, kwargs.keys())

    def log(self, **kwargs):
        """
        Log the given values in a new line of the CSV file.

        :param kwargs: {column name: column value}
        """
        if self._do_log:
            with self._path.open("a") as file:
                file.write(",".join(f"{kwargs[col]:{fmt}}" for col, fmt in self._format_by_col.items()) + "\n")


class MatrixLogger:

    @staticmethod
    def as_str(mat, fmt: str, sep: str):
        # Following https://stackoverflow.com/questions/78632524/ (20241008)
        raw_rows = [[f"{v:{fmt}}" for v in row] for row in mat]
        len_i, len_v = len(str(mat.shape[0])), max(len(str(mat.shape[1])), *(len(v) for row in raw_rows for v in row))
        rows = [f"{' ' * len_i}" + sep + sep.join(f"{i:>{len_v}}" for i in range(mat.shape[1]))]
        rows += [f"{i:>{len_i}}" + sep + sep.join(f"{v:>{len_v}}" for v in row) for i, row in enumerate(raw_rows)]
        return "\n".join(rows)

    def __init__(self, path: Path, *, do_log: bool, fmt: str | None = None, separation: int = 1):
        self._path = path
        self._do_log = do_log
        self._fmt = "" if fmt is None else fmt
        self._sep = " " * separation

    def log(self, matrix, *, prepend_line: str | None = None, append_line: str | None = None):
        if self._do_log:
            with self._path.open("a") as file:
                lines = [prepend_line] if prepend_line is not None else []
                lines += [self.as_str(matrix, self._fmt, self._sep)]
                lines += [append_line] if append_line is not None else []
                file.write("\n".join(lines) + "\n")


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def adamw_logger(optimizer):
    """ logging magnitude of first and second momentum buffers in adamw """
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get('state')
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))
    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}
