# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from contextlib import contextmanager
import os

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


@contextmanager
def distributed(*, port: int | None = None, world_size: int, rank: int):
    """
    Convenience context manager that wraps ``init_distributed()`` and ``shutdown_distributed()``.
    """
    init_distributed(port=port, world_size=world_size, rank=rank)
    try:
        yield
    finally:
        shutdown_distributed()


def world_size_and_rank() -> tuple[int, int]:
    """
    :return: current world size, current rank
    """
    assert dist.is_available() and dist.is_initialized()
    return dist.get_world_size(), dist.get_rank()


def init_distributed(*, port: int | None = None, world_size: int, rank: int) -> None:
    """
    Init environment for distributed training.

    :param port: a free port (default: 37123)
        (cf. https://docs.pytorch.org/docs/stable/distributed.html#environment-variable-initialization)
    :param world_size: total number of processes
    :param rank: current process ID (0 <= ``rank`` < ``world_size``)
    """
    # TODO: Add SLURM support back in
    assert dist.is_available() and not dist.is_initialized()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(37123 if port is None else port)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, device_id=torch.device("cuda:0"))


def shutdown_distributed():
    # Fix "WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit,
    # the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this
    # process. In rare cases this process can exit before this point and block the progress of another member of the
    # process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4
    # (function operator())" – also see https://pytorch.org/docs/stable/distributed.html#shutdown (20241001) and
    # https://github.com/pytorch/pytorch/issues/75097#issuecomment-1088027600 (20241001)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
