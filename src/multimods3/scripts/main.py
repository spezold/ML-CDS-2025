# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from argparse import BooleanOptionalAction
import importlib
from logging import getLogger, INFO, ERROR
import multiprocessing as mp

from multimods3.utils.distributed import distributed
from multimods3.utils.info import base_dir
from multimods3.utils.misc import prettified, config_from

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, help='name of config file to load', default='default.yaml')
parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'], help='which devices to use on local machine')
parser.add_argument('--mp', default=True, action=BooleanOptionalAction,
                    help='use multiprocessing? (yes: mp (default), no: no-mp)')
parser.add_argument('--port', type=int, required=False, default=None,
                    help='Optional value for MASTER_PORT (use to launch multiple) '
                         'multiprocessing runs on the same machine (default: None)')


def wrap_main(script_name, args_dict):

    getLogger().info(f'Running script: {script_name}')
    return importlib.import_module(f'multimods3.scripts.{script_name}').main(args_dict)


def process_main(rank, fname, world_size, devices, port):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    from multimods3.utils.logs import init_logger, post_init_logger_from

    logger = init_logger(force=True)
    if rank == 0:
        logger.setLevel(INFO)
    else:
        logger.setLevel(ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    args_dict = config_from(base_dir() / "configs" / fname)
    post_init_logger_from(args_dict, rank)
    logger.info(f"{rank=}, {fname=}, {world_size=}, {devices=}")
    logger.info("Loaded params:")
    logger.info("\n" + prettified(args_dict))

    logger.info(f"Preparing to run script '{args_dict['script']}.py'")

    with distributed(port=port, world_size=world_size, rank=rank):  # Init distributed
        logger.info(f"Running... (rank: {rank}/{world_size})")
        wrap_main(args_dict["script"], args_dict=args_dict)  # Launch the script with loaded config


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    if not args.mp:
        assert num_gpus == 1, "If '--no-mp' is specified, only one GPU should be given"
        process_main(0, args.fname, num_gpus, args.devices, args.port)
    else:
        mp.set_start_method('spawn')
        for rank_ in range(num_gpus):
            mp.Process(target=process_main, args=(rank_, args.fname, num_gpus, args.devices, args.port)).start()
