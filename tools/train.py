# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

import numpy as np

if not hasattr(np, 'sctypes'):
    np.sctypes = dict(
        float=[np.float16, np.float32, np.float64],
        int=[np.int8, np.int16, np.int32, np.int64],
        uint=[np.uint8, np.uint16, np.uint32, np.uint64],
        complex=[np.complex64, np.complex128],
        others=[np.bool_, np.object_, np.bytes_, np.str_])
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = np.bool_

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def find_next_run_dir(base_dir):
    """Find the next indexed run directory under a base work directory."""
    if not osp.isdir(base_dir):
        return osp.join(base_dir, 'run_001')

    indices = []
    for name in os.listdir(base_dir):
        if not name.startswith('run_'):
            continue
        index = name[4:]
        if index.isdigit():
            indices.append(int(index))

    next_index = max(indices, default=0) + 1
    return osp.join(base_dir, f'run_{next_index:03d}')


def find_latest_run_dir(base_dir):
    """Find the latest indexed run directory under a base work directory."""
    if not osp.isdir(base_dir):
        return None

    runs = []
    for name in os.listdir(base_dir):
        if not name.startswith('run_'):
            continue
        index = name[4:]
        run_dir = osp.join(base_dir, name)
        if index.isdigit() and osp.isdir(run_dir):
            runs.append((int(index), run_dir))

    if not runs:
        return None
    return max(runs)[1]


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename.
    # Fresh runs are saved in indexed child directories to avoid overwriting
    # checkpoints. Resuming reuses the specified work_dir, or the latest indexed
    # child directory when work_dir is inferred from the config.
    if args.work_dir is not None:
        base_work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        base_work_dir = osp.join('./work_dirs',
                                 osp.splitext(osp.basename(args.config))[0])
    else:
        base_work_dir = cfg.work_dir

    if args.resume:
        if args.work_dir is not None:
            cfg.work_dir = base_work_dir
        else:
            cfg.work_dir = find_latest_run_dir(base_work_dir) or base_work_dir
    else:
        cfg.work_dir = find_next_run_dir(base_work_dir)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
