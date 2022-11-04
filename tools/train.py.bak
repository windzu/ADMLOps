# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings


import torch
import torch.distributed as dist

# mmcv
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

# mmdet
from mmdet import __version__
from mmdet import __version__ as mmdet_version
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (
    collect_env,
    get_device,
    get_root_logger,
    replace_cfg_vals,
    setup_multi_processes,
    update_data_root,
)

# mmdet3d
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger

# mmseg
from mmseg import __version__ as mmseg_version

# local
from mmdet_ext.models import *
from mmdet_ext.datasets import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from the latest checkpoint automatically",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff-seed",
        action="store_true",
        help="Whether or not set different seeds for different ranks",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--auto-scale-lr",
        action="store_true",
        help="enable automatically scaling LR.",
    )
    parser.add_argument(
        "--task-type", type=str, help="det or det3d or seg", default="det3d"
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def det_main():
    args = parse_args()
    # 1. 加载配置文件为dict 和 从 args.cfg_options中更新配置文件
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 3. 设置工作目录
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # 2. 通过 config 中 workers_per_gpu 设置多线程数
    setup_multi_processes(cfg)

    # replace the ${key} with the value of cfg.key
    # 将dict格式的配置文件中的${key}替换为cfg.key的值
    # cfg = replace_cfg_vals(cfg)

    # check 是否启用了 auto_scale_lr
    if args.auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn(
                'Can not find "auto_scale_lr" or '
                '"auto_scale_lr.enable" or '
                '"auto_scale_lr.base_batch_size" in your'
                " configuration file. Please update all the "
                "configuration files to mmdet >= 2.24.1."
            )

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # resume设置
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # 单卡gpu id设置
    cfg.gpu_ids = [args.gpu_id]

    # 分布式训练设置
    distributed = False

    # 创建work_dir并将配置文件保存到work_dir路径下
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    ###########################################
    ########## init logger and meta ###########
    ###########################################

    # 初始化logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(
        f"Set random seed to {seed}, " f"deterministic: {args.deterministic}"
    )
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    ###########################################
    ######### build model and dataset #########
    ###########################################

    # model = build_detector(cfg.model)
    model = build_model(cfg.model)
    model.init_weights()

    # 根据dataset config 构建 dataset
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # start train
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


def det3d_main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn(
            "`--auto-resume` is only supported when mmdet"
            "version >= 2.20.0 for 3D detection model or"
            "mmsegmentation verision >= 0.21.0 for 3D"
            "segmentation model"
        )

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ["EncoderDecoder3D"]:
        logger_name = "mmseg"
    else:
        logger_name = "mmdet"
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(
        f"Set random seed to {seed}, " f"deterministic: {args.deterministic}"
    )
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    logger.info(f"Model:\n{model}")
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], "PALETTE")
            else None,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


def main():
    args = parse_args()
    if args.task_type == "det":
        det_main()
    elif args.task_type == "det3d":
        det3d_main()
    elif args.task_type == "seg":
        pass
    ###########################################
    ############### init config ###############
    ###########################################


if __name__ == "__main__":
    main()
