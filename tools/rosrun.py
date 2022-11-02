# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser


import sys
import os
from argparse import ArgumentParser
import numpy as np
import torch
from copy import deepcopy
import time
import cv2

# mmcv
import mmcv

# mmdet
from mmdet.models import build_detector
from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor

# mmdet3d
from mmdet3d.apis import init_model


# local
from ros_utils import ROSExtension
from mmdet_ext.models import *
from mmdet_ext.datasets import *
from mmdet3d_ext.datasets import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--topic", help="input msg ros topic")
    parser.add_argument(
        "--device", default="cuda:0", help="Device used for inference"
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.0,
        help="bbox score threshold",
    )

    parser.add_argument(
        "--msg_type",
        type=str,
        default="img",
        help="msg type, img or ponitcloud",
    )
    parser.add_argument(
        "--republish",
        action="store_true",
        help="republish for sync msg and detected result",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="if compressed image",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)

    # init model
    # NOTE: this is a temporary solution for loading models from MMDetection3D
    if args.msg_type == "img":
        model = init_detector(config, args.checkpoint, device=args.device)
    elif args.msg_type == "pointcloud":
        model = init_model(config, args.checkpoint, device=args.device)
    else:
        raise NotImplementedError

    print("---model init done---")

    ros_extension = ROSExtension(
        model=model,
        sub_topic=args.topic,
        pub_topic=args.topic + "/detected_objects",
        score_thr=args.score_thr,
        msg_type=args.msg_type,
        republish=args.republish,
        compressed=args.compressed,
    )
    print("---waiting for topic %s msgs---:" % args.topic)
    ros_extension.start()


if __name__ == "__main__":
    main()
