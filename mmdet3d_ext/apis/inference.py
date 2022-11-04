# Copyright (c) OpenMMLab. All rights reserved.
import re
from copy import deepcopy
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core import (
    Box3DMode,
    CameraInstance3DBoxes,
    Coord3DMode,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    show_multi_modality_result,
    show_result,
    show_seg_result,
)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == "norm_cfg":
                config[item]["type"] = config[item]["type"].replace("naiveSyncBN", "BN")
            else:
                convert_SyncBN(config[item])


def inference_detector(model, pc):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): ros pointcloud.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    data = dict(
        pts_filename="none",
        pc=pc,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[],
    )

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data["img_metas"] = data["img_metas"][0].data
        data["points"] = data["points"][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data
