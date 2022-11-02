# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image
from skimage import io


def get_usd_info(path, label_path_list: list, num_worker=8):
    """获取lidar数据的信息

    Args:
        path (str): 数据集的根路径
        label_path_list (list): 需要读取的label的文件名list
        num_worker (int): 并行处理的数量

    """
    root_path = Path(path)

    def map_func(label_path):
        """根据文件名获取该文件名所对应的原始数据、label等相关信息

        Args:
            label_path (str): label相交于跟目录的路径(包含label的后缀)

        Returns:
            dict: info dict
        """
        label_path = os.path.join(root_path, label_path)
        # load json file and convert to dict
        label = json.load(open(label_path))
        # label = json.loads(label)  # convert str to dict

        # 将从json文件中读取的label进行一些补全操作
        return __label_postprocess(label)

    with futures.ThreadPoolExecutor(num_worker) as executor:
        usd_infos = executor.map(map_func, label_path_list)

    return list(usd_infos)


def __label_postprocess(label):
    """将原始的label进行一些补全操作

    主要工作包括：
        1. 如果数据为None,则返回None
        2. 如果数据为为list且有数据,则转换为np.ndarray
        2. 如果数据为为list但没有有数据,将需要补全
    Args:
        label (dict): 直接从json文件中读取的label

    Returns:
        dict: 转换后的label
    """

    def annos_postprocess(annos):
        annos["class_names"] = np.array(annos["class_names"]).reshape(-1)
        annos["track_ids"] = np.array(annos["track_ids"]).reshape(-1)
        annos["bbox2d"] = np.array(annos["bbox2d"], dtype=np.int32).reshape(-1, 4)

        # bbox3d原始标注中仅有7个值,但是为了支持mmdetetion3d中的pipeline需要转换9个值,因此尾部补全两个0
        annos["bbox3d"] = np.array(annos["bbox3d"], dtype=np.float32).reshape(-1, 7)
        annos["bbox3d"] = np.concatenate([annos["bbox3d"], np.zeros((annos["bbox3d"].shape[0], 2))], axis=1)

        annos["truncated"] = np.array(annos["truncated"], dtype=np.int32).reshape(-1)
        annos["occluded"] = np.array(annos["occluded"], dtype=np.int32).reshape(-1)
        if annos["num_points_in_gt"] is None:
            annos["num_points_in_gt"] = np.zeros(len(annos["class_names"], dtype=np.int32))
        else:
            annos["num_points_in_gt"] = np.array(annos["num_points_in_gt"]).reshape(-1)
        return annos

    def calib_postprocess(calib):
        """calib字段的后处理"""
        if calib is None:
            return None
        if "intrinsics" in calib:
            for key in calib["intrinsics"]:
                calib["intrinsics"][key] = np.array(calib["intrinsics"][key]).reshape(3, 3)
        for key in calib:
            if key != "intrinsics":
                calib[key] = np.array(calib[key]).reshape(4, 4)

    # images postprocess
    if label["images"] is None:
        label["images"] = None
    else:
        for key, value in label["images"].items():
            value["shape"] = np.array(value["shape"])
            value["annos"] = annos_postprocess(value["annos"])

    # point_clouds postprocess
    if label["point_clouds"] is None:
        label["point_clouds"] = None
    else:
        for key, value in label["point_clouds"].items():
            value["shape"] = np.array(value["shape"])
            value["annos"] = annos_postprocess(value["annos"])

    # calib postprocess
    if label["calib"] is None:
        label["calib"] = None
    else:
        label["calib"] = calib_postprocess(label["calib"])

    return label
