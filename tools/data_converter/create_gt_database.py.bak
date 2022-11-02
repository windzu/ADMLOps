# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from os import path as osp

import mmcv
import numpy as np
from mmcv import track_iter_progress

# from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
):
    """通过标注数据 原始数据等构建 lidar_gt_database.pkl 文件
    NOTE : 此函数是为了配合自定义的lidar数据集而写的,目前仅支持 lidar 一种数据集,目前没有计划支持其他数据集
    数据格式：
    {
        "class_name_0": [
            {
                "name": str, # class name
                "path": str, # path to the gt file
                "idx": None, # raw file index (euqal to file name)
                "gt_idx": int, # gt index in each raw file
                "box3d_lidar": None,
                "num_points_in_gt": int,
                "difficulty": int (0,1,2),
                "group_id": int, # 0,1,2,...
            },
            ...
        ],
        ...
    }

    Args:
        dataset_class_name (str): 数据集格式名称,也对应该类的名称,目前仅支持 LidarDataset .
        data_path (str): 数据集根路径.
        info_prefix (str): 数据集前缀,默认为lidar.
        info_path (str, optional): 之前生成的 xxx_lidar_info.pkl 文件的路径,默认为None.
        used_classes (list[str], optional): Classes have been used. Default: None.
        database_save_path (str, optional): 保存gt_database的路径,默认为None.
        db_info_save_path (str, optional): Path to save db_info.
    """
    print(f"Create GT Database of {dataset_class_name}")

    assert dataset_class_name == "LidarDataset"

    # 构建一个 LidarDataset config,用于注册该类
    # TODO ： 通过配置文件加载 而不是在此临时构建一个config dict
    dataset_cfg = dict(
        type="LidarDataset",
        data_root=data_path,
        ann_file=info_path,
        test_mode=False,
        modality=dict(
            use_lidar=True,
            use_depth=False,
            use_lidar_intensity=True,
            use_camera=False,
        ),
        pipeline=[
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend="disk"),
            ),
            dict(
                type="LoadAnnotations3D",
                with_bbox_3d=True,
                with_label_3d=True,
                file_client_args=dict(backend="disk"),
            ),
        ],
    )
    dataset = build_dataset(dataset_cfg)

    # 确保待保存的文件夹存在
    database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)

    # TODO : 应该先创建该数据结构,后续往其中填充。或者直接在dataset中实现一个方法，将需要进行的操作进行封装。这样方便理解代码
    all_db_infos = dict()
    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)

        idx = example["sample_idx"]  # (str) filename
        points = example["points"].tensor.numpy()
        gt_boxes_3d = example["ann_info"]["gt_bboxes_3d"].tensor.numpy()
        names = example["ann_info"]["gt_names"]
        group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)  # default euqal to [0 1 2 ...]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        group_dict = dict()
        for i in range(num_obj):
            filename = f"{idx}_{names[i]}_{i}.bin"  # {raw_filename}_{class_name}_{gt_bbox_id}.bin
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds  for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]  # shift to center (每个点坐标减去对应的gt_box的中心)

            # save points which in gt_box to disk
            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1

                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "idx": idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": 0,  # 给一个固定的difficulty
                    "group_id": group_dict[local_group_id],
                }

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
