# Copyright (c) windzu. All rights reserved.
"""
**Train**
```bash
# single GPU
python3 tools/mmdet3d_train.py ./configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus-4d.py

# multi GPU
./tools/mmdet3d_dist_train.sh ./configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus-4d.py 1

```

**Resume**
```bash
python3 tools/train.py ./configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus-4d.py \
--resume-from ./work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus-4d/latest.pth
```

**导出ONNX**
```bash
python3 tools/onnx_tools/centerpoint/export_onnx.py ./demo/data/nuscenes/test.bin \
./configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus-4d.py \
./work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus-4d/epoch_1.pth
```

"""

import os
"""
Datasets Settings
-----------------
关于数据集的设置，设置的内容包括:
    1. 数据集的路径
    2. 数据集的类型和类别
    3. 数据集的预处理方式(pipeline)
"""
data_root = os.path.join(os.environ['ADMLOPS'], 'data', 'nuscenes_v1.0_mini')
dataset_type = 'NuScenesDataset'
class_names = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=os.path.join(data_root, 'nuscenes_dbinfos_train.pkl'),
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5,
        ),
    ),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2,
    ),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args,
    ),
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False,
            ),
            dict(type='Collect3D', keys=['points']),
        ],
    ),
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_train.pkl'),
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_val.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'nuscenes_infos_val.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
)

evaluation = dict(interval=20, pipeline=eval_pipeline)
"""
Models Settings
---------------
关于模型的设置，设置的内容包括:
    1. 模型的类型
    2. 模型的结构细节
    3. 模型的损失函数等
"""
voxel_size = [0.2, 0.2, 8]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range,
    ),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=(512, 512),
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7,
            pc_range=point_cloud_range[:2],
        ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        )),
)
"""
Schedules Settings
---------------
关于训练时候Schedules的设置,设置的内容包括:
    1. optimizer 的设置
    2. lr_config 的设置
    3. runner 的设置
"""
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
"""
Runtime Settings
---------------
关于训练时候Runtime的设置,设置的内容包括:
    1. logger 的设置
    2. load_from 的设置 (加载预训练模型)
    3. workflow 的设置
"""
runner = dict(type='EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')],
)
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# load_from = os.path.join(
#     os.environ['ADMLOPS'],
#     'checkpoints',
#     'centerpoint',
#     'centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.pth',
# )
