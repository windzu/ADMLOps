# Copyright (c) windzu. All rights reserved.
import os
"""
Datasets Settings
-----------------
- 数据集的基础设置,设置的内容包括:
    - 数据集类型
    - 数据集路径
    - 数据集类别
    - 使用的点云范围
    - 使用的传感器数据
    - 文件的存储client
- 数据集的pipeline设置,设置的内容包括:
    - db_sampler : 数据集的采样设置,用来辅助解决数据集不平衡的问题
    - train_pipeline : 训练数据集的预处理方式
    - test_pipeline : 测试数据集的预处理方式,可能会多出一些适用于OTA数据增强的操作
    - eval_pipeline : 评估数据集的预处理方式，一般不涉及数据增强
"""
data_root = os.path.join(os.environ['ADMLOPS'], 'data', 'test')
dataset_type = 'KittiExtensionDataset'
class_names = [
    'small_vehicles',
    'big_vehicles',
    'pedestrian',
    'bicyclist',
    'traffic_cone',
    'others',
]
point_cloud_range = [-60, -60, -3, 60, 60, 3]
input_modality = dict(use_lidar=True, use_camera=False)
# NOTE : sampler的设置需要和class对应
db_sampler = dict(
    type="DataBaseSampler",
    data_root=data_root,
    info_path=os.path.join(data_root, 'kitti_dbinfos_train.pkl'),
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        # 设置各个类别滤除的最小点数量,根据数据集的情况进行设置
        # 一般来说,较大的目标需要更多的点
        filter_by_min_points=dict(
            small_vehicles=5,
            big_vehicles=5,
            pedestrian=5,
            bicyclist=5,
            traffic_cone=5,
            others=5,
        ),
    ),
    classes=class_names,
    # 设置单帧中各个类别的最大采样数量,如果多了则剔除一些，如果少了则采用伪造数据
    sample_groups=dict(
        small_vehicles=15,
        big_vehicles=15,
        pedestrian=15,
        bicyclist=15,
        traffic_cone=15,
        others=15,
    ),
)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args,
    ),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,  # 默认为True,但是这里没有2d数据
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
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
        load_dim=4,
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
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=os.path.join(data_root, 'kitti_infos_train.pkl'),
            split='training',
            only_lidar=True,  # 只使用LiDAR数据
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'kitti_infos_val.pkl'),
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'kitti_infos_val.pkl'),
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
)

evaluation = dict(interval=1, pipeline=eval_pipeline)
"""
Models Settings
---------------
关于模型的设置，设置的内容包括:
    1. 模型的类型
    2. 模型的结构细节
    3. 模型的损失函数等
"""
voxel_size = [0.2, 0.2, 6]  # 与 point_cloud_range 相关
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
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        ],
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
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car']),
        ],
        common_heads=dict(
            # kitti数据不包含速度 所以去掉了 vel=(2, 2)
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            # 对中心点的范围限制，可以比设置的点云范围稍大一些
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7,  # 不包含速度是7个参数
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
            grid_size=[
                int((point_cloud_range[3] - point_cloud_range[0]) /
                    voxel_size[0]),
                int((point_cloud_range[4] - point_cloud_range[1]) /
                    voxel_size[1]),
                int((point_cloud_range[5] - point_cloud_range[2]) /
                    voxel_size[2])
            ],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # 去掉了速度的权重
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
            out_size_factor=4,  # 未知
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
