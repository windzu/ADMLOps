import os

admlops_path = os.environ["ADMLOPS_PATH"]

###########################################
########### datasets settings #############
###########################################
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# used classes:
class_names = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]

dataset_type = "USDDataset"
infos_prefix = "usd"

# choose different data for training and testing
data_root = admlops_path + "/data/mmdet3d/USD_Apollo/"
data_root = admlops_path + "/data/mmdet3d/USD_Apollo_SUB/"

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)

file_client_args = dict(backend="disk")
# 修改nus 训练阶段的 pipeline
# * 删除nus专属的 LoadPointsFromMultiSweeps
# * 修改默认的 ann_file 路径
train_pipeline = [
    dict(
        type="LoadPointsFromFileExtension",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, file_client_args=file_client_args),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFileExtension",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, file_client_args=file_client_args),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="GlobalRotScaleTrans", rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]
eval_pipeline = [
    dict(
        type="LoadPointsFromFileExtension",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["points"]),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + infos_prefix + "_" + "infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + infos_prefix + "_" + "infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + infos_prefix + "_" + "infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)
evaluation = dict(interval=1, pipeline=eval_pipeline)  # interval=24 or 1


###########################################
############ models settings ##############
###########################################
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.25, 0.25, 8]
model = dict(
    type="MVXFasterRCNN",
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000),  # max_voxels=(30000, 40000),
        deterministic=False,
    ),
    pts_voxel_encoder=dict(
        type="HardVFE",
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=[400, 400],
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    pts_neck=dict(
        type="FPN",
        norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
        act_cfg=dict(type="ReLU"),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3,
    ),
    pts_bbox_head=dict(
        type="Anchor3DHead",
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [2.5981, 0.8660, 1.0],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.0],  # 1 / sqrt(3)
                [1.0, 1.0, 1.0],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],  # 默认生成的3danchor的为7位编码,为了与9位编码对应，这里尾部补全两个0
            rotations=[0, 1.57],
            reshape_out=True,
        ),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder", code_size=9),  # 3dbbox默认编码为7，这里改为9
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],  # 与9位编码对应
            pos_weight=-1,
            debug=False,
        )
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500,
        )
    ),
)

###########################################
########### schedules settings ############
###########################################
# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy="step", warmup="linear", warmup_iters=1000, warmup_ratio=1.0 / 1000, step=[20, 23])
momentum_config = None

###########################################
############ runtime settings #############
###########################################
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=30)  # max_epochs = 24

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
resume_from = None
workflow = [("train", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"


load_from = (
    mmlab_extension_path
    + "/mmdetection3d/checkpoints/pointpillars/"
    + "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
)
