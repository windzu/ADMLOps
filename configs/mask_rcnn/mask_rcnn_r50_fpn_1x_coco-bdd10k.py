# Copyright (c) windzu. All rights reserved.
import os
"""
Datasets Settings
-----------------
关于数据集的设置，设置的内容包括:
    1. 数据集的路径
    2. 数据集的类型和类别
    3. 数据集的预处理方式(pipeline)
"""
data_root = os.path.join(os.environ['ADMLOPS'], 'data', 'COCO_BDD10K')
dataset_type = 'CocoDataset'
class_names = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=os.path.join(data_root, 'annotations',
                              'instances_train.json'),
        img_prefix=os.path.join(data_root, 'train'),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=os.path.join(data_root, 'annotations', 'instances_val.json'),
        img_prefix=os.path.join(data_root, 'val'),
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=os.path.join(data_root, 'annotations', 'instances_val.json'),
        img_prefix=os.path.join(data_root, 'val'),
        pipeline=test_pipeline,
    ),
)
evaluation = dict(metric=['bbox', 'segm'])
"""
Models Settings
---------------
关于模型的设置，设置的内容包括:
    1. 模型的类型
    2. 模型的结构细节
    3. 模型的损失函数等
"""
# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',  # backbone 类型为 ResNet
        depth=50,  # ResNet常用深度为{18, 34, 50, 101, 152}，这里使用50层
        num_stages=4,  # 其对应 (Bottleneck, (3, 4, 6, 3))
        out_indices=(0, 1, 2, 3),  # 输出的 stage 索引
        frozen_stages=1,  # 冻结的 stage 索引
        norm_cfg=dict(type='BN', requires_grad=True),  # BN 层配置
        norm_eval=True,  # eval时是否冻结 BN 层
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN',  # neck 类型为 FPN
        in_channels=[256, 512, 1024, 2048],  # 每个尺度下输入的通道数，这里与 ResNet50 对应
        out_channels=256,  # 每个尺度下输出的通道数
        num_outs=5,  # 输出的特征图数量
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,  # 输入feat的通道数，与 FPN 输出的通道数对应
        feat_channels=256,  # RPNHead 中间层的通道数
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(  # 根据 RoI 从特征图中提取 RoI 特征
            # 从单尺度的特征图中提取 RoI，如果输入是多尺度的特征图，则根据其对应的尺度映射提取
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',  # 使用 RoIAlign 对齐所提取的特征图
                output_size=7,  # 输出的特征图大小为 7x7
                sampling_ratio=0,  # 采样率
            ),
            out_channels=256,  # 输出的通道数为 256
            featmap_strides=[4, 8, 16, 32],  # 对应的特征图的 stride
        ),
        bbox_head=dict(  # bbox_head 包含两个分支，一个是分类分支，一个是回归分支
            #                             /-> cls convs -> cls fcs -> cls
            # shared convs -> shared fcs
            #                             \-> reg convs -> reg fcs -> reg
            # 使用shared head,包含 0个conv -> 2个fc,然后分别进行分类和回归
            type='Shared2FCBBoxHead',
            in_channels=256,  # 输入的特征图通道数为 256，与 bbox_roi_extractor 输出的通道数对应
            fc_out_channels=1024,  # fc层的输出通道数
            roi_feat_size=7,  # RoI 特征图大小为 7x7
            num_classes=len(class_names),  #
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type='FCNMaskHead',  # 通过FCN进行mask预测
            num_convs=4,  # 4个卷积层
            in_channels=256,  # 输入特征的通道数
            conv_out_channels=256,  # 卷积层输出的通道数
            num_classes=len(class_names),
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(  # rpn网络中proposal的配置
            # nms操作前，通过score筛选的proposal数量，然后取前topk个有效结果，这里topk为2000
            nms_pre=2000,
            max_per_img=1000,  # nms操作后，最终保留的proposal数量，这里为1000
            nms=dict(type='nms', iou_threshold=0.7),  # nms操作中的iou阈值
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        ),
    ),
)
"""
Schedules Settings
---------------
关于训练时候Schedules的设置,设置的内容包括:
    1. optimizer 的设置
    2. lr_config 的设置
    3. runner 的设置
"""
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)
runner = dict(type='EpochBasedRunner', max_epochs=12)
"""
Runtime Settings
---------------
关于训练时候Runtime的设置,设置的内容包括:
    1. logger 的设置
    2. load_from 的设置 (加载预训练模型)
    3. workflow 的设置
"""
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

load_from = os.path.join(
    os.environ['ADMLOPS'],
    'checkpoints',
    'mask_rcnn',
    'mask_rcnn_r50_fpn_1x_coco.pth',
)
