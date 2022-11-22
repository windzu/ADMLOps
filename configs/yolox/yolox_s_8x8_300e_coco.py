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
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
img_scale = (640, 640)  # height, width

# 配置 pipeline 和 dataset
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0)),
            ),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            classes=class_names,
            ann_file=os.path.join(data_root, 'annotations',
                                  'instances_train.json'),
            img_prefix=os.path.join(data_root, 'train'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            filter_empty_gt=False,
        ),
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
"""
Models Settings
---------------
关于模型的设置，设置的内容包括:
    1. 模型的类型
    2. 模型的结构细节
    3. 模型的损失函数等
"""
model = dict(
    type='YOLOX',
    input_size=img_scale,  # 网络输入尺寸
    random_size_range=(15, 25),  # 多尺度训练时随机乘以的尺度范围
    random_size_interval=10,  # 多尺度训练时在尺度变化中变化的间隔
    backbone=dict(
        # CSPDarknet 默认使用 P5 结构
        # 其结构如下：
        # [
        #   p4 [512, 1024, 3, False, True]
        #   p3 [256, 512, 9, True, False],
        #   p2 [128, 256, 9, True, False],
        #   p1 [64, 128, 3, True, False],
        # ]
        # 其中，每个元素的含义为：
        # [ in_channels, out_channels, num_blocks, add_identity, use_spp ]
        type='CSPDarknet',
        deepen_factor=0.33,  # 深度因子，用于调整 CSPDarknet 的深度
        widen_factor=0.5,  # 宽度因子，用于调整 CSPDarknet 的宽度
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[
            128,
            256,
            512,
        ],  # 输入的每个scale的channel数,与backbone的out_channels对应
        out_channels=128,  # 输出的channel数
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=len(class_names),
        in_channels=128,  # neck的输出channel数
        feat_channels=128,  # 在stacked conv中间的channel数
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
)
"""
Schedules Settings
---------------
关于训练时候Schedules的设置,设置的内容包括:
    1. optimizer 的设置
    2. lr_config 的设置
    3. runner 的设置
"""
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,  # default: False 不知道什么意思
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),  # 不知道什么意思
)
optimizer_config = dict(grad_clip=None)

num_last_epochs = 15
resume_from = None
interval = 10  # default: 10 使用1快速测试eval是否有问题

# learning policy
lr_config = dict(
    policy='YOLOX',  # 不知道什么意思
    warmup='exp',  # default: linear
    by_epoch=False,  # 不知道什么意思
    warmup_by_epoch=True,  # 不知道什么意思
    warmup_ratio=1,  # default: 0.001
    warmup_iters=5,  # 5 epoch
    num_last_epochs=15,  # 不知道什么意思
    min_lr_ratio=0.05,  # 不知道什么意思
)

runner = dict(type='EpochBasedRunner', max_epochs=300)  # default: 12
"""
Runtime Settings
---------------
关于训练时候Runtime的设置,设置的内容包括:
    1. logger 的设置
    2. load_from 的设置 (加载预训练模型)
    3. workflow 的设置
"""
checkpoint_config = dict(interval=interval)  # default: 1
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(300 - num_last_epochs, 1)],
    metric='bbox',
)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')],
)
# yapf:enable
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48,
    ),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49,
    ),
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (x GPUs) x (y samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

load_from = os.path.join(
    os.environ['ADMLOPS'],
    'checkpoints',
    'yolox',
    'yolox_s_8x8_300e_coco.pth',
)
