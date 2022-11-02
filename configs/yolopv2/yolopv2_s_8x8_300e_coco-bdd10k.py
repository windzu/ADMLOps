import os

###########################################
########### datasets settings #############
###########################################
data_root = os.path.join(os.environ["ADMLOPS"], "data", "COCO_BDD10K")
dataset_type = "CocoDataset"
class_names = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
img_scale = (640, 640)  # height, width

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=os.path.join(data_root, "annotations", "instances_train.json"),
        img_prefix=os.path.join(data_root, "train"),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=os.path.join(data_root, "annotations", "instances_val.json"),
        img_prefix=os.path.join(data_root, "val"),
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=os.path.join(data_root, "annotations", "instances_val.json"),
        img_prefix=os.path.join(data_root, "val"),
        pipeline=test_pipeline,
    ),
)
evaluation = dict(metric=["bbox", "segm"])


###########################################
############ models settings ##############
###########################################
# model settings
model = dict(
    type="YOLOPV2",
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
        type="CSPDarknet",
        deepen_factor=0.33,  # 深度因子，用于调整 CSPDarknet 的深度
        widen_factor=0.5,  # 宽度因子，用于调整 CSPDarknet 的宽度
    ),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[128, 256, 512],  # 输入的每个scale的channel数,与backbone的out_channels对应
        out_channels=128,  # 输出的channel数
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=len(class_names),
        in_channels=128,  # neck的输出channel数
        feat_channels=128,  # 在stacked conv中间的channel数
    ),
    drivable_head=dict(
        type="FCNMaskHead",  # 通过FCN进行mask预测
        num_convs=4,  # 4个卷积层
        in_channels=128,  # 输入特征的通道数
        conv_out_channels=256,  # 卷积层输出的通道数
        num_classes=80,
        loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
    ),
    lane_head=dict(
        type="FCNMaskHead",  # 通过FCN进行mask预测
        num_convs=4,  # 4个卷积层
        in_channels=128,  # 输入特征的通道数
        conv_out_channels=256,  # 卷积层输出的通道数
        num_classes=80,
        loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)


###########################################
########### schedules settings ############
###########################################
# optimizer
optimizer = dict(
    type="SGD",
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,  # default: False 不知道什么意思
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),  # 不知道什么意思
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)
runner = dict(type="EpochBasedRunner", max_epochs=12)

###########################################
############ runtime settings #############
###########################################
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"  # 控制log的输出级别
load_from = None
resume_from = None
workflow = [("train", 1)]  # 1个epoch的workflow，即一个epoch进行一个epoch的train

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


load_from = os.path.join(
    os.environ["ADMLOPS"],
    "checkpoints",
    "mask_rcnn",
    "mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
)
