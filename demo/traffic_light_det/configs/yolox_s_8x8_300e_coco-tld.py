import os

###########################################
########### datasets settings #############
###########################################
# NOTE ： 这里使用多个coco数据集
data_root = os.path.join(os.environ["ADMLOPS"], "data", "COCO_TLD")
dataset_list = ["bstld", "huawei", "shanjiaoke", "weitang"]
# 遍历指定的数据集，将数据集组合成一个数据集列表
train_ann_file_list = [
    os.path.join(data_root, dataset, "annotations", "instances_train.json") for dataset in dataset_list
]
train_img_prefix_list = [os.path.join(data_root, dataset, "train") for dataset in dataset_list]

val_ann_file_list = [os.path.join(data_root, dataset, "annotations", "instances_val.json") for dataset in dataset_list]
val_img_prefix_list = [os.path.join(data_root, dataset, "val") for dataset in dataset_list]

dataset_type = "CocoDataset"
class_names = [
    "green_circle",
    "green_arrow_left",
    "green_arrow_straight",
    "green_arrow_right",
    "red_circle",
    "red_arrow_left",
    "red_arrow_straight",
    "red_arrow_right",
    "yellow_circle",
    "yellow_arrow_left",
    "yellow_arrow_straight",
    "yellow_arrow_right",
    "off",
    "unkown",
]

img_scale = (640, 640)  # height, width

# 配置 pipeline 和 dataset
train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(type="RandomAffine", scaling_ratio_range=(0.1, 2), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=train_ann_file_list,
        img_prefix=train_img_prefix_list,
        pipeline=[dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True)],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)


test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,  # 使用了数据增强,在通用的数据集配置上加了一个wrapper
    val=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=val_ann_file_list,
        img_prefix=val_img_prefix_list,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=class_names,
        ann_file=val_ann_file_list,
        img_prefix=val_img_prefix_list,
        pipeline=test_pipeline,
    ),
)


###########################################
############ models settings ##############
###########################################
model = dict(
    type="YOLOX",
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.5),
    neck=dict(type="YOLOXPAFPN", in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(type="YOLOXHead", num_classes=len(class_names), in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)

###########################################
########### schedules settings ############
###########################################
# default 8 gpu
optimizer = dict(
    type="SGD",
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
    policy="YOLOX",  # 不知道什么意思
    warmup="exp",  # default: linear
    by_epoch=False,  # 不知道什么意思
    warmup_by_epoch=True,  # 不知道什么意思
    warmup_ratio=1,  # default: 0.001
    warmup_iters=5,  # 5 epoch
    num_last_epochs=15,  # 不知道什么意思
    min_lr_ratio=0.05,  # 不知道什么意思
)

runner = dict(type="EpochBasedRunner", max_epochs=300)  # default: 12


###########################################
############ runtime settings #############
###########################################
checkpoint_config = dict(interval=interval)  # default: 1
evaluation = dict(
    save_best="auto",
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(300 - num_last_epochs, 1)],
    metric="bbox",
)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
# yapf:enable
custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="SyncNormHook", num_last_epochs=num_last_epochs, interval=interval, priority=48),
    dict(type="ExpMomentumEMAHook", resume_from=None, momentum=0.0001, priority=49),
]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (x GPUs) x (y samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

load_from = os.path.join(
    ADMLOPS_PATH, "checkpoints", "mmdet", "yolox", "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
)
