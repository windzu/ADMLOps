# BDD100K

## Introduction

* [官网](https://www.bdd100k.com/)
* [下载地址](https://bdd-data.berkeley.edu/portal.html#download)

BDD100K数据集是一个面向自动驾驶任务的`纯图像`数据集，包含了多种任务的数据，包括目标检测，目标跟踪，全景分割等。

包含的任务如下：
* Detection
* Instance Segmentation
* Pose Estimation
* Panoptic Segmentation
* Semantic Segmentation
* Drivable Area
* Lane Marking
* Multiple Object Tracking
* Multi Object Tracking and Segmentation (Segmentation Tracking)

其数据包含两个部分
* BDD100K数据集：共包含`90k`张图像，其中训练集包含了`70k`张图像，测试集和验证集合各包含了`10k`张图像。其覆盖的任务如下：
    * Detection
    * Drivable Area
    * Lane Marking
    * Pose Estimation
* BDD10K数据集：这部分算是随着自动驾驶任务的拓展而更新拓展数据集，其包含了`10k`张图像，其中训练集包含了`7k`张图像，测试集和验证集合各包含了`1k`张图像。其覆盖的任务如下：
    * Detection
    * Instance Segmentation
    * Panoptic Segmentation
    * Semantic Segmentation


数据的组织结构如下：
> Note 没有列出所以任务的数据组织结构
```bash
BDD100K
├── images # 所有图像数据
│   ├── 100k
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── 10k 
│       ├── test
│       ├── train
│       └── val
├── labels
│   ├── bdd100k_labels_images_train.json
│   ├── bdd100k_labels_images_val.json
│   ├── det_20 # 2020年针对100k修正后的2D检测label
│   │   ├── det_train.json
│   │   └── det_val.json
│   ├── drivable
│   │   ├── colormaps
│   │   ├── masks
│   │   ├── polygons
│   │   └── rles
│   ├── ins_seg
│   │   ├── bitmasks
│   │   ├── colormaps
│   │   ├── polygons
│   │   └── rles
│   ├── lane
│   │   ├── colormaps
│   │   ├── masks
│   │   └── polygons
│   ├── pan_seg
│   │   ├── bitmasks
│   │   ├── colormaps
│   │   └── polygons
│   └── sem_seg
│       ├── colormaps
│       ├── masks
│       ├── polygons
│       └── rles
...

```


## Usage

### Format

BDD100K官方提供了数据格式转换的工具和文档，具体可以参考 [官方格式转换文档](https://doc.bdd100k.com/format.html#format-conversion)

下面是一些具体的转换示例

**instance segmentation to coco**
> Note 转换过程中丢弃了`traffic light` 和 `traffic sign`两个类别，转换后的label中包含bbox 和seg信息
```bash
# train
python3 -m bdd100k.label.to_coco -m ins_seg \
-i $DATASET/BDD100K/labels/ins_seg/rles/ins_seg_train.json \
-o $DATASET/COCO_BDD10K/annotations/instances_train.json

# val
python3 -m bdd100k.label.to_coco -m ins_seg \
-i $DATASET/BDD100K/labels/ins_seg/rles/ins_seg_val.json \
-o $DATASET/COCO_BDD10K/annotations/instances_val.json
```

## Make Dataset