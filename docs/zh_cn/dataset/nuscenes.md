# nuScenes

## Introduction

* [官网]()

## Usage

### Format

**to open-mmlab**
在open-mmlab的mmdetection3d中，使用nuscenes数据集前需要将其转换为pkl格式，官方文档参考[这里](https://mmdetection3d.readthedocs.io/en/latest/data_preparation.html#nuscenes)
在本工程中转换使用如下命令：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuScenes --out-dir ./data/nuScenes --extra-tag nuscenes
```

## Make Dataset