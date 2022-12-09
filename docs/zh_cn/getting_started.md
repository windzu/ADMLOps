## 依赖

- Ubuntu 20.04 (其他发行版可能也可以，但是没有测试过)
- Python 3.8
- CUDA 11.3
- PyTorch 1.12.0

## 准备工作

`dataset` 和 `checkpoint` 是开发过程中需要长期维护的数据，为了保证操作的一致性，我们推荐将它们按照如下的名称和结构进行存放管理

**Dataset文件夹**

在一块大容量的独立的磁盘(或数据服务器)上创建一个 `Dataset` 文件夹，用于存放数据集，结构如下所示。其内部结构将在介绍数据集的章节详细介绍

```bash
Dataset
├── ApolloScape
├── BDD100K
├── COCO2017
├── nuScenes
├── KITTI
├── CULane
├── VOC
...
```

然后将该路径添加到系统环境变量中，例如将下面的内容添加到 `~/.bashrc` 中

```bash
export DATASET=xxx/Dataset
```

**ModelZoo文件夹**

在一块大容量的独立的磁盘(或数据服务器)上创建一个 `ModelZoo` 文件夹，用于存放模型的checkpoint，结构如下所示。其内部结构将在介绍checkpoint的章节详细介绍


```bash
ModelZoo
├── centerpoint
├── faster_rcnn
├── fcos3d
├── pointpillars
├── yolo
├── yolox
...
```

然后将该路径添加到系统环境变量中，例如将下面的内容添加到 `~/.bashrc` 中

```bash
export MODELZOO=xxx/ModelZoo
```

## 安装

### 开发环境

假设当前已按照上述依赖安装好了开发环境，接下来我们通过 conda 安装 ADMLOps的开发环境
> Note 目前仅提供了ADMLOps1.x的开发环境，2.x需要等 mmengine 完善再做跟进

**创建 conda 环境**

```bash
export CONDA_ENV_NAME=ADMLOps && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export TORCH_VERSION=1.12.0 && \
export MMCV_VERSION=1.6.2 && \
export MMDET_VERSION=2.25.1 && \
export MMSEG_VERSION=0.29.0 && \
export MMDET3D_VERSION=1.0.0rc4 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip install openmim && \
mim install mmcv-full==$MMCV_VERSION && \
mim install mmdet==$MMDET_VERSION && \
mim install mmsegmentation==$MMSEG_VERSION && \
mim install mmdet3d==$MMDET3D_VERSION
```

**安装 ADMLOps**

1. 拉取 ADMLOps 工程

    ```bash
    git clone https://github.com/windzu/ADMLOps.git 
    ```

2. 将工程路径添加到系统环境变量中，例如将下面的内容添加到 `~/.bashrc` 中

    ```bash
    export ADMLOPS=xxx/ADMLOps
    ```

3. 添加数据软链接

    > Note 数据软链接在docker环境下无效，需要使用docker挂载

    ```bash
    ln -s $DATASET $ADMLOPS/data
    ln -s $MODELZOO $ADMLOPS/checkpoints
    ```

4. 安装 ADMLOps

    ```bash
    export CONDA_ENV_NAME=ADMLOps && \
    conda activate $CONDA_ENV_NAME && \
    cd $ADMLOPS && \
    pip3 install -e .
    ```

### 部署环境

On The Way

## 验证

这里使用yolox-s的模型作为例子，验证ADMLOps的安装是否成功

1. 下载yolox-s的模型，[下载地址](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth)，将其放到 `ModelZoo/yolox` 文件夹下，并重命名为 `yolox_s_8x8_300e_coco.pth`

2. 执行如下命令，验证yolox-s的模型是否能够正常推理

    ```bash
    export CONDA_ENV_NAME=ADMLOps && \
    conda activate $CONDA_ENV_NAME && \
    cd $ADMLOPS && \
    python demo/image_demo.py demo/data/demo.jpg \
    configs/yolox/yolox_s_8x8_300e_coco.py \
    checkpoints/yolox/yolox_s_8x8_300e_coco.pth \
    --device cuda:0
    ```


## 其他

### Wandb

Wandb 是一个对大规模训练和远程监控很友好的工具，本工程的训练框架是基于 open-mmlab 系列的，而它们都支持 wandb，可以在训练过程中开启 Wandb 作为 logger ，以提高炼丹效率，具体开启方法可以参考[官方文档](https://docs.wandb.ai/guides/integrations/mmdetection)





