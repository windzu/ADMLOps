# Introduction

`ADMLOps`意为用于自动驾驶感知任务的`MLOps`

自动驾驶感知任务应该是一个闭环任务，其生命周期应该至少包括`数据`、`模型`和`部署`三个阶段。 因此，在MLOps的设计过程中，至少需要涵盖这三个方向。

## Data

数据对于深度学习的重要性不言而喻，一个好的数据集应该兼顾“深度”和“广度”。

* 广度 : 可以借助开源数据集 + 自我采集

* 深度 : 需要结合实际测试，针对性分析，重新收集来完成，这类数据一般也被称为`Corner Case`

### Dataset

* 开源数据集
  
  * 采集：采集和下载自动驾驶相关数据集
  
  * 分析：对采集到的数据集进行数据分析，评估其包含的数据分布是否满足当前任务的需求
  
  * 总结整理：对于有用的开源数据，按照任务的类别进行分类，与同一任务的数据集一起管理
  
  * 转换：根据流水线的需要，开发相应的转换脚本，转换成流水线需要的数据集格式

* 自定义数据
  
  * 数据集定义：根据任务需要自定义所需数据，包括其类别、标注标准、标注格式等。
  
  * 数据采集：开发数据采集工具，需要满足上述定义要求，需要考虑数据筛选、持续采集等任务的需要
  
  * 数据标注：根据定义，写出详细的标注需求，然后交给数据标注公司
  
  * 分析、归纳、转化：参考开源数据集中的描述

### 数据集迭代

多次采集数据不可能解决所有场景问题，因为自动驾驶任务中的极端情况很难一次性解决，需要根据实际测试情况不断采集新的数据补充已有的 . 数据集逐渐覆盖尽可能多的场景

### Dataset Pipeline Design

随着自动驾驶行业的发展，必然会出现更多性能更好的模型，这些模型加载数据的方式很可能会与之前的通用方式有所不同，比如不久前流行的BEV方案。 因此，为了适应这些模型，需要编写相应的加载数据的管道，以便在现有数据集的基础上快速完成新模型的训练和测试。

## Modeling

模型部分基于open-mmlab实验室开源的一系列优秀框架，它可以轻松复现过去的一些经典模型，并在其基础上构建了一个新模型来验证自己的想法，感谢 open-mmlab

### 模型复现 & 框架扩展 & 网络重构

根据论文和开源代码等信息，在ADMLOps中复制了一些自动驾驶领域的经典和前沿模型。 对于一些比较新的或者不是主流的网络结构，open-mmlab 还没有支持。 这些网络结构需要手动实现，我通常在扩展模块中编写这些实现。 另外，如果想验证自己的一些想法，可以通过简单的修改配置文件来重构模型结构，从而快速完成一些实验

### 训练 & 测试 & 评估

方便训练模型，通过测试和评估的结果帮助我们评估模型的效果和数据集的效果

## Deploy

模型的部署是一个完整的工程问题，而部署中最大的问题往往是所使用的部署平台提供的operator不支持部署网络中的一些operator。

尽管如此，我们还是尽量基于一些成熟的部署框架，比如tensorrt、openvino等，这将大大简化部署任务。

如果要进一步优化部署，还需要知识蒸馏、剪枝等部署技巧。

# Preparation

> Note 为了保证操作的一致性，请按照下述步骤完成相关的准备工作

### 数据管理

在磁盘中按照如下名称和结构设置两个文件夹，分别用于存储数据集和模型

**Dataset文件夹**

```bash
Dataset
├── ApolloScape
├── BDD100K
├── COCO
├── nuScenes
├── KITTI
├── CULane
├── VOC
...
```

**ModelZoo文件夹**

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

# General Configuration

## 拉取工程

```bash
git clone https://github.com/windzu/ADMLOps.git --depth 1 
```

## 添加环境变量

> Note 根据自己的shell选择`.bashrc` 或者`.zshrc`,将如下环境变量添加到其中

```bash
export ADMLOPS=xxx/admlops
export DATASET=xxx/Dataset
export MODELZOO=xxx/ModelZoo
```

## 添加数据软链接

> Note 数据软链接在docker环境下无效，需要使用docker挂载

```bash
ln -s $DATASET $ADMLOPS/data
ln -s $MODELZOO $ADMLOPS/checkpoints
```

至此准备工作全部完成 ！！！

## Development Environment Setup

> Note 
> 
> open-mmlab系列的框架存在依赖关系，如mmdetection3d依赖mmdetection，但是他们之间的兼容性并不是很好，往往有着严格的版本依赖，所以出现了openmim这样用来解决依赖的工具。但即便如此，随着各个版本的更新速度不一，依然存在版本不兼容的bug，所以为了能构建一个稳定的开发环境，最好基于某个稳定的版本，并记录下该版本号，在安装时候指定版本号以确保不在开发环境上浪费过多时间，下面给出的是个人的一些 best practice

### ADMLOps1.0

- python : 3.8
- cuda : 11.3
- torch : 1.12.0
- mmcv : 1.6.2
- mmdet : 2.25.1
- mmdet3d : 1.0.0rc4

```bash
export CONDA_ENV_NAME=ADMLOps1.0 && \
export PYTHON_VERSION=3.8 && \
export CUDA_VERSION=11.3 && \
export TORCH_VERSION=1.12.0 && \
export MMCV_VERSION=1.6.2 && \
export MMDET_VERSION=2.25.1 && \
export MMDET3D_VERSION=1.0.0rc4 && \
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
pip install openmim && \
mim install mmcv-full==$MMCV_VERSION && \
mim install mmdet==$MMDET_VERSION && \
mim install mmsegmentation && \
mim install mmdet3d==$MMDET3D_VERSION
```

### ADMLOps2.0

待mmegine完善后，将会基于mmegine构建ADMLOps2.0

### Local Development

将本地开发的内容安装到环境中，建议使用`pip install -e .`的方式，这样可以在本地修改代码后，不需要重新安装

```bash
export CONDA_ENV_NAME=ADMLOps1.0 && \
conda activate $CONDA_ENV_NAME && \
cd $ADMLOPS && \
pip install -v -e . 
```

### Wandb

open-mmlab系列的框架都支持wandb，这是一个对大规模训练和远程监控很友好的工具，可以在训练过程中实时查看训练过程中的loss，acc等指标，以及训练过程中的图片，模型等，具体使用方法可以参考[官方文档](https://docs.wandb.ai/quickstart)
启用此功能需要在config文件中的hook中增加wandb的配置，具体使用见[文档](https://docs.wandb.ai/guides/integrations/mmdetection)


## Tutorials
为了方便大家快速上手使用以及提高对本工程的理解，本工程还提供了一系列Tutorials，一般是结合自动驾驶中常见的任务而展开的，希望能给大家提供一些思路

## Last but not least
如果在使用过程中发现bug或者文档错误，非常欢迎您能提`issus`或`pr`，我将在收到通知后的第一时间尽快修复

如果觉得本工程对您有帮助，希望能给一个star，我将会非常开心，感谢！