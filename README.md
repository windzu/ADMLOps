# ADMLOps
MLOps for autonomous driving perception tasks


## Quicl Start

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
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y && \
conda activate $CONDA_ENV_NAME && \
conda install pytorch=$TORCH_VERSION torchvision torchaudio cudatoolkit=$CUDA_VERSION -c pytorch -y && \
export MMCV_VERSION=1.6.2 && \
export MMDET_VERSION=2.25.1 && \
export MMDET3D_VERSION=1.0.0rc4 && \
pip install openmim && \
mim install mmcv-full==$MMCV_VERSION && \
mim install mmdet==$MMDET_VERSION && \
mim install mmsegmentation && \
mim install mmdet3d==$MMDET3D_VERSION
```