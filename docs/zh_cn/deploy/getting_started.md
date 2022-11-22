## Introduction

本文档是 ADMLOps 的 Deploy 模块的快速入门文档，主要介绍了如何使用 ADMLOps 的 Deploy 模块进行模型的部署。

## Environment Configuration

环境的搭建需要分两个步骤

- build docker image

- create docker container

### build docker image

ADMLOps的dockerfile是在mmdeploy的dockerfile基础上添加了一些内容，具体内容在mmdeploy_extension/docker/GPU下的dockerfile中，里面有详细的说明

```bash
cd $ADMLOPS_PATH/mmdeploy_extension && \
docker build docker/GPU -t mmdeploy:main \
--build-arg USE_SRC_INSIDE=true
```

### create docker container

为了实现ROS多机器间的通信，ADMLOps的docker-compose中做了一些与网络配置,具体内容在`mmdeploy_extension/docker-compose.yml`文件中有详细说明

但仅依靠docker-compose中的网络设置还无法达到多机器通信的效果，我们还需要对host做一些网络配置，具需要做的工作如下：

1. 创建docker bridge网络

2. docker-compose中指定网络

3. container与host进行ROS主从机配置

**创建docker bridge网络**

- 网络类型：`bridge`
- 网络名称：`admlops`
- 网段：`172.28.0.0`

```bash
docker network create \
  --driver=bridge \
  --subnet=172.28.0.0/16 \
  admlops
```

**docker-compose中指定网络**

```bash
services:
  mmdeploy:
    networks:
      - admlops

# use already existing network
networks:
  admlops:
    external: true
```

**container与host进行ROS主从机配置**

> Note 因为container只有一个ip 所以直接获取其地址填入中，而不是通过手动分配固定地址的方式

```bash
# 创建container
cd $ADMLOPS_PATH/mmdeploy_extension && \
docker-compose up -d

# 1. 进入container
docker exec -it mmdeploy /bin/bash

# 2. 设置主从机
echo export ROS_IP=`hostname -I` >> ~/.bashrc && \
echo export ROS_HOSTNAME=`hostname -I` >> ~/.bashrc && \
echo export ROS_MASTER_URI=http://172.28.0.1:11311 >> ~/.bashrc
```

## 基本使用

基本使用包括:

- 模型转换
- 模型评估
- 模型推理

### 模型转换

通过 `tools/deploy.py` 脚本支持将模型转换为 `TensorRT` , `ONNX` 模型，转换后的模型可以用于推理, 详细参数如下:

```bash
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
```

**参数描述**

- `deploy_cfg` : 针对此模型的部署配置，包含推理框架类型、是否量化、输入 shape 是否动态等, 详细配置请参考 [mmdeploy 配置](./mmdeploy_config.md)
- `model_cfg` : 模型的配置文件路径
- `checkpoint` : 模型的checkpoint路径
- `img` : 模型转换时，用做测试的图像或点云文件路径。
- `--test-img` : 用于测试模型的图像文件路径。默认设置成None。
- `--work-dir` : 工作目录，用来保存日志和模型文件。
- `--calib-dataset-cfg` : 此参数只有int8模式下生效，用于校准数据集配置文件。若在int8模式下未传入参数，则会自动使用模型配置文件中的’val’数据集进行校准。
- `--device` : 用于模型转换的设备。 默认是 `cpu`，对于 trt 可使用 `cuda:0` 这种形式。
- `--log-level` : 设置日记的等级，选项包括'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'。 默认是INFO。
- `--show` : 是否显示检测的结果。
- `--dump-info` : 是否输出 SDK 信息。

### 模型评估

通过 `tools/deploy_eval.py` 脚本评估转换后的模型的在给定的输入下的运行速度, 详细参数如下:

```bash
python tools/test.py \
${DEPLOY_CFG} \
${MODEL_CFG} \
--model ${BACKEND_MODEL_FILES} \
[--speed-test] \
[--warmup ${WARM_UP}] \
[--log-interval ${LOG_INTERVERL}] \
[--log2file ${LOG_RESULT_TO_FILE}]
```

**参数描述**

- `deploy_cfg` : 针对此模型的部署配置，包含推理框架类型、是否量化、输入 shape 是否动态等, 详细配置请参考 [mmdeploy 配置](./mmdeploy_config.md)
- `model_cfg` : 模型的配置文件路径
- `--model` : 转换后的模型文件路径
- `--speed-test` : 是否做速度测试
- `--warm-up` : 执行前是否 warm-up
- `--log-interval` : 日志打印间隔
- `--log2file` : 保存日志和运行文件的路径

### 模型推理

On the way
