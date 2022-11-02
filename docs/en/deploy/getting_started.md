# Introduction

On The Way

## Environment Configuration

环境的搭建需要分两个步骤

* build docker image 

* create docker container

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

# use alread existing network
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