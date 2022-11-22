FROM nvcr.io/nvidia/l4t-ml:r35.1.0-py3


ARG CUDA=11.4
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.12.0
ARG TORCHVISION_VERSION=0.13.0
ARG ONNXRUNTIME_VERSION=1.12.0
ARG MMCV_VERSION=1.6.2
ARG PPLCV_VERSION=0.7.0
ENV FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

### change the system source for installing libs
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
    sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    echo "Use aliyun source for installing libs" ; \
    else \
    echo "Keep the download source unchanged" ; \
    fi

### update apt and install libs
# - vim
# - libssl-dev
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

### pip3 install libs
RUN pip3 install pyyaml numpy ipython cython typing typing_extensions ninja

RUN pip3 install -U openmim && \
    mim install mmcv-full

### install mmcv-full
# RUN pip3 install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html
