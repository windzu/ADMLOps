FROM nvcr.io/nvidia/tensorrt:22.04-py3

ARG CUDA=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG ONNXRUNTIME_VERSION=1.8.1
ARG MMCV_VERSION=1.5.3
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
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c pytorch
ENV PATH /opt/conda/bin:$PATH

### install mmcv-full
RUN /opt/conda/bin/pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html

WORKDIR /root/workspace
### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}

### cp trt from pip to conda
RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/

### install mmdeploy
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ARG VERSION
RUN git clone https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
    git submodule update --init --recursive &&\
    mkdir -p build &&\
    cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. &&\
    make -j$(nproc) &&\
    cd .. &&\
    pip install -e .

### build sdk
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

RUN cd /root/workspace/mmdeploy &&\
    rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc &&\
    mkdir -p build && cd build &&\
    cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
    -DMMDEPLOY_CODEBASES=all &&\
    make -j$(nproc) && make install &&\
    export SPDLOG_LEVEL=warn &&\
    if [ -z ${VERSION} ] ; then echo "Built MMDeploy master for GPU devices successfully!" ; else echo "Built MMDeploy version v${VERSION} for GPU devices successfully!" ; fi

ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"


######################################################################################################
################################# additional steps for mmdeploy_extension ############################
######################################################################################################
# set python lib path
ENV PYTHONPATH=/root/workspace/mmdeploy/build/lib:$PYTHONPATH

# install ros
RUN apt update && apt install -y lsb-release && \
    sh -c '. /etc/lsb-release && echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && apt update && \
    apt install -y ros-noetic-desktop-full && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# install common libs or tools
# - apt install: ros-xx-msgs
# - pip install rospkg wadda pycuda
# - wadda
RUN apt update && apt install -y ros-noetic-autoware-msgs ros-noetic-foxglove-msgs && \
    /opt/conda/bin/pip install rospkg wadda pycuda

# install open-mmlab serial frameworks
# - min install mmdet
# - min install mmsegmentation
# - build install mmdet3d from source (pip or mim install will cause error)
# - pip install pycuda
RUN /opt/conda/bin/pip install -U openmim && \
    /opt/conda/bin/mim install mmdet && \
    /opt/conda/bin/mim install mmsegmentation && \
    cd /root/workspace && git clone https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && /opt/conda/bin/pip install -e .
