# test onnx model
```bash
conda create -n mmdeploy_test python=3.8 && \
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia && \
pip install onnxruntime-gpu==1.8.1 && \
pip install openmim && \
mim install mmcv && \
mim install mmsegmentation && \
git clone https://github.com/open-mmlab/mmdetection3d.git && \
cd mmdetection3d && \
pip install -e . && \
cp -r mmdet3d /home/wind/miniconda3/envs/mmdeploy_test/lib/python3.8/site-packages/ && \
cd .. && \
rm -rf mmdetection3d
pip install rospkg

