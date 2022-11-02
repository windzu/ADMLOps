# Jeston

---

对于 Jeston 系列计算设备的部署，目前支持的型号与JetPack 版本如下表

> Note 通过 `sudo apt-cache show nvidia-jetpack` 可以查看 JetPack 版本

| Device            | Version                                                        |     |
| ----------------- | -------------------------------------------------------------- | --- |
| Jeston AGX Xavier | [4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461) |     |
| Jeston AGX Orin   | [5.0.2](https://developer.nvidia.com/embedded/jetpack-sdk-502) |     |

## Develop

### Jeston AGX Orin

以 [nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3]([NVIDIA L4T ML | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml)) image为基础进行搭建，其中所包含的 Package 信息如下：

- TensorFlow 1.15.5
- PyTorch v1.12.0
- torchvision v0.13.0
- torchaudio v0.12.0
- onnx 1.12.0
- CuPy 10.2.0
- numpy 1.22.4
- numba 0.56.0
- PyCUDA 2022.1
- OpenCV 4.5.0 (with CUDA)
- pandas 1.4.3
- scipy 1.9.0
- scikit-learn 1.1.1
- JupyterLab 3.4.4
- **TensorRT 8.4.1**
- **cuDNN 8.4.1**
- **CUDA 11.4.14**

```yml
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
```

### Jeston AGX Xavier

```yml
on the way
```

## Deploy

### Jeston AGX Orin

```yml
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
```

### Jeston AGX Xavier

```yml
on the way
```
