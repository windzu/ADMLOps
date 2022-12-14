# Introduction

自动驾驶的感知传感器包括：
* Camera
* Lidar
* Radar
* USS：目前很少使用了

每种传感器都有自己的优缺点：
* Camera
    * 优点：成本低，信息稠密
    * 缺点：视野有限，受天气影响大，受阳光影响
* Lidar
    * 优点：可靠的距离信息，不受阳光影响
    * 缺点：成本高，信息稀疏，受雨雾响大
* Radar
    * 优点：成本低，穿透能力强，可靠的距离信息，不受阳光影响
    * 缺点：信息稀疏，受金属物体影响

针对不同传感器的特性，有不同的感知算法与之对应，例如纯视觉算法、纯激光雷达算法、纯雷达算法、视觉+激光雷达算法、视觉+雷达算法等

本工程是一个包含多种检测任务的框架，可以帮助使用者快速的完成数据准备、模型搭建、模型训练等任务，从而提高使用者的工作效率

## Supported
本工程提供了对多种任务的支持，并提供了一些小工具，可以帮助使用者提高工作效率

### Model

- [x] YOLOX
- [x] YOLOPv2
- [ ] Nanodet
- [ ] UFLD
- [ ] PointPillars
- [ ] CenterPoint
- [ ] BEVFusion

### Tool

- [x] 自动标注数据为scalabel格式
- [x] ROS快速测试接口


## Tutorials

为了方便大家快速上手使用以及提高对本工程的理解，本工程还提供了一系列Tutorials，一般是结合自动驾驶中常见的任务而展开的，希望能给大家提供一些思路

如果在使用过程中发现bug或者文档错误，非常欢迎您能提`issus`或`pr`，我将在收到通知后的第一时间尽快修复

最后，如果觉得本工程对您有帮助，希望能给一个star，我将会非常开心，感谢！
