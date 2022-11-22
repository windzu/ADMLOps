## 简介

这是一个借助 opencv 的单目标定程序，支持 `pinehole` 和 `fisheye` 的相机模型

本方法使用的是`九宫格`采样法，需要`标定板`以及`人`的参与，如果有机械臂，可以使用机械臂来标定。

**九宫格基本思想**：

1. 固定相机，是将相机的画面按照九宫格的方式进行九等分
2. 调整标定板与相机的距离，使标定板在图像中大概占据一个格子
3. 开始采样，每个格子位置采集五张图像，分别是正、上下左右偏转30度左右
4. 九个格子均按照3的步骤采集，共采集45张图像，然后进行标定

虽然这种方法比较麻烦，但是可以保证标定的精度，请务必遵循如下步骤进行标定，以及相关的注意事项。

## 步骤

### 1. 准备标定板

本标定程序支持两种标定板，分别是

- [Chessboard](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
- [ChArUco](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)

生成标定板的方法可以参考 [这里](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)

> Note
>
> 在设置 pattern 的 rows 和 columns , 尽量设置的不要相同(不要正方形)，且数值最好大于5， 这样可以防止方向上的混淆，以及有较多的角点可以保证标定的精度

**TODO**:
实现一个生成 pattern 的方法

### 2. 准备稳定的光源

所有寻找标定板角点的算法以及亚像素优化算法都非常依赖图像的灰度变化质量，所以为了保证标定的精度，需要保证标定板的光照是稳定的，可以使用柔光灯，或者使用一盏灯光加上一块白色的板子，来保证光照的稳定。

如果光照不稳定，会导致标定的精度下降、无法找到角点，甚至无法标定。

### 3. 相机和标定板摆放初始化

> Note
>
> 这里叫初始化更多是为了机械臂带标定板的移动而准备的，如果是手动可以跳过这一步

将相机水平固定且稳定出图后，然后将标定板竖直放置在一个大概可以占据相机九宫格中间的位置，以此位置作为标定的初始位置。强烈建议使用固定的支架或者机械臂来固定标定板，以保证标定板的稳定性

### 4. 开始标定

```bash
python3 main.py --camera_model pinhole \
    --pattern chessboard \
    --rows 7 --cols 7 \
    --size 0.025 \
    --device /dev/video0
```

程序启动后，会自动打开摄像头，然后开始采集图像，显示图像的窗口被切分为九宫格，每个格子有一个编号，编号如下

| 1   | 2   | 3   |
| --- | --- | --- |
| 8   | 0   | 4   |
| 7   | 6   | 5   |

在初始化后，标定板应该放置在中间的格子中，按照0-8的编号顺序遍历格子，然后在每个格子中分别采集上述的5张图像，与相机的角度分别是：正、上下左右偏转30度

具体的执行步骤如下

1. 在初始化后，标定板应该在`0号区域`中，此时`0号区域`会高亮，并显示数字`1`和标定板需要摆放的角度指示，将标定板按照指示方式摆放后，当角点可以稳定检出后，按下 `s` 键，程序会采集当前的图像，然后指示会变换，继续按照指示采集
2. 当一个区域的图像采集完成后，程序会自动切换到下一个区域，继续按照指示采集，直到所有区域的图像都采集完成，程序会开始标定
3. 标定完成后，会同时显示标定前和标定后的图像，如果标定的结果不理想，可以按下 `r` 键，重新开始标定，如果标定的结果理想，可以按下 `q` 键，退出程序
4. 标定完成后，会在当前目录下生成一个文件夹，文件夹名称为该相机的`device`名称，文件夹中包含了标定的结果和本次标定所采集的图像

如果采集到的图像中没有找到标定板，会在窗口中显示 `No pattern found` ，如果找到了标定板，会在窗口中显示 `Found pattern` ，然后会在窗口中显示标定板的角点，以及标定板的中心点

## Roadmap

> Note 记录一下后续的计划，完成后删除

- [ ] 实现图像切分的功能，将图像切分为九宫格，并让指定的区域高亮
- [ ] 实现两种标定板的检测方式
- [ ] 基于上种方式实现在低分辨率下检测的模块，以达到实时检测的目的