# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import time

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.datasets.builder import PIPELINES

from wadda.pypcd import pypcd


@PIPELINES.register_module()
class LoadPointsFromPointCloud2(object):
    """Load Points From LoadPointsFromPointCloud2.
    加载式 point cloud 2 格式的点云数据

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _pc_format_converter(self, input_data):
        """将输入的点云数据转换为符合mmdetection3d中模型输入需要的点云格式。
        并且点云目标检测模型因为训练时候所采用的数据集的不同，对输入点云的维度需求不同。
        因为实例化模型是根据配置文件来的，所以转换后的点云要与配置文件中一致，否则会导致维度出错。
        例如：
            nuscenes数据集的原始点云是[x, y, z, intensity, ring]5维,
        那么采用nuscenes数据集的配置文件所实例化的模型就需要提供5维的点云(尽管最后只是使用了其中的前4维参与了训练)
            waymo数据集的原始点云是[x, y, z, intensity,ring,未知]6维,
        那么采用waymo数据集的配置文件所实例化的模型就需要提供6维的点云(尽管最后只是使用了其中的前4维参与了训练)

        Args:
            input_data (PointCloud2): 输入的点云数据,来自ros1订阅,为ros1 sensor_msgs 的PointCloud2类型的点云数据
        """
        pc = pypcd.PointCloud.from_msg(input_data)
        # 所有点云至少包含xyz这三个维度
        x = pc.pc_data["x"].flatten()
        y = pc.pc_data["y"].flatten()
        z = pc.pc_data["z"].flatten()
        intensity = pc.pc_data["intensity"].flatten()
        ring = pc.pc_data["ring"].flatten()

        # 首先将输入的点云数据转换为6维的点云，原始点云中不存在的维度设置为0，然后根据需求转换为需要的维度
        pc_array_6d = np.zeros((x.shape[0], 6))
        pc_array_6d[:, 0] = x
        pc_array_6d[:, 1] = y
        pc_array_6d[:, 2] = z
        pc_array_6d[:, 3] = intensity
        pc_array_6d[:, 4] = ring

        return pc_array_6d

    def _load_pointcloud2(self, pointcloud2):
        """Private function to load point clouds data.

        Args:
            pointcloud2 (pointcloud2): pointcloud2 format point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """

        points = self._pc_format_converter(pointcloud2)

        return points

    def __call__(self, results):
        """Call function to load points data from pointcloud2.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """

        pointcloud2 = results["pointcloud2"]
        points = self._load_pointcloud2(pointcloud2)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + "("
        repr_str += f"shift_height={self.shift_height}, "
        repr_str += f"use_color={self.use_color}, "
        repr_str += f"file_client_args={self.file_client_args}, "
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim})"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFileExtension(object):
    """修改自 LoadPointsFromFile , 解决加载数据维度问题
    nuscenes数据集的原始点云是[x, y, z, intensity,ring] 5维,但是一般数据集只有4维度,
    这里采用自己扩增fake数据来解决这个问题,

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        # assert max(use_dim) < load_dim, f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        # judeg load dims and use dims
        self.dim_diff = 0
        self.dim_diff_flag = False
        if self.load_dim < len(self.use_dim):
            self.dim_diff = len(self.use_dim) - self.load_dim
            self.dim_diff_flag = True

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)

        points = points.reshape(-1, self.load_dim)

        if self.dim_diff_flag:
            points = np.concatenate([points, np.zeros((points.shape[0], self.dim_diff))], axis=1)

        points = points[:, self.use_dim]

        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + "("
        repr_str += f"shift_height={self.shift_height}, "
        repr_str += f"use_color={self.use_color}, "
        repr_str += f"file_client_args={self.file_client_args}, "
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim})"
        return repr_str
