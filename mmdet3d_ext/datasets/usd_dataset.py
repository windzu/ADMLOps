# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset
from mmcv.utils import print_log

from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.utils import extract_result_dict, get_loading_pipeline


@DATASETS.register_module()
class USDDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    Annotation format:
    [
        {
            'scene_name': 'scene_name',
            'seq' : '000000',
            'images': {
                'CAM_00':{
                    'file_name': '000000.jpg',
                    'shape': array([370,1224,3], dtype=int32),
                    'annos': {
                        'class_names': <np.ndarray> (N,),
                        'track_ids': <np.ndarray> (N,),
                        'bbox2d'': <np.ndarray> (N, 4),
                        'bbox3d': <np.ndarray> (N, 7),
                        'box_type_3d'="LiDAR",
                        'truncated': <np.ndarray> (N,),
                        'occluded': <np.ndarray> (N,),
                        'num_points_in_gt': <np.ndarray> (n),
                    }
                },
                ...
            },
            'point_clouds': {
                'LIDAR_00':{
                    'frame_id': 'LIDAR_00',
                    'file_name': '000000.bin',
                    'shape': array([1224,4], dtype=int32),
                    'annos':...
                },
                ...
            },
            'calib': {
                'CAM_00': <np.ndarray> (4, 4),
                ...
                'LIDAR_00': <np.ndarray> (4, 4),
                ...
                "intrinsics":{
                    'CAM_00': <np.ndarray> (3, 3),
                    ...
                },
            },
        },
        ...
    ]

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(
        self,
        data_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        file_client_args=dict(backend="disk"),
    ):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # load annotations
        if hasattr(self.file_client, "get_local_path"):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, "rb"))
        else:
            warnings.warn(
                "The used MMCV version does not have get_local_path. "
                f"We treat the {self.ann_file} as local paths and it "
                "might cause errors if the path is not a local path. "
                "Please use MMCV>= 1.3.16 if you meet errors."
            )
            self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format="pkl")

    def get_data_info(self, index):
        """从标注文件中获取满足条件的数据信息
        返回的数据格式示例：
        {
            "seq": "0000", # 该数据在该scene中的序列号
            "pts_filename" : "path/000000.bin", # 点云文件路径
            "ann_info":{
                "gt_bboxes_3d":<np.ndarray> (N, 7),
                "gt_labels_3d":<np.ndarray> (N, 1)
            }
            # NOTE : 以下字段仅为pipeline中的 LoadPointsFromMultiSweeps 需要
            "timestamp": 0.0, # 如果点云是5维且最后一维包含的是时间戳,则返回时间戳,否则返回0.0
            "sweeps":[] # 如果包含多帧点云,则返回多帧点云的路径，否则返回空列表
        }
        """
        raw_info = self.data_infos[index]

        # - parse images_info
        # TODO : 解析图像的相关信息， 但是现在用不到，暂且搁置
        # images_info = raw_info["images"]

        # - parse point_clouds_info
        # TODO: support multi-lidar，暂时只使用一个默认的lidar
        # -- lidar是否应该只有一个呢？如果有多个，那么应该如何处理呢？
        point_clouds_info = raw_info["point_clouds"]
        pts_filename = osp.join(
            self.data_root, raw_info["scene_name"], "LIDAR", point_clouds_info["LIDAR"]["file_name"]
        )
        # 如果点云文件不存在，直接返回None,会跳过并进行下一个数据的选择
        if not osp.exists(pts_filename):
            return None

        # - parse annotations_info
        # TODO : 解析图像的标注信息
        raw_gt_bboxes_3d = point_clouds_info["LIDAR"]["annos"]["bbox3d"]
        raw_gt_names_3d = point_clouds_info["LIDAR"]["annos"]["class_names"]
        gt_labels_3d = []
        for cat in raw_gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        ##  对3d bbox 进行转换
        ori_box_type_3d = point_clouds_info["LIDAR"]["annos"]["box_type_3d"]  # default: LiDAR
        ori_box_type_3d, _ = get_box_type(ori_box_type_3d)
        # turn original box type to target box type

        gt_bboxes_3d = ori_box_type_3d(
            raw_gt_bboxes_3d, box_dim=raw_gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)
        ann_info = {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
        }

        result = {
            "seq": raw_info["seq"],
            "pts_filename": pts_filename,
            "ann_info": ann_info,
            "timestamp": 0.0,
            "sweeps": [],
        }
        return result

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
            return None

        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_dt_annos(self, dt_annos):
        """将检测结果格式化,便于与标注文件进行eval计算.
        返回的格式如下：
        - 对于3d检测任务的格式化结果示例如下:
            [
                {
                    "sample_idx": -1,
                    "name": <np.ndarray> (N, 1), # string type
                    "location": <np.ndarray> (N, 3),
                    "dimensions": <np.ndarray> (N, 3),
                    "rotation_y": <np.ndarray> (N, 1),
                    "score": <np.ndarray> (N, 1),
                },
                ...
            ]
        - 对于2d检测任务的格式化结果示例如下:
            待补充...

        Args:
            dt_annos (list[dict]): test模式下验证集的输出结果.
        Returns:
            (list[dict]): 格式化后的dt
        """

        # - 输入输出检查，其长度应该相同
        assert len(dt_annos) == len(self.data_infos), "invalid list length of network outputs"

        # - 将网络的在test模式下的输出结果转换为eval所需的格式
        # TODO ： 针对eval任务的不同，需要做一些修改，目前仅仅针对的是3d检测的任务
        # - 增加对3d检测任务的判断以及处理支持
        # - 增加对3d分割任务的判断以及处理支持
        result_annos = []
        print("\nConverting prediction to lidar format")
        for idx, pred_dicts in enumerate(mmcv.track_iter_progress(dt_annos)):

            annos = []
            # info = self.data_infos[idx]
            # sample_idx = info["point_cloud"]["idx"]  # 不知sample_idx的用途是什么
            sample_idx = -1

            anno = {
                "name": [],
                "location": [],
                "dimensions": [],
                "rotation_y": [],
                "score": [],
            }

            # 在3d检测任务中，pred_dicts中包含了多个类别的检测结果
            # 点云的3d检测任务的结果有时候存放在 pts_bbox 这个key中，所以需呀要做一下判断
            if "pts_bbox" in pred_dicts:
                pred_dicts = pred_dicts["pts_bbox"]

            if len(pred_dicts["boxes_3d"]) > 0:
                boxes_3d = pred_dicts["boxes_3d"]
                scores_3d = pred_dicts["scores_3d"]
                labels_3d = pred_dicts["labels_3d"]

                # TODO：这个填充的方式太丑陋了，不应该需要循环，有待改进
                for (box_3d, score_3d, label_3d) in zip(boxes_3d, scores_3d, labels_3d):
                    anno["name"].append(self.CLASSES[int(label_3d)])
                    anno["location"].append(box_3d[:3])
                    anno["dimensions"].append(box_3d[3:6])
                    anno["rotation_y"].append(box_3d[6])
                    anno["score"].append(score_3d)
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    "name": np.array([]),
                    "location": np.zeros([0, 3]),
                    "dimensions": np.zeros([0, 3]),
                    "rotation_y": np.array([]),
                    "score": np.array([]),
                }
                annos.append(anno)
            annos[-1]["sample_idx"] = np.array([sample_idx] * len(annos[-1]["score"]), dtype=np.int64)

            result_annos += annos

        return result_annos

    def format_gt_annos(self, gt_annos):
        """将gt格式化,便于进行eval计算.
        返回的格式如下：
        - 对于3d检测任务的格式化结果示例如下:
            [
                {
                    "sample_idx": -1,
                    "name": <np.ndarray> (N, 1), # string type
                    "location": <np.ndarray> (N, 3),
                    "dimensions": <np.ndarray> (N, 3),
                    "rotation_y": <np.ndarray> (N, 1),
                    "score": <np.ndarray> (N, 1), # gt中没有该字段,全部填充为0
                },
                ...
            ]
        - 对于2d检测任务的格式化结果示例如下:
            待补充...

        Args:
            gt_annos (list[dict]): self.data_infos 中的 anno 字段构成的list
        Returns:
            (list[dict]): 格式化后的gt
        """
        # - 输入输出检查，其长度应该相同
        assert len(gt_annos) == len(self.data_infos), "invalid list length of network outputs"

        # - 将网络的在test模式下的输出结果转换为eval所需的格式
        # TODO ： 针对eval任务的不同，需要做一些修改，目前仅仅针对的是3d检测的任务
        # - 增加对3d检测任务的判断以及处理支持
        # - 增加对3d分割任务的判断以及处理支持
        result_annos = []
        print("\nConverting prediction to lidar format")
        for idx, gt_dicts in enumerate(mmcv.track_iter_progress(gt_annos)):

            annos = []
            # info = self.data_infos[idx]
            # sample_idx = info["point_cloud"]["idx"]  # 不知sample_idx的用途是什么
            sample_idx = -1

            anno = {
                "name": [],
                "location": [],
                "dimensions": [],
                "rotation_y": [],
                "score": [],
            }

            if len(gt_dicts["gt_bboxes_3d"]) > 0:
                boxes_3d = gt_dicts["gt_bboxes_3d"]
                # scores_3d = gt_dicts["scores_3d"] 没有scores_3d字段
                labels_3d = gt_dicts["gt_labels_3d"]

                for (box_3d, label_3d) in zip(boxes_3d, labels_3d):
                    anno["name"].append(self.CLASSES[int(label_3d)])
                    anno["location"].append(box_3d[:3])
                    anno["dimensions"].append(box_3d[3:6])
                    anno["rotation_y"].append(box_3d[6])
                    anno["score"].append(0)
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    "name": np.array([]),
                    "location": np.zeros([0, 3]),
                    "dimensions": np.zeros([0, 3]),
                    "rotation_y": np.array([]),
                    "score": np.array([]),
                }
                annos.append(anno)
            annos[-1]["sample_idx"] = np.array([sample_idx] * len(annos[-1]["score"]), dtype=np.int64)

            result_annos += annos

        return result_annos

    def evaluate(
        self,
        results,
        metric=["bev", "3d"],
        logger=None,
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluate.

        Args:
            results (list[dict]): test模式下的检测结果.
            metric (str | list[str], optional): Metrics to be evaluated.Defaults to ["bev", "3d"].
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        gt_annos = [self.get_data_info(i)["ann_info"] for i in range(len(self.data_infos))]

        dt_annos_after_format = self.format_dt_annos(results)
        gt_annos_after_format = self.format_gt_annos(gt_annos)
        from mmdet3d_extension.core.evaluation import usd_eval

        ap_result_str, ap_dict = usd_eval(
            gt_annos=gt_annos_after_format,
            dt_annos=dt_annos_after_format,
            current_classes=self.CLASSES,
            eval_types=metric,  # default evaluate bev and 3d
        )

        print_log("\n" + ap_result_str, logger=logger)

        # TODO : 可视化的查看评估结果曲线
        # if show or out_dir:
        #     self.show(results, out_dir, show=show, pipeline=pipeline)

        return ap_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError(
            "_build_default_pipeline is not implemented " f"for dataset {self.__class__.__name__}"
        )

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, "pipeline") or self.pipeline is None:
                warnings.warn("Use default pipeline for data loading, this may cause " "errors when data is on ceph")
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, "data loading pipeline is not provided"
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
