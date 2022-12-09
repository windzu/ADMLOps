# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from pathlib import Path
from concurrent import futures as futures
import random
import mmcv
import json
import numpy as np
from tqdm import tqdm
from tools.visualizer import pypcd
from mmdet3d.core.bbox import box_np_ops
from .kitti_data_utils import (get_velodyne_path, get_label_path, 
                               get_label_anno, add_difficulty_to_annos)
from .kitti_converter import _NumPointsInGTCalculater


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def _read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line for line in lines]

def _read_result(path):
    relative_paths = []
    label_infos = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            infos = json.loads(line.split('\t')[1])
            if 'labelData' not in infos.keys() and 'standardData' not in infos.keys():
                continue
            relative_paths.append(infos['datasetsRelatedFiles'][0]['localRelativePath'])
            label_infos.append(infos)

    sorted_index = argsort(relative_paths)
    _write_imageset_file(path.split('result')[0] + 'pcd.txt', relative_paths)
    return [label_infos[i] for i in sorted_index]


def _split_imageset(imageset_dir, frame_num):
    trainval_idx = [*range(frame_num)]
    mmcv.mkdir_or_exist(imageset_dir)
    if osp.exists(osp.join(imageset_dir, 'trainval.txt')) and \
       osp.exists(osp.join(imageset_dir, 'train.txt')) and \
       osp.exists(osp.join(imageset_dir, 'val.txt')):
       return
    _write_imageset_file(osp.join(imageset_dir, 'trainval.txt'), trainval_idx)

    random.shuffle(trainval_idx)
    total = len(trainval_idx)
    split_ratio = 0.8
    split = (int)(total * split_ratio)
    print('{} splits {} to train, {} to val'.format(total, split, total-split))

    train_idx = trainval_idx[0: split]
    val_idx = trainval_idx[split: total]
    _write_imageset_file(osp.join(imageset_dir, 'train.txt'), train_idx)
    _write_imageset_file(osp.join(imageset_dir, 'val.txt'), val_idx)


def _write_imageset_file(path, split_idx):
    split_idx.sort()
    with open(path, 'w+') as f:
        for idx in split_idx:
            f.write(str(idx) + '\n')


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


class AICV2KITTI(object):
    """AICV to KITTI converter.

    This class serves as the converter to change the aicv raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load aicv raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 enable_rotate_45degree=False,
                 test_mode=False,
                 workers=64):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_aicv_locations = None
        self.save_track_id = False

        self.type_list = [
            'smallMot', 'bigMot', 'OnlyTricycle', 'OnlyBicycle', 
            'Tricyclist', 'bicyclist', 'motorcyclist', 'pedestrian', 
            'TrafficCone', 'others', 'fog', 'stopBar', 'smallMovable', 
            'smallUnmovable', 'crashBarrel', 'safetyBarrier', 'sign'
        ]

        self.aicv_to_kitti_class_map = {
            'smallMot': 'Car',
            'bigMot': 'Car',
            'OnlyTricycle': 'Bicycle',
            'OnlyBicycle': 'Bicycle',
            'Tricyclist': 'Bicycle',
            'bicyclist': 'Bicycle',
            'motorcyclist': 'Bicycle',
            'pedestrian': 'Pedestrian',
            'TrafficCone': 'TrafficCone', 
            'stopBar': 'DontCare', 
            'crashBarrel': 'DontCare', 
            'safetyBarrier': 'DontCare', 
            'sign': 'DontCare', 
            'smallMovable': 'DontCare',
            'smallUnmovable': 'DontCare', 
            'fog': 'DontCare',
            'others': 'DontCare'
        }
        # self.selected_kitti_classes = [
        #     'Car', 'Pedestrian', 'Bicycle', 'TrafficCone', 'Barrier'
        # ]
        self.selected_kitti_classes = [
            'Car', 'Pedestrian', 'Bicycle', 'TrafficCone'
        ]
        
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.enable_rotate_45degree = enable_rotate_45degree
        self.test_mode = test_mode
        self.workers = int(workers)

        self.rotate_matrix = np.eye(4)
        if enable_rotate_45degree:
            self.rotate_matrix = np.array(
                [[np.cos(np.pi/4),  np.sin(np.pi/4), 0, 0, 0], 
                 [-np.sin(np.pi/4), np.cos(np.pi/4), 0, 0, 0], 
                 [0, 0, 1, 0, 0], 
                 [0, 0, 0, 1, 0], 
                 [0, 0, 0, 0, 1]])

        self.label_infos = _read_result(load_dir + f'/result.txt')

        self.imageset_dir = f'{self.save_dir}/kitti_format/ImageSets'
        _split_imageset(self.imageset_dir, len(self))

        self.label_save_dir = f'{self.save_dir}/label'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        for frame_idx in tqdm(range(len(self))):
            infos = self.label_infos[frame_idx]
            pcd_pathname = osp.join(
                self.load_dir, 
                infos['datasetsRelatedFiles'][0]['localRelativePath'], 
                infos['datasetsRelatedFiles'][0]['fileName'])
            if 'standardData' in infos.keys():
                annotations = infos['standardData']
            elif 'labelData' in infos.keys():
                annotations = infos['labelData']['result']
            # frame_id timestamp x y z qx qy qz qw
            pose = infos['poses']['velodyne_points']
            timestamp = infos['frameTimestamp']

            self.save_lidar(pcd_pathname, frame_idx, timestamp)
            self.save_pose(pose, frame_idx)
            self.save_timestamp(timestamp, frame_idx)
            self.save_label(annotations, frame_idx)
        print('\nFinished ...')

    # def convert(self):
    #     """Convert action."""
    #     print('Start converting ...')
    #     mmcv.track_parallel_progress(self.convert_one, range(len(self)),
    #                                  self.workers)
    #     print('\nFinished ...')

    def convert_one(self, frame_idx):
        infos = self.label_infos[frame_idx]
        pcd_pathname = osp.join(
            self.load_dir, 
            infos['datasetsRelatedFiles'][0]['localRelativePath'], 
            infos['datasetsRelatedFiles'][0]['fileName'])
        if 'standardData' in infos.keys():
            annotations = infos['standardData']
        elif 'labelData' in infos.keys():
            annotations = infos['labelData']['result']
        # frame_id timestamp x y z qx qy qz qw
        pose = infos['poses']['velodyne_points']
        timestamp = infos['frameTimestamp']

        self.save_lidar(pcd_pathname, frame_idx, timestamp)
        self.save_pose(pose, frame_idx)
        self.save_timestamp(timestamp, frame_idx)
        self.save_label(annotations, frame_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.label_infos)

    def save_lidar(self, pcd_pathname, frame_idx, timestamp):
        point_cloud_path = f'{self.point_cloud_save_dir}/{str(frame_idx).zfill(6)}.bin'

        pcd = pypcd.PointCloud.from_path(pcd_pathname)
        if 'timestamp' not in pcd.fields:
            stamp = np.zeros_like(pcd.pc_data['x']) + float(timestamp)/1000.
        else:
            stamp = pcd.pc_data['timestamp']
        points = np.stack([pcd.pc_data['x'], pcd.pc_data['y'], pcd.pc_data['z'], 
                           pcd.pc_data['intensity'], stamp]).transpose(1, 0)
        points = np.dot(points, self.rotate_matrix)
        points.astype(np.float32).tofile(point_cloud_path)

    def save_label(self, annotations, frame_idx):
        """Parse and save the label data in txt format.
        The relation between aicv and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (aicv) -> w, h, l
        3. bbox origin at volumetric center (aicv) -> bottom center (kitti)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label = open(
            f'{self.label_save_dir}/{str(frame_idx).zfill(6)}.txt', 'w+')
        fp_label.close()
        for annotation in annotations:
            position = annotation['position']
            rotation = annotation['rotation']
            type = annotation['type']
            size = annotation['size']

            type = self.aicv_to_kitti_class_map[type]
            if type not in self.selected_kitti_classes:
                continue

            # not available
            truncated = 0
            occluded = 0
            alpha = -10
            bounding_box = [0, 0, 100, 100]

            l = size[0]
            w = size[1]
            h = size[2]
            x = position['x'] * self.rotate_matrix[0, 0] + \
                position['y'] * self.rotate_matrix[1, 0]
            y = position['x'] * self.rotate_matrix[0, 1] + \
                position['y'] * self.rotate_matrix[1, 1]
            z = position['z'] - h / 2
            rotation_y = rotation['phi']
            if self.enable_rotate_45degree:
                rotation_y += np.pi / 4
            if rotation_y > np.pi:
                rotation_y -= 2 * np.pi

            # if self.clean_data(type, l, w, h):
            #     continue

            # [w, h, l] will transfose to [l, w, h] in get_label_anno() of kitti_data_utils.py
            line = type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(w, 2), round(h, 2), round(l, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            fp_label = open(
                f'{self.label_save_dir}/{str(frame_idx).zfill(6)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

    def save_timestamp(self, timestamp, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        with open(osp.join(f'{self.timestamp_save_dir}/{str(frame_idx).zfill(6)}.txt'), 'w') as f:
            f.write(str(timestamp))

    def save_pose(self, pose, frame_idx):
        with open(osp.join(f'{self.pose_save_dir}/{str(frame_idx).zfill(6)}.txt'), 'w') as f:
            f.write(pose)

    def clean_data(self, type, l, w, h):
        """
        clean data according to anchor_size, the up and down ratio is 4 and 0.25
        """
        if type == 'Car' and (
            l > 17.8 or w > 7.68 or h > 6.6 or l < 1.1 or w < 0.48 or h < 0.1):
            return True
        elif type == 'Pedestrian' and (
            l > 2.2 or w > 2.4 or h > 6.6 or l < 0.14 or w < 0.15 or h < 0.1):
            return True
        elif type == 'Bicycle' and (
            l > 7.4 or w > 3.24 or h > 5.2 or l < 0.46 or w < 0.2 or h < 0.3):
            return True
        elif type == 'TrafficCone' and (
            l > 1.44 or w > 1.44 or h > 2.6 or l < 0.09 or w < 0.09 or h < 0.16):
            return True
        else:
            return False

    def create_folder(self):
        """Create folder for data preprocessing."""
        dir_list = [
            self.point_cloud_save_dir, 
            self.pose_save_dir, 
            self.label_save_dir, 
            self.timestamp_save_dir
        ]
        for d in dir_list:
            mmcv.mkdir_or_exist(d)


def get_aicv_image_info(path,
                        training=True,
                        label_info=True,
                        velodyne=False,
                        image_ids=7481,
                        num_worker=8,
                        relative_path=True):
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        image_info = {'image_idx': idx}

        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path, 
                                        info_type='label')
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['point_cloud'] = pc_info
        info['image'] = image_info

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info) # all [-1]

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)

def _calculate_num_points_in_gt(data_path,
                                      infos,
                                      relative_path,
                                      num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def create_aicv_info_file(data_path,
                          pkl_prefix='aicv',
                          save_path=None,
                          relative_path=True,
                          max_sweeps=0,
                          workers=8):
    """Create info file of aicv dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'aicv'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        max_sweeps (int, optional): Max sweeps before the detection frame
            to be used. Default: 0.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    aicv_infos_gatherer = AicvInfoGatherer(
        data_path,
        training=True,
        velodyne=True,
        relative_path=relative_path,
        num_worker=workers)
    num_points_in_gt_calculater = _NumPointsInGTCalculater(
        data_path,
        relative_path,
        calib=False,
        num_features=5,
        remove_outside=False,
        num_worker=workers)

    aicv_infos_train = aicv_infos_gatherer.gather(train_img_ids)
    num_points_in_gt_calculater.calculate(aicv_infos_train)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Aicv info train file is saved to {filename}')
    mmcv.dump(aicv_infos_train, filename)
    aicv_infos_val = aicv_infos_gatherer.gather(val_img_ids)
    num_points_in_gt_calculater.calculate(aicv_infos_val)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Aicv info val file is saved to {filename}')
    mmcv.dump(aicv_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Aicv info trainval file is saved to {filename}')
    mmcv.dump(aicv_infos_train + aicv_infos_val, filename)


class AicvInfoGatherer:
    """
    Parallel version of AICV dataset information gathering.
    AICV annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
        }
        point_cloud: {
            num_features: 5
            velodyne_path: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """

    def __init__(self,
                 path,
                 training=True,
                 label_info=True,
                 velodyne=True,
                 pose=False,
                 relative_path=True,
                 num_worker=8) -> None:
        self.path = path
        self.training = training
        self.label_info = label_info
        self.velodyne = velodyne
        self.pose = pose
        self.relative_path = relative_path
        self.num_worker = num_worker

    def gather_single(self, idx):
        root_path = Path(self.path)
        info = {}
        pc_info = {'num_features': 4}
        image_info = {'image_idx': idx}

        annotations = None
        if self.velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, self.path, self.training, self.relative_path)
        if self.label_info:
            label_path = get_label_path(idx, self.path, self.training, 
                                        self.relative_path, info_type='label')
            if self.relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['point_cloud'] = pc_info
        info['image'] = image_info

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info) # all [-1]

        return info

    def gather(self, image_ids):
        if not isinstance(image_ids, list):
            image_ids = list(range(image_ids))
        image_infos = mmcv.track_parallel_progress(self.gather_single,
                                                   image_ids, self.num_worker)
        return list(image_infos)