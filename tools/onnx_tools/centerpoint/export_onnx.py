from argparse import ArgumentParser
from copy import deepcopy

import mmcv
import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmcv.parallel import collate, scatter
from mmcv.runner import auto_fp16, force_fp32, load_checkpoint
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.models.builder import BACKBONES, HEADS, NECKS
from mmdet.core import build_bbox_coder
from torch import nn
from torch.nn import functional as F


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        last_layer (bool, optional): If last_layer, there is no
            concatenation of features. Defaults to False.
        mode (str, optional): Pooling model to gather features inside voxels.
            Defaults to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, voxel_features):
        for pfn in self.pfn_layers:
            voxel_features = pfn(voxel_features)
        return voxel_features.squeeze(1)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_backbone = BACKBONES.build(cfg['pts_backbone'])
        self.pts_neck = NECKS.build(cfg['pts_neck'])
        self.cfg = cfg
        pts_bbox_head = cfg['pts_bbox_head']
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        pts_train_cfg = train_cfg.pts if train_cfg else None
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        pts_bbox_head.update(test_cfg=pts_test_cfg)
        self.pts_bbox_head = HEADS.build(pts_bbox_head)
        self.bbox_coder = build_bbox_coder(cfg['pts_bbox_head']['bbox_coder'])
        self.box_code_size = self.bbox_coder.code_size

    def forward(self, x):
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        outs = self.pts_bbox_head(x)
        # for task in outs:
        #     heatmap = torch.sigmoid(task[0]['heatmap'])
        #     scores, labels = torch.max(heatmap, dim=1)
        #     task[0]['heatmap'] = (scores, labels)
        #     task[0]['dim'] = torch.exp(task[0]['dim'])

        bbox_preds, scores, dir_scores = [], [], []
        for task_res in outs:
            bbox_preds.append(task_res[0]['reg'])
            bbox_preds.append(task_res[0]['height'])
            bbox_preds.append(task_res[0]['dim'])
            if 'vel' in task_res[0].keys():
                bbox_preds.append(task_res[0]['vel'])
            scores.append(task_res[0]['heatmap'])
            dir_scores.append(task_res[0]['rot'])
        bbox_preds = torch.cat(bbox_preds, dim=1)
        scores = torch.cat(scores, dim=1)
        dir_scores = torch.cat(dir_scores, dim=1)
        return scores, bbox_preds, dir_scores


def parse_model(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def load_data(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['points'] = data['points'][0].data
    return data


def build_pfn_model(cfg, checkpoint=None, device='cuda:0'):
    pts_voxel_encoder = PillarFeatureNet(
        in_channels=cfg.model['pts_voxel_encoder']['in_channels'],
        feat_channels=cfg.model['pts_voxel_encoder']['feat_channels'],
        with_distance=cfg.model['pts_voxel_encoder']['with_distance'],
        voxel_size=cfg.model['pts_voxel_encoder']['voxel_size'],
        point_cloud_range=cfg.model['pts_voxel_encoder']['point_cloud_range'],
        norm_cfg=cfg.model['pts_voxel_encoder']['norm_cfg'],
        legacy=cfg.model['pts_voxel_encoder']['legacy'],)
    pts_voxel_encoder.to(device).eval()
    checkpoint_pts_load = torch.load(checkpoint, map_location=device)
    dicts = {}
    for key in checkpoint_pts_load['state_dict'].keys():
        if 'pfn' in key:
            dicts[key.split('pts_voxel_encoder.')[1]
                  ] = checkpoint_pts_load['state_dict'][key]
    pts_voxel_encoder.load_state_dict(dicts)
    print('-----------------------')
    parse_model(pts_voxel_encoder)
    return pts_voxel_encoder


def build_backbone_model(cfg, checkpoint=None, device='cuda:0'):
    backbone = Backbone(cfg.model)
    backbone.to('cuda').eval()

    checkpoint = torch.load(checkpoint, map_location='cuda')
    dicts = {}
    for key in checkpoint["state_dict"].keys():
        if "backbone" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "neck" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "bbox_head" in key:
            dicts[key] = checkpoint["state_dict"][key]
    backbone.load_state_dict(dicts)

    torch.cuda.set_device(device)
    backbone.to(device)
    backbone.eval()
    return backbone


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    if isinstance(args.config, str):
        cfg = mmcv.Config.fromfile(args.config)
    elif not isinstance(args.config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(args.config)}')
    cfg.model.pretrained = None
    convert_SyncBN(cfg.model)
    cfg.model.train_cfg = None
    device = args.device

    # original model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint_load = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint_load['meta']:
            model.CLASSES = checkpoint_load['meta']['CLASSES']
        else:
            model.CLASSES = cfg.class_names
        if 'PALETTE' in checkpoint_load['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint_load['meta']['PALETTE']
    model.cfg = cfg  # save the config in the model for convenience
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    parse_model(model)

    # PillarFeatureNet
    pts_voxel_encoder = build_pfn_model(cfg, args.checkpoint, device=device)
    # max_voxels * 20 * 10
    voxel_features = torch.ones(cfg.model['pts_voxel_layer']['max_voxels'][1],
                                cfg.model['pts_voxel_layer']['max_num_points'],
                                pts_voxel_encoder.in_channels).cuda()
    torch.onnx.export(pts_voxel_encoder,
                      voxel_features,
                      f='./tools/onnx_tools/centerpoint/pfe.onnx',
                      opset_version=12,
                      verbose=True,
                      input_names=['voxels'],
                      output_names=['voxel_features'],
                      do_constant_folding=True)

    # Backbone
    data = load_data(model, args.pcd)
    pts = data['points'][0]
    voxels, num_points, coors = model.voxelize(pts)
    voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0] + 1
    # 64 * 800 * 800
    scattered_features = model.pts_middle_encoder(
        voxel_features, coors, batch_size)
    backbone_model = build_backbone_model(cfg, args.checkpoint, device)
    torch.onnx.export(backbone_model,
                      scattered_features,
                      f='./tools/onnx_tools/centerpoint/backbone.onnx',
                      opset_version=12,
                      verbose=True,
                      input_names=['scattered_features'],
                      output_names=['scores', 'bbox_preds', 'dir_scores'],
                      do_constant_folding=True)


if __name__ == '__main__':
    main()