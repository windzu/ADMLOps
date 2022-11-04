# Quick Test

## COCO_BDD10K
### Train

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/mmseg_train.py $ADMLOPS/configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py
```

### Resume

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/train.py $ADMLOPS/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
--resume-from $ADMLOPS/work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth
```

### Eval

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/test.py $ADMLOPS/local/configs/yolox/yolox_s_8x8_300e_coco-bdd10k.py \
./work_dirs/yolox_s_8x8_300e_coco-bdd10k/latest.pth \
--eval bbox
```

### ROS TEST

```bash
conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py ./configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py \
./checkpoints/segformer/segformer_mit-b0_512x512_160k_ade20k.pth \
--topic /CAM_FRONT/image_rect_compressed \
--device cuda:0 \
--score_thr 0.3 \
--msg_type img \
--republish \
--compressed
```

conda activate ADMLOps1.0 && \
cd $ADMLOPS && \
python3 tools/rosrun.py --config $ADMLOPS/configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py \
--checkpoint ./checkpoints/segformer/segformer_mit-b0_512x512_160k_ade20k.pth \
--score_thr 0.3 \
--device cuda:0 \
--task_type det3d \
--sub_topic /LIDAR_TOP \
--sub_msg_type pc \
--republish