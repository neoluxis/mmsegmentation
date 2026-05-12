# SWRD

SWRD steel tube defect dataset support in this repository converts YOLO box
annotations into coarse semantic segmentation masks for MMSegmentation.

## Prepare Dataset

The default config reads the dataset from ``DATASET_PATH``:

```text
$DATASET_PATH/SWRD/steel-tube-dataset-all/yolo
```

From the workspace root:

```shell
source envs
python mmsegmentation/tools/dataset_converters/swrd.py \
  "$DATASET_PATH/SWRD/steel-tube-dataset-all/yolo" \
  --overwrite
python mmsegmentation/tools/dataset_converters/check_swrd.py \
  "$DATASET_PATH/SWRD/steel-tube-dataset-all/yolo"
```

The script writes masks to:

```text
/datasets/SWRD/steel-tube-dataset-all/yolo/annotations/train2021
/datasets/SWRD/steel-tube-dataset-all/yolo/annotations/val2021
```

Mask label `0` is background. Original YOLO labels `0-7` are converted to
semantic labels `1-8`.

## Train and Compare

Run from the `mmsegmentation` project directory:

```shell
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/unet-s5-d16_fcn_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/pspnet_r50-d8_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/deeplabv3plus_r50-d8_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/mask2former_r50_4xb2-40k_swrd-512x512.py
```

## Preprocessing Ablation

```shell
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512_gaussian.py
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512_clahe.py
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512_gamma.py
```

## RIAWELC Qualitative Inference

RIAWELC currently has classification-style folders and no pixel-level masks in
this workspace. Use it for qualitative cross-dataset inference:

```shell
python tools/analysis_tools/riawelc_infer.py \
  configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py \
  work_dirs/segformer_mit-b0_4xb2-40k_swrd-512x512/best_mIoU_iter_*.pth \
  --data-root "$DATASET_PATH/RIAWELC/DB/testing" \
  --out-dir work_dirs/riawelc_infer
```
