# SWRD

SWRD steel tube defect dataset support in this repository converts YOLO box
annotations into coarse semantic segmentation masks for MMSegmentation.

## Prepare Dataset

The default config expects the dataset to be mounted in Docker as:

```text
/datasets/SWRD/steel-tube-dataset-all/yolo
```

From the workspace root:

```shell
python onedl-mmsegmentation/tools/dataset_converters/swrd.py \
  "/datasets/SWRD/steel-tube-dataset-all/yolo" \
  --overwrite
```

The script writes masks to:

```text
/datasets/SWRD/steel-tube-dataset-all/yolo/annotations/train2021
/datasets/SWRD/steel-tube-dataset-all/yolo/annotations/val2021
```

Mask label `0` is background. Original YOLO labels `0-7` are converted to
semantic labels `1-8`.

## Train

Run from the `onedl-mmsegmentation` project directory:

```shell
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py
```

The PSPNet config is also available for comparison:

```shell
python tools/train.py configs/swrd/pspnet_r50-d8_4xb2-40k_swrd-512x512.py
```
