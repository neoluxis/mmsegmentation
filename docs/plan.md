# 基于 MMSegmentation 的焊缝 X 射线缺陷分割快速开发计划

## Summary

- 主线使用 SWRD 数据集训练缺陷分割模型：YOLO 框标注 -> 粗语义分割 mask -> MMSegmentation 训练/验证/可视化。
- 主模型为 SegFormer；对比模型固定为 UNet、PSPNet、DeepLabV3+、Mask2Former。
- 预处理保持四组：baseline、Gaussian 去噪、CLAHE、Gamma/对比度增强。
- RIAWELC 当前更像分类裁片集，未看到像素级 mask；默认用于跨数据集推理与可视化泛化观察。

## Implemented Framework

- 数据集路径来自 `envs`：`DATASET_PATH=$GRAD/../datasets`。
- SWRD 根目录：`SWRD/steel-tube-dataset-all/yolo`。
- SWRD mask 转换脚本：`mmsegmentation/tools/dataset_converters/swrd.py`。
- SWRD 检查脚本：`mmsegmentation/tools/dataset_converters/check_swrd.py`。
- RIAWELC 跨域推理脚本：`mmsegmentation/tools/analysis_tools/riawelc_infer.py`。
- 新增 2D 高斯去噪 transform：`GaussianDenoise`。

## Configs

- SegFormer baseline：`configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py`
- SegFormer Gaussian：`configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512_gaussian.py`
- SegFormer CLAHE：`configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512_clahe.py`
- SegFormer Gamma：`configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512_gamma.py`
- UNet：`configs/swrd/unet-s5-d16_fcn_4xb2-40k_swrd-512x512.py`
- PSPNet：`configs/swrd/pspnet_r50-d8_4xb2-40k_swrd-512x512.py`
- DeepLabV3+：`configs/swrd/deeplabv3plus_r50-d8_4xb2-40k_swrd-512x512.py`
- Mask2Former：`configs/swrd/mask2former_r50_4xb2-40k_swrd-512x512.py`

## Commands

```bash
source envs

python mmsegmentation/tools/dataset_converters/swrd.py \
  "$DATASET_PATH/SWRD/steel-tube-dataset-all/yolo" \
  --overwrite

python mmsegmentation/tools/dataset_converters/check_swrd.py \
  "$DATASET_PATH/SWRD/steel-tube-dataset-all/yolo"
```

Run training from `mmsegmentation`:

```bash
python tools/train.py configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/unet-s5-d16_fcn_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/pspnet_r50-d8_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/deeplabv3plus_r50-d8_4xb2-40k_swrd-512x512.py
python tools/train.py configs/swrd/mask2former_r50_4xb2-40k_swrd-512x512.py
```

Run RIAWELC qualitative cross-dataset inference:

```bash
python tools/analysis_tools/riawelc_infer.py \
  configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py \
  work_dirs/segformer_mit-b0_4xb2-40k_swrd-512x512/best_mIoU_iter_*.pth \
  --data-root "$DATASET_PATH/RIAWELC/DB/testing" \
  --out-dir work_dirs/riawelc_infer
```

## Test Plan

- 数据转换检查：SWRD `train=2726`、`val=682`，mask 数量与图像数量一致，mask 标签值只在 `0-8`。
- 预处理消融：固定 SegFormer，对 baseline、Gaussian、CLAHE、Gamma 输出 `mIoU/mAcc/aAcc`。
- 模型对比：在最佳预处理下训练 SegFormer、UNet、PSPNet、DeepLabV3+、Mask2Former。
- 跨域测试：最佳模型对 RIAWELC `testing` 目录推理，保存 overlay 和 `summary.csv`。

## Assumptions

- SWRD 是检测框标注，转换得到的是粗分割 mask，论文中需要说明该标签来源。
- RIAWELC 当前未发现像素级 mask，因此不默认做 mIoU 定量评估。
- DeepLab 默认采用 DeepLabV3+，因为它在 MMSeg 中配置成熟，也更适合作为强基线。
