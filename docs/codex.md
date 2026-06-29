套餐

基于 MMSegmentation 的焊缝 X 射线缺陷分割快速开发计划
Summary
主线使用 SWRD 数据集训练缺陷分割模型：YOLO 框标注 -> 粗语义分割 mask -> MMSegmentation 训练/验证/可视化。
主模型为 SegFormer；对比模型固定为 UNet、PSPNet、DeepLab、Mask2Former。
预处理保持当前计划：baseline、Gaussian 去噪、CLAHE、Gamma/对比度增强。
RIAWELC 当前更像分类裁片集，未看到像素级 mask；默认用于跨数据集推理与可视化泛化观察。
Key Changes
数据集处理：

使用 envs 中 DATASET_PATH=$GRAD/../datasets。
SWRD 根目录为 SWRD/steel-tube-dataset-all/yolo。
复用 tools/dataset_converters/swrd.py，将 YOLO 框转换为 annotations/train2021、annotations/val2021 分割 mask。
mask 标签规则保持 0=background，YOLO 0-7 映射为分割标签 1-8，模型统一 num_classes=9。
预处理实验：

保持四组：原图 baseline、Gaussian 去噪、CLAHE、Gamma/对比度增强。
每组固定数据划分、模型配置、训练轮数、随机种子，只改变预处理策略。
先在 SegFormer 上做预处理消融，选出最优预处理后再用于模型对比。
模型对比框架：

SegFormer：基于现有 configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py，快速阶段用 MiT-B0，正式阶段可升级 MiT-B1/B2。
UNet：基于 MMSeg 内置 configs/_base_/models/fcn_unet_s5-d16.py，改 SWRD 数据集、num_classes=9、512x512 crop。
PSPNet：基于现有 configs/swrd/pspnet_r50-d8_4xb2-40k_swrd-512x512.py。
DeepLab：优先使用 DeepLabV3+ R50 作为 DeepLab 对比项；若论文命名需要，统一写作 DeepLabV3+。
Mask2Former：使用 R50 backbone 的 MMSeg 配置改成 SWRD 数据集和 num_classes=9。
Test Plan
数据转换检查：确认 SWRD train=2726、val=682，mask 数量与图像数量一致，mask 标签值只在 0-8。
配置检查：用 tools/misc/print_config.py 验证 SegFormer、UNet、PSPNet、DeepLab、Mask2Former 配置可展开。
冒烟训练：每个模型先跑 100-500 iter，确认 dataloader、loss、checkpoint、visualization hook 正常。
预处理消融：固定 SegFormer，对四种预处理输出 mIoU/mAcc/aAcc，选择最佳方案。
模型对比：在最佳预处理下训练 SegFormer、UNet、PSPNet、DeepLab、Mask2Former，统一比较 mIoU/mAcc/aAcc 和单类 IoU。
跨域测试：最佳模型对 RIAWELC testing 目录推理，保存 overlay 和按目录类别汇总的预测结果。
Assumptions
SWRD 是检测框标注，转换得到的是粗分割 mask，论文中需要说明该标签来源。
RIAWELC 当前未发现像素级 mask，因此不默认做 mIoU 定量评估。
DeepLab 默认采用 DeepLabV3+，因为它在 MMSeg 中配置成熟，也更适合作为强基线。
