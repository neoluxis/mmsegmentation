# 工具说明

本目录包含 MMSegmentation 常用命令行工具，用于训练、测试、日志分析、数据集转换、权重转换和部署。除非特别说明，以下命令都应在仓库根目录下运行。

## 训练与测试

| 工具 | 用途 | 基本用法 |
| --- | --- | --- |
| `train.py` | 根据配置文件训练分割模型。新的训练会保存到 `work_dirs/<config>/run_001` 这类编号子目录；`--resume` 会从指定目录或最新 run 目录恢复训练。 | `python tools/train.py <config> [--work-dir <dir>] [--resume] [--amp]` |
| `test.py` | 使用 checkpoint 进行测试或评估。支持结果导出、TTA、不同 launcher 以及可视化相关选项。 | `python tools/test.py <config> <checkpoint> [--out <file.pkl>] [--tta]` |
| `dist_train.sh` | 使用 PyTorch distributed 启动多卡训练。 | `bash tools/dist_train.sh <config> <gpus> [extra args]` |
| `dist_test.sh` | 使用 PyTorch distributed 启动多卡测试。 | `bash tools/dist_test.sh <config> <checkpoint> <gpus> [extra args]` |
| `slurm_train.sh` | 在 Slurm 集群上提交训练任务。 | `bash tools/slurm_train.sh <partition> <job_name> <config> [extra args]` |
| `slurm_test.sh` | 在 Slurm 集群上提交测试任务。 | `bash tools/slurm_test.sh <partition> <job_name> <config> <checkpoint> [extra args]` |

## 分析工具

`analysis_tools/` 目录用于检查数据集、分析日志、统计模型复杂度、测试推理速度和生成定性结果。

| 工具 | 用途 | 基本用法 |
| --- | --- | --- |
| `analysis_tools/analyze_logs.py` | 从 JSON 日志中绘制指定指标曲线。 | `python tools/analysis_tools/analyze_logs.py <log.json> --keys mIoU --out curve.png` |
| `analysis_tools/benchmark.py` | 统计指定 config 和 checkpoint 的推理 FPS。 | `python tools/analysis_tools/benchmark.py <config> <checkpoint>` |
| `analysis_tools/browse_dataset.py` | 根据配置文件可视化经过 pipeline 处理后的数据样本。 | `python tools/analysis_tools/browse_dataset.py <config> [--output-dir <dir>]` |
| `analysis_tools/confusion_matrix.py` | 根据分割预测结果生成混淆矩阵。 | `python tools/analysis_tools/confusion_matrix.py <config> <prediction_path> <save_dir>` |
| `analysis_tools/get_flops.py` | 统计模型 FLOPs 和参数量。 | `python tools/analysis_tools/get_flops.py <config> --shape 512 512` |
| `analysis_tools/plot_scalars.py` | 绘制 MMEngine 本地 visualizer 写出的 scalar 曲线，并导出 `scalars.csv`。可传入 work dir、编号 run 目录、timestamp 目录、`vis_data` 目录或 `scalars.json`。 | `python tools/analysis_tools/plot_scalars.py <work_dir_or_scalars_json>` |
| `analysis_tools/riawelc_infer.py` | 对 RIAWELC 这类按类别文件夹组织、没有像素级标注的数据做跨数据集推理，输出 `summary.csv`、`folder_summary.csv`，可选叠加图。 | `python tools/analysis_tools/riawelc_infer.py <config> <checkpoint> --data-root <root> [--no-overlays]` |
| `analysis_tools/run_riawelc_cross_dataset.sh` | 批量对 `configs/swrd/*40k_swrd-512x512*.py` 与对应 `iter_40000.pth` 做 RIAWELC 跨数据集推理。 | `bash tools/analysis_tools/run_riawelc_cross_dataset.sh` |
| `analysis_tools/run_swrd_val_visualizations.sh` | 批量使用 40000 轮权重在 SWRD 验证集上测试，并保存预测可视化图片。 | `bash tools/analysis_tools/run_swrd_val_visualizations.sh` |
| `analysis_tools/visualization_cam.py` | 使用 `pytorch-grad-cam` 可视化类别激活图。 | `python tools/analysis_tools/visualization_cam.py <img> <config> <checkpoint>` |

## 数据集转换工具

`dataset_converters/` 目录用于把原始数据集转换成 MMSegmentation 常用目录结构，通常为 `img_dir/<split>` 和 `ann_dir/<split>`。

| 工具 | 数据集或任务 | 基本用法 |
| --- | --- | --- |
| `dataset_converters/chase_db1.py` | 转换 CHASE_DB1 视网膜血管数据集。 | `python tools/dataset_converters/chase_db1.py <CHASEDB1.zip> -o data/CHASE_DB1` |
| `dataset_converters/check_swrd.py` | 检查由 YOLO 框生成的 SWRD mask；也可通过 `--convert` 先补齐缺失 mask。 | `python tools/dataset_converters/check_swrd.py <swrd_yolo_root> [--convert]` |
| `dataset_converters/cityscapes.py` | 将 Cityscapes 标注转换为 `labelTrainIds`。 | `python tools/dataset_converters/cityscapes.py <cityscapes_path> --gt-dir gtFine -o data/cityscapes` |
| `dataset_converters/coco_stuff10k.py` | 转换 COCO-Stuff 10k 标注。 | `python tools/dataset_converters/coco_stuff10k.py <coco_stuff10k_path> -o data/coco_stuff10k` |
| `dataset_converters/coco_stuff164k.py` | 转换 COCO-Stuff 164k 标注。 | `python tools/dataset_converters/coco_stuff164k.py <coco_stuff164k_path> -o data/coco_stuff164k` |
| `dataset_converters/drive.py` | 转换 DRIVE 视网膜血管数据集。 | `python tools/dataset_converters/drive.py <training.zip> <test.zip> -o data/DRIVE` |
| `dataset_converters/hrf.py` | 转换 HRF 视网膜血管数据集。 | `python tools/dataset_converters/hrf.py <healthy.zip> <healthy_manualsegm.zip> <glaucoma.zip> <glaucoma_manualsegm.zip> <diabetic_retinopathy.zip> <diabetic_retinopathy_manualsegm.zip> -o data/HRF` |
| `dataset_converters/isaid.py` | 转换 iSAID 遥感分割数据集，并进行滑窗裁剪。 | `python tools/dataset_converters/isaid.py <isaid_path> -o data/iSAID` |
| `dataset_converters/levircd.py` | 转换 LEVIR-CD 变化检测数据集。 | `python tools/dataset_converters/levircd.py --dataset_path <root> -o data/LEVIR-CD` |
| `dataset_converters/loveda.py` | 转换 LoveDA 遥感数据集。 | `python tools/dataset_converters/loveda.py <loveda_path> -o data/LoveDA` |
| `dataset_converters/nyu.py` | 转换 NYU Depth 数据集。 | `python tools/dataset_converters/nyu.py <raw_data> -o data/NYU` |
| `dataset_converters/pascal_context.py` | 转换 PASCAL Context 标注。 | `python tools/dataset_converters/pascal_context.py <VOCdevkit_path> <trainval_merged.json> -o data/pascal_context` |
| `dataset_converters/potsdam.py` | 转换 ISPRS Potsdam 数据集，并进行 patch 生成。 | `python tools/dataset_converters/potsdam.py <potsdam_path> -o data/potsdam` |
| `dataset_converters/refuge.py` | 转换 REFUGE 视盘/视杯数据集。 | `python tools/dataset_converters/refuge.py --raw_data_root <root> -o data/REFUGE` |
| `dataset_converters/stare.py` | 转换 STARE 视网膜血管数据集。 | `python tools/dataset_converters/stare.py <stare-images.tar> <labels-ah.tar> <labels-vk.tar> -o data/STARE` |
| `dataset_converters/swrd.py` | 将 SWRD 的 YOLO 框标注转换成粗粒度语义分割 mask。 | `python tools/dataset_converters/swrd.py <swrd_yolo_root> [--overwrite]` |
| `dataset_converters/synapse.py` | 转换 Synapse 多器官分割数据集。 | `python tools/dataset_converters/synapse.py --dataset-path <root> --save-path data/synapse` |
| `dataset_converters/vaihingen.py` | 转换 ISPRS Vaihingen 数据集，并进行 patch 生成。 | `python tools/dataset_converters/vaihingen.py <vaihingen_path> -o data/vaihingen` |
| `dataset_converters/voc_aug.py` | 转换 PASCAL VOC 增强标注。 | `python tools/dataset_converters/voc_aug.py <VOCdevkit_path> <aug_path> -o data/VOCdevkit` |

## 模型权重转换工具

`model_converters/` 目录用于把官方或第三方预训练权重转换为 MMSegmentation 使用的 key 命名。

| 工具 | 来源模型 | 基本用法 |
| --- | --- | --- |
| `model_converters/beit2mmseg.py` | BEiT 预训练权重。 | `python tools/model_converters/beit2mmseg.py <src> <dst>` |
| `model_converters/clip2mmseg.py` | CLIP 权重，用于 MMSeg 兼容的开放词汇模型。 | `python tools/model_converters/clip2mmseg.py <src> <dst>` |
| `model_converters/mit2mmseg.py` | SegFormer MiT 预训练权重。 | `python tools/model_converters/mit2mmseg.py <src> <dst>` |
| `model_converters/san2mmseg.py` | SAN 权重。 | `python tools/model_converters/san2mmseg.py <src> <dst>` |
| `model_converters/stdc2mmseg.py` | STDC1/STDC2 预训练权重。 | `python tools/model_converters/stdc2mmseg.py <src> <dst> <STDC1|STDC2>` |
| `model_converters/swin2mmseg.py` | Swin Transformer 预训练权重。 | `python tools/model_converters/swin2mmseg.py <src> <dst>` |
| `model_converters/twins2mmseg.py` | Twins PCPVT/SVT 预训练权重。 | `python tools/model_converters/twins2mmseg.py <src> <dst> <pcpvt|svt>` |
| `model_converters/vit2mmseg.py` | timm ViT 预训练权重。 | `python tools/model_converters/vit2mmseg.py <src> <dst>` |
| `model_converters/vitjax2mmseg.py` | 官方 JAX ViT 预训练权重。 | `python tools/model_converters/vitjax2mmseg.py <src> <dst>` |

## 部署工具

| 工具 | 用途 | 基本用法 |
| --- | --- | --- |
| `deployment/pytorch2onnx.py` | 将 MMSegmentation 模型导出为 ONNX 文件。支持固定输入尺寸、动态 batch/高/宽，以及可选输出 resize。 | `python tools/deployment/pytorch2onnx.py <config> --checkpoint <checkpoint> --output-file model.onnx --shape 512 512` |
| `deployment/pytorch2torchscript.py` | trace MMSegmentation 模型并导出 TorchScript 文件。 | `python tools/deployment/pytorch2torchscript.py <config> --checkpoint <checkpoint> --output-file model.pt --shape 512 512` |

## 其他工具

| 工具 | 用途 | 基本用法 |
| --- | --- | --- |
| `misc/browse_dataset.py` | 根据配置文件可视化数据集样本，功能与 `analysis_tools/browse_dataset.py` 类似。 | `python tools/misc/browse_dataset.py <config> [--output-dir <dir>]` |
| `misc/print_config.py` | 打印解析后的完整配置，支持 import 和 `cfg-options` 覆盖。 | `python tools/misc/print_config.py <config> [--cfg-options key=value]` |
| `misc/publish_model.py` | 清理 checkpoint 中不适合发布的内容，并在输出文件名中追加哈希。 | `python tools/misc/publish_model.py <input.pth> <output.pth>` |

## TorchServe

| 工具 | 用途 | 基本用法 |
| --- | --- | --- |
| `torchserve/mmseg2torchserve.py` | 将 config 和 checkpoint 打包为 TorchServe `.mar` 模型归档。 | `python tools/torchserve/mmseg2torchserve.py <config> <checkpoint> --output-folder <dir> --model-name <name>` |
| `torchserve/mmseg_handler.py` | TorchServe 推理 handler，由 `mmseg2torchserve.py` 打包引用，通常不需要直接运行。 | 不适用 |
| `torchserve/test_torchserve.py` | 对比 TorchServe 推理结果和本地 PyTorch 推理结果。 | `python tools/torchserve/test_torchserve.py <img> <config> <checkpoint> <model_name>` |

## 使用提示

- 运行 `python <tool> --help` 可以查看脚本的完整参数列表。
- 数据集转换脚本可能依赖特定 Python 包或特定原始数据目录结构。如果官方数据存在多种下载格式，遇到路径错误时建议先查看对应 converter 源码。
- `__pycache__/` 这类运行时生成文件不是工具，因此没有列入本文档。
