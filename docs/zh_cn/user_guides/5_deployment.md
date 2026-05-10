# 教程 5：模型部署

# MMSegmentation 模型部署

- [教程 5：模型部署](#教程-5模型部署)
- [MMSegmentation 模型部署](#mmsegmentation-模型部署)
  - [安装](#安装)
    - [安装 mmseg](#安装-mmseg)
    - [安装 mmdeploy](#安装-mmdeploy)
  - [模型转换](#模型转换)
  - [模型规范](#模型规范)
  - [模型推理](#模型推理)
    - [后端模型推理](#后端模型推理)
    - [SDK 模型推理](#sdk-模型推理)
  - [支持的模型](#支持的模型)
  - [注意事项](#注意事项)

______________________________________________________________________

[MMSegmentation](https://github.com/vbti-development/onedl-mmsegmentation/tree/main)，也称为 `mmseg`，是一个基于 PyTorch 的开源语义分割工具箱。

## 安装

### 安装 mmseg

请参考[安装指南](https://onedl-mmsegmentation.readthedocs.io/en/latest/get_started.html)。

### 安装 mmdeploy

可以通过以下方式安装 `mmdeploy`：

**方式 1：** 安装预编译包

请参考 [Installation overview](https://onedl-mmdeploy.readthedocs.io/en/latest/get_started.html#mmdeploy)。

**方式 2：** 自动安装脚本

如果部署平台是 **Ubuntu 18.04+**，请参考[脚本安装](../01-how-to-build/build_from_script.md)。
下面的命令展示了如何安装 mmdeploy 以及推理引擎 `ONNX Runtime`。

```shell
git clone --recursive -b main https://github.com/vbti-development/onedl-mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**注意**：

- 将 `$(pwd)/build/lib` 添加到 `PYTHONPATH` 后，可以加载 mmdeploy SDK 的 Python 包 `mmdeploy_runtime`。更多信息请参见 [SDK 模型推理](#sdk-模型推理)。
- 使用 [ONNX Runtime 模型推理](#后端模型推理) 时，需要加载自定义算子库，并将 ONNX Runtime 库路径添加到 `LD_LIBRARY_PATH`。

**方式 3：** 使用 mim 安装

1. 使用 mim 安装 mmcv

```shell
pip install -U onedl-mim
mim install "onedl-mmcv"
```

2. 安装 mmdeploy

```shell
git clone https://github.com/vbti-development/onedl-mmdeploy.git
cd mmdeploy
mim install -e .
```

**方式 4：** 从源码构建 MMDeploy

如果前三种方式不适用，请参考[从源码构建 MMDeploy](<(../01-how-to-build/build_from_source.md)>)。

## 模型转换

[tools/deploy.py](https://github.com/vbti-development/onedl-mmdeploy/tree/main/tools/deploy.py) 可以方便地将 mmseg 模型转换为后端模型。详细用法请参考[这里](https://github.com/vbti-development/onedl-mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage)。

下面以将 `unet` 转换为 onnx 模型为例：

```shell
cd onedl-mmdeploy

# 从 mmseg 模型库下载 unet 模型
mim download onedl-mmsegmentation --config unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024 --dest .

# 将 mmseg 模型转换为动态 shape 的 onnxruntime 模型
python tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_dynamic.py \
    unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py \
    fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth \
    demo/resources/cityscapes.png \
    --work-dir mmdeploy_models/mmseg/ort \
    --device cpu \
    --show \
    --dump-info
```

模型转换时，指定正确的部署配置非常关键。MMDeploy 已经为 mmsegmentation 支持的所有后端提供了内置部署配置[文件](https://github.com/vbti-development/onedl-mmdeploy/tree/main/configs/mmseg)，配置文件路径遵循以下模式：

```text
segmentation_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** 推理后端，例如 onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等。
- **{precision}:** fp16、int8。如果为空，则表示 fp32。
- **{static | dynamic}:** 静态 shape 或动态 shape。
- **{shape}:** 模型的输入 shape 或 shape 范围。

因此，在上面的示例中，也可以通过 `segmentation_tensorrt-fp16_dynamic-512x1024-2048x2048.py` 将 `unet` 转换为 tensorrt-fp16 模型。

```{tip}
将 mmsegmentation 模型转换为 tensorrt 模型时，--device 应设置为 "cuda"。
```

## 模型规范

在进入模型推理章节之前，先了解转换后的模型结构。它对后续推理非常重要。

以上一个示例为例，转换后的模型位于工作目录 `mmdeploy_models/mmseg/ort`，其中包括：

```text
mmdeploy_models/mmseg/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

其中：

- **end2end.onnx**：可由 ONNX Runtime 推理的后端模型。
- ***xxx*.json**：mmdeploy SDK 所需的必要信息。

整个 **mmdeploy_models/mmseg/ort** 包被定义为 **mmdeploy SDK model**。也就是说，**mmdeploy SDK model** 同时包含后端模型和推理元信息。

## 模型推理

### 后端模型推理

以前面转换得到的 `end2end.onnx` 模型为例，可以使用下面的代码进行模型推理并可视化结果：

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmseg/segmentation_onnxruntime_dynamic.py'
model_cfg = './unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmseg/ort/end2end.onnx']
image = './demo/resources/cityscapes.png'

# 读取 deploy_cfg 和 model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# 构建任务和后端模型
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# 处理输入图像
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# 执行模型推理
with torch.no_grad():
    result = model.test_step(model_inputs)

# 可视化结果
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='./output_segmentation.png')
```

### SDK 模型推理

也可以按如下方式使用 SDK 进行模型推理：

```python
from mmdeploy_runtime import Segmentor
import cv2
import numpy as np

img = cv2.imread('./demo/resources/cityscapes.png')

# 创建 segmentor
segmentor = Segmentor(model_path='./mmdeploy_models/mmseg/ort', device_name='cpu', device_id=0)
# 执行推理
seg = segmentor(img)

# 可视化推理结果
## 随机生成大小为 256x3 的调色板
palette = np.random.randint(0, 256, size=(256, 3))
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
    color_seg[seg == label, :] = color
# 转换为 BGR
color_seg = color_seg[..., ::-1]
img = img * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)
cv2.imwrite('output_segmentation.png', img)
```

除 Python API 外，mmdeploy SDK 还提供其他 FFI（Foreign Function Interface，外部函数接口），例如 C、C++、C#、Java 等。可以从 [demo](https://github.com/vbti-development/onedl-mmdeploy/tree/main/demo) 学习它们的用法。

## 支持的模型

| Model                                                                                                                 | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVino |
| :-------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: | :------: | :--: | :---: | :------: |
| [FCN](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/fcn)                                 |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [PSPNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/pspnet)[\*](#static_shape)        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [DeepLabV3](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/deeplabv3)                     |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [DeepLabV3+](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/deeplabv3plus)                |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [Fast-SCNN](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/fastscnn)[\*](#static_shape)   |      Y      |      Y      |    Y     |  N   |   Y   |    Y     |
| [UNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/unet)                               |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [ANN](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/ann)[\*](#static_shape)              |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [APCNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/apcnet)                           |      Y      |      Y      |    Y     |  Y   |   N   |    N     |
| [BiSeNetV1](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/bisenetv1)                     |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [BiSeNetV2](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/bisenetv2)                     |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [CGNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/cgnet)                             |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [DMNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/dmnet)                             |      ?      |      Y      |    N     |  N   |   N   |    N     |
| [DNLNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/dnlnet)                           |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |
| [EMANet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/emanet)                           |      Y      |      Y      |    Y     |  N   |   N   |    Y     |
| [EncNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/encnet)                           |      Y      |      Y      |    Y     |  N   |   N   |    Y     |
| [ERFNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/erfnet)                           |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [FastFCN](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/fastfcn)                         |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [GCNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/gcnet)                             |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [ICNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/icnet)[\*](#static_shape)          |      Y      |      Y      |    Y     |  N   |   N   |    Y     |
| [ISANet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/isanet)[\*](#static_shape)        |      N      |      Y      |    Y     |  N   |   N   |    Y     |
| [NonLocal Net](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/nonlocal_net)               |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |
| [OCRNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/ocrnet)                           |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [PointRend](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/point_rend)[\*](#static_shape) |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [Semantic FPN](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/sem_fpn)                    |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [STDC](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/stdc)                               |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [UPerNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/upernet)[\*](#static_shape)      |      N      |      Y      |    Y     |  N   |   N   |    N     |
| [DANet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/danet)                             |      ?      |      Y      |    Y     |  N   |   N   |    Y     |
| [Segmenter](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/segmenter)[\*](#static_shape)  |      N      |      Y      |    Y     |  Y   |   N   |    Y     |
| [SegFormer](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/segformer)[\*](#static_shape)  |      ?      |      Y      |    Y     |  N   |   N   |    Y     |
| [SETR](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/setr)                               |      ?      |      Y      |    N     |  N   |   N   |    Y     |
| [CCNet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/ccnet)                             |      ?      |      N      |    N     |  N   |   N   |    N     |
| [PSANet](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/psanet)                           |      ?      |      N      |    N     |  N   |   N   |    N     |
| [DPT](https://github.com/vbti-development/onedl-mmsegmentation/tree/main/configs/dpt)                                 |      ?      |      N      |    N     |  N   |   N   |    N     |

## 注意事项

- 所有 mmseg 模型仅支持 `'whole'` 推理模式。

- <i id="static_shape">PSPNet、Fast-SCNN</i> 仅支持静态输入，因为多数推理框架的 [nn.AdaptiveAvgPool2d](https://github.com/vbti-development/onedl-mmsegmentation/blob/0c87f7a0c9099844eff8e90fa3db5b0d0ca02fee/mmseg/models/decode_heads/psp_head.py#L38) 不支持动态输入。

- 对于只支持静态 shape 的模型，应使用静态 shape 部署配置文件，例如 `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`。

- 如果要部署用于生成概率特征图的模型，请在部署配置文件中添加 `codebase_config = dict(with_argmax=False)`。
