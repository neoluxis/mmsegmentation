# MMCV / Mask2Former 环境排障复盘

日期：2026-05-13  

Commit: 
- `mmsegmentation` @ `5934b3d`
- `onedl-mmcv` @ `bdafaa1`

相关目录：

- MMSegmentation：`/home/neolux/workspace/2026/grad/code/mmsegmentation`
- MMCV 源码：`/home/neolux/workspace/2026/grad/code/onedl-mmcv`
- Conda 环境：`mmseg1`
- 目标显卡：NVIDIA GeForce RTX 5080

## 现象

运行：

```bash
./task.sh configs/swrd/mask2former_r50_4xb2-40k_swrd-512x512.py
```

先后遇到的问题：

1. `KeyError: Duplicate key is not allowed among bases`
2. `ImportError: Failed to import mmdet.models`
3. 手动 `import mmdet` 可以，但 `import mmdet.models` 失败
4. 根因进一步定位为 `ModuleNotFoundError: No module named 'mmcv._ext'`
5. 编译出 `mmcv._ext` 后，又遇到 NumPy 2 与旧版 `imgaug` 的兼容问题
6. 继续导入后缺少 `faster_coco_eval`

## 已修改的配置

文件：

```text
configs/swrd/mask2former_r50_4xb2-40k_swrd-512x512.py
```

处理了重复 base 问题：

- `_base_` 只保留 `../mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py`
- 去掉重复引入的 dataset / schedule base
- 在当前配置中显式保留 SWRD 数据集、dataloader、pipeline、schedule 等设置

同时加入了 NumPy 2 兼容 shim，用于绕过旧依赖 `imgaug` 访问 `np.sctypes` / `np.complex` / `np.bool` 的问题。

## Duplicate key 的原因

原配置同时继承了 Mask2Former ADE20K 配置和 SWRD dataset / schedule base。多个 base 中重复定义了这些 key：

```text
test_pipeline
val_dataloader
train_dataloader
test_evaluator
tta_pipeline
val_evaluator
data_root
test_dataloader
train_pipeline
img_ratios
dataset_type
crop_size
```

MMEngine 不允许多个 base 中出现同名顶层 key，所以需要让当前配置只继承一个完整模型 base，然后在当前文件里覆盖数据集和训练策略。

## mmcv._ext 的根因

`mmdet.models` 会触发 MMCV ops 的导入，进而需要 `mmcv._ext`。虽然 `import mmdet` 可以成功，但那只是导入包入口，并不会加载需要 C++/CUDA 扩展的模型组件。

检查命令：

```bash
source /home/neolux/.zshrc
conda activate mmseg1
python - <<'PY'
import importlib.util
import mmcv
print(mmcv.__version__, mmcv.__file__)
print(importlib.util.find_spec('mmcv._ext'))
PY
```

修复前 `find_spec('mmcv._ext')` 为 `None`。

## 编译 MMCV 扩展

环境信息：

- Python 3.14.4
- PyTorch 2.10.0
- CUDA runtime 12.8
- 本机 CUDA / conda CUDA dev packages 使用 12.9
- RTX 5080 需要显式指定较新的架构，本次使用 `TORCH_CUDA_ARCH_LIST=12.0`

安装构建所需 conda 包：

```bash
conda install -y cuda-cudart-dev=12.9 cuda-driver-dev=12.9
conda install -y cuda-crt-dev_linux-64=12.9.86 cuda-nvcc-dev_linux-64=12.9.86
conda install -y cuda-libraries-dev=12.9.1
conda install -y gcc_linux-64=14 gxx_linux-64=14
```

最终成功的构建命令：

```bash
cd /home/neolux/workspace/2026/grad/code/onedl-mmcv
source /home/neolux/.zshrc
conda activate mmseg1

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=6

python setup.py build_ext --inplace -v
```

产物：

```text
/home/neolux/workspace/2026/grad/code/onedl-mmcv/mmcv/_ext.cpython-314-x86_64-linux-gnu.so
```

## 中间踩坑

### cuda_runtime.h 缺失

原因：只安装 runtime 不够，缺少 CUDA dev headers。

处理：安装 `cuda-cudart-dev`、`cuda-driver-dev`。

### crt/host_config.h 缺失

原因：缺少 CUDA CRT / NVCC dev 组件。

处理：安装 `cuda-crt-dev_linux-64`、`cuda-nvcc-dev_linux-64`。

### cusparse.h 缺失

原因：缺少 CUDA libraries dev headers。

处理：安装 `cuda-libraries-dev`。

### GCC 版本不兼容

系统 GCC 太新，NVCC 编译失败。

处理：安装并使用 conda 的 GCC/G++ 14：

```bash
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
```

### cuda_runtime_api.h 缺失

原因：C++ 编译阶段没有找到 conda CUDA target include。

处理：

```bash
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
```

## mmdet.models 后续问题

编译 `mmcv._ext` 后，`mmdet.models` 继续失败于：

```text
AttributeError: np.sctypes was removed in NumPy 2.0
```

这是旧版 `imgaug` 与 NumPy 2 的兼容问题。当前在 Mask2Former 配置中加入了运行时 shim。

之后又缺少：

```text
ModuleNotFoundError: No module named 'faster_coco_eval'
```

已安装：

```bash
python -m pip install faster-coco-eval
```

## 验证结果

验证 `mmcv._ext`：

```bash
source /home/neolux/.zshrc
conda activate mmseg1
python - <<'PY'
import importlib.util
import mmcv
print('mmcv', mmcv.__version__, mmcv.__file__)
print('ext spec', importlib.util.find_spec('mmcv._ext'))
import mmcv._ext
print('mmcv._ext ok', mmcv._ext.__file__)
PY
```

结果：

```text
mmcv 2.3.5 /home/neolux/workspace/2026/grad/code/onedl-mmcv/mmcv/__init__.py
mmcv._ext ok /home/neolux/workspace/2026/grad/code/onedl-mmcv/mmcv/_ext.cpython-314-x86_64-linux-gnu.so
```

验证 `mmdet.models`：

```bash
source /home/neolux/.zshrc
conda activate mmseg1
python - <<'PY'
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'float': [np.float16, np.float32, np.float64],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'complex': [np.complex64, np.complex128],
        'others': [np.bool_, np.object_, np.bytes_, np.str_],
    }
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import mmdet.models
print('mmdet.models ok')
PY
```

结果：

```text
mmdet.models ok
```

## 当前 task.sh 运行状态

再次运行：

```bash
./task.sh configs/swrd/mask2former_r50_4xb2-40k_swrd-512x512.py
```

已经越过了：

- Duplicate key
- `mmdet.models`
- `mmcv._ext`
- NumPy 2 / `imgaug`
- `faster_coco_eval`

在当前受限执行环境里，后续失败点变为：

```text
OSError: [Errno 30] Read-only file system:
'/home/neolux/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth...partial'
```

这是因为 PyTorch 正在下载 ResNet50 预训练权重到只读的 `~/.cache`。真实本机 shell 中如果 `~/.cache` 可写，通常可以继续；或者提前设置：

```bash
export TORCH_HOME=/tmp/torch-cache
```

另外，当前工具沙箱里还出现了 multiprocessing socket 权限限制：

```text
PermissionError: [Errno 1] Operation not permitted
```

这是沙箱限制，不代表本机正常终端一定会失败。

## 后续建议

1. 在真实终端中重新运行 `task.sh`，确认 CUDA 是否可用：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
PY
```

2. 如果还要重新编译 MMCV，优先复用本文的最终构建命令。
3. 如果不想使用 config 内的 NumPy shim，更稳的长期方案是将 NumPy 降到 1.26.x，或替换/升级依赖链中使用 `imgaug` 的部分。
4. 如果预训练权重下载受限，可以手动下载 `resnet50-0676ba61.pth` 并放到 `TORCH_HOME/hub/checkpoints/`。
