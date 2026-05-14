# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import numpy as np
import torch
from mmengine import Config, digit_version
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch import nn
from torch.nn import functional as F

if not hasattr(np, 'sctypes'):
    np.sctypes = dict(
        float=[np.float16, np.float32, np.float64],
        int=[np.int8, np.int16, np.int32, np.int64],
        uint=[np.uint8, np.uint16, np.uint32, np.uint64],
        complex=[np.complex64, np.complex128],
        others=[np.bool_, np.object_, np.bytes_, np.str_])
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = np.bool_

from mmseg.models import build_segmentor


def check_torch_version():
    torch_minimum_version = '1.8.0'
    torch_version = digit_version(torch.__version__)

    assert torch_version >= digit_version(torch_minimum_version), (
        f'Torch=={torch.__version__} does not support reliable ONNX export. '
        f'Please install pytorch>={torch_minimum_version}.')


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


class ONNXSegmentorWrapper(nn.Module):
    """Wrap a MMSegmentation segmentor with a tensor-only forward."""

    def __init__(self, model: nn.Module, resize_output: bool = False):
        super().__init__()
        self.model = model
        self.resize_output = resize_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model.decode_head.__class__.__name__ == 'Mask2FormerHead':
            image_shape = tuple(inputs.shape[2:])
            batch_img_metas = [
                dict(
                    ori_shape=image_shape,
                    img_shape=image_shape,
                    pad_shape=image_shape,
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
            seg_logits = self.model.inference(inputs, batch_img_metas)
        else:
            seg_logits = self.model(inputs, mode='tensor')
        if self.resize_output:
            seg_logits = F.interpolate(
                seg_logits,
                size=inputs.shape[2:],
                mode='bilinear',
                align_corners=self.model.align_corners)
        return seg_logits


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMSeg to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('tmp.onnx'),
        help='path to save the exported ONNX model')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size. Use one value for square input, or two values '
        'for height and width')
    parser.add_argument(
        '--opset-version',
        type=int,
        default=11,
        help='ONNX opset version. Default: 11')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='export ONNX with dynamic batch, height, and width axes')
    parser.add_argument(
        '--resize-output',
        action='store_true',
        help='resize output logits to the input spatial size')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='run onnx.checker after export. Requires the onnx package')
    return parser.parse_args()


def build_input_shape(shape):
    if len(shape) == 1:
        return (1, 3, shape[0], shape[0])
    if len(shape) == 2:
        return (1, 3, shape[0], shape[1])
    raise ValueError('invalid input shape')


def build_model(config, checkpoint=None):
    init_default_scope('mmseg')
    cfg = Config.fromfile(config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=None)
    model = _convert_batchnorm(model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    return model


def export_onnx(model, input_shape, output_file, opset_version,
                dynamic_export, resize_output):
    wrapper = ONNXSegmentorWrapper(model, resize_output=resize_output).eval()
    dummy_input = torch.randn(*input_shape)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    use_legacy_export = model.decode_head.__class__.__name__ == 'Mask2FormerHead'

    dynamic_axes = None
    if dynamic_export:
        output_height = 'height' if resize_output else 'out_height'
        output_width = 'width' if resize_output else 'out_width'
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'seg_logits': {
                0: 'batch',
                2: output_height,
                3: output_width
            }
        }

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_file),
        input_names=['input'],
        output_names=['seg_logits'],
        dynamic_axes=dynamic_axes,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        opset_version=opset_version,
        dynamo=not use_legacy_export)


def verify_onnx(output_file):
    try:
        import onnx
    except ImportError as exc:
        raise ImportError('Please install onnx to use --verify.') from exc

    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)


def main():
    args = parse_args()
    check_torch_version()

    input_shape = build_input_shape(args.shape)
    model = build_model(args.config, args.checkpoint)
    export_onnx(
        model,
        input_shape,
        args.output_file,
        opset_version=args.opset_version,
        dynamic_export=args.dynamic_export,
        resize_output=args.resize_output)

    if args.verify:
        verify_onnx(args.output_file)

    print(f'Successfully exported ONNX model: {args.output_file}')


if __name__ == '__main__':
    main()
