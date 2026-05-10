# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SWRDDataset(BaseSegDataset):
    """SWRD steel tube defect segmentation dataset.

    The original SWRD annotations in this workspace are YOLO detection boxes.
    Use ``tools/dataset_converters/swrd.py`` to convert them into semantic
    segmentation masks before training MMSegmentation models.
    """

    METAINFO = dict(
        classes=(
            'background',
            'air-hole12(hollow-bead)-142',
            'air-hole7-028',
            'broken-arc2-064',
            'broken-arc2-156',
            'air-hole4(hollow-bead)-105',
            'broken-arc3(air-hole)-282',
            'crack043',
            'air-hole10-026',
        ),
        palette=[
            [0, 0, 0],
            [220, 20, 60],
            [255, 127, 14],
            [44, 160, 44],
            [31, 119, 180],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [188, 189, 34],
        ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
