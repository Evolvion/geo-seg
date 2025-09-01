import torch
from geo_seg.metrics import binary_iou


def test_iou_all_zeros():
    logits = torch.full((2, 1, 4, 4), -10.0)
    target = torch.zeros((2, 4, 4), dtype=torch.long)
    iou = binary_iou(logits, target)
    assert iou == 1.0  # by convention when union==0


def test_iou_all_ones():
    logits = torch.full((1, 1, 4, 4), 10.0)
    target = torch.ones((1, 4, 4), dtype=torch.long)
    iou = binary_iou(logits, target)
    assert abs(iou - 1.0) < 1e-6


def test_iou_partial_overlap():
    logits = torch.tensor(
        [
            [
                [
                    [10, 10, -10, -10],
                    [10, 10, -10, -10],
                    [-10, -10, -10, -10],
                    [-10, -10, -10, -10],
                ]
            ]
        ],
        dtype=torch.float,
    )
    target = torch.tensor(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long
    )
    iou = binary_iou(logits, target)
    # intersection=3, union=4 -> IoU=0.75
    assert abs(iou - 0.75) < 1e-6
