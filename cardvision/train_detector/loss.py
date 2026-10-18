# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX),
# Copyright (c) Megvii Inc., Apache-2.0; modified:
#
#   * IgnoreRegionHead.get_losses is YOLOXHead.get_losses from yolox/models/yolo_head.py at
#     commit 6ddff4824372906469a7fae2dc3206c7aa4bbaee, written out again with one change to
#     the loss: the objectness term of an anchor is multiplied by a per-anchor weight, which
#     is 0 for an anchor that is not assigned to a target and falls in an ignore region, and
#     1 for every other anchor. Also changed: the retry of the assignment on the CPU after an
#     out-of-memory error, and the clearing of the CUDA cache, are left out; the number of
#     anchors given weight 0 is kept for diagnostics.
#
# The upstream file carries the notice "Copyright (c) Megvii Inc. All rights reserved."
# The assignment of anchors to targets (get_assignments), the box, class and L1 terms, the
# normalizer and the weights between the terms are the library's and are called or kept as
# they are. ignore_anchor_mask is not derived from the library.
"""The detector's loss with ignore regions.

A frame often holds things that are neither a card to find nor plain background: a card too
hidden to ask for, a stack whose members cannot be boxed one by one, a face nobody could
judge. Training on them as background would teach the detector to stay silent on cards, and
training on them as cards would teach it boxes nobody wants. An *ignore region* says: no
opinion here.

The detector scores every anchor (a cell of one of its three output grids) for objectness.
With ignore regions, an anchor's objectness term is left out of the loss when both hold:

* the anchor was not assigned to a face or a back (assigned anchors always count: a target
  that overlaps an ignore region is still learned), and
* the anchor's center lies inside an ignore region, or at least ``ignore_overlap`` (half, by
  default) of the box the anchor currently predicts lies inside one single ignore region.

Nothing else changes. In particular the loss is still divided by the number of assigned
anchors, as in the library.

Ignore regions are handed to the head before each forward pass, as ``head.ignore_boxes``: a
float tensor ``(batch, m, 4)`` of ``(cx, cy, w, h)`` in input pixels, where a row of zeros is
padding. ``None`` means no ignore regions, and the loss is then the library's.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from yolox.models.yolo_head import YOLOXHead

DEFAULT_IGNORE_OVERLAP = 0.5


def ignore_anchor_mask(ignore_boxes: torch.Tensor, anchor_cx: torch.Tensor, anchor_cy: torch.Tensor,
                       predicted: torch.Tensor, overlap: float = DEFAULT_IGNORE_OVERLAP) -> torch.Tensor:
    """Which anchors fall in an ignore region.

    Args:
        ignore_boxes: ``(m, 4)`` ignore regions as ``(cx, cy, w, h)`` in input pixels, without
            padding rows.
        anchor_cx, anchor_cy: ``(a,)`` anchor centers in input pixels.
        predicted: ``(a, 4)`` the boxes the anchors predict, ``(cx, cy, w, h)`` in input pixels.
        overlap: the share of a predicted box's area that must lie inside one ignore region.

    Returns a boolean ``(a,)`` tensor: true where the anchor's center is inside a region (edges
    included) or its predicted box overlaps a region by at least ``overlap`` of its own area.
    """
    if ignore_boxes.numel() == 0:
        return torch.zeros_like(anchor_cx, dtype=torch.bool)
    ix0 = (ignore_boxes[:, 0] - ignore_boxes[:, 2] / 2)[None]
    iy0 = (ignore_boxes[:, 1] - ignore_boxes[:, 3] / 2)[None]
    ix1 = (ignore_boxes[:, 0] + ignore_boxes[:, 2] / 2)[None]
    iy1 = (ignore_boxes[:, 1] + ignore_boxes[:, 3] / 2)[None]
    inside = ((anchor_cx[:, None] >= ix0) & (anchor_cx[:, None] <= ix1) &
              (anchor_cy[:, None] >= iy0) & (anchor_cy[:, None] <= iy1)).any(1)
    px0 = (predicted[:, 0] - predicted[:, 2] / 2)[:, None]
    py0 = (predicted[:, 1] - predicted[:, 3] / 2)[:, None]
    px1 = (predicted[:, 0] + predicted[:, 2] / 2)[:, None]
    py1 = (predicted[:, 1] + predicted[:, 3] / 2)[:, None]
    shared_w = (torch.minimum(px1, ix1) - torch.maximum(px0, ix0)).clamp(min=0)
    shared_h = (torch.minimum(py1, iy1) - torch.maximum(py0, iy0)).clamp(min=0)
    area = ((px1 - px0).clamp(min=0) * (py1 - py0).clamp(min=0)).clamp(min=1e-6)
    return inside | ((shared_w * shared_h) / area >= overlap).any(1)


class IgnoreRegionHead(YOLOXHead):
    """The YOLOX head with ignore regions in its objectness loss. It adds no parameters."""

    ignore_boxes: torch.Tensor | None = None  # set by the caller before each forward pass
    ignore_overlap: float = DEFAULT_IGNORE_OVERLAP
    last_ignored_anchors: int = 0  # anchors given weight 0 in the last batch

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        anchor_cx = (x_shifts[0] + 0.5) * expanded_strides[0]
        anchor_cy = (y_shifts[0] + 0.5) * expanded_strides[0]

        cls_targets, reg_targets, l1_targets, obj_targets, fg_masks, obj_weights = [], [], [], [], [], []
        num_fg, num_gts, ignored = 0.0, 0.0, 0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds,
                 num_fg_img) = self.get_assignments(
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds)
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask])

            # The change: weight 0 on the objectness of unassigned anchors inside ignore regions.
            weight = outputs.new_ones((total_num_anchors, 1))
            if self.ignore_boxes is not None:
                regions = self.ignore_boxes[batch_idx]
                regions = regions[(regions[:, 2] > 0) & (regions[:, 3] > 0)]
                if regions.numel():
                    with torch.no_grad():
                        masked = ignore_anchor_mask(regions.to(outputs.dtype), anchor_cx, anchor_cy,
                                                    bbox_preds[batch_idx].detach(), self.ignore_overlap)
                        masked = masked & ~fg_mask
                    weight[masked] = 0.0
                    ignored += int(masked.sum())

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            obj_weights.append(weight)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        obj_weights = torch.cat(obj_weights, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        self.last_ignored_anchors = ignored

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets) * obj_weights).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return (loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1))


def set_ignore_regions(model: Any, ignore_boxes: torch.Tensor | None) -> None:
    """Hand a batch's ignore regions to the model's head, for the next forward pass."""
    model.head.ignore_boxes = ignore_boxes
