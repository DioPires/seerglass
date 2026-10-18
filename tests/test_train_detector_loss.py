# SPDX-License-Identifier: Apache-2.0
"""The ignore-region loss: the anchor mask against cases worked out by hand, and the loss on
small made-up predictions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yolox")

from cardvision.train_detector import loss as L  # noqa: E402
from cardvision.train_detector import model as M  # noqa: E402

# A 96 x 64 input: grids of 12 x 8, 6 x 4 and 3 x 2 cells at strides 8, 16 and 32.
INPUT_W, INPUT_H = 96, 64
STRIDES = (8, 16, 32)
ANCHORS = 96 + 24 + 6


def grids():
    """The head's own bookkeeping for that input: per stride, each anchor's cell and stride."""
    x_shifts, y_shifts, strides = [], [], []
    for stride in STRIDES:
        rows, columns = INPUT_H // stride, INPUT_W // stride
        ys, xs = torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij")
        x_shifts.append(xs.reshape(1, -1).float())
        y_shifts.append(ys.reshape(1, -1).float())
        strides.append(torch.full((1, rows * columns), float(stride)))
    return x_shifts, y_shifts, strides


def centers():
    x_shifts, y_shifts, strides = (torch.cat(part, 1)[0] for part in grids())
    return (x_shifts + 0.5) * strides, (y_shifts + 0.5) * strides


def predictions(box_size: float = 2.0, seed: int = 0) -> torch.Tensor:
    """``(1, anchors, 7)``: a small box on each anchor's center, and random objectness and
    class logits."""
    cx, cy = centers()
    generator = torch.Generator().manual_seed(seed)
    out = torch.zeros((1, ANCHORS, 7))
    out[0, :, 0], out[0, :, 1] = cx, cy
    out[0, :, 2:4] = box_size
    out[0, :, 4:] = torch.randn((ANCHORS, 3), generator=generator)
    return out


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(0)
    return M.build_model(M.nano_experiment(2)).head


def losses(head, outputs, labels, ignore, use_l1=False):
    head.use_l1 = use_l1
    head.ignore_boxes = ignore
    x_shifts, y_shifts, strides = grids()
    origin = [torch.zeros((1, part.shape[1], 4)) for part in x_shifts]
    return head.get_losses(None, x_shifts, y_shifts, strides, labels, outputs, origin, dtype=outputs.dtype)


# ----------------------------------------------------------------------------------- the mask
def test_the_mask_by_hand():
    region = torch.tensor([[50.0, 50.0, 20.0, 20.0]])  # x 40..60, y 40..60
    anchor_cx = torch.tensor([50.0, 40.0, 60.0, 39.9, 80.0, 80.0, 80.0, 80.0])
    anchor_cy = torch.tensor([50.0, 40.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    predicted = torch.tensor([
        [200.0, 200.0, 4.0, 4.0],   # center in the middle of the region
        [200.0, 200.0, 4.0, 4.0],   # center on its corner: edges count as inside
        [200.0, 200.0, 4.0, 4.0],   # center on the opposite corner
        [200.0, 200.0, 4.0, 4.0],   # center just left of it
        [50.0, 50.0, 10.0, 10.0],   # center outside, predicted box wholly inside
        [60.0, 50.0, 20.0, 20.0],   # predicted box x 50..70: exactly half inside
        [61.0, 50.0, 20.0, 20.0],   # x 51..71: 45 % inside
        [50.0, 50.0, 40.0, 40.0],   # covers the region, but only a quarter of it is inside
    ])
    mask = L.ignore_anchor_mask(region, anchor_cx, anchor_cy, predicted)
    assert mask.tolist() == [True, True, True, False, True, True, False, False]
    # The overlap is with one region at a time: two regions holding a quarter each do not add up.
    two = torch.tensor([[45.0, 50.0, 10.0, 20.0], [65.0, 50.0, 10.0, 20.0]])  # x 40..50 and 60..70
    wide = torch.tensor([[55.0, 50.0, 40.0, 20.0]])  # x 35..75: a quarter in each
    far = torch.tensor([200.0])
    assert L.ignore_anchor_mask(two, far, far, wide).tolist() == [False]
    assert L.ignore_anchor_mask(two, far, far, wide, overlap=0.25).tolist() == [True]
    assert L.ignore_anchor_mask(region[:0], anchor_cx, anchor_cy, predicted).tolist() == [False] * 8
    # A predicted box without area overlaps nothing.
    assert L.ignore_anchor_mask(region, far, far, torch.tensor([[50.0, 50.0, 0.0, 0.0]])).tolist() == [False]


# ----------------------------------------------------------------------------------- the loss
def test_objectness_inside_an_ignore_region_is_left_out(head):
    outputs = predictions()
    labels = torch.zeros((1, 5, 5))  # nothing to find
    cx, _ = centers()
    left = cx <= 48  # the region below: the left half, edge included
    assert int(left.sum()) == 48 + 12 + 4
    ignore = torch.zeros((1, 3, 4))
    ignore[0, 0] = torch.tensor([24.0, 32.0, 48.0, 64.0])  # x 0..48, y 0..64; the other rows are padding

    total, iou, obj, cls, l1, _ = losses(head, outputs, labels, ignore)
    # With nothing assigned, the loss is the objectness cost of saying "something" where
    # nothing is, summed over the anchors that are not in the region.
    expected = torch.nn.functional.softplus(outputs[0, ~left, 4]).sum()
    assert float(total) == pytest.approx(float(expected), rel=1e-6)
    assert float(obj) == float(total) and float(iou) == float(cls) == 0.0 and l1 == 0.0
    assert head.last_ignored_anchors == 64

    # Changing what is predicted inside the region changes nothing...
    changed = outputs.clone()
    changed[0, left, 4:] += 3.0 * torch.randn((int(left.sum()), 3), generator=torch.Generator().manual_seed(1))
    assert float(losses(head, changed, labels, ignore)[0]) == float(total)
    # ...changing it anywhere else does...
    outside = outputs.clone()
    outside[0, int((~left).nonzero()[0]), 4] += 1.0
    assert float(losses(head, outside, labels, ignore)[0]) != float(total)
    # ...and without the region every anchor counts.
    everything = torch.nn.functional.softplus(outputs[0, :, 4]).sum()
    assert float(losses(head, outputs, labels, None)[0]) == pytest.approx(float(everything), rel=1e-6)
    assert head.last_ignored_anchors == 0


def test_the_gradient_is_zero_inside_an_ignore_region(head):
    outputs = predictions().requires_grad_(True)
    cx, cy = centers()
    inside = (cx >= 16) & (cx <= 80) & (cy >= 0) & (cy <= 32)
    ignore = torch.tensor([[[48.0, 16.0, 64.0, 32.0]]])  # x 16..80, y 0..32
    losses(head, outputs, torch.zeros((1, 5, 5)), ignore)[0].backward()
    gradient = outputs.grad[0, :, 4]
    assert (gradient[inside] == 0).all() and (gradient[~inside] != 0).all()
    assert head.last_ignored_anchors == int(inside.sum()) > 0


def test_an_anchor_whose_predicted_box_lies_in_a_region_is_left_out_too(head):
    outputs = predictions()
    cx, _ = centers()
    right = int((cx > 80).nonzero()[0])  # an anchor well right of the region below
    ignore = torch.tensor([[[16.0, 32.0, 32.0, 64.0]]])  # x 0..32
    base = float(losses(head, outputs, torch.zeros((1, 5, 5)), ignore)[0])
    counted = head.last_ignored_anchors
    moved = outputs.clone()
    moved[0, right, :4] = torch.tensor([16.0, 32.0, 8.0, 8.0])  # it now predicts a box inside the region
    less = float(losses(head, moved, torch.zeros((1, 5, 5)), ignore)[0])
    assert head.last_ignored_anchors == counted + 1
    assert base - less == pytest.approx(float(torch.nn.functional.softplus(outputs[0, right, 4])), rel=1e-4)


def test_assigned_anchors_count_even_inside_an_ignore_region(head):
    # One face in the middle, 24 x 32 px, and an ignore region over the whole input.
    labels = torch.zeros((1, 5, 5))
    labels[0, 0] = torch.tensor([0.0, 48.0, 32.0, 24.0, 32.0])
    outputs = predictions(box_size=28.0).requires_grad_(True)
    everywhere = torch.tensor([[[48.0, 32.0, 96.0, 64.0]]])
    total, _, _, _, _, per_target = losses(head, outputs, labels, everywhere)
    assigned = int(round(float(per_target)))  # one target, so this is the number of assigned anchors
    assert assigned >= 1 and torch.isfinite(total)
    assert head.last_ignored_anchors == ANCHORS - assigned
    total.backward()
    gradient = outputs.grad[0, :, 4]
    counted = gradient != 0
    # Only the assigned anchors have an objectness term, and they sit on the face.
    assert int(counted.sum()) == assigned
    cx, cy = centers()
    assert ((cx[counted] - 48).abs() <= 44).all() and ((cy[counted] - 32).abs() <= 48).all()


def test_without_ignore_regions_it_is_the_library_loss(head):
    from yolox.models.yolo_head import YOLOXHead

    labels = torch.zeros((2, 5, 5))
    labels[0, 0] = torch.tensor([1.0, 40.0, 30.0, 30.0, 40.0])
    labels[0, 1] = torch.tensor([0.0, 70.0, 20.0, 20.0, 28.0])
    outputs = torch.cat([predictions(24.0, seed=3), predictions(24.0, seed=4)])
    x_shifts, y_shifts, strides = grids()
    origin = [torch.randn((2, part.shape[1], 4), generator=torch.Generator().manual_seed(5)) for part in x_shifts]
    for use_l1 in (False, True):
        head.use_l1, head.ignore_boxes = use_l1, None
        here = head.get_losses(None, x_shifts, y_shifts, strides, labels, outputs, origin, dtype=outputs.dtype)
        library = YOLOXHead.get_losses(head, None, x_shifts, y_shifts, strides, labels, outputs, origin,
                                      dtype=outputs.dtype)
        assert [float(value) for value in here] == [float(value) for value in library]
    head.use_l1 = False


def test_the_loss_of_a_tiny_batch_through_the_network():
    torch.manual_seed(1)
    model = M.build_model(M.nano_experiment(2))
    model.train()
    images = torch.rand((2, 3, 128, 160)) * 255
    targets = torch.zeros((2, 4, 5))
    targets[0, 0] = torch.tensor([0.0, 60.0, 50.0, 40.0, 56.0])
    targets[0, 1] = torch.tensor([1.0, 120.0, 90.0, 36.0, 50.0])
    ignore = torch.zeros((2, 3, 4))
    ignore[0, 0] = torch.tensor([30.0, 100.0, 50.0, 40.0])
    ignore[1, 0] = torch.tensor([80.0, 64.0, 160.0, 128.0])  # the whole second image
    L.set_ignore_regions(model, ignore)
    out = model(images, targets)
    assert set(out) == {"total_loss", "iou_loss", "l1_loss", "conf_loss", "cls_loss", "num_fg"}
    assert torch.isfinite(out["total_loss"]) and float(out["total_loss"].detach()) > 0
    anchors = 20 * 16 + 10 * 8 + 5 * 4
    assert anchors <= model.head.last_ignored_anchors < 2 * anchors  # all of image two and some of image one
    out["total_loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    L.set_ignore_regions(model, None)
    plain = model(images, targets)
    assert model.head.last_ignored_anchors == 0 and float(plain["conf_loss"].detach()) > float(out["conf_loss"].detach())
