# SPDX-License-Identifier: Apache-2.0
"""The losses on tiny cases worked out by hand, and the encoder's shape. Skipped without
PyTorch."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from cardvision.train_recognizer import losses as L  # noqa: E402


def unit(*rows) -> torch.Tensor:
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=1)


# --------------------------------------------------------------------------------- ArcFace
def test_arcface_adds_the_margin_to_the_true_class_s_angle():
    head = L.ArcFace(n_classes=2, dim=2, scale=10.0, margin=0.3)
    with torch.no_grad():
        head.W.copy_(torch.tensor([[2.0, 0.0], [0.0, 5.0]]))                      # the rows' lengths do not matter
    angle = 0.4
    emb = torch.tensor([[math.cos(angle), math.sin(angle)]])
    loss = head(emb, torch.tensor([0])).detach()
    logits = torch.tensor([10.0 * math.cos(angle + 0.3), 10.0 * math.sin(angle)])
    assert float(loss) == pytest.approx(float(-torch.log_softmax(logits, 0)[0]), abs=1e-5)
    # With no margin it is the plain softmax over scaled cosines.
    plain = L.ArcFace(2, 2, 10.0, 0.0)
    with torch.no_grad():
        plain.W.copy_(head.W)
    logits = torch.tensor([10.0 * math.cos(angle), 10.0 * math.sin(angle)])
    assert float(plain(emb, torch.tensor([0])).detach()) == pytest.approx(float(-torch.log_softmax(logits, 0)[0]), abs=1e-5)


def test_masked_arcface_gives_no_gradient_to_the_other_classes_of_the_same_name():
    torch.manual_seed(0)
    # Six classes; classes 0, 1 and 4 are printings of one name, 2 and 3 of another, 5 alone.
    name_index = torch.tensor([0, 0, 1, 1, 0, 2])
    emb = F.normalize(torch.randn(3, 8), dim=1)
    labels = torch.tensor([0, 2, 5])

    head = L.ArcFace(6, 8, 32.0, 0.2)
    head(emb, labels, name_index).backward()
    grad = head.W.grad
    # Sample 0 (class 0) leaves 1 and 4 alone; sample 1 (class 2) leaves 3 alone. But 1, 3 and 4
    # are still wrong classes for the samples of the other names, so check sample by sample.
    for sample, same_name in ((0, [1, 4]), (1, [3]), (2, [])):
        head.zero_grad()
        head(emb[sample:sample + 1], labels[sample:sample + 1], name_index).backward()
        others = [c for c in range(6) if c != int(labels[sample]) and c not in same_name]
        assert all(float(head.W.grad[c].abs().sum()) == 0.0 for c in same_name)
        assert all(float(head.W.grad[c].abs().sum()) > 0.0 for c in [int(labels[sample]), *others])
    assert float(grad.abs().sum()) > 0

    # Without the mask every other class is pushed away, printings of the same name included.
    head.zero_grad()
    head(emb[:1], labels[:1]).backward()
    assert all(float(head.W.grad[c].abs().sum()) > 0.0 for c in range(6))

    # The masked loss is the loss over the classes that are left.
    kept = [0, 2, 3, 5]
    small = L.ArcFace(4, 8, 32.0, 0.2)
    with torch.no_grad():
        small.W.copy_(head.W[kept])
    with torch.no_grad():
        assert float(head(emb[:1], labels[:1], name_index)) == pytest.approx(float(small(emb[:1], torch.tensor([0]))), abs=1e-5)


# ----------------------------------------------------------------------------- contrastive
def test_the_contrastive_term_pulls_a_crop_to_its_own_card_s_clean_image():
    t = 0.5
    crops = unit([1, 0, 0], [0, 1, 0])
    arts = unit([1, 0, 0], [0, 1, 0])
    labels = torch.tensor([0, 1])
    # Each crop: similarity 1 to its own image and 0 to the other, over the temperature.
    expected = -math.log(math.exp(1 / t) / (math.exp(1 / t) + math.exp(0)))
    assert float(L.contrastive(crops, arts, labels, t)) == pytest.approx(expected, abs=1e-6)
    # Swapped images: the crop's own image is the far one.
    swapped = -math.log(math.exp(0) / (math.exp(1 / t) + math.exp(0)))
    assert float(L.contrastive(crops, arts[[1, 0]], labels, t)) == pytest.approx(swapped, abs=1e-6)
    # A lower temperature sharpens it.
    assert float(L.contrastive(crops, arts, labels, 0.07)) < expected


def test_other_clean_images_of_the_same_card_or_name_are_not_negatives():
    t = 0.5
    crops = unit([1, 0, 0], [1, 0, 0], [0, 1, 0])
    arts = unit([1, 0, 0], [1, 0, 0], [0, 1, 0])
    own = math.exp(1 / t)                                                         # a crop against its own image
    third = -math.log(own / (own + 2))                                            # row 2: two unrelated images at 0
    # Rows 0 and 1 are the same card: each one's denominator leaves the other's image out.
    same_card = torch.tensor([4, 4, 9])
    masked = -math.log(own / (own + 1))
    assert float(L.contrastive(crops, arts, same_card, t)) == pytest.approx((2 * masked + third) / 3, abs=1e-6)
    # As different cards, the twin image is a negative at similarity 1.
    different = torch.tensor([4, 5, 9])
    unmasked = -math.log(own / (2 * own + 1))
    assert float(L.contrastive(crops, arts, different, t)) == pytest.approx((2 * unmasked + third) / 3, abs=1e-6)
    # Different cards of one name, with the name index: masked again.
    name_index = torch.zeros(10, dtype=torch.long)
    name_index[9] = 1
    assert float(L.contrastive(crops, arts, different, t, name_index)) == pytest.approx((2 * masked + third) / 3, abs=1e-6)
    # The masked pair gets no gradient through the masked entry.
    crops.requires_grad_(True)
    L.contrastive(crops, arts, same_card, t).backward()
    assert crops.grad is not None and torch.isfinite(crops.grad).all()


# ---------------------------------------------------------------------------- the hinge
def test_a_negative_is_scored_by_its_best_match_among_bank_and_batch_images():
    negatives = unit([1, 0, 0], [0, 1, 0], [0, 0, 1])
    bank = unit([1, 0, 0], [0, 0.6, 0.8])
    arts = unit([0, 0.8, 0.6])
    scores = L.negative_scores(negatives, bank, arts)
    assert scores.tolist() == pytest.approx([1.0, 0.8, 0.8])                      # bank, batch image, bank
    assert L.negative_scores(negatives, bank).tolist() == pytest.approx([1.0, 0.6, 0.8])
    assert L.negative_scores(negatives, bank, arts[:0]).tolist() == pytest.approx([1.0, 0.6, 0.8])


def test_the_hinge_is_zero_below_the_margin_and_weighs_card_backs_more():
    scores = torch.tensor([0.75, 0.50, 0.60, 0.30])
    backs = torch.tensor([False, False, True, True])
    plain, excess = L.negative_hinge(scores, backs, margin=0.55)
    assert float(plain) == pytest.approx((0.20 + 0 + 0.05 + 0) / 4) and excess.tolist() == pytest.approx([0.20, -0.05, 0.05, -0.25])
    weighted, excess = L.negative_hinge(scores, backs, margin=0.55, back_margin=0.50, back_weight=4.0)
    # backs: 4 x relu(0.60 - 0.50) and 4 x relu(0.30 - 0.50); the others as before
    assert float(weighted) == pytest.approx((0.20 + 0 + 4 * 0.10 + 0) / 4)
    assert excess.tolist() == pytest.approx([0.20, -0.05, 0.10, -0.20])
    # Below the margin there is nothing to learn.
    scores = torch.tensor([0.2, 0.1], requires_grad=True)
    loss, _ = L.negative_hinge(scores, torch.tensor([False, True]), 0.55, 0.50, 4.0)
    loss.backward()
    assert float(loss.detach()) == 0.0 and scores.grad.tolist() == [0.0, 0.0]


def test_gradient_reaches_the_negative_and_the_batch_image_but_not_the_bank():
    negative = unit([1.0, 0.2]).requires_grad_(True)
    bank = unit([1.0, 0.0]).requires_grad_(True)
    art = unit([0.2, 1.0]).requires_grad_(True)
    loss, _ = L.negative_hinge(L.negative_scores(negative, bank.detach(), art), torch.tensor([False]), 0.55)
    loss.backward()
    assert float(negative.grad.abs().sum()) > 0 and bank.grad is None
    # Here the bank is the better match, so the batch image gets nothing either.
    assert float(art.grad.abs().sum()) == 0.0


# --------------------------------------------------------------------------- the encoder
def test_the_encoder_returns_unit_length_embeddings():
    pytest.importorskip("torchvision")
    from cardvision.train_recognizer.model import Encoder, build_encoder, build_export_encoder

    torch.manual_seed(0)
    model = build_encoder("random")
    assert [name.split(".")[0] for name in model.state_dict()][0] == "features"
    assert {name.split(".")[0] for name in model.state_dict()} == {"features", "proj", "bn"}
    assert model.proj.in_features == 576 and model.proj.out_features == 128
    out = model(torch.randn(4, 3, 128, 128))
    assert out.shape == (4, 128) and torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5)
    assert Encoder("random", dim=32)(torch.randn(2, 3, 128, 128)).shape == (2, 32)
    with pytest.raises(ValueError, match="init must be one of"):
        Encoder("pretrained")
    # The export network takes a trained encoder's weights; weights of another network are refused.
    export = build_export_encoder()
    export.load_state_dict(model.state_dict())
    model.eval(), export.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        assert torch.equal(model(x), export(x))
    with pytest.raises(RuntimeError, match="pass --model MODULE:FUNCTION"):
        export.load_state_dict({"conv.weight": torch.zeros(1)})
