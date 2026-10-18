# SPDX-License-Identifier: Apache-2.0
"""Train the recognizer: an image encoder whose embeddings of a card crop and of that card's
clean image lie close together.

The parts:

``config``
    The TOML config and its defaults: seeds, optimizer, loss weights, data paths, the epoch's
    make-up, the components.
``sources``
    The data seam. The trainer reads named streams of rows in one documented format; real
    crops, composite crops, card images and background patches are built in, and a plug-in
    hook adds others.
``dataset``
    An epoch's composition, the per-row augmentation chain, balanced draws and the pair batch
    sampler.
``model``
    The encoder: MobileNetV3-Small features, projected to 128 values, batch-normalized and
    scaled to unit length.
``losses``
    ArcFace over the training cards, the cross-domain contrastive term, the hinge on
    negatives, same-name masking and card-back weighting.
``validate``
    Top-1 retrieval against clean card images on two held-out sets, and the choice of the
    checkpoint to keep.
``loop``
    The step, the schedule, the loop and the run record.
``cli``
    ``cardvision train-recognizer --config FILE --out DIR``.

PyTorch and torchvision are optional (the ``train`` extra). ``config`` and ``sources`` import
neither; the other modules do.
"""
