# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the card detector: YOLOX-Nano with two classes, face and back.

::

    cardvision train-detector --config cardvision/train_detector/default.toml --out runs/first \
        --download-pretrained
    cardvision export detector runs/first/last.pth --out runs/first

What you need
=============

* the ``train`` extra and the YOLOX package (``cardvision.export.detector`` says how to
  install it);
* a label manifest and its frames (``cardvision fetch``), a folder of generated scenes
  (``cardvision make-composites``) and a folder of your own images without cards;
* starting weights: by default the public COCO-pretrained YOLOX-Nano checkpoint, downloaded
  only when you pass ``--download-pretrained``.

Any of the three streams can be left out of a config's mix, and others can be added through a
plug-in function: ``sources`` documents the format.

Modules
=======

``config``
    The config file: every key, its meaning and its default.
``sources``
    The data seam: named streams of frames with targets, the three built-in streams and the
    plug-in hook. Nothing else here knows where a frame comes from.
``augment``
    From streams to training images: the tile mix, mosaic, color jitter, flip and canvas.
``loss``
    The YOLOX head with ignore regions in its objectness loss.
``model``
    The network and optimizer, both built by the YOLOX package; starting weights, widening a
    one-class checkpoint to two classes; the learning-rate schedule.
``trainer``
    The loop, the checkpoint (the last step is the result) and the run record.
``cli``
    The ``cardvision train-detector`` subcommand.

PyTorch, OpenCV and YOLOX are imported only when a run starts; ``config`` and ``sources`` need
none of them.

Third-party code
================

The network, the assignment of anchors to targets, the optimizer's set-up, the weight
averaging and the image-warping functions are the YOLOX package's (Megvii, Apache-2.0), called
through its public names. Two modules contain code adapted from YOLOX and say so at the top:
``loss`` (the loss function, from ``yolox/models/yolo_head.py``) and ``augment`` (the mosaic
and the order of the per-image steps, from ``yolox/data/datasets/mosaicdetection.py`` and
``yolox/data/data_augment.py``).
"""
