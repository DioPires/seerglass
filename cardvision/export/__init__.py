# SPDX-License-Identifier: Apache-2.0
"""Export: from a training checkpoint to the files a runtime loads.

Each export writes a float32 ONNX file, a copy whose weights are stored in half precision, the
weights alone in the safetensors format, and a report of how closely ONNX Runtime reproduces
PyTorch on the same inputs.

PyTorch, ONNX and safetensors are optional (the ``train`` extra) and are imported only when an
export runs, so importing this package costs nothing. The functions live in the submodules:
``recognizer.export_recognizer``, ``detector.export_detector``, ``detector.build_yolox_nano``,
``onnx_tools.fp16_weights`` and ``checkpoint`` for reading checkpoints and writing safetensors.
"""
