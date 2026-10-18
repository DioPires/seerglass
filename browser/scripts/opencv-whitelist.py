# SPDX-License-Identifier: Apache-2.0
# The OpenCV functions and classes the browser's geometric stage calls, and nothing else: the
# list OpenCV's JavaScript binding generator (modules/js/generator/embindgen.py) reads in place of
# its own platforms/js/opencv_js.config.py. scripts/build-opencv.mjs passes it with --config.
#
# The generator runs this file with exec(), with makeWhiteList already defined. Each name is
# what cardvision/geometric calls in Python:
#
#   cvtColor, resize                     features.scaled_grey (grey, then INTER_AREA / INTER_LINEAR)
#   ORB.create, detectAndCompute         features.make_detector, features.extract
#   BFMatcher.create, knnMatch           stage.inliers (and the CPU check of the vote)
#   estimateAffine2D                     stage._fit (model "affine")
#   transform, isContourConvex,
#   contourArea                          stage._plausible
#   setRNGSeed                           stage.verify seeds OpenCV's generator before each crop
#
# Mat, the vectors (MatVector, KeyPointVector, DMatchVector, DMatchVectorVector) and the
# constants come with OpenCV's own core bindings (modules/js/src/core_bindings.cpp).

core = {
    '': ['setRNGSeed', 'transform'],
    'Algorithm': [],
}

imgproc = {
    '': ['contourArea', 'cvtColor', 'isContourConvex', 'resize'],
}

features = {
    'Feature2D': ['detect', 'compute', 'detectAndCompute', 'descriptorSize', 'descriptorType', 'defaultNorm', 'empty'],
    'ORB': ['create', 'getMaxFeatures', 'getScaleFactor', 'getNLevels', 'getEdgeThreshold', 'getFirstLevel',
            'getWTA_K', 'getScoreType', 'getPatchSize', 'getFastThreshold'],
    'DescriptorMatcher': ['knnMatch', 'match', 'add', 'clear', 'train', 'empty'],
    'BFMatcher': ['create'],
}

geometry = {
    '': ['estimateAffine2D'],
}

white_list = makeWhiteList([core, imgproc, features, geometry])  # noqa: F821 (defined by embindgen.py)
