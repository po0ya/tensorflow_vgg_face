#!/usr/bin/env bash
PYTHONPATH=`pwd`:$PYTHONPATH
ipdb /cfarhomes/pouya/gleuclid-data/pyproj/github/face_verification/tools/classify.py \
    --model_path data/vgg_face.npy \
    --image_test_pairs data/lfw/pairs.txt \
    --base_path data/lfw/imgs/ \
    $1
