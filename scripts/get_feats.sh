  #!/usr/bin/env bash
set +x
set -e

PYTHONPATH=`pwd`:$PYTHONPATH
python /cfarhomes/pouya/gleuclid-data/pyproj/github/face_verification/tools/vgg_features.py \
    --model_path data/vgg_face.npy \
    --save_feats_path debug/feats.pckl \
    --image_paths $1
