  #!/usr/bin/env bash
PYTHONPATH=`pwd`:$PYTHONPATH
python /cfarhomes/pouya/gleuclid-data/pyproj/github/face_verification/tools/classify.py \
    --model_path data/vgg_face.npy \
    --image_test_pairs data/lfw/pairs.txt \
    --base_path data/lfw/imgs/ \
    --bbox_file data/lfw/imgs/img_list_bboxes.pckl \
    $1
