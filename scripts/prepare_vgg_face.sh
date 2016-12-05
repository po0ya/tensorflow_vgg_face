#!/usr/bin/env bash

# CAFFE_HOME should be set to root of caffe dir
if [ -z ${CAFFE_HOME+x} ]; then
   echo "err: CAFFE_HOME is not set"
   exit
fi

CUR_DIR=`pwd`

#download vgg_face if necessary
if ! [ -a data ]; then
    echo "Create or link a directory named 'data' in the root of the project first"
    exit
fi

if ! [ -d data/vgg_face_caffe ]; then
echo "Downloading vgg_face"
wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz data/vgg_face_caffe
cd data
tar xvf vgg_face_caffe.tar.gz
cd $CUR_DIR
fi

echo "Upgrading prototxt and weights"
#upgrade prototxt and binary
./${CAFFE_HOME}/.build_release/tools/upgrade_net_proto_text data/vgg_face_caffe/VGG_FACE_deploy.prototxt data/vgg_face_caffe/vgg_face_deploy_new.prototxt
./${CAFFE_HOME}/.build_release/tools/upgrade_net_proto_binary data/vgg_face_caffe/VGG_FACE.caffemodel data/vgg_face_caffe/VGG_FACE_new.caffemodel

cd caffe-tensorflow
python convert.py --caffemodel ../data/vgg_face_caffe/VGG_FACE_new.caffemodel ../data/vgg_face_caffe/vgg_face_deploy_new.prototxt --data-output-path ../data/vgg_face.npy --code-output-path ../models/vgg/vgg_face.py
cd $CUR_DIR