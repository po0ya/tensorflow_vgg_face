#!/usr/bin/env bash
set -x
set -e

# CAFFE_HOME should be set to root of caffe dir
if [ -z ${CAFFE_HOME+x} ]; then
   echo "err: CAFFE_HOME is not set"
   exit
fi

CUR_DIR=`pwd -P`

cd test

echo "addpath ${CAFFE_HOME}/matlab;
model = '${CUR_DIR}/data/vgg_face_caffe/vgg_face_deploy_new.prototxt';
weights = '${CUR_DIR}/data/vgg_face_caffe/VGG_FACE_new.caffemodel';
" > init.m
matlab -nodesktop -r "init,test,exit"
cd ..

python test/test.py

cd $CUR_DIR

