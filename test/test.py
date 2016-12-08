#!/usr/bin/env python

import _init_paths_tfvgg

import argparse

import cPickle
import numpy as np
import tensorflow as tf
import os.path as osp

import datasets
import datasets.dataset as dataset
from models.vgg.vgg_face_embedding import VGG_FACE_16
from config_tfvgg import cfg
import caffe
import scipy.io as sio

def main():

    model_data_path = 'data/vgg_face.npy'
    with tf.Session() as sesh:
        # Load the converted parameters
        base_path = './test/'
        image_paths = ['test.png']
        cfg.FLIP = False
        cfg.CROP = False
        print('Loading the model')
        spec = datasets.get_data_spec(model_class=VGG_FACE_16)
        spec.scale_size = spec.crop_size # matlab code does not crop
        spec.isotropic = False
        spec.batch_size = 1

        image_producer = dataset.LFWProducer(image_paths,base_path, data_spec=spec,num_concurrent=1)

        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        input_node = tf.placeholder(tf.float32,
                                    shape=(None, spec.crop_size, spec.crop_size, spec.channels))

        indices, input_image = sesh.run([image_producer.ind_deq, image_producer.img_deq])


        # net = VGG_FACE_16({'data':  image_producer.img_deq})
        net = VGG_FACE_16({'data': input_node})
        net.load(model_data_path, sesh, ignore_missing=True)

        caffe.set_mode_cpu()
        caffe_net = caffe.Net('./data/vgg_face_caffe/vgg_face_deploy_new.prototxt',
                              './data/vgg_face_caffe/VGG_FACE_new.caffemodel', caffe.TEST)

        probs = sesh.run(net.get_output(), feed_dict={input_node: input_image})[0].flatten()

        input_blobs = {'data': np.transpose(input_image, [0, 3, 1, 2])}
        caffe_net.blobs['data'].reshape(*(input_blobs['data'].shape))

        caffe_net.forward(data=input_blobs['data'].astype(np.float32, copy=False))
        caffe_ft = caffe_net.blobs['fc7'].data[0].flatten()

        print('MSE pycaffe and tensorflow: {}'.format(np.sum(np.square(caffe_ft - probs)) / (np.sum(np.square(caffe_ft) + np.square(probs)))))

        matcaffe_ft = sio.loadmat('test/matcaffe_ft.mat')

        matcaffe_ft = matcaffe_ft['caffe_ft'].flatten()
        print('MSE matcaffe and tensorflow: {}'.format(np.sum(np.square(matcaffe_ft - probs)) / (np.sum(np.square(matcaffe_ft) + np.square(probs)))))


        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)




if __name__ == '__main__':
    main()
