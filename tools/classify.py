#!/usr/bin/env python
import argparse

import cv2
import numpy as np
import tensorflow as tf
import os.path as osp

import datasets
import datasets.dataset as dataset
from datasets.dataset import process_image
from models.vgg.vgg_face_embedding import VGG_FACE_16


def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))

def classify_simple(model_data_path,image_path):
    spec = datasets.get_data_spec(model_class=VGG_FACE_16)

    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))
    net = VGG_FACE_16({'data': input_node})

    with tf.Session() as sesh:
        # Start the image processing workers
        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sesh)

        # Load the input image
        print('Loading the images')
        img = cv2.imread(image_path)
        processed_img = process_image(img=img,
                                      scale=spec.scale_size,
                                      isotropic=spec.isotropic,
                                      crop=spec.crop_size,
                                      mean=spec.mean)



        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        embeddings = sesh.run(net.get_output(), feed_dict={input_node: input_image})

def classify(model_data_path, image_pairs_fp,base_path):
    '''Classify the given images using VGG FACE.'''

    # Get the data specifications for the GoogleNet model
    spec = datasets.get_data_spec(model_class=VGG_FACE_16)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    net = VGG_FACE_16({'data': input_node})

    image_paths = []
    with open(image_pairs_fp,'r') as f:
        im_pairs_lines = f.readlines()
    num = int(im_pairs_lines[0].strip())
    ctr = 1
    for _ in range(num):
        l = im_pairs_lines[ctr].split()
        image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[1])))
        image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[2])))
        ctr = ctr+1

    for _ in range(num):
        l = im_pairs_lines[ctr].split()
        image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[1])))
        image_paths.append('{}/{}_{:04d}.jpg'.format(l[2],l[2],int(l[3])))
        ctr = ctr+1

    # Create an image producer (loads and processes images in parallel)
    image_producer = dataset.LFWProducer(image_paths,base_path, data_spec=spec)
    all_embeddings = {}

    with tf.Session() as sesh:
        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sesh,ignore_missing=True)

        # Load the input image
        print('Loading the images')
        indices, input_images = image_producer.get(sesh)

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})
        for i in range(indices.shape[0]):
            all_embeddings[image_paths[indices[i]]]=probs[i,:]

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)

    print all_embeddings


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the GoogleNet model')
    parser.add_argument('image_pairs_fp', help='LFW image pairs')
    parser.add_argument('base_path', help='The root of LFW where the directories with names_lastname reside')

    args = parser.parse_args()

    # Classify the image
    classify(args.model_path, args.image_pairs_fp,args.base_path)


if __name__ == '__main__':
    main()
