#!/usr/bin/env python

import _init_paths

import argparse

import cPickle
import cv2
import numpy as np
import tensorflow as tf
import os.path as osp

import datasets
import datasets.dataset as dataset
from config import cfg_from_list, cfg_from_file
from datasets.dataset import process_image
from models.vgg.vgg_face_embedding import VGG_FACE_16
from config import cfg
import matplotlib.pyplot as plt
import sys


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

def classify(model_data_path, image_pairs_fp, base_path, save_feats_fp=''):
    '''Classify the given images using VGG FACE.'''

    # Get the data specifications for the GoogleNet model
    spec = datasets.get_data_spec(model_class=VGG_FACE_16)

    # Create a placeholder for the input image
    #input_node = tf.placeholder(tf.float32,
    #                            shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network

    image_paths = []
    with open(image_pairs_fp,'r') as f:
        im_pairs_lines = f.readlines()
    temp = im_pairs_lines[0].split()
    # num_splits = int(temp[0])
    # num_pairs = int(temp[1])
    num_splits = 1
    num_pairs = int(temp[0])
    ctr = 1
    # for _ in range(2):
    #     l = im_pairs_lines[ctr].split()
    #     image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[1])))
    #     image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[2])))
    #     ctr = ctr+1
    for _ in range(num_splits):
        for _ in range(num_pairs):
            l = im_pairs_lines[ctr].split()
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[1])))
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[2])))
            ctr = ctr+1

        for _ in range(num_pairs):
            l = im_pairs_lines[ctr].split()
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[0],l[0],int(l[1])))
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[2],l[2],int(l[3])))
            ctr = ctr+1

    # Create an image producer (loads and processes images in parallel)
    get_ems = True
    if save_feats_fp is not None and osp.exists(save_feats_fp):
        try:
            with open(save_feats_fp, 'r') as f:
                print('Loading from {}'.format(save_feats_fp))
                all_embeddings = cPickle.load(f)
                get_ems = False
        except:
            print('Error reading {}'.format(save_feats_fp))

    if get_ems:
        image_producer = dataset.LFWProducer(image_paths,base_path, data_spec=spec)
        all_embeddings = {}

        input_node = tf.placeholder(tf.float32,
                                    shape=(None, spec.crop_size, spec.crop_size, spec.channels))

        num_feats_per_image = 1
        if cfg.FLIP:
            num_feats_per_image = num_feats_per_image * 2

        if cfg.CROP:
            num_feats_per_image = num_feats_per_image * 5

        with tf.Session() as sesh:
            # Start the image processing workers
            coordinator = tf.train.Coordinator()
            threads = image_producer.start(session=sesh, coordinator=coordinator)

            # Load the converted parameters
            print('Loading the model')

            #net = VGG_FACE_16({'data':  image_producer.img_deq})
            net = VGG_FACE_16({'data':  input_node})
            net.load(model_data_path, sesh,ignore_missing=True)

            num_loaded = 0
            num_imgs_in_queue = len(image_paths)*num_feats_per_image


            while num_loaded < num_imgs_in_queue:
                # Perform a forward pass through the network to get the class probabilities
                # print('Classifying {}'.format(num_loaded*1.0/len(image_paths)))
                if coordinator.should_stop():
                    break
                indices, input_images = image_producer.get(sesh)

                # indices, input_images = sesh.run([image_producer.ind_deq, image_producer.img_deq])

                # for k in range(indices.shape[0]):
                #     cv2.imwrite('./debug/{}.jpg'.format(num_loaded+k),input_images[k,...].squeeze())
                probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})

                #[probs,indices,imgs] = sesh.run([net.get_output(),image_producer.ind_deq.name,image_producer.img_deq.name])
                for i in range(0,indices.shape[0]):
                    # cv2.imwrite('./debug/{}_{}.jpg'.format(indices[i],i),imgs[i,...].squeeze())
                    if not image_paths[indices[i]] in all_embeddings.keys():
                        all_embeddings[image_paths[indices[i]]]=probs[i,:]
                    else:
                        all_embeddings[image_paths[indices[i]]]+=probs[i,:]
                num_loaded = num_loaded + indices.shape[0]
                print('Classified {}'.format(num_loaded*1.0/num_imgs_in_queue))


            # for imp in image_paths:
            #     all_embeddings[imp] = all_embeddings[imp]/5.0

            # Stop the worker threads
            coordinator.request_stop()
            coordinator.join(threads, stop_grace_period_secs=2)
        if save_feats_fp is not None:
            with open(save_feats_fp,'w') as f:
                cPickle.dump(all_embeddings,f,cPickle.HIGHEST_PROTOCOL)
    else:
        with open(save_feats_fp, 'r') as f:
            print('Loading from {}'.format(save_feats_fp))
            all_embeddings = cPickle.load(f)

    verif_labels = np.concatenate((np.ones(num_pairs),np.zeros(num_pairs)))
    for _ in range(num_splits-1):
        verif_labels = np.concatenate((verif_labels,np.concatenate((np.ones(num_pairs), np.zeros(num_pairs)))))

    scores = np.zeros(2*num_pairs*num_splits)
    euc_dist = lambda x,y: np.sqrt(np.sum(np.square(x-y)))
    for i in range(2*num_pairs*num_splits):
        em1 = all_embeddings[image_paths[2*i]]
        em2 = all_embeddings[image_paths[2*i+1]]
        scores[i] = euc_dist(em1,em2)
    score_sort_inds = np.argsort(scores)
    sorted_labels = verif_labels[score_sort_inds]
    tp = 0.0
    fp = 0.0
    tpr = [0.0,]
    fpr = [0.0,]
    for i in range(sorted_labels.shape[0]):
        if sorted_labels[i]==1:
            tp+=1
        else:
            fp+=1
            tpr.append(tp*1.0/(num_splits*num_pairs))
            fpr.append(fp*1.0/(num_splits*num_pairs))

    plt.plot(fpr,tpr)
    plt.savefig('data/roc.pdf')
    plt.cla()

    with open('data/out.txt','w') as f:
        for i in range(len(tpr)):
            f.write('{} {} \n'.format(tpr[i],fpr[i]))

    ### Get accuracy
    if num_splits > 1:
        split_accuracies = np.zeros(num_splits)
        for i in range(num_splits):
            flags = np.zeros(2*num_pairs*num_splits,dtype=np.bool)
            flags[2*num_pairs*i:2*num_pairs*(i+1)] = True
            test_scores = scores[flags]
            test_labels = verif_labels[flags]
            dev_scores = scores[np.logical_not(flags)]
            dev_labels = verif_labels[np.logical_not(flags)].astype(np.bool)
            max_acc = 0
            final_th = 0
            for j in range(dev_scores.shape[0]-1):
                th = (dev_scores[j]+dev_scores[j+1])/2.0
                correct_preds = (dev_scores < th) == dev_labels
                if max_acc < np.mean(correct_preds):
                    max_acc = np.mean(correct_preds)
                    final_th = th

            correct_test_preds = ((test_scores < final_th) == test_labels)
            split_accuracies[i] = np.mean(correct_test_preds)

        print('Accuracy: {:.4} +- {:.4}'.format(np.mean(split_accuracies),np.std(split_accuracies)))



def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required = True, help='Converted parameters for the GoogleNet model')
    parser.add_argument('--image_test_pairs', required = True, dest='image_test_pairs', help='LFW pairs.txt path')
    parser.add_argument('--base_path', required = True, help='The root of LFW where the directories with Firstname_Lastname reside')
    parser.add_argument('--save_feats_path',default=None,help='Save fc7 embeddings to a file (optional)')
    parser.add_argument('--cfg',dest='cfg_file',default=None,help='Config file for testing (optional)')


    args = parser.parse_args()
    if len(sys.argv) == 1:
       parser.print_help()
       sys.exit(1)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # Classify the image
    classify(args.model_path, args.image_test_pairs,args.base_path,args.save_feats_path)


if __name__ == '__main__':
    main()
