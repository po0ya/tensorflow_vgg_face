#!/usr/bin/env python

import _init_paths_tfvgg

import argparse

import cPickle
import cv2
import numpy as np
import tensorflow as tf
import os.path as osp

import datasets
import datasets.dataset as dataset
from config_tfvgg import cfg_from_list, cfg_from_file
from datasets.dataset import process_image
from models.vgg.vgg_face_embedding import VGG_FACE_16
from config_tfvgg import cfg
import matplotlib.pyplot as plt
import sys
from vgg_features import get_features

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

def classify_lfw(model_data_path, image_pairs_fp, base_path, save_feats_fp='',bbox_fp=None):
    '''
    :param model_data_path:
    :param image_pairs_fp:
    :param base_path:
    :param save_feats_fp:
    :param bbox_fp:
    :return:
    '''
    image_paths = []
    with open(image_pairs_fp,'r') as f:
        im_pairs_lines = f.readlines()
    temp = im_pairs_lines[0].split()
    num_splits = int(temp[0])
    num_pairs = int(temp[1])
    ctr = 1

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


    spec = datasets.get_data_spec(model_class=VGG_FACE_16)

    image_producer = dataset.LFWProducer(image_paths, base_path, data_spec=spec, bbox_fp=bbox_fp)

    all_embeddings = get_features(model_data_path, image_producer, save_feats_fp=save_feats_fp, bbox_fp=bbox_fp)

    verif_labels = np.concatenate((np.ones(num_pairs),np.zeros(num_pairs)))
    for _ in range(num_splits-1):
        verif_labels = np.concatenate((verif_labels,np.concatenate((np.ones(num_pairs), np.zeros(num_pairs)))))

    image_paths = image_producer.image_paths

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
    parser.add_argument('--bbox_file',dest='bbox_file',default=None,help='File containing dictionary of face bboxes (optional)')


    args = parser.parse_args()
    if len(sys.argv) == 1:
       parser.print_help()
       sys.exit(1)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # Classify the image
    classify_lfw(args.model_path, args.image_test_pairs, args.base_path, args.save_feats_path, bbox_fp = args.bbox_file)


if __name__ == '__main__':
    main()
