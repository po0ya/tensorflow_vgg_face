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


def get_features(model_data_path, image_producer, save_feats_fp='', bbox_fp=None):
    '''
    Gets fc7 embeddings of images in image_
    :param model_data_path: the converted caffemodel to .npy
    :param image_paths: a list of paths to images
    :param save_feats_fp:
    :param bbox_fp: a pckl path to dictionary with {im_path:[x1,y1,x2,y2]} of faces in the image
    :return: {img_path: 4096 face embeddings}
    '''
    spec = datasets.get_data_spec(model_class=VGG_FACE_16)

    get_ems = True

    if save_feats_fp is not None and osp.exists(save_feats_fp):
        try:
            with open(save_feats_fp, 'r') as f:
                print('Loading from {}'.format(save_feats_fp))
                all_embeddings = cPickle.load(f)
                return all_embeddings
        except:
            print('Error reading {}'.format(save_feats_fp))

    if get_ems:
        all_embeddings = {}

        # input_node = tf.placeholder(tf.float32,
        #                             shape=(None, spec.crop_size, spec.crop_size, spec.channels))


        with tf.Session() as sesh:
            # Start the image processing workers
            coordinator = tf.train.Coordinator()
            threads = image_producer.start(session=sesh, coordinator=coordinator)

            # Load the converted parameters
            print('Loading the model')

            net = VGG_FACE_16({'data': image_producer.img_deq})
            # net = VGG_FACE_16({'data':  input_node})
            net.load(model_data_path, sesh, ignore_missing=True)

            num_loaded = 0
            num_imgs_in_queue = image_producer.num_imgs
            num_feats_per_image = image_producer.num_feats_per_image
            image_paths = image_producer.image_paths

            while num_loaded < num_imgs_in_queue:
                # Perform a forward pass through the network to get the class probabilities
                # print('Classifying {}'.format(num_loaded*1.0/len(image_paths)))
                if coordinator.should_stop():
                    break
                # indices, input_images = image_producer.get(sesh)

                # indices, input_images = sesh.run([image_producer.ind_deq, image_producer.img_deq])

                # for k in range(indices.shape[0]):
                #     cv2.imwrite('./debug/{}.jpg'.format(num_loaded+k),input_images[k,...].squeeze())
                # probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})
                if cfg.DEBUG:
                    [probs, indices, imgs] = sesh.run(
                        [net.get_output(), image_producer.ind_deq.name, image_producer.img_deq.name])
                    for i in range(0, indices.shape[0]):
                        cv2.imwrite('./debug/{}_{}.jpg'.format(indices[i], i), imgs[i, ...].squeeze())
                        if not image_paths[indices[i]] in all_embeddings.keys():
                            all_embeddings[image_paths[indices[i]]] = probs[i, :]
                        else:
                            all_embeddings[image_paths[indices[i]]] += probs[i, :]
                            exit(0)
                else:
                    [probs, indices] = sesh.run([net.get_output(), image_producer.ind_deq.name])
                    for i in range(0, indices.shape[0]):
                        if not image_paths[indices[i]] in all_embeddings.keys():
                            all_embeddings[image_paths[indices[i]]] = probs[i, :]
                        else:
                            all_embeddings[image_paths[indices[i]]] += probs[i, :]
                    num_loaded = num_loaded + indices.shape[0]
                print('Classified {}'.format(num_loaded * 1.0 / num_imgs_in_queue))

            for imp in image_paths:
                 all_embeddings[imp] = all_embeddings[imp]/num_feats_per_image

            # Stop the worker threads
            coordinator.request_stop()
            coordinator.join(threads, stop_grace_period_secs=2)
        if save_feats_fp is not None:
            with open(save_feats_fp, 'w') as f:
                cPickle.dump(all_embeddings, f, cPickle.HIGHEST_PROTOCOL)
    else:
        with open(save_feats_fp, 'r') as f:
            print('Loading from {}'.format(save_feats_fp))
            all_embeddings = cPickle.load(f)
    return all_embeddings


def classify_lfw(model_data_path, image_pairs_fp, base_path, save_feats_fp='', bbox_fp=None):
    '''
    :param model_data_path:
    :param image_pairs_fp:
    :param base_path:
    :param save_feats_fp:
    :param bbox_fp:
    :return:
    '''

    image_paths = []
    with open(image_pairs_fp, 'r') as f:
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
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[0], l[0], int(l[1])))
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[0], l[0], int(l[2])))
            ctr = ctr + 1

        for _ in range(num_pairs):
            l = im_pairs_lines[ctr].split()
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[0], l[0], int(l[1])))
            image_paths.append('{}/{}_{:04d}.jpg'.format(l[2], l[2], int(l[3])))
            ctr = ctr + 1

    # Create an image producer (loads and processes images in parallel)


    all_embeddings = get_features(model_data_path)
    verif_labels = np.concatenate((np.ones(num_pairs), np.zeros(num_pairs)))
    for _ in range(num_splits - 1):
        verif_labels = np.concatenate((verif_labels, np.concatenate((np.ones(num_pairs), np.zeros(num_pairs)))))

    scores = np.zeros(2 * num_pairs * num_splits)
    euc_dist = lambda x, y: np.sqrt(np.sum(np.square(x - y)))
    for i in range(2 * num_pairs * num_splits):
        em1 = all_embeddings[image_paths[2 * i]]
        em2 = all_embeddings[image_paths[2 * i + 1]]
        scores[i] = euc_dist(em1, em2)
    score_sort_inds = np.argsort(scores)
    sorted_labels = verif_labels[score_sort_inds]
    tp = 0.0
    fp = 0.0
    tpr = [0.0, ]
    fpr = [0.0, ]
    for i in range(sorted_labels.shape[0]):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
            tpr.append(tp * 1.0 / (num_splits * num_pairs))
            fpr.append(fp * 1.0 / (num_splits * num_pairs))

    plt.plot(fpr, tpr)
    plt.savefig('data/roc.pdf')
    plt.cla()

    with open('data/out.txt', 'w') as f:
        for i in range(len(tpr)):
            f.write('{} {} \n'.format(tpr[i], fpr[i]))

    ### Get accuracy
    if num_splits > 1:
        split_accuracies = np.zeros(num_splits)
        for i in range(num_splits):
            flags = np.zeros(2 * num_pairs * num_splits, dtype=np.bool)
            flags[2 * num_pairs * i:2 * num_pairs * (i + 1)] = True
            test_scores = scores[flags]
            test_labels = verif_labels[flags]
            dev_scores = scores[np.logical_not(flags)]
            dev_labels = verif_labels[np.logical_not(flags)].astype(np.bool)
            max_acc = 0
            final_th = 0
            for j in range(dev_scores.shape[0] - 1):
                th = (dev_scores[j] + dev_scores[j + 1]) / 2.0
                correct_preds = (dev_scores < th) == dev_labels
                if max_acc < np.mean(correct_preds):
                    max_acc = np.mean(correct_preds)
                    final_th = th

            correct_test_preds = ((test_scores < final_th) == test_labels)
            split_accuracies[i] = np.mean(correct_test_preds)

        print('Accuracy: {:.4} +- {:.4}'.format(np.mean(split_accuracies), np.std(split_accuracies)))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Converted parameters for the GoogleNet model')
    parser.add_argument('--image_paths', required=True, dest='image_paths', help='LFW pairs.txt path')
    parser.add_argument('--save_feats_path', default=None, help='Save fc7 embeddings to a file (optional)')
    parser.add_argument('--cfg', dest='cfg_file', default=None, help='Config file for testing (optional)')
    parser.add_argument('--bbox_file', dest='bbox_file', default=None,
                        help='File containing dictionary of face bboxes (optional)')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)


    with open(args.image_paths,'r') as f:
        img_paths = [l.strip() for l in f.readlines()]

    spec = datasets.get_data_spec(model_class=VGG_FACE_16)

    ext_num = spec.batch_size-len(img_paths)%spec.batch_size
    img_paths.extend(img_paths[:ext_num])

    image_producer = dataset.VGGFaceProducer(img_paths, data_spec=spec, bbox_fp=args.bbox_file)

    get_features(args.model_path, image_producer, args.save_feats_path, bbox_fp=args.bbox_file)


if __name__ == '__main__':
    main()
