import sys
import os
import argparse
import cv2 as cv
import numpy as np
import mxnet as mx
import face_model
import pickle as pkl
import operator
import random

from time import time
from operator import itemgetter
from collections import defaultdict
from imgaug import augmenters as iaa
from arcface_classifier import Classifier


def main(args):
    arcface_class = Classifier(args)
    arcface_class.load_model()
    #arcface_class.get_embeddings()
    #arcface_class.create_classifier()
    arcface_class.identification()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, help='Path to folder with videos for classifier', default='./video')
    parser.add_argument('--model_path', type=str, help='Path to folder with arcface model', default='model/r50/model, 0')
    parser.add_argument('--emb_file', type=str, help='Name for embeddings file', default='embs.pkl')
    parser.add_argument('--cent_file', type=str, help='Name for centroids file', default='cents.emb')
    parser.add_argument('--final_cent_file', type=str, help='Name for file with centroids/radius',
                        default='final_centroids.pkl')
    parser.add_argument('--img_size', default='112, 112', help='Size for croped/aligned img')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
    parser.add_argument('--test', type=str, help='Path to folder for test identification')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
