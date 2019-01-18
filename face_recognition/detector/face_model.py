from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocess'))

import argparse
import face_image
import face_preprocess
import numpy as np
import mxnet as mx
import random
import cv2 as cv
import sklearn
from scipy.spatial import distance
from scipy import misc
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from six.moves import xrange


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix, epoch = _vec[0], int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, gpu, image_size, model, threshold, det):
        ctx = mx.gpu(gpu)
        _vec = image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None

        if len(model) > 0:
            self.model = get_model(ctx, image_size, model, 'fc1')

        self.det = det
        self.threshold = threshold
        self.det_minsize = 120
        self.det_threshold = [0.6, 0.7, 0.8]
        self.image_size = image_size
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        if det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
        self.detector = detector

    def get_input(self, face_img):
        """Aligning face and resizing to (112, 112)"""
        ret = self.detector.detect_face(face_img, det_type=self.det)
        if ret is None:
            return None

        bbox, points = ret
        if bbox.shape[0] == 0:
            return None

        max_box_area, box_id = 0, None
        for i, (x, y, x, y, p) in enumerate(bbox):
            box_area = (x - x) * (y - y)
            if box_area < max_box_area or p < 0.99:
                continue
            max_box_area = box_area
            box_id = i

        bbox = bbox[box_id, 0:4]
        if len(points[box_id]) != 10:
            return None
        points = points[box_id, :].reshape((2, 5)).T

        # fp1, fp2, fp3, fp4, fp5 = points[0], points[1], points[2], points[3], points[4]  # left eye, right eye, nose, left mouth, rigth mouth
        # if abs(distance.euclidean(fp1, fp3) - distance.euclidean(fp2, fp3)) > 10 or \
        #                 abs(distance.euclidean(fp4, fp3) - distance.euclidean(fp5, fp3)) > 10:
        #     return None

        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112, 112')
        nimg = cv.cvtColor(nimg, cv.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))

        return aligned

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()

        return embedding
