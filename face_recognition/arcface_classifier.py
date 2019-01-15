import numpy as np
import mxnet
import cv2 as cv
import sys
import os
import random
import pickle
import imgaug
import face_model

from collections import defaultdict
from imgaug import augmenters as iaa
from time import time


def sqeuclidean_dist(emb_1, emb_2):
    return np.sum(np.square(emb_1 - emb_2))


class Classifier:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.embeddings = defaultdict(list)
        self.first_centorids = defaultdict(list)

    def load_model(self):
        self.model = face_model.FaceModel(self.args.gpu, self.args.img_size, self.args.model_path, self.args.threshold,
                                          self.args.det)

    def parse_video(self, video_path, name):
        capture = cv.VideoCapture(video_path)
        while True:
            ret, img = capture.read()

            if ret is False:
                break

            img = self.augment_video(img)
            aligned_face = self.model.get_input(img)

            if aligned_face is not None:
                print(aligned_face.shape)
                face_embedding = self.model.get_feature(aligned_face)
                self.embeddings[name].append(face_embedding)

        centroid = np.mean(self.embeddings[name], axis=0)
        self.first_centorids[name].append(centroid)
        print('First mean centroid {} for person {}:'.format(centroid.mean(), name))

    def get_embeddings(self):
        names = [name for name in os.listdir(self.args.video_folder)]
        for name in names:
            video_path = os.path.join(self.args.video_folder, name,
                                      os.listdir(os.path.join(self.args.video_folder, name))[0])
            self.parse_video(video_path, name)

        with open(self.args.emb_file, 'wb') as file:
            pickle.dump(self.embeddings, file)

        with open(self.args.cent_file, 'wb') as file:
            pickle.dump(self.first_centorids, file)

    # static
    def augment_video(self, img):
        """Augmentation to get wider cluster"""
        random_counter = random.randint(0, 5)

        ag_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
        flipper = iaa.Fliplr(0.5)
        contrast_normalizer = iaa.ContrastNormalization((0.75, 1.5))
        dropouter = iaa.Dropout(0.015)

        if random_counter == 0:
            return cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if random_counter == 1:
            return ag_noise.augment_image(img)

        if random_counter == 2:
            return flipper.augment_image(img)

        if random_counter == 3:
            return contrast_normalizer.augment_image(img)

        if random_counter == 4:
            return dropouter.augment_image(img)

        if random_counter == 5:
            return img

    def create_classifier(self):
        with open(self.args.emb_file, 'rb') as file:
            embs_loaded = pickle.load(file)

        with open(self.args.cent_file, 'rb') as file:
            centers_loaded = pickle.load(file)

        dists_in_clusters, distance_between_clusters = defaultdict(list), defaultdict(list)
        max_dist_in_cluster, available_embs, centroids = defaultdict(int), defaultdict(list), defaultdict(list)

        for person in embs_loaded:
            for emb in embs_loaded[person]:
                """get all distances in cluster between embeddings"""
                dists_in_clusters[person].append(sqeuclidean_dist(emb, centers_loaded[person]))

            max_dist_in_cluster[person] = max(dists_in_clusters[person])
            print('Maximum distance = {} in cluster for person {}'.format(max_dist_in_cluster[person], person))

        for center in centers_loaded:
            nearest_clusters, dists = list(), list()
            for _center in centers_loaded:
                if center is not _center:
                    nearest_clusters.append((_center, sqeuclidean_dist(centers_loaded[center][0],
                                                                       centers_loaded[_center][0])))

            nearest_cluster, dist_to_cluster = sorted(nearest_clusters, key=lambda dist: dist[1])[0]

            if dist_to_cluster < max_dist_in_cluster[center]:
                available_distance = dist_to_cluster
            else:
                available_distance = max_dist_in_cluster[center]

            for emb in embs_loaded[center]:
                if sqeuclidean_dist(emb, centers_loaded[center]) < available_distance:
                    available_embs[center].append(emb)

            final_centroid = np.mean(available_embs[center], axis=0)
            for emb in available_embs[center]:
                dists.append(sqeuclidean_dist(emb, final_centroid))

            radius = max(dists).round(3)
            centroids[center].append((np.mean(available_embs[center], axis=0), radius))

        for person in centroids:
            print("Person`s name is {} | radius = {}".format(person, centroids[person][0][1]))

        with open(self.args.final_cent_file, 'wb') as file:
            pickle.dump(centroids, file)

    def identification(self):
        with open(self.args.final_cent_file, 'rb') as file:
            centroids = pickle.load(file)

        for dirs, _, files in os.walk(self.args.test):
            for file in files:
                img_path = os.path.join(self.args.test, file)
                img = cv.imread(img_path)
                aligned_face = self.model.get_input(img)

                if aligned_face is not None:
                    face_emb = self.model.get_feature(aligned_face)

                    cosine_similarities, face_dists = list(), list()

                    for center in centroids:
                        sim = np.dot(face_emb, centroids[center][0][0].T)
                        face_dists.append((center, sqeuclidean_dist(face_emb, centroids[center][0][0])))
                        cosine_similarities.append((center, sim))

                    sorted_distances = sorted(face_dists, key=lambda dist: dist[1])
                    face_ident, dist_to_centroid = sorted_distances[0]

                    sorted_similarities = sorted(cosine_similarities, key=lambda cos: cos[1])
                    second_place, first_place = sorted_similarities[-2:]
                    # print('1_st place: {} | 2_nd place: {} | face ident {}:'.format
                    #      (first_place, second_place, face_ident))
                    if dist_to_centroid < centroids[face_ident][0][1]:
                        print('Identificated by dists: {}'.format(face_ident))
                    elif (first_place[1] / second_place[1] >= 2 and 0.4 > first_place[1] > 0.3) or first_place[1] > 0.4:
                        print('Identificated: {}'.format(first_place[0]))
                    else:
                        print('Cant identificate!')
