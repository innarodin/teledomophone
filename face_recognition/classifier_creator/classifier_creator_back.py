#!/usr/bin/env python
import postgresql
from scipy import spatial
import numpy as np
import time
import sys
import logging
import redis
import pickle
import psycopg2.extensions
import select
import json


def get_embeddings():
    with postgresql.open('pq://postgres:postgres@db:5432/recognition') as db:
        result = db.query("SELECT name, embedding FROM embeddings;")

    embs = {}
    for row in result:
        emb_array = [[]]
        emb_array[0] = list(map(float, row[1]))
        emb_array = np.asarray(emb_array)

        if row[0] in embs:
            embs[row[0]].append(emb_array)
        else:
            embs[row[0]] = []
            embs[row[0]].append(emb_array)

    return embs


def face_distance(face_encodings, face_to_compare):
    if not np.isnan(face_encodings).any():
        dist = spatial.distance.sqeuclidean(face_encodings, face_to_compare)
        return dist
    else:
        return np.empty(0)


def write_centers_to_db(centers):
    with postgresql.open('pq://postgres:postgres@db:5432/recognition') as db:
        upd = db.prepare("UPDATE centers SET embedding=$1, distance=$2 WHERE name=$3;")
        ins = db.prepare("INSERT INTO centers (name, embedding, distance) VALUES ($1, $2, $3);")

        for name in centers:
            emb = centers[name][0]
            dist = centers[name][1]
            if r.get('centers') is None:
                ins(name, emb[0], dist)
            elif name in pickle.loads(r.get('centers')):
                upd(emb[0], dist, name)
            else:
                ins(name, emb[0], dist)


def create_classifier():
    centers = {}
    d_o_e_means = []  # first time centers
    max_dists = {}  # max distance to cluster items' embeddings

    # get all centers (first time) -> d_o_e_means
    # get max cluster distance -> max_dists
    embs_loaded = get_embeddings()

    for dataset_object in embs_loaded:
        if len(embs_loaded[dataset_object]) < 5:
            continue

        d_o_e_mean = np.mean(embs_loaded[dataset_object], axis=0)
        d_o_e_means.append((dataset_object, d_o_e_mean))

        dists = []
        for dataset_obj_emb in embs_loaded[dataset_object]:
            dist = face_distance(dataset_obj_emb[0], d_o_e_mean[0])
            dists.append(dist)

        max_dists[dataset_object] = max(dists)  # max cluster distance

    # print(time.ctime())
    for name, center in d_o_e_means:
        distances = {}

        # get the nearest center -> close_center
        for _name, emb in d_o_e_means:
            if name == _name:
                continue
            distances[_name] = face_distance(emb[0], center[0])

        close_centers = list(sorted(distances.items(), key=lambda x: x[1]))

        # get min distance to another cluster
        dists_to_other_clusters = {}

        for _name, d in close_centers:
            dists_to_other_clusters[_name] = []
            for dataset_obj_emb in embs_loaded[_name]:
                dist = face_distance(dataset_obj_emb[0], center[0])
                dists_to_other_clusters[_name].append(dist)

        dists_to_clusters = {}
        for i in dists_to_other_clusters.items():
            dists_to_clusters[i[0]] = min(i[1])

        nearest_cluster = sorted(dists_to_clusters.items(), key=lambda x: x[1])[0]  # name, dist

        # get available radius for cluster
        if nearest_cluster[1] < max_dists[name]:
            available_dist = nearest_cluster[1]
        else:
            available_dist = max_dists[name]

        if available_dist > 0.8:
            available_dist = 0.8

        # get all embeddings in radius
        dataset_available_embeddings = []
        for obj_emb in embs_loaded[name]:
            dist = face_distance(obj_emb[0], center[0])
            if dist <= available_dist:
                dataset_available_embeddings.append(obj_emb)
            else:
                continue

        # get center
        center = np.mean(dataset_available_embeddings, axis=0)
        centers[name] = (center, available_dist)
        logger.info("Name: {}, all embs: {}, available embs: {}, radius: {}".format(name, len(embs_loaded[name]),
                                                                                    len(dataset_available_embeddings),
                                                                                    available_dist))

    write_centers_to_db(centers)
    r.set('centers', pickle.dumps(centers))


if __name__ == '__main__':
    logger = logging.getLogger("Classifier_creator")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/classifier_creator.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    r = redis.StrictRedis(host='redis', port=6379, db=0)
    create_classifier()

    conn = psycopg2.connect(dbname='recognition', user='postgres', password='postgres', host='db')
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    curs = conn.cursor()
    curs.execute("LISTEN embeddings;")

    inserts = 0

    logger.info("Waiting for notifications on channel 'embeddings'")

    while True:
        if select.select([conn], [], [], 5) != ([], [], []):
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                logger.info("Got NOTIFY: {} {}".format(notify.pid, notify.payload))

                name = json.loads(notify.payload)["name"]

                if r.get('centers') is None:
                    create_classifier()
                elif name in pickle.loads(r.get('centers')):
                    inserts += 1
                    if inserts >= 10:
                        inserts = 0
                        create_classifier()
                else:
                    inserts += 1
                    if inserts >= 50:
                        inserts = 0
                        logger.info("NEW USER: {}".format(name))
                        create_classifier()


