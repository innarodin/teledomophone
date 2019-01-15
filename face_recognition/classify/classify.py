#!/usr/bin/python3.5

import argparse
import time
import sys
from scipy import spatial
import numpy as np
import pickle
import operator
import os
import postgresql
from configobj import ConfigObj
import redis
from rabbitmq import RabbitClass
import json
import decimal
import logging


def face_distance(face_encodings, face_to_compare):
    if not np.isnan(face_encodings).any():
        dist = spatial.distance.sqeuclidean(face_encodings, face_to_compare)
        return dist
    else:
        return np.empty(0)


def save_emb_to_db(found_cluster, embedding):
    with postgresql.open('pq://postgres:postgres@db:5432/recognition') as db:
        ins = db.prepare("INSERT INTO embeddings (name, embedding) VALUES ($1, $2);")
        ins(found_cluster, np.array(embedding, dtype=np.dtype(decimal.Decimal)))


def get_centers_from_db():
    with postgresql.open('pq://postgres:postgres@db:5432/recognition') as db:
        result = db.query("SELECT name, embedding, distance FROM centers;")
        centers = {}
        for i in result:
            emb_array = [[]]
            emb_array[0] = list(map(float, i[1]))
            emb_array = np.asarray(emb_array)
            centers[i[0]] = (emb_array, float(i[2]))

    return centers


def classification(channel, method, props, body):
    data = pickle.loads(body)
    path = data['path']
    service_id = data['service_id']
    embedding = data['embeddings']
    t0 = float(data['t0'])
    t1 = float(data['t1'])
    t2 = float(data['t2'])
    t3 = float(data['t3'])
    session_id = data['session_id']

    t6 = time.time()

    if r.get(str(session_id)) is not None:
        return

    if r.get('centers') is None:
        centers = get_centers_from_db()
        logger.debug("Centers from db")
    else:
        centers = pickle.loads(r.get('centers'))

    face_distances = {}
    cosine_similarities, face_dists = list(), list()
    for name in centers:
        distance = face_distance(centers[name][0], embedding)
        face_distances[name] = distance

        sim = np.dot(embedding, centers[name][0][0].T)
        cosine_similarities.append((name, sim))

    min_dist_cluster = min(face_distances.items(), key=operator.itemgetter(1))[0]

    sorted_similarities = sorted(cosine_similarities, key=lambda cos: cos[1])
    second_place, first_place = sorted_similarities[-2:]

    if face_distances[min_dist_cluster] <= centers[min_dist_cluster][1]:
        found_cluster = min_dist_cluster
        dist = face_distances[found_cluster]
        save_emb_to_db(found_cluster, embedding)
    elif (first_place[1] / second_place[1] >= 2 and 0.4 > first_place[1] > 0.3) or first_place[1] > 0.4:
        found_cluster = first_place[0]
        save_emb_to_db(found_cluster, embedding)
        dist = "cosine_similarities: {}".format(first_place[1])
    else:
        found_cluster = "Unknown"
        dist = None

    msg = {
        'predict': (found_cluster, dist),
        'service_id': service_id,
        'session_id': session_id,
        'type': 'face'
    }

    logger.debug("redis {}".format(r.get(str(session_id))))
    if r.get(str(session_id)) is not None:
        return
    
    queue.send_message(msg, 'voice_face')
    logger.debug("sent")

    # if found_cluster != 'Unknown':
    #     with postgresql.open('pq://postgres:postgres@db:5432/recognition') as db:
    #         result = db.query("SELECT username FROM username_id WHERE user_id={};".format(found_cluster))
    #         if result[0][0] is not None:
    #             found_cluster = result[0][0]

    msg = {
        "service_id": service_id,
        "session_id": session_id,
        "status": "NAME: {}, DIST: {}".format(found_cluster, dist)
    }
    logger.info(msg)

    t7 = time.time()

    logger.debug("Photo push: {}".format(t1 - t0))
    logger.debug("Photo push -> detector: {}".format(t2 - t1))
    logger.debug("detector: {}".format(t3 - t2))
    logger.debug("detector -> classify: {}".format(t6 - t3))
    logger.debug("classify: {}".format(t7 - t6))
    logger.debug("All time: {}".format(t7 - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Video stream capturing component for face detection and recognition service')

    parser.add_argument('--config', '-c',
                        dest='config', type=str,
                        default="/volumes/recognition.cfg",
                        help='Path to configuration file'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("No such file or directory: %s" % args.config)

    config = ConfigObj(args.config)

    for config_section in ('rabbitmq',):
        if config_section not in config:
            sys.exit("Mandatory section missing: %s" % config_section)

    logger = logging.getLogger("classifyApp")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/classify.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    # queue = RabbitQueue(args.config)
    # queue_handler = QueueHandler()
    # queue_handler.set_queue(queue)
    # queue_handler.setLevel(logging.INFO)
    # queue_handler.setFormatter(formatter)
    # logger.addHandler(queue_handler)

    logger.debug("Start classify")

    r = redis.StrictRedis(host='redis', port=6379, db=0)

    queue = RabbitClass(args.config, logger)
    # queue.create_exchange(config['rabbitmq']['exchange'])
    queue.create_queue('classify')
    queue.read_queue(classification)
