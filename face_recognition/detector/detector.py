import argparse
import sys
import time
import cv2
import pickle
import os
import numpy as np
import pytz
import datetime
from configobj import ConfigObj
#import mtcnn_detector
from rabbitmq import RabbitClass
import tensorflow as tf
# from detect_face import detect_face, create_mtcnn
import face_model
import logging
import redis
from rabbit_queues import RabbitQueue, QueueHandler


def make_photo(path, face, session_id):
    tz = pytz.timezone('Asia/Yekaterinburg')
    data = str(datetime.datetime.now(tz=tz).date())
    if not os.path.exists(path):
        os.mkdir(path)

    if data not in os.listdir(path):
        os.mkdir(os.path.join(path, data))

    new_path = os.path.join(path, data)
    cur_time = str(datetime.datetime.now(tz=tz).time())
    img_name = cur_time + '.jpg'
    path_to_write = str(os.path.join(new_path, img_name))
    logger.debug("{} {}".format(path_to_write, session_id))
    face = np.swapaxes(face, 0, 2)
    face = np.swapaxes(face, 0, 1)
    cv2.imwrite(path_to_write, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return path_to_write.split("photo/")[1]


def detect(channel, method_frame, header_frame, body):
    storage_path = None
    t2 = time.time()

    if 'storage' in detector_config:
        storage_path = detector_config['storage']

    data = pickle.loads(body)
    face_string = data['face']
    t0 = data['t0']
    t1 = data['t1']
    service_id = data['service_id']
    session_id = data['session_id']

    if r.get(str(session_id)) is not None:
        return

    image = np.asarray(bytearray(face_string), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face = model.get_input(image)

    if face is None:
        if r.get("detect" + str(session_id)) is None:
            r.set("detect" + str(session_id), 1)
            logger.debug("No face {}".format(session_id))
        else:
            r.incr("detect" + str(session_id))
            logger.debug("No face {}".format(session_id))

        if r.get("detect" + str(session_id)) == b'3':
            msg = {
                "service_id": service_id,
                "session_id": session_id,
                "status": "No face"
            }
            logger.info(msg)
            r.delete("detect" + str(session_id))
        return

    emb_array = model.get_feature(face)
    if emb_array is None:
        return

    path = make_photo(os.path.join(storage_path, service_id), face, session_id)

    t3 = time.time()
    msg = {
        'embeddings': emb_array,
        'path': os.path.join(path),
        't0': t0,
        't1': t1,
        't2': t2,
        't3': t3,
        'service_id': service_id,
        "session_id": session_id
    }

    queue.send_message(msg, 'classify')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video stream capturing component for face detection and recognition service')

    parser.add_argument('--config', '-c',
                        dest='config', type=str,
                        default="/volumes/recognition.cfg",
                        help='Path to configuration file'
                        )
    parser.add_argument('--model_path', type=str, help='Path to folder with arcface model', default='model/r50/model, 0')
    parser.add_argument('--img_size', default='112, 112', help='Size for croped/aligned img')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("No such file or directory: %s" % args.config)

    config = ConfigObj(args.config)

    for config_section in ('rabbitmq', 'detector'):
        if config_section not in config:
            sys.exit("Mandatory section missing: %s" % config_section)

    # get configuration parameters
    detector_config = config['detector']

    model = face_model.FaceModel(args.gpu, args.img_size, args.model_path, args.threshold, args.det)

    r = redis.StrictRedis(host='redis', port=6379, db=0)

    logger = logging.getLogger("detectorApp")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/detector.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    queue = RabbitQueue(args.config)
    queue_handler = QueueHandler()
    queue_handler.set_queue(queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    queue = RabbitClass(args.config, logger)
    queue.create_queue('detect')
    logger.debug("Start detector")
    queue.read_queue(detect)
