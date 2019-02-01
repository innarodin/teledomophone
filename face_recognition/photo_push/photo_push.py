#!/usr/bin/env python3

import argparse
import json
import cv2
import os
import sys
import time
import numpy as np
from configobj import ConfigObj
from imutils.video import VideoStream
from rabbit_queues import RabbitQueue
from rabbitmq import RabbitConnection
import logging
import pika
import redis
import logstash


class PhotoCreator:
    def __init__(self):
        self.logger = self.create_logger()
        self.logger.debug("Start photo push")
        self.service_dict = self.get_streams()

    @staticmethod
    def get_streams():
        vs6 = VideoStream(src="rtsp://user:cVis_288@10.100.146.221/Streaming/Channels/1?tcp").start()
        vs7 = VideoStream(src="rtsp://admin:cVis_288@10.100.158.125/Streaming/Channels/101").start()
        vs8 = VideoStream(src="rtsp://admin:cVis_288@10.100.158.127/Streaming/Channels/1").start()

        service_dict = {'etalon6': vs6,
                        'etalon7': vs7,
                        'etalon8': vs8
                        }
        return service_dict

    @staticmethod
    def create_logger():
        logger = logging.getLogger("photo_pushApp")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("/volumes/logs/push.log")  # create the logging file handler
        formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)  # add handler to logger object

        host = '10.80.0.30'
        lh = logstash.TCPLogstashHandler(host, 5000, version=1, tags=['photo_push'])
        lh.setLevel(logging.INFO)
        logger.addHandler(lh)
        return logger

    def create_photo(self, ch, method, properties, body):
        try:
            msg = json.loads(body.decode('utf-8'))
        except pika.exceptions.ConnectionClosed:
            queue.create_connection()
            self.logger.debug("Restart photo push")
            msg = json.loads(body.decode('utf-8'))

        session_id = msg['session_id']
        service_id = msg['service_id']
        received_time = msg['time']

        extra = {
            'service_id': service_id,
            'session_id': session_id,
        }
        self.logger.info(extra, extra=extra)

        if service_id not in ['etalon6', 'etalon7', 'etalon8']:
            return

        self.logger.debug("Time1: {}".format(time.time() - received_time))

        for i in range(3):
            t0 = time.time()
            if r_cache.get(str(session_id)) is not None:
                return

            frame = self.service_dict[service_id].read()
            if frame is None:
                msg = {
                        "session_id": session_id,
                        "service_id": service_id,
                        "status": 'Error frame'
                    }
                self.logger.error(msg)
                sys.exit(255)

            t1 = time.time()
            self.logger.debug("time2: {}".format(t1 - t0))

            retval, frame_jpg = cv2.imencode(".jpg", frame)
            msg = {
                "t0": t0,
                "t1": t1,
                "face": frame_jpg.tobytes(),
                "service_id": service_id,
                "session_id": session_id
            }

            queue.send_message(msg, 'detect')
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video stream capturing component for face detection and recognition service')
    parser.add_argument('--config', '-c',
                        dest='config', type=str,
                        default='/volumes/recognition.cfg',
                        help='Path to configuration file'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("No such file or directory: %s" % args.config)

    r_cache = redis.StrictRedis(host='redis', port=6379, db=0)
    config = ConfigObj(args.config)

    if 'rabbitmq' not in config:
        sys.exit("Mandatory section missing: %s" % 'rabbitmq')

    photo_push = PhotoCreator()

    queue_vad = RabbitQueue(args.config)
    queue = RabbitConnection(args.config, photo_push.logger)
    queue_vad.read_queue_with_direct_exchange(photo_push.create_photo, 'kicks', 'kick_face')
