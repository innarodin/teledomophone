# coding=utf-8
import argparse
import os

import sys
import postgresql
import telebot
import logging
import pickle
import pytz
from rabbitmq import RabbitClass
from rabbit_queues import RabbitQueue, QueueHandler
from configobj import ConfigObj

_token = '738125364:AAFyEJm9A0noRUyaNVr58O-91cr0Fz-BG54'
_timeout = 1
_limit = 100

bot = telebot.TeleBot(_token)


def get_username(user_id):
    with postgresql.open('pq://postgres:postgres@db:5432/recognition') as db:
        result = db.query("SELECT username FROM username_id WHERE user_id={};".format(user_id))
        if len(result) == 0:
            return user_id
        return result[0][0]


def read_feedback(channel, method_frame, header_frame, body):
    data = pickle.loads(body)
    reason = data['reason']
    face_pred = data['face_pred']
    voice_pred = data['voice_pred']
    conf = data['conf']
    time = data['time']

    bot.send_message(face_pred, "{} ЛИЦО: {} ГОЛОС: {} ВЕРОЯТНОСТЬ РАСПОЗНАВАНИЯ ГОЛОСА: {} ВРЕМЯ: {}".format(reason, get_username(face_pred),
                                                                                 get_username(voice_pred), conf, time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Video stream capturing component for face detection and recognition service')

    parser.add_argument('--config', '-c',
                        dest='config', type=str,
                        default="/volumes/feedback.cfg",
                        help='Path to configuration file'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("No such file or directory: %s" % args.config)

    config = ConfigObj(args.config)

    for config_section in ('rabbitmq',):
        if config_section not in config:
            sys.exit("Mandatory section missing: %s" % config_section)

    logger = logging.getLogger("feedbackApp")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("/volumes/logs/feedback.log")  # create the logging file handler
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add handler to logger object

    queue = RabbitQueue("/volumes/feedback.cfg")
    queue_handler = QueueHandler()
    queue_handler.set_queue(queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    queue = RabbitClass("/volumes/feedback.cfg", logger)
    queue.read_queue(read_feedback, 'feedback')
