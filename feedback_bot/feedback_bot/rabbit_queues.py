import pika
import time
import json
import sys
from configobj import ConfigObj
import logging
import ast


class QueueHandler(logging.Handler):
    def set_queue(self, rabbit_queue):
        self._queue = rabbit_queue
        self._queue.create_exchange_direct('logs')

    def emit(self, record):
        service_id = ast.literal_eval(record.message)['service_id']
        log_entry = self.format(record)
        self._queue.send_message(log_entry, service_id)


class RabbitQueue():
    def __init__(self, path_config):
        self.name_queue = None
        self.exchange = None
        self.config = ConfigObj(path_config)

        amqp_url = 'amqp://%s:%s@%s:%s/%s' % (
            self.config['rabbitmq_vad']['user'],
            self.config['rabbitmq_vad']['pass'],
            self.config['rabbitmq_vad']['host'],
            self.config['rabbitmq_vad']['port'],
            self.config['rabbitmq_vad']['vhost'],
        )
        amqp_url_query = {
            'heartbeat_interval': 600
        }

        self.amqp_parameters = pika.URLParameters(
            amqp_url + '?' + '&'.join(['%s=%s' % (k, v) for k, v in amqp_url_query.items()]))
        self.create_connection()

    def create_connection(self):
        try:
            self.connection = pika.BlockingConnection(self.amqp_parameters)
            self.channel = self.connection.channel()
        except pika.exceptions.AMQPConnectionError as err:
            print("Pika exception:", err)
            sys.exit(255)

    def create_queue(self, name_queue):
        # name_queue = 'intercom'
        self.name_queue = name_queue
        self.channel.queue_declare(queue=self.name_queue, durable=True)
        # connection.close()

    def send_message(self, body, name_queue=None):
        try:
            self.channel.basic_publish('' if self.exchange is None else self.exchange,
                                       routing_key=name_queue if self.name_queue is None else self.name_queue,
                                       body=body,
                                       properties=pika.BasicProperties(content_type='application/json', delivery_mode=2,)

            )
        except pika.exceptions.ConnectionClosed:
            self.create_connection()
            self.channel.basic_publish('' if self.exchange is None else self.exchange,
                                       routing_key=name_queue if self.name_queue is None else self.name_queue,
                                       body=body,
                                       properties=pika.BasicProperties(content_type='application/json',
                                                                       delivery_mode=2, )

                                       )

    def create_exchange_direct(self, name_exchange):
        self.exchange = name_exchange
        self.channel.exchange_declare(exchange=self.exchange,
                                      exchange_type='direct',
                                      durable=True)

    def read_queue_with_direct_exchange(self, callback, exchange_name, routing_key):
        result = self.channel.queue_declare(exclusive=True, durable=True)
        queue_name = result.method.queue
        self.channel.queue_bind(exchange=exchange_name,
                                queue=queue_name,
                                routing_key=routing_key)    # config['voice']['routing_key'])

        self.channel.basic_consume(
        lambda channel, method_frame, header_frame, body: callback(channel, method_frame, header_frame, body),
                                   queue=queue_name,
                                   no_ack=True)

        self.channel.start_consuming()


def callback(ch, method, properties, body):
    data = json.loads(body)
    print(" [x] Received session_id: %s. Error time: %f" % (data['session_id'], time.time() - data['time']))


