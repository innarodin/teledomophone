import random
from configobj import ConfigObj
import pika
import sys
import pickle


class RabbitClass:
    def __init__(self, path_config, logger):
        self.queue_name = None
        self.exchange = None
        self.config = ConfigObj(path_config)
        self.logger = logger

        amqp_url = 'amqp://%s:%s@%s:%s/%s' % (
            self.config['rabbitmq']['user'],
            self.config['rabbitmq']['pass'],
            self.config['rabbitmq']['host'],
            self.config['rabbitmq']['port'],
            self.config['rabbitmq']['vhost'],
        )
        amqp_url_query = {
            'heartbeat_interval': 600
        }

        self.exchange = self.config['rabbitmq']['exchange']
        self.amqp_parameters = pika.URLParameters(
            amqp_url + '?' + '&'.join(['%s=%s' % (k, v) for k, v in amqp_url_query.items()]))

        self.create_connection()

    def create_connection(self):
        try:
            self.connection = pika.BlockingConnection(self.amqp_parameters)
            self.channel = self.connection.channel()
        except pika.exceptions.AMQPConnectionError as err:
            msg = {
                "service_id": None,
                "session_id": None,
                "status": "Pika exception:".format(err)
            }
            self.logger.error(msg)
            sys.exit(255)

    def create_queue(self, queue_name):
        self.queue_name = queue_name
        self.channel.queue_declare(queue=self.queue_name)

    def read_queue(self, callback):
        result = self.channel.queue_declare(self.queue_name, durable=False)
        self.channel.queue_bind(exchange=self.exchange,
                                queue=self.queue_name,
                                routing_key=self.queue_name)

        consumer_id = ''.join(['%02X' % random.getrandbits(8) for _ in range(8)])

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
        lambda channel, method_frame, header_frame, body: callback(channel, method_frame, header_frame, body),
            queue=self.queue_name, consumer_tag='{}.{}'.format(self.queue_name, consumer_id), no_ack=True)

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
        except pika.exceptions.ConnectionClosed as err:
            msg = {
                "service_id": None,
                "session_id": None,
                "status": "RabbitMQ connection closed. Reason: {}".format(err)
            }
            self.logger.error(msg)
            sys.exit(255)

    def send_message(self, body, name_queue):
        try:
            self.channel.basic_publish(self.exchange,
                                       name_queue,
                                       pickle.dumps(body),
                                       pika.BasicProperties(
                                           content_type='application/json',
                                           delivery_mode=2,
                                       )
                                       )
        except pika.exceptions.ConnectionClosed:
            self.create_connection()
            self.channel.basic_publish(self.exchange,
                                       name_queue,
                                       pickle.dumps(body),
                                       pika.BasicProperties(
                                           content_type='application/json',
                                           delivery_mode=2,
                                       )
                                       )
