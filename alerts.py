from datetime import datetime
from camera_functions import *

from kafka import KafkaProducer
from kafka.errors import KafkaError


class KafkaMessager:
    def __init__(self, bootstrap_servers=['10.0.21.11:32100']):
        self.bootstrap_servers = bootstrap_servers
        self.building = get_cameras_list()

        self.incidents = {
            'MAX_OCCUPATION': 'La ocupación ha llegado a su límite en la sala.',
            'CHAOS_PREDICTED': '¡ATENCIÓN! No es posible una evacuación ordenada.',
            'GENERAL_WARNING': 'Incidencia detectada.',
            'PEAK_PREDICTED': '¡ATENCIÓN! se prevé un pico de ocupación durante la próxima hora.'  # To be implemented
        }

        self.topics = dict(
            THE_ONLY_TOPIC='mqtt-odin-platform-alertsystem-control-kafka'
        )

    def send_message(self, camera_id, incident, t_stamp=datetime.now()):
        producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)

        alert = self.incidents[incident]
        room = self.building[camera_id]

        body = alert + ' ' + room

        message = {
            'version': '0.0.1.0',
            'id': camera_id,
            'timestamp': t_stamp,
            'sequence': 'xxx',
            'ker': 'har-tech',
            'message': 'status',
            'data': {
                'alertType': incident,
                'alertID': 'alert_id',  # Unique ID for each alert
                'alertMessage': body,
                'timestamp': t_stamp,
                'sound': True,
                'location': camera_id
            }
        }

        # Asynchronous by default
        future = producer.send(client, message)  # (client = 'my-topic2', message = b'raw_bytes')

        # Block for 'synchronous' sends
        try:
            record_metadata = future.get(timeout=10)
        except KafkaError:
            # Decide what to do if produce request failed...
            print('El mensaje no se pudo enviar por un error con Python.')
            pass

        # Successful result returns assigned partition and offset
        print(record_metadata.topic)
        print(record_metadata.partition)
        print(record_metadata.offset)


kfk = KafkaMessager()



