
from digi.xbee.devices import ZigBeeDevice # dodano
from digi.xbee.models.address import XBee64BitAddress, XBee16BitAddress # dodano
from digi.xbee.exception import TransmitException # dodano

class XBeeDevice():
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        
        self.device.open()

    def add_data_received_callback(self, callback):
        self.device.add_data_received_callback(callback)

    
