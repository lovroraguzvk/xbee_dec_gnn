
import time, json
from digi.xbee.devices import ZigBeeDevice, ZigBeeNetwork
from digi.xbee.models.address import XBee64BitAddress, XBee16BitAddress # dodano
from digi.xbee.exception import TransmitException # dodano


class ZigBeeMessage():
    def __init__(self, msg_name, payload):
        self.msg_name = msg_name
        self.payload = payload

class ZigBeeNode():
    def __init__(self, port, baud_rate):
        communicator = ZigBeeCommunicator(port, baud_rate)
        communicator.start()

class ZigBeeCommunicator():
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        self.network = None

    def start(self):
        self.device = ZigBeeDevice(self.port, self.baud_rate)
        self.device.open()
        self.device.add_data_received_callback(self._data_receive_callback)
        self.network = self.device.get_network()

    def add_msg_type(self, msg_type):
        pass

    def _data_receive_callback(self, xbee_message):
        pass

    def send_message(self, msg, addr, node_id):
        data = json.dumps(msg).encode("utf-8")

        if isinstance(addr, str):
            addr = XBee64BitAddress.from_hex_string(addr)

        msg_type = msg.get("type")
        ok = False
        for attempt in range(1, 5):
            try:
                self.device.send_data_64_16(addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                ok = True
                if attempt == 1:
                    self.get_logger().debug("TX: %s -> node %s", msg_type, node_id)
                else:
                    self.get_logger().debug("TX: %s -> node %s (retry %d)", msg_type, node_id, attempt)
                break
            except TransmitException as e:
                status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                self.get_logger().warning("TX fail: %s (attempt %d, %s)", msg_type, attempt, status)
                time.sleep(0.1)

        if not ok:
            self.get_logger().error("TX gave up: %s to node %s", msg_type, node_id)

        time.sleep(0.05)
    
    def discover_others(self, num_nodes=0):
        self.network.start_discovery_process()
