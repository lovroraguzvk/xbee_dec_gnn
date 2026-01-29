#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, logging, json, threading, socket, argparse, torch, random
from collections import defaultdict
from typing import Dict, Any
from digi.xbee.devices import ZigBeeDevice
from digi.xbee.models.address import XBee64BitAddress, XBee16BitAddress
from digi.xbee.exception import TransmitException, TimeoutException

import networkx as nx
import numpy as np
from colorlog import ColoredFormatter
from prettytable import PrettyTable

from xbee_dec_gnn.decentralized_gnns.dec_gnn import DecentralizedGNN
from xbee_dec_gnn.utils.led_matrix import LEDMatrix

from xbee_dec_gnn.encoder import encode_msg, decode_msg, pack_tensor, unpack_tensor


def load_config(path: str) -> Dict[str, Any]:    # dodano TODO: move to utils.py or something
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class ObjectWithLogger:
    def __init__(self):
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
            datefmt="%M:%S.%f",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )

        self.logger = logging.getLogger("xbee_dec_gnn")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def get_logger(self):
        return self.logger



class Node(ObjectWithLogger):
    def __init__(self, port: str = '/dev/ttyUSB0', baud: int = 9600):
        super().__init__()

        # Unique identifier for the node. # DOC: We assume all nodes have the same format of the name.
        # self.node_id = node_id
        self.node_prefix = "node_"
        self.hostname = socket.gethostname()
        self.node_name = self.node_prefix + self.hostname
        self.node_id = None
        self.id_to_addr = None
        self.data = None

        self.get_logger().info("Node online: hostname=%s", self.hostname)
        # print getReceivedTimout

        self.value = torch.Tensor()  # The current representation of the node.
        self.output = torch.Tensor()  # The interpretable output of the GNN after each layer.
        self.layer = 0  # The current layer of the GNN being processed.
        self.received_mp = defaultdict(dict)  # Message passing values received from neighbors, indexed by iteration
                                              # number. Also used for synchronization.
        self.received_pooling = defaultdict(dict)  # Pooling values received from neighbors, indexed by iteration number

        self.round_counter = 0
        self.local_subgraph = nx.Graph()
        self.active_neighbors = []

        # cfg = load_config('config.json')
        # self.id_to_addr = cfg["id_to_addr"]

        self.stats = {"inference_time": [], "message_passing_time": [], "pooling_time": [], "round_time": []}

        # TODO: Load parameters
        default_model_path = "/root/other_ws/xbee_dec_gnn/xbee_dec_gnn/xbee_dec_gnn/data/MIDS_model.pth"
        self.num_nodes = 3
        self.gnn_model_path = default_model_path

        # # TODO: Load the GNN model. # DOC: Change this to customize how the model is loaded.
        dist_model_kwargs = dict(pooling_protocol="consensus", consensus_sigma=1 / self.num_nodes)
        self.decentralized_model = DecentralizedGNN.from_simple_gnn_wrapper(self.gnn_model_path, **dist_model_kwargs)

        # Initialize the LED matrix if available.
        self.led = LEDMatrix()

        self.port = port
        self.baud = baud

        self.central_addr = None

        self.device = ZigBeeDevice(port, baud)
        self.bcast_lock = threading.Event()
        self.init_id_lock = threading.Event()
        self.graph_lock = threading.Event()

    def run(self):
        # Main loop of the node.
        while True:
            time.sleep(0.1)
            self.compute_gnn()

    def receive_message_xbee(self, xbee_message):
        #check if message is json or pickled



        try:
            msg = json.loads(xbee_message.data.decode("utf-8"))
        except Exception:
            msg = decode_msg(xbee_message.data)

        if msg.get("type") == "DISCOVERY":
            self.central_addr = msg.get("addr")

            new_msg = {
                "type" : "NODE_REGISTER",
                "hostname" : self.hostname
            }

            self.send_message_xbee(new_msg, msg.get("addr"), "CENTRAL")

            self.bcast_lock.set()
            return

        if msg.get("type") == "REGISTER_ACK":
            self.id_to_addr = {int(k): v for k, v in msg["id_to_addr"].items()}
            self.node_id = msg.get("id")

            new_msg = {
                "type" : "ID_CONFIRM",
                "id" : self.node_id
            }

            self.send_message_xbee(new_msg, self.central_addr, "CENTRAL")

            self.node_name = self.node_prefix + str(self.node_id)

            self.init_id_lock.set()
            return

        if msg.get("type") == "GRAPH":
            # Legacy full-graph payload (graph6 + full feature matrix)
            if "graph6_str" in msg:
                G = nx.from_graph6_bytes(bytes(msg["graph6_str"].strip(), "ascii"))
                self.local_subgraph = G.subgraph([self.node_id] + list(G.neighbors(self.node_id)))

                lambda2 = nx.laplacian_spectrum(G)[1]
                self.get_logger().debug(
                    "RX: GRAPH received (\u03bb\u2082=%.4f, nodes=%d, edges=%d)",
                    lambda2,
                    G.number_of_nodes(),
                    G.number_of_edges(),
                )

                self.data = torch.tensor(msg.get("data")).reshape(tuple(msg.get("shape")))
                self.graph_lock.set()
                return

            # Compact per-node payload (neighbors + own features)
            if "n" in msg and "x" in msg:
                graph_node_id = msg.get("id", self.node_id)
                self._apply_graph_payload(graph_node_id, msg.get("n", []), msg.get("x"))
                return

        if msg.get("t") == "G":  # short type for compact payloads
            graph_node_id = msg.get("id", self.node_id)
            self._apply_graph_payload(graph_node_id, msg.get("n", []), msg.get("x"))
            return


        if msg.get("t") == "MP":
            self.receive_message_passing(msg)
            return
        if msg.get("t") == "pooling":
            self.receive_pooling(msg)
            return

    def start(self):
        self.device.open()
        self.device.add_data_received_callback(self.receive_message_xbee)
        self.get_logger().info("XBee receive timeout: %s seconds", self.device.get_sync_ops_timeout())

        self.get_logger().info(f"Port: {self.port} @ {self.baud}")
        self.get_logger().info(f"XBee addr64: {self.device.get_64bit_addr()}")

        self.get_logger().info("Waiting for DISCOVERY from central...")

        self.bcast_lock.wait()
        self.get_logger().info("Handshake: DISCOVERY received; INIT sent; awaiting REGISTER_ACK (id assignment)")

        self.init_id_lock.wait()
        self.get_logger().info("Handshake: REGISTER_ACK received; ID_CONFIRM sent; awaiting GRAPH payload")

        self.graph_lock.wait()
        self.get_logger().info("Setup complete: graph/features loaded; starting GNN loop")

    def get_neighbors(self):
        self.active_neighbors = []

        # We know the whole graph in development mode.
        if self.local_subgraph.number_of_nodes() > 0:
            # Use integer IDs directly to match id_to_addr keys
            self.active_neighbors = list(self.local_subgraph.neighbors(self.node_id))

        if not nx.is_connected(self.local_subgraph):
            self.get_logger().fatal(
                "Local subgraph is not connected! This should never happen. If it does, it is probably due to a "
                "mismatch of communication radius used in this node and for the creation of the discovery messages."
            )
            raise RuntimeError("Local subgraph is not connected.")
        ready = len(self.active_neighbors) > 0

        if ready:
            self.get_logger().debug("Neighbors: %s", self.active_neighbors)
        return ready

    def get_initial_features(self):
        # Compute the initial feature vector for this node (already received over XBee).
        self.value = self.data
        return self.value

    def _apply_graph_payload(self, node_id, neighbors, features):
        # Use provided node_id, fallback to self.node_id
        if node_id is not None:
            self.node_id = node_id
        
        self.get_logger().info("Graph received: id=%s neighbors=%s", self.node_id, neighbors)
        
        self.local_subgraph = nx.Graph()
        self.local_subgraph.add_node(self.node_id)
        for nb in neighbors:
            self.local_subgraph.add_edge(self.node_id, nb)

        if features is None:
            raise ValueError("Missing node features in GRAPH message")

        x_tensor = torch.tensor(features, dtype=torch.float32)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)

        self.data = x_tensor
        self.graph_lock.set()

    def run_message_passing(self, initial_features: torch.Tensor):
        inference_time = 0.0

        mp_start = time.perf_counter()
        node_value = initial_features
        for layer in range(self.decentralized_model.num_layers):
            # Send the current node representation to neighbors.
            self.send_message_passing(layer, node_value)

            # Wait until values are received from all neighbors.
            wait_time_start = time.time()
            key = (self.round_counter, layer)
            while len(self.received_mp[key]) < len(self.active_neighbors):
                if time.time() - wait_time_start > 30:  # 30 seconds timeout
                    raise TimeoutError("Timeout waiting for message passing messages.")
                # self.get_logger().debug("Waiting for MP layer %d: %d/%d received", layer, len(self.received_mp[layer]), len(self.active_neighbors))
                time.sleep(0.1)


            # Update the node's representation using the GNN layer.
            neighbor_values = list(self.received_mp[key].values())
            inference_start = time.perf_counter()
            node_value = self.decentralized_model.update_gnn(layer, node_value, neighbor_values)
            inference_time += time.perf_counter() - inference_start
            del self.received_mp[key]
            
            self.get_logger().debug("MP layer %d complete", layer)

        self.stats["inference_time"].append(inference_time)
        self.stats["message_passing_time"].append(time.perf_counter() - mp_start)
        self.get_logger().info("  Message passing complete (%.2fs)", time.perf_counter() - mp_start)

        return node_value

    def run_pooling(self, node_value: torch.Tensor) -> torch.Tensor:
        pooling_start = time.perf_counter()

        if self.decentralized_model.pooling is None:
            return node_value

        node_value = self.decentralized_model.init_pooling(node_value)
        for iteration in range(self.num_nodes - 1 + self.decentralized_model.pooling.convergence_min):
            # Send the current node representation to neighbors.
            self.send_pooling(iteration, node_value)

            # Wait until values are received from all neighbors.
            wait_time_start = time.time()
            while len(self.received_pooling[iteration]) < len(self.active_neighbors):
                if time.time() - wait_time_start > 30:  # 30 seconds timeout
                    raise TimeoutError("Timeout waiting for pooling messages.")
                time.sleep(0.1)

            # Update the pooled value at current iteration.
            node_value, final_value, error = self.decentralized_model.update_pooling(
                node_value, list(self.received_pooling[iteration].values())
            )
            del self.received_pooling[iteration]

            self.get_logger().debug("Pooling iteration %d complete (error=%.6f)", iteration, error)

        if final_value is None:
            raise RuntimeError("Pooling did not converge.")

        self.stats["pooling_time"].append(time.perf_counter() - pooling_start)
        self.get_logger().info("  Pooling complete (%.2fs)", time.perf_counter() - pooling_start)
        return final_value

    def run_prediction(self, graph_value: torch.Tensor):
        with torch.no_grad():
            graph_value = self.decentralized_model.predictor_model(graph_value)
        return graph_value
    
    def send_message_xbee(self, msg, addr, node_id):
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

    def send_message_passing(self, layer: int, value: torch.Tensor):

        blob, shape = pack_tensor(value)
        msg = {
            "t": "MP",
            "id" : self.node_id,
            "i" : layer,
            "r" : self.round_counter,
            "x" : blob,
            "s" : shape
        }

        data = encode_msg(msg)

        for neighbor in self.active_neighbors:
            time.sleep(random.uniform(0.05, 0.2))

            node_id = neighbor
            addr = self.id_to_addr[neighbor]

            if isinstance(addr, str):
                addr = XBee64BitAddress.from_hex_string(addr)

            ok = False
            time_wait_exc = 0.05
            for attempt in range(1, 5):
                try:
                    self.device.send_data_64_16(addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                    ok = True
                    if attempt == 1:
                        self.get_logger().debug("TX: %s -> node %s", "MP at iteration " + str(layer), node_id)
                    else:
                        self.get_logger().debug("TX: %s -> node %s (retry %d)", "MP at iteration " + str(layer), node_id, attempt)
                    break
                except (TransmitException, TimeoutException) as e:
                    status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                    self.get_logger().warning("TX fail: %s (attempt %d, %s)", "MP", attempt, status)
                    time.sleep(time_wait_exc)
                    time_wait_exc *= 2  # Exponential backoff

            if not ok:
                self.get_logger().error("TX gave up: %s to node %s", "MP", node_id)
            time.sleep(0.05)

    def receive_message_passing(self, msg):        
        r = msg["r"]
        layer = msg["i"]
        sender = msg["id"]

        tensor_data = unpack_tensor(msg["x"], msg["s"])
        self.received_mp[(r, layer)][sender] = tensor_data

        self.get_logger().debug(
            "RX: MP stored from node %s (r=%s, layer=%s) now %d/%d",
            sender, r, layer, len(self.received_mp[(r, layer)]), len(self.active_neighbors)
        )

    def send_pooling(self, iteration: int, value: dict[str, torch.Tensor] | torch.Tensor):
        msg = {
            "t": "pooling",
            "id" : self.node_id,
            "i" : iteration
        }

        if isinstance(value, dict):  # This enables pooling by flooding
            msg["ss"] = list(value.keys()) # sources

            data = torch.stack(list(value.values()), dim=0)
            msg["x"], msg["s"] = pack_tensor(data)
        else:
            msg["x"], msg["s"] = pack_tensor(value)

        data = encode_msg(msg)
        for neighbor in self.active_neighbors:
            node_id = neighbor
            addr = self.id_to_addr[neighbor]

            if isinstance(addr, str):
                addr = XBee64BitAddress.from_hex_string(addr)

            ok = False
            for attempt in range(1, 5):
                try:
                    self.device.send_data_64_16(addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                    ok = True
                    if attempt == 1:
                        self.get_logger().debug("TX: %s -> node %s", "pooling", node_id)
                    else:
                        self.get_logger().debug("TX: %s -> node %s (retry %d)", "pooling", node_id, attempt)
                    break
                except (TransmitException, TimeoutException) as e:
                    status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                    self.get_logger().warning("TX fail: %s (attempt %d, %s)", "pooling", attempt, status)
                    time.sleep(0.1)

            if not ok:
                self.get_logger().error("TX gave up: %s to node %s", "pooling", node_id)
            time.sleep(0.05)

    def receive_pooling(self, msg):
        sources = msg.get("ss", [])
        if len(sources) > 0:
            # Reconstruct dict of tensors from flattened data and shape
            data = torch.tensor(msg["x"]).reshape(tuple(msg["s"]))
            tensor_data = {source: data[i] for i, source in enumerate(sources)}
        else:
            # Reconstruct tensor from flattened data and shape
            tensor_data = torch.tensor(msg["x"]).reshape(tuple(msg["s"]))
        self.received_pooling[msg["i"]][msg["id"]] = tensor_data

    def compute_gnn(self):
        ready = self.get_neighbors()
        if not ready:
            self.get_logger().info("Waiting for neighbors...")
            time.sleep(1.0)
            return
        
        time.sleep(1)

        self.round_counter += 1
        self.get_logger().info("─" * 40)
        self.get_logger().info("ROUND %d", self.round_counter)
        round_start = time.perf_counter()

        initial_features = self.get_initial_features()

        time.sleep(0.1)  # Small delay to ensure all nodes are ready

        try:
            node_value = self.run_message_passing(initial_features)
            graph_value = self.run_pooling(node_value)
            graph_value = self.run_prediction(graph_value)
            graph_value = graph_value.item()
        except TimeoutError as e:
            self.get_logger().warning("Timeout: %s", e)
            return

        elapsed = time.perf_counter() - round_start
        self.stats["round_time"].append(elapsed)
        self.get_logger().info("ROUND %d DONE: value=%.4f (%.2fs)", self.round_counter, graph_value, elapsed)

        if graph_value < 0.5:
            mids = "not "
            color = (50, 50, 50)
        else:
            mids = ""
            color = (50, 0, 0)
        self.get_logger().info(f"The node is \033[7m{mids}in MIDS\033[0m.\n")
        self.led.set_all(color)

    def print_stats(self):
        table = PrettyTable()
        table.add_column("", ["Mean", "Std", "Max", "Min"])
        for key, values in self.stats.items():
            table.add_column(
                key,
                [f"{np.mean(values):.4f}", f"{np.std(values):.4f}", f"{np.max(values):.4f}", f"{np.min(values):.4f}"],
            )
        self.get_logger().info("\n%s", table)

    def stop(self):
        self.device.close()
        self.get_logger().info("Node stopped.")
        # self.print_stats()
        self.led.exit()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    cli_args = parser.parse_args(args=args)

    gnn_node = Node(port=cli_args.port, baud=cli_args.baud)
    gnn_node.start()
    try:
        gnn_node.run()
    except KeyboardInterrupt:
        gnn_node.stop()


if __name__ == "__main__":
    main()
