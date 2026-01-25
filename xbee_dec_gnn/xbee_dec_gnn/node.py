#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, logging, json, threading, socket
from collections import defaultdict
from typing import Dict, Any # dodano
from digi.xbee.devices import ZigBeeDevice # dodano
from digi.xbee.models.address import XBee64BitAddress, XBee16BitAddress # dodano
from digi.xbee.exception import TransmitException # dodano

import networkx as nx
import numpy as np
import torch
from colorlog import ColoredFormatter
from prettytable import PrettyTable

from xbee_dec_gnn.decentralized_gnns.dec_gnn import DecentralizedGNN
from xbee_dec_gnn.utils.led_matrix import LEDMatrix


def load_config(path: str) -> Dict[str, Any]:    # dodano TODO: move to utils.py or something
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class ObjectWithLogger:
    def __init__(self):
        """Return a logger with a default ColoredFormatter."""
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
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

    def get_logger(self):
        return self.logger


class Node(ObjectWithLogger):
    def __init__(self):
        super().__init__()

        # Unique identifier for the node. # DOC: We assume all nodes have the same format of the name.
        # self.node_id = node_id
        self.node_prefix = "node_"
        self.hostname = socket.gethostname()
        self.node_name = self.node_prefix + self.hostname
        self.node_id = None
        self.id_to_addr = None
        self.data = None

        self.get_logger().info("GNN node process started: name=%s hostname=%s", self.node_name, self.hostname)

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
        self.num_nodes = 5
        self.gnn_model_path = default_model_path

        # # TODO: Load the GNN model. # DOC: Change this to customize how the model is loaded.
        dist_model_kwargs = dict(pooling_protocol="consensus", consensus_sigma=1 / self.num_nodes)
        self.decentralized_model = DecentralizedGNN.from_gnn_wrapper(self.gnn_model_path, **dist_model_kwargs)

        # Initialize the LED matrix if available.
        self.led = LEDMatrix()

        self.device = ZigBeeDevice('/dev/ttyUSB0', 9600)  # CHANGE SO ITS NOT HARDCODED
        self.bcast_lock = threading.Event()
        self.init_id_lock = threading.Event()
        self.graph_lock = threading.Event()

    def run(self):
        # Main loop of the node.
        while True:
            self.compute_gnn()

    # def graph_cb(self, msg):
    #     G = nx.from_graph6_bytes(bytes(msg.data.strip(), "ascii"))
    #     lambda2 = nx.laplacian_spectrum(G)[1]
    #     self.get_logger().debug(f"Received graph {msg.data} with algebraic connectivity λ₂: {lambda2:.4f}")

    #     self.local_subgraph: nx.Graph = G.subgraph([self.node_id] + list(G.neighbors(self.node_id)))

    def receive_message_xbee(self, xbee_message):
        try:
            msg = json.loads(xbee_message.data.decode("utf-8"))
        except Exception:
            self.get_logger().exception("RX: failed to decode JSON payload from XBee message")
            return

        if msg.get("type") == "DISCOVERY":
            self.central_addr = msg.get("addr")

            new_msg = {
                "type" : "INIT",
                "hostname" : self.hostname
            }

            self.send_message_xbee(new_msg, msg.get("addr"), "CENTRAL")

            self.bcast_lock.set()

        if msg.get("type") == "REGISTER_ACK":
            self.node_id = msg.get("id")
            self.id_to_addr = msg.get("id_to_addr")

            new_msg = {
                "type" : "ID_CONFIRM",
                "id" : self.node_id
            }

            self.send_message_xbee(new_msg, self.central_addr, "CENTRAL")

            self.node_name = self.node_prefix + str(self.node_id)

            self.init_id_lock.set()

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
                self._apply_graph_payload(msg.get("n", []), msg.get("x"))
                return

        if msg.get("t") == "G":  # short type for compact payloads
            self._apply_graph_payload(msg.get("n", []), msg.get("x"))
            return


        if msg.get("type") in ("MP", "mp"):
            self.receive_message_passing(msg)
        if msg.get("type") == "pooling":
            self.receive_pooling(msg)

    def start(self):
        self.device.open()
        self.device.add_data_received_callback(self.receive_message_xbee)

        self.get_logger().info(f"[{self.node_name}] Port: {'/dev/ttyUSB0'} @ {9600}")
        self.get_logger().info(f"[{self.node_name}] Adresa: {self.device.get_64bit_addr()}")

        self.get_logger().info("Node initialized; waiting for discovery broadcast (DISCOVERY) from central")

        self.bcast_lock.wait()
        self.get_logger().info("Handshake: received DISCOVERY; sent INIT; waiting for REGISTER_ACK (id assignment)")

        self.init_id_lock.wait()
        self.get_logger().info("Handshake: received REGISTER_ACK; sent ID_CONFIRM; waiting for GRAPH payload")

        self.graph_lock.wait()
        self.get_logger().info("Setup: graph/features received; entering GNN compute loop")

    def get_neighbors(self):
        self.active_neighbors = []

        # We know the whole graph in development mode.
        if self.local_subgraph.number_of_nodes() > 0:
            self.active_neighbors = [f"{self.node_prefix}{i}" for i in self.local_subgraph.neighbors(self.node_id)]

        if not nx.is_connected(self.local_subgraph):
            self.get_logger().fatal(
                "Local subgraph is not connected! This should never happen. If it does, it is probably due to a "
                "mismatch of communication radius used in this node and for the creation of the discovery messages."
            )
            raise RuntimeError("Local subgraph is not connected.")
        ready = len(self.active_neighbors) > 0

        if ready:
            self.get_logger().info(f"Active neighbors: {self.active_neighbors}")
        return ready

    def get_initial_features(self):
        # Compute the initial feature vector for this node (already received over XBee).
        self.value = self.data
        self.get_logger().debug(f"Initial feature vector for node: {self.value}")
        self.get_logger().debug(f"Local subgraph edges: {list(self.local_subgraph.edges())}")
        return self.value

    def _apply_graph_payload(self, neighbors, features):
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
            while len(self.received_mp[layer]) < len(self.active_neighbors):
                if time.time() - wait_time_start > 2.0:  # 2 seconds timeout
                    raise TimeoutError("Timeout waiting for message passing messages.")
                time.sleep(0.1)

            # Update the node's representation using the GNN layer.
            neighbor_values = list(self.received_mp[layer].values())
            inference_start = time.perf_counter()
            node_value = self.decentralized_model.update_gnn(layer, node_value, neighbor_values)
            inference_time += time.perf_counter() - inference_start
            del self.received_mp[layer]

        self.stats["inference_time"].append(inference_time)
        self.stats["message_passing_time"].append(time.perf_counter() - mp_start)
        self.get_logger().debug("GNN: message passing done (mean=%.6f)", float(node_value.mean()))

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
                if time.time() - wait_time_start > 2.0:  # 2 seconds timeout
                    raise TimeoutError("Timeout waiting for pooling messages.")
                time.sleep(0.1)

            # Update the pooled value at current iteration.
            node_value, final_value, error = self.decentralized_model.update_pooling(
                node_value, list(self.received_pooling[iteration].values())
            )
            del self.received_pooling[iteration]

        if final_value is None:
            raise RuntimeError("Pooling did not converge.")

        self.stats["pooling_time"].append(time.perf_counter() - pooling_start)
        self.get_logger().debug("GNN: pooling done (mean=%.6f)", float(final_value.mean()))
        return final_value

    def run_prediction(self, graph_value: torch.Tensor):
        with torch.no_grad():
            graph_value = self.decentralized_model.predictor_model(graph_value)
        return graph_value
    
    def send_message_xbee(self, msg, addr, node_id):
        data = json.dumps(msg).encode("utf-8")
        ok = False

        if isinstance(addr, str):
            addr = XBee64BitAddress.from_hex_string(addr)

        for attempt in range(1, 5): # TODO: make retries variable
            try:
                self.device.send_data_64_16(addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                ok = True
                self.get_logger().debug("TX: %s -> %s (mac=%s attempt=%d)", msg.get("type"), node_id, addr, attempt)
                break
            except TransmitException as e:
                status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                self.get_logger().warning("TX: fail -> %s (attempt=%d status=%s)", node_id, attempt, status)
                time.sleep(0.1)

        if not ok:
            self.get_logger().error("TX: giving up delivering %s to %s (addr=%s)", msg.get("type"), node_id, addr)

        time.sleep(0.1)

    def send_message_passing(self, layer: int, value: torch.Tensor):
        # msg = GNNmessage()
        # msg.sender = self.node_name
        # msg.iteration = layer
        # msg.data = value.flatten().tolist()  # Flatten tensor to 1D list
        # msg.shape = list(value.shape)  # Store original shape
        # for neighbor in self.active_neighbors:
        #     self.mp_pubs[neighbor].publish(msg)
        #     self.get_logger().debug(f"Sent message to {neighbor} at layer {layer}")
        # TODO: Adapt for Xbee

        msg = {
            "type": "MP",
            "sender" : self.node_name,
            "iter" : layer,
            "data" : value.flatten().tolist(),
            "shape" : list(value.shape)
        }

        for neighbor in self.active_neighbors:
            self.send_message_xbee(msg, self.id_to_addr[neighbor], neighbor)

    def receive_message_passing(self, msg):
        # Reconstruct tensor from flattened data and shape
        # tensor_data = torch.tensor(msg.data).reshape(tuple(msg.shape))
        # self.received_mp[msg.iteration][msg.sender] = tensor_data
        # TODO: Adapt for Xbee

        tensor_data = torch.tensor(msg.get("data")).reshape(tuple(msg.get("shape")))
        self.received_mp[msg.get("iter")][msg.get("sender")] = tensor_data

    def send_pooling(self, iteration: int, value: dict[str, torch.Tensor] | torch.Tensor):
        # msg = GNNmessage()
        # msg.sender = self.node_name
        # msg.iteration = iteration

        # if isinstance(value, dict):  # This enables pooling by flooding
        #     msg.sources = list(value.keys())
        #     data = torch.stack(list(value.values()), dim=0)
        #     msg.data = data.flatten().tolist()  # Flatten tensor to 1D list
        #     msg.shape = list(data.shape)  # Store original shape
        # else:
        #     msg.data = value.flatten().tolist()  # Flatten tensor to 1D list
        #     msg.shape = list(value.shape)  # Store original shape

        # for neighbor in self.active_neighbors:
        #     self.pooling_pubs[neighbor].publish(msg)
        #     self.get_logger().debug(f"Sent pooling message to {neighbor} at iteration {iteration}")
        # TODO: Adapt for Xbee


        msg = {
            "type": "pooling",
            "sender" : self.node_name,
            "i" : iteration
        }

        if isinstance(value, dict):  # This enables pooling by flooding
            msg["sources"] = list(value.keys)

            data = torch.stack(list(value.values()), dim=0)
            msg["data"] = data.flatten().tolist()  # Flatten tensor to 1D list
            msg["shape"] = list(data.shape)  # Store original shape
        else:
            msg["data"] = value.flatten().tolist()  # Flatten tensor to 1D list
            msg["shape"] = list(value.shape)  # Store original shape

        for neighbor in self.active_neighbors:
            self.send_message_xbee(msg, self.id_to_addr[neighbor], neighbor)

        

    def receive_pooling(self, msg):
        # self.get_logger().debug(f"Received pooling message from {msg.sender} at iteration {msg.iteration}")
        # if len(msg.sources) > 0:
        #     # Reconstruct dict of tensors from flattened data and shape
        #     data = torch.tensor(msg.data).reshape(tuple(msg.shape))
        #     tensor_data = {source: data[i] for i, source in enumerate(msg.sources)}
        # else:
        #     # Reconstruct tensor from flattened data and shape
        #     tensor_data = torch.tensor(msg.data).reshape(tuple(msg.shape))
        # self.received_pooling[msg.iteration][msg.sender] = tensor_data
        # TODO: Adapt for Xbee

        self.get_logger().debug(f"Received pooling message from {msg['sender']} at iteration {msg['i']}")
        if len(msg.sources) > 0:
            # Reconstruct dict of tensors from flattened data and shape
            data = torch.tensor(msg["data"]).reshape(tuple(msg["shape"]))
            tensor_data = {source: data[i] for i, source in enumerate(msg["sources"])}
        else:
            # Reconstruct tensor from flattened data and shape
            tensor_data = torch.tensor(msg["data"]).reshape(tuple(msg["shape"]))
        self.received_pooling[msg["i"]][msg["sender"]] = tensor_data

    def compute_gnn(self):
        ready = self.get_neighbors()
        if not ready:
            self.get_logger().info("Waiting for neighbors...")
            time.sleep(1.0)
            return

        self.round_counter += 1
        self.get_logger().info("Round %d started", self.round_counter)
        round_start = time.perf_counter()

        initial_features = self.get_initial_features()


        try:
            node_value = self.run_message_passing(initial_features)
            graph_value = self.run_pooling(node_value)
            graph_value = self.run_prediction(graph_value)
            graph_value = graph_value.item()
        except TimeoutError as e:
            self.get_logger().warning("Timeout: %s (canceling round %d)", e, self.round_counter)
            return

        self.stats["round_time"].append(time.perf_counter() - round_start)
        self.get_logger().info("Round %d complete: graph_value=%.3f", self.round_counter, graph_value)

        # TODO: Adatpt for Xbee
        # led_color = LEDMatrix.from_colormap(graph_value / self.num_nodes, color_space="hsv", cmap_name="jet")
        # led_color = (led_color[0], led_color[1], led_color[2] * 0.2)  # Full brightness
        # self.led.set_all(led_color, color_space="hsv")

    def print_stats(self):
        table = PrettyTable()
        table.add_column("", ["Mean", "Std", "Max", "Min"])
        for key, values in self.stats.items():
            table.add_column(
                key,
                [f"{np.mean(values):.4f}", f"{np.std(values):.4f}", f"{np.max(values):.4f}", f"{np.min(values):.4f}"],
            )
        self.get_logger().info("\n%s", table)


def main(args=None):

    # TODO: Adapt for Xbee
    gnn_node = Node()
    
    gnn_node.start()
    gnn_node.run()


if __name__ == "__main__":
    main()
