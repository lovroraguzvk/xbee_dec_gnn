#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
import io
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import rclpy
# import rclpy.qos
import torch_geometric.utils as tg_utils
# from rclpy.node import Node
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from torch_geometric.data import InMemoryDataset
# from sensor_msgs.msg import Image

# from ros2_dec_gnn_msgs.msg import GraphData
from my_graphs_dataset import GraphDataset

import json 
import time
import threading
from typing import Dict, Any 
from digi.xbee.devices import ZigBeeDevice 
from digi.xbee.models.address import XBee64BitAddress, XBee16BitAddress
from digi.xbee.exception import TransmitException 

BCAST_64 = XBee64BitAddress.from_hex_string("000000000000FFFF")
BCAST_16 = XBee16BitAddress.from_hex_string("FFFE")

# msg_qos = rclpy.qos.QoSProfile(
#     history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
#     depth=10,
#     reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
#     durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
# )

def load_config(path: str) -> Dict[str, Any]:    # dodano TODO: move to utils.py or something
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class GraphGenerator():
    def __init__(self, graph_mode="load", gui_mode=False, port="/dev/ttyUSB0", baud_rate=9600, config="config.json"):
        self.num_nodes = 5
        self.graph_mode = graph_mode
        self.gui_mode = gui_mode

        self.G = nx.cycle_graph(self.num_nodes)
        self.node_positions = {}

        # Load graph dataset.
        self.dataset = InMemoryDataset()
        self.dataset.load(str(pathlib.Path(__file__).parents[1] / "config" / "data" / "MIDS_data.pt"))
        dataset_range = {}
        curr_index = 0
        for size, num_graphs in zip(range(3, 9), [2, 6, 21, 112, 853, 11117]):
            dataset_range[size] = (curr_index, curr_index + num_graphs - 1)
            curr_index += num_graphs
        self.dataset_range = dataset_range[self.num_nodes]
        self.current_graph_index = 0

        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Dynamic Graph Visualization")
        self.ax.format_coord = lambda x, y: ''

        # Create button for GUI mode
        if self.gui_mode:
            # Add button - will be repositioned after each graph update
            self.button_ax = self.fig.add_axes([0.02, 0.02, 0.12, 0.05])
            self.next_button = Button(self.button_ax, 'Next')
            self.next_button.on_clicked(self.on_next_button_clicked)

        self.prev_image = None
        self.ax_range = None

        # Set up a callback group for the publishers
        # self.blocked_group = MutuallyExclusiveCallbackGroup()
        # self.continuous_group = MutuallyExclusiveCallbackGroup()

        # self.graph_pub = self.create_publisher(GraphData, "/graph_topic", msg_qos)
        # self.image_pub = self.create_publisher(Image, "/graph_visualization", 1)
        # self.image_pub_prev = self.create_publisher(Image, "/graph_visualization_prev", 1)

        #TODO: xbee: init central node device, set callback and fetch addresses

        cfg = load_config(config)
        self.hostnames_to_id = cfg["hostnames_to_id"]

        self.id_to_addr = dict.fromkeys(list(self.hostnames_to_id.values()))
        self._received_responses = set()
        self.final_init_event = threading.Event()
        self.final_ack_event = threading.Event()
        self.wait_forever = True
        self.init_timeout_s = 30.0


        self.device = ZigBeeDevice(port, baud_rate)

        self.start()

        # In GUI mode, generate first graph immediately and wait for button clicks
        # In non-GUI mode, use timer with input() blocking
        if self.gui_mode:
            self.process_next_graph()
        else:
            # self.create_timer(0.5, self.graph_timer_cb, callback_group=self.blocked_group)
            print("Non-gui not implemented yet") # TODO: implement

    def start(self):
        self.device.open()
        self.device.add_data_received_callback(self.receive_message)

        print(f"[{self.node_id}] Port: {self.port} @ {self.baud}")
        print(f"[{self.node_id}] Adresa: {self.device.get_64bit_addr()}")
        print(f"[{self.node_id}] Waiting for INIT from others (broadcasting MAC)")

        my_addr = str(self.device.get_64bit_addr())

        msg = {
            "type": "BCAST",
            "addr": my_addr,
        }
        
        data = json.dumps(msg).encode("utf-8")

        interval_s = 1.0
        start_time = time.time()

        while not self.final_init_event.is_set():
            if not self.wait_forever:
                elapsed = time.time() - start_time
                if elapsed >= self.init_timeout_s:
                    raise TimeoutError(
                        f"[CENTRAL] Only received {len(self.final_acks)}/{self.num_nodes} INITs"
                    )

            # --- broadcast INIT ---
            try:
                self.device.send_data_64_16(BCAST_64, BCAST_16, data)
                print(f"[CENTRAL] Broadcast BCAST (mac={my_addr})")
            except TransmitException as e:
                status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                print(f"[CENTRAL] Broadcast failed: status={status}")

            # --- wait but wake early if ACKs complete ---
            self.final_init_event.wait(timeout=interval_s)

        print("[CENTRAL] All INITs received, waiting for ACK_IDs")

        for node_id, addr in self.id_to_addr:
            msg = {
                    "type": "ACK_INIT",
                    "id" : node_id,
                    "id_to_addr" : self.id_to_addr
                }

            self.send_message_xbee(msg, addr, node_id)

        self.final_ack_event.wait(timeout=interval_s)

        print("[CENTRAL] All ACK_IDs received")



        
    def send_message_xbee(self, msg, addr, node_id):
        data = json.dumps(msg).encode("utf-8")
        ok = False
        for attempt in range(1, 5): # TODO: make retries variable
            try:
                self.device.send_data_64_16(addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                ok = True
                print(f"[CENTRAL] {msg.get("type")} -> {node_id}, MAC -> {addr} (attempt {attempt})")
                break
            except TransmitException as e:
                status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                print(f"[CENTRAL] TX FAIL -> {node_id} attempt={attempt} status={status}")
                time.sleep(0.1)

        if not ok:
            print(f"[CENTRAL] ERROR: Could not deliver ACK_INIT to {node_id}")

        time.sleep(0.1)

    def receive_message(self, xbee_message):
        try:
            msg = json.loads(xbee_message.data.decode("utf-8"))
        except Exception:
            return
        
        if msg.get("type") == "INIT":
            hostname = msg.get("hostname")
            sender_64 = xbee_message.remote_device.get_64bit_addr()
            node_id = self.hostnames_to_id[hostname]

            self.id_to_addr[node_id] = sender_64

            print(f'[CENTRAL] Received INIT from {node_id} ({sender_64}), sending ACK_INIT')

            if len(self._received_responses) >= self.num_nodes:
                self._received_responses = set()
                self.final_init_event.set()

        if msg.get("type") == "ACK_ID":
            self.final_acks.add(msg.get("id"))

            print(f'[CENTRAL] Received ACK_ID from {msg.get("id")}')

            if len(self.final_acks) >= self.num_nodes:
                self.final_ack_event.set()
            

    def send_node_info(self):

        # TODO: finish this and make it so each only gets its own neighbourhood

        for id, addr in self.id_to_addr:
            msg = {
                "type" : "GRAPH",
                "graph6_str" :  GraphDataset.to_graph6(self.G),
                "data" : self.data.x.flatten().tolist(),
                "shape" : list(self.data.x.shape)
            }

            self.send_message_xbee(msg, addr, id)

    def process_next_graph(self):
        """Generate/load and publish the next graph."""
        if self.graph_mode == "generate":
            self.generate_graph()
        elif self.graph_mode == "load":
            self.load_next_graph()
        else:
            self.get_logger().error(f"Unknown graph mode: {self.graph_mode}")
            return

        self.publish_graph_image() # changed for xbee

        # Publish the graph in graph6 format
        # graph_data = GraphData()
        # graph_data.graph6_str = GraphDataset.to_graph6(self.G)
        # graph_data.data = self.data.x.flatten().tolist()  # Flatten tensor to 1D list
        # graph_data.shape = list(self.data.x.shape)  # Store original shape
        # self.graph_pub.publish(graph_data)

        # TODO: xbee: send new graph to nodes
        
        self.send_node_info()

        # Update GUI display if in GUI mode
        if self.gui_mode:
            # Reposition button to fixed pixel location after figure resize
            self.reposition_button()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def graph_timer_cb(self):
        """Timer callback for non-GUI mode."""
        self.process_next_graph()
        input("Ready for next graph?")

    def on_next_button_clicked(self, event):
        """Callback when Next button is clicked in GUI mode."""
        self.process_next_graph()

    def reposition_button(self):
        """Reposition button to fixed pixel location after figure resize."""
        # Get current figure size in pixels
        fig_width_px = self.fig.get_figwidth() * self.fig.dpi
        fig_height_px = self.fig.get_figheight() * self.fig.dpi

        # Button dimensions in pixels
        button_width_px = 100
        button_height_px = 40
        button_x_px = 10  # 10 pixels from left
        button_y_px = 10  # 10 pixels from bottom

        # Convert to figure coordinates (0-1)
        button_left = button_x_px / fig_width_px
        button_bottom = button_y_px / fig_height_px
        button_width = button_width_px / fig_width_px
        button_height = button_height_px / fig_height_px

        # Update button position
        self.button_ax.set_position([button_left, button_bottom, button_width, button_height])

    def generate_graph(self):
        graph_ok = False
        while not graph_ok:
            if random.random() < nx.density(self.G):
                # When number of possible edges is large, we are more likely to
                # enter this branch and we should remove an edge.
                edge_to_remove = random.choice(list(self.G.edges))
                self.G.remove_edge(*edge_to_remove)
                graph_ok = nx.is_connected(self.G)
                if not graph_ok:
                    self.G.add_edge(*edge_to_remove)

            else:
                # When number of possible edges is small, we are more likely to
                # enter this branch and we should add an edge.
                non_edges = list(nx.non_edges(self.G))
                if non_edges:  # Check if there are any non-edges to add
                    self.G.add_edge(*random.choice(non_edges))
                graph_ok = True

    def load_next_graph(self):
        """Load the next graph from the dataset."""
        self.current_graph_index = random.randint(self.dataset_range[0], self.dataset_range[1])
        self.data = self.dataset[self.current_graph_index]

        # Create graph based on positions and communication radius
        self.G = tg_utils.to_networkx(self.data, to_undirected=True)

def publish_graph_image(self):
    """Draw/update the graph in the matplotlib window (no ROS publishing)."""

    # Clear the previous plot and draw the updated graph
    self.ax.clear()
    self.ax.set_title(f"Graph index: {self.current_graph_index}")
    self.ax.format_coord = lambda x, y: ''

    cmap = LinearSegmentedColormap.from_list('simple', ['white', 'red'])

    # Use loaded positions if in load mode, otherwise use circular layout
    if self.graph_mode == "load" and self.node_positions:
        # Convert string keys to integers and use loaded positions
        pos = {int(node_id): position for node_id, position in self.node_positions.items()}
    else:
        # Use a fixed layout for consistent positioning
        pos = nx.circular_layout(self.G)

    offset_x = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values()) + 0.5

    # NOTE: i is used later for figure size, so keep it initialized.
    i = 0
    for i in range(self.data.y.shape[1]):
        if self.data.y[0][i] == -1:
            break

        new_pos = {k: v.copy() for k, v in pos.items()}
        for node in new_pos:
            new_pos[node][0] += offset_x

        nx.draw(
            self.G,
            pos=new_pos,
            ax=self.ax,
            with_labels=True,
            edgecolors="black",
            node_size=500,
            font_size=16,
            font_weight="bold",
            node_color=self.data.y[:, i].cpu().numpy(),
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

    # Resize figure (optional; you had this before)
    self.fig.set_size_inches(max(2.5 * max(i, 1), 6), 4)

    # Keep the "Next" button placed correctly
    if self.gui_mode:
        self.reposition_button()
        self.button_ax.set_visible(True)

    # >>> CHANGED: Just draw/flush; no ROS, no buffer, no image conversions
    self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()
    plt.draw()


def main(args):
    # Configure matplotlib backend before any figures are created
    if args.gui:
        # Use TkAgg backend for GUI mode (or Qt5Agg if available)
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                print("Warning: No interactive backend available. Install python3-tk or pyqt5.")
                args.gui = False
                matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')

    # rclpy.init()

    graph_generator = GraphGenerator(
        graph_mode=args.mode, 
        gui_mode=args.gui, 
        baud_rate=args.baud, 
        port=args.port,
        config=args.config
    )

    try:
        if args.gui:
            # Interactive updates during runtime (your publish_graph_image uses draw/flush)
            plt.ion()
            plt.show(block=False)

            # Keep the GUI + RX callbacks alive until window closed or Ctrl+C
            while plt.fignum_exists(graph_generator.fig.number):
                plt.pause(0.05)  # lets matplotlib process events

        else:
            # Non-GUI: do whatever your non-gui loop will be (placeholder)
            # You can still periodically call process_next_graph() here if desired.
            while True:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C).")

    finally:
        # --- Zigbee shutdown ---
        try:
            if hasattr(graph_generator, "device") and graph_generator.device is not None:
                if graph_generator.device.is_open():
                    graph_generator.device.close()
                    print("[CENTRAL] XBee device closed.")
        except Exception as e:
            print(f"[CENTRAL] Warning: failed to close XBee device: {e}")

        # --- Plot handling ---
        # IMPORTANT: do NOT plt.close("all") if you want to show the final figure.
        if args.gui:
            plt.ioff()
            plt.show()   # final blocking show (keeps window open)
        else:
            plt.close("all")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--mode",
        type=str,
        default="load",
        choices=["load", "generate"],
        help='Mode for graph generation: "load" to load from file, "generate" to create random graphs.',
    )
    args.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="If set, display graphs in an interactive GUI window with a Next button.",
    )
    args.add_argument(
        "--continuous", action="store_true", default=False, help="If set, the graph will change continuously over time."
    )
    args.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps to interpolate between graphs when in load mode and continuous is set.",
    )
    args.add_argument("--port", default="/dev/ttyUSB0")
    args.add_argument("--baud", type=int, default=9600)
    args.add_argument("--config", default="config.json")
    parsed_args = args.parse_args()
    main(parsed_args)