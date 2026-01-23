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

msg_qos = rclpy.qos.QoSProfile(
    history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class GraphGenerator(Node):
    def __init__(self, graph_mode="load", gui_mode=False):
        super().__init__("graph_generator")

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
        self.blocked_group = MutuallyExclusiveCallbackGroup()
        self.continuous_group = MutuallyExclusiveCallbackGroup()

        self.graph_pub = self.create_publisher(GraphData, "/graph_topic", msg_qos)
        self.image_pub = self.create_publisher(Image, "/graph_visualization", 1)
        self.image_pub_prev = self.create_publisher(Image, "/graph_visualization_prev", 1)

        # In GUI mode, generate first graph immediately and wait for button clicks
        # In non-GUI mode, use timer with input() blocking
        if self.gui_mode:
            self.process_next_graph()
        else:
            self.create_timer(0.5, self.graph_timer_cb, callback_group=self.blocked_group)


    def process_next_graph(self):
        """Generate/load and publish the next graph."""
        if self.graph_mode == "generate":
            self.generate_graph()
        elif self.graph_mode == "load":
            self.load_next_graph()
        else:
            self.get_logger().error(f"Unknown graph mode: {self.graph_mode}")
            return

        # Convert matplotlib figure to image and publish
        self.publish_graph_image()

        # Publish the graph in graph6 format
        graph_data = GraphData()
        graph_data.graph6_str = GraphDataset.to_graph6(self.G)
        graph_data.data = self.data.x.flatten().tolist()  # Flatten tensor to 1D list
        graph_data.shape = list(self.data.x.shape)  # Store original shape
        self.graph_pub.publish(graph_data)

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
        """Convert matplotlib figure to ROS Image message and publish it."""
        # Clear the previous plot and draw the updated graph
        self.ax.clear()
        self.ax.set_title(f"Graph index: {self.current_graph_index}")
        cmap = LinearSegmentedColormap.from_list('simple', ['white', 'red'])

        # Use loaded positions if in load mode, otherwise use circular layout
        if self.graph_mode == "load" and self.node_positions:
            # Convert string keys to integers and use loaded positions
            pos = {int(node_id): position for node_id, position in self.node_positions.items()}
        else:
            # Use a fixed layout for consistent positioning
            pos = nx.circular_layout(self.G)
        offset_x = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values()) + 0.5

        for i in range(self.data.y.shape[1]):
            if self.data.y[0][i] == -1:
                break

            new_pos = pos.copy()
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

        self.fig.set_size_inches(2.5 * i, 2.5)

        # Hide button before rendering image for publishing
        if self.gui_mode:
            self.button_ax.set_visible(False)

        # Render the figure to a buffer
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)

        # Show button again for GUI
        if self.gui_mode:
            self.button_ax.set_visible(True)

        # Convert buffer to numpy array
        img_array = plt.imread(buf, format="png")
        buf.close()

        # Convert RGBA to RGB if necessary (remove alpha channel)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Convert to uint8 format (0-255 range)
        img_array = (img_array * 255).astype(np.uint8)

        # Create ROS Image message manually
        ros_image = Image()
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = "graph_visualization"
        ros_image.height = img_array.shape[0]
        ros_image.width = img_array.shape[1]
        ros_image.encoding = "rgb8"
        ros_image.is_bigendian = False
        ros_image.step = 3 * ros_image.width  # 3 bytes per pixel for RGB
        ros_image.data = img_array.flatten().tobytes()

        self.image_pub.publish(ros_image)

        if self.prev_image is not None:
            self.image_pub_prev.publish(self.prev_image)
        self.prev_image = ros_image


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

    rclpy.init()

    graph_generator = GraphGenerator(graph_mode=args.mode, gui_mode=args.gui)

    try:
        if args.gui:
            # Use SingleThreadedExecutor for GUI mode to avoid threading issues
            executor = SingleThreadedExecutor()
            executor.add_node(graph_generator)
            # Start matplotlib event loop in non-blocking mode
            plt.ion()
            plt.show(block=False)
            # Integrate ROS spinning with matplotlib event loop
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
                plt.pause(0.01)  # Process matplotlib events
        else:
            executor = MultiThreadedExecutor()
            executor.add_node(graph_generator)
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        graph_generator.destroy_node()
        rclpy.shutdown()
        plt.close("all")  # Close all matplotlib figures


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
    parsed_args = args.parse_args()
    main(parsed_args)