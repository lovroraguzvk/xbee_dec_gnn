import argparse
import json
import time
import sys
from typing import Dict, Any
from dataset import SignalGraphDataset
from util import visualize_graph
import matplotlib.pyplot as plt

# from digi.xbee.devices import DigiMeshDevice  # ORIGINAL (DigiMesh)
from digi.xbee.devices import ZigBeeDevice  # >>> CHANGED: use ZigBee firmware/device class
from digi.xbee.models.address import XBee64BitAddress, XBee16BitAddress  # >>> CHANGED
from digi.xbee.exception import TransmitException


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--retry_delay", type=float, default=0.4)
    args = ap.parse_args()

    cfg = load_config(args.config)

    id_to_addr = cfg["id_to_addr"]
    nodes_cfg = cfg.get("nodes", None)

    dataset = SignalGraphDataset()
    G = dataset.getGraph()
    nodes_cfg = G["nodes_letters"]
    visualize_graph(G)

    # device = DigiMeshDevice(args.port, args.baud)
    device = ZigBeeDevice(args.port, args.baud)
    acks = set()

    def on_rx(xbee_message):
        try:
            msg = json.loads(xbee_message.data.decode("utf-8"))
        except Exception:
            return
        if isinstance(msg, dict) and msg.get("type") == "ACK_INIT":
            nid = msg.get("id")
            if nid:
                acks.add(str(nid))

    for i in range(5):
        try:
            device.open()
            break
        except:
            print("Device couldn't open, trying again...")
            time.sleep(0.5)


    device.add_data_received_callback(on_rx)

    print(f"[CENTRAL] Port: {args.port} @ {args.baud}")
    print(f"[CENTRAL] Addr: {device.get_64bit_addr()}")

    t = 15
    print(f"Waiting {t} seconds for others to start...")
    time.sleep(t)

    # >>> CHANGED: show max RF payload (NP) as reported by firmware.
    try:
        np_bytes = device.get_parameter("NP")
        np_val = int.from_bytes(np_bytes, byteorder="big") if np_bytes is not None else None
        print(f"[CENTRAL] NP (max RF payload bytes) = {np_val}")
    except Exception as e:
        print(f"[CENTRAL] NP read failed: {e}")
    print(f"[CENTRAL] Sending INIT to: {sorted(nodes_cfg.keys())}")

    # print(nodes_cfg)

    for node_id, node_info in nodes_cfg.items():
        if node_id not in id_to_addr:
            print(f"[CENTRAL] WARN: node '{node_id}' missing from id_to_addr, skipping")
            continue

        print(node_id)
        neighbors = node_info.get("neighbours")
        value0 = node_info.get("value")

        init_msg = {
            "t": True,
            "n": list(neighbors),
            "v": value0
        }
        data = json.dumps(init_msg).encode("utf-8")
        print(f"[CENTRAL] INIT payload_len={len(data)} bytes -> {node_id}")
        addr = XBee64BitAddress.from_hex_string(id_to_addr[node_id])

        ok = False
        for attempt in range(1, args.retries + 1):
            try:
                # device.send_data_64(addr, data)
                device.send_data_64_16(addr, XBee16BitAddress.UNKNOWN_ADDRESS, data)
                ok = True
                print(f"[CENTRAL] INIT -> {node_id}, MAC -> {addr} (attempt {attempt}) neighbours={neighbors} value0={value0}")
                break
            except TransmitException as e:
                status = getattr(e, "transmit_status", None) or getattr(e, "status", None)
                print(f"[CENTRAL] TX FAIL -> {node_id} attempt={attempt} status={status}")
                time.sleep(args.retry_delay)

        if not ok:
            print(f"[CENTRAL] ERROR: Could not deliver INIT to {node_id}")

        time.sleep(0.1)

    device.close()
    plt.show()

def test():
    dataset = SignalGraphDataset()
    G = dataset.getGraph()
    nodes_cfg = G["nodes_letters"]

    init_msg = {
        "n": ["D","B", "E","C"],
        "v": float(0.854857),
    }
    data = json.dumps(init_msg).encode("utf-8")
    # print(f"[CENTRAL] INIT payload_len={len(data)} bytes -> {node_id}")

    print("SIZEOF INIT PAYLOAD: ", sys.getsizeof(data), len(data))

if __name__ == "__main__":
    main()

