
import json
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:    # dodano TODO: move to utils.py or something
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def main():
    cfg = load_config("config.json")
    hostnames_to_id = cfg["hostnames_to_id"]

    print(dict.fromkeys(list(hostnames_to_id.values())))


if __name__=="__main__":
    main()