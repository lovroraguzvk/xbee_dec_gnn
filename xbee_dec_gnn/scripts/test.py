import torch
src="/root/other_ws/xbee_dec_gnn/xbee_dec_gnn/xbee_dec_gnn/data/MIDS_model.pth"
d=torch.load(src, map_location="cpu")
print("TOP-LEVEL KEYS:", list(d.keys()))
for k,v in d.items():
    if isinstance(v, dict):
        print(f"{k}: dict keys ->", list(v.keys())[:40])
