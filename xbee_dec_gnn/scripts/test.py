import torch, copy
src="/root/other_ws/xbee_dec_gnn/xbee_dec_gnn/xbee_dec_gnn/data/MIDS_model.pth"
dst="/root/other_ws/xbee_dec_gnn/xbee_dec_gnn/xbee_dec_gnn/data/MIDS_model_fixed.pth"
d=torch.load(src, map_location="cpu")
cfg=copy.deepcopy(d["config"])
if "gnn_layers" not in cfg and "num_layers" in cfg:
    cfg["gnn_layers"]=cfg.pop("num_layers")
d["config"]=cfg
torch.save(d,dst)
print("Wrote",dst,"config keys:",sorted(cfg.keys()))