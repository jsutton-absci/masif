"""
masif_site_train.py: Entry point to train MaSIF-site.
"""

import os
import sys
import importlib
import torch
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_site import MaSIF_site
from masif_modules.train_masif_site import train_masif_site

params = masif_opts["site"]

if len(sys.argv) > 1:
    custom_params = importlib.import_module(sys.argv[1]).custom_params
    for key, val in custom_params.items():
        print(f"Setting {key} to {val}")
        params[key] = val

if "pids" not in params:
    params["pids"] = ["p1", "p2"]

n_thetas    = params.get("n_theta",     4)
n_rhos      = params.get("n_rho",       3)
n_rotations = params.get("n_rotations", 4)

model = MaSIF_site(
    params["max_distance"],
    n_thetas=n_thetas,
    n_rhos=n_rhos,
    n_rotations=n_rotations,
    feat_mask=params["feat_mask"],
    n_conv_layers=params["n_conv_layers"],
)

model_path = os.path.join(params["model_dir"], "model.pt")
if os.path.exists(params["model_dir"]) and os.path.exists(model_path):
    print("Reading pre-trained network from", model_path)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
else:
    os.makedirs(params["model_dir"], exist_ok=True)

train_masif_site(model, params)
