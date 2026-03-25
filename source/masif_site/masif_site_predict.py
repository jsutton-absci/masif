"""
masif_site_predict.py: Evaluate one or multiple proteins with MaSIF-site.
"""

import os
import sys
import importlib
import numpy as np
import time
import torch
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_site import MaSIF_site
from masif_modules.train_masif_site import run_masif_site, mask_input_feat, pad_indices

params = masif_opts["site"]
custom_params = importlib.import_module(sys.argv[1]).custom_params
for key, val in custom_params.items():
    print(f"Setting {key} to {val}")
    params[key] = val

parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

if len(sys.argv) == 3:
    ppi_pair_ids = [sys.argv[2]]
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    eval_list = [line.rstrip() for line in open(sys.argv[3])]
    ppi_pair_ids = os.listdir(parent_in_dir)
else:
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaSIF_site(
    params["max_distance"],
    n_thetas=params.get("n_theta", 4),
    n_rhos=params.get("n_rho", 3),
    n_rotations=params.get("n_rotations", 4),
    feat_mask=params["feat_mask"],
    n_conv_layers=params["n_conv_layers"],
)

model_path = os.path.join(params["model_dir"], "model.pt")
print("Restoring model from:", model_path)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

os.makedirs(params["out_pred_dir"], exist_ok=True)

for ppi_pair_id in ppi_pair_ids:
    print(ppi_pair_id)
    in_dir = os.path.join(parent_in_dir, ppi_pair_id, "")
    fields = ppi_pair_id.split("_")
    if len(fields) < 2:
        continue

    pdbid  = fields[0]
    chain1 = fields[1]
    pids, chains = ["p1"], [chain1]
    if len(fields) == 3 and fields[2]:
        pids.append("p2")
        chains.append(fields[2])

    for ix, pid in enumerate(pids):
        pdb_chain_id = pdbid + "_" + chains[ix]
        if eval_list and pdb_chain_id not in eval_list and pdb_chain_id + "_" not in eval_list:
            continue

        print(f"Evaluating {pdb_chain_id}")
        try:
            rho = np.load(in_dir + pid + "_rho_wrt_center.npy")
        except FileNotFoundError:
            print(f"File not found: {in_dir}{pid}_rho_wrt_center.npy")
            continue

        theta   = np.load(in_dir + pid + "_theta_wrt_center.npy")
        feat    = np.load(in_dir + pid + "_input_feat.npy")
        feat    = mask_input_feat(feat, params["feat_mask"])
        mask    = np.load(in_dir + pid + "_mask.npy")
        indices = np.load(in_dir + pid + "_list_indices.npy",
                          encoding="latin1", allow_pickle=True)

        print(f"Total patches: {len(mask)}")
        tic = time.time()
        scores = run_masif_site(params, model, rho, theta, feat, mask, indices, device=device)
        print(f"Inference time: {time.time()-tic:.3f}s  patches: {len(scores)}")

        np.save(
            os.path.join(params["out_pred_dir"], f"pred_{pdbid}_{chains[ix]}.npy"),
            scores,
        )
