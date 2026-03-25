"""
masif_site_label_surface.py: Colour a protein PLY surface by the MaSIF-site
interface score and optionally compute ROC AUC against ground truth.
"""

import os
import sys
import importlib
import numpy as np
from sklearn.metrics import roc_auc_score
from plyfile import PlyData, PlyElement
from default_config.masif_opts import masif_opts

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
    print("Not enough parameters")
    sys.exit(1)

os.makedirs(params["out_surf_dir"], exist_ok=True)
all_roc_auc = []

for ppi_pair_id in ppi_pair_ids:
    fields = ppi_pair_id.split("_")
    pdbid  = fields[0]
    chains = [fields[1]]
    if len(fields) == 3 and fields[2]:
        chains.append(fields[2])
    pids = ["p1"] if len(chains) == 1 else ["p1", "p2"]

    for ix, pid in enumerate(pids):
        pdb_chain_id = pdbid + "_" + chains[ix]
        if eval_list and pdb_chain_id not in eval_list and pdb_chain_id + "_" not in eval_list:
            continue

        ply_file = masif_opts["ply_file_template"].format(pdbid, chains[ix])
        try:
            plydata = PlyData.read(ply_file)
        except Exception:
            print(f"File does not exist: {ply_file}")
            continue

        pred_file = os.path.join(params["out_pred_dir"], f"pred_{pdbid}_{chains[ix]}.npy")
        try:
            scores = np.load(pred_file)
        except Exception:
            print(f"Prediction not found: {pred_file}")
            continue

        v = plydata["vertex"]
        prop_names = {p.name for p in v.properties}

        if "vertex_iface" in prop_names:
            ground_truth = np.array(v["vertex_iface"], dtype=np.float64)
            try:
                roc = roc_auc_score(ground_truth, scores)
                all_roc_auc.append(roc)
                print(f"ROC AUC for {pdb_chain_id}: {roc:.2f}")
            except Exception:
                print("No ROC AUC (possibly no ground truth).")

        # Re-save the PLY with the prediction score replacing vertex_iface
        n = len(v)
        verts = np.stack([
            np.array(v["x"], dtype=np.float32),
            np.array(v["y"], dtype=np.float32),
            np.array(v["z"], dtype=np.float32),
        ], axis=1)

        dtype = [("x","f4"),("y","f4"),("z","f4")]
        cols  = [verts[:,0], verts[:,1], verts[:,2]]

        for attr in ["vertex_nx","vertex_ny","vertex_nz",
                     "vertex_charge","vertex_hbond","vertex_hphob","vertex_cb"]:
            if attr in prop_names:
                dtype.append((attr, "f4"))
                cols.append(np.array(v[attr], dtype=np.float32))

        # Add prediction score
        dtype.append(("vertex_iface", "f4"))
        cols.append(scores.astype(np.float32))

        vertex_data = np.array(list(zip(*cols)), dtype=dtype)
        vertex_el   = PlyElement.describe(vertex_data, "vertex")

        face_data = plydata["face"]["vertex_indices"]
        faces = np.vstack([np.asarray(f, dtype=np.int32) for f in face_data])
        face_records = np.array(
            [([int(f[0]), int(f[1]), int(f[2])],) for f in faces],
            dtype=[("vertex_indices", "O")],
        )
        face_el = PlyElement.describe(face_records, "face")

        out_path = os.path.join(params["out_surf_dir"], pdb_chain_id + ".ply")
        PlyData([vertex_el, face_el], text=True).write(out_path)
        print(f"Saved: {out_path}")

if all_roc_auc:
    print(f"Computed ROC AUC for {len(all_roc_auc)} proteins")
    print(f"Median ROC AUC: {np.median(all_roc_auc):.4f}")
