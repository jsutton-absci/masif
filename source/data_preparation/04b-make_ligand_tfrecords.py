import numpy as np
from random import shuffle
import os
import glob
from scipy import spatial
from default_config.masif_opts import masif_opts

"""
04b-make_ligand_dataset.py: Build the MaSIF-ligand dataset.

For each protein in the precomputation directory, compute per-vertex
pocket labels (one column per ligand) and save them as
``p1_pocket_labels.npy`` alongside the other precomputed arrays.

Also saves train/val/test PDB split lists as .npy files in ``lists/``.

Replaces the original TFRecords-based pipeline.
Pablo Gainza / Freyr Sverrisson - LPDI STI EPFL 2019 (PyTorch port 2024)
Released under an Apache License 2.0
"""

params = masif_opts["ligand"]
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
labels_dict = {lig: i + 1 for i, lig in enumerate(ligands)}

precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]

# List all structures that have been preprocessed
precomputed_pdbs = glob.glob(os.path.join(precom_dir, "*", "p1_X.npy"))
precomputed_pdbs = [p.split("/")[-2] for p in precomputed_pdbs]

# Only use those selected based on sequence homology
selected_pdbs = np.load(os.path.join("lists", "selected_pdb_ids_30.npy")).astype(str)
all_pdbs = [p for p in precomputed_pdbs if p.split("_")[0] in selected_pdbs]

# Use previously saved splits if present, otherwise create new ones
splits_exist = all(
    os.path.exists(f"lists/{s}_pdbs_sequence.npy")
    for s in ("train", "val", "test")
)
if splits_exist:
    train_pdbs = np.load("lists/train_pdbs_sequence.npy").astype(str)
    val_pdbs = np.load("lists/val_pdbs_sequence.npy").astype(str)
    test_pdbs = np.load("lists/test_pdbs_sequence.npy").astype(str)
else:
    shuffle(all_pdbs)
    train = int(len(all_pdbs) * params["train_fract"])
    val = int(len(all_pdbs) * params["val_fract"])
    train_pdbs = all_pdbs[:train]
    val_pdbs = all_pdbs[train:train + val]
    test_pdbs = all_pdbs[train + val:]
    np.save("lists/train_pdbs_sequence.npy", train_pdbs)
    np.save("lists/val_pdbs_sequence.npy", val_pdbs)
    np.save("lists/test_pdbs_sequence.npy", test_pdbs)

print(f"Train: {len(train_pdbs)}  Val: {len(val_pdbs)}  Test: {len(test_pdbs)}")


def compute_and_save_labels(pdb_list, split_name):
    success = 0
    for i, pdb in enumerate(pdb_list):
        d = os.path.join(precom_dir, pdb + "_")
        out_path = os.path.join(d, "p1_pocket_labels.npy")
        if os.path.exists(out_path):
            continue  # already computed
        try:
            X = np.load(os.path.join(d, "p1_X.npy"))
            Y = np.load(os.path.join(d, "p1_Y.npy"))
            Z = np.load(os.path.join(d, "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(ligand_coord_dir, f"{pdb.split('_')[0]}_ligand_coords.npy")
            )
            all_ligand_types = np.load(
                os.path.join(ligand_coord_dir, f"{pdb.split('_')[0]}_ligand_types.npy")
            ).astype(str)
        except Exception as e:
            print(f"  Skipping {pdb}: {e}")
            continue

        if len(all_ligand_types) == 0:
            continue

        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int32
        )
        for j, structure_ligand in enumerate(all_ligand_types):
            if structure_ligand not in labels_dict:
                continue
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flat = list(set(pp for p in pocket_points for pp in p))
            pocket_labels[pocket_points_flat, j] = labels_dict[structure_ligand]

        np.save(out_path, pocket_labels)
        success += 1
        if i % 50 == 0:
            print(f"  [{split_name}] {i}/{len(pdb_list)} — {pdb}")

    print(f"[{split_name}] saved pocket_labels for {success} proteins")


compute_and_save_labels(train_pdbs, "train")
compute_and_save_labels(val_pdbs, "val")
compute_and_save_labels(test_pdbs, "test")
