import os
import numpy as np
import torch
from torch.utils.data import Dataset

"""
read_ligand_tfrecords.py: PyTorch Dataset for MaSIF-ligand training data.
Replaces the original TensorFlow TFRecords pipeline.

Each sample is one protein; data are loaded from .npy files in the
masif precomputation directory. pocket_labels.npy must be pre-computed
by 04b-make_ligand_dataset.py.

Pablo Gainza / Freyr Sverrisson - LPDI STI EPFL 2019 (PyTorch port 2024)
Released under an Apache License 2.0
"""


class LigandDataset(Dataset):
    """Dataset of precomputed MaSIF-ligand surface patches.

    Args:
        pdb_list:    list of PDB IDs (strings) to include.
        precom_dir:  path to the masif precomputation directory, where each
                     protein has a subdirectory ``<pdb>_/`` containing:
                     p1_input_feat.npy, p1_rho_wrt_center.npy,
                     p1_theta_wrt_center.npy, p1_mask.npy,
                     p1_pocket_labels.npy.
    """

    def __init__(self, pdb_list, precom_dir):
        self.pdbs = []
        for pdb in pdb_list:
            d = os.path.join(precom_dir, pdb + "_")
            if os.path.exists(os.path.join(d, "p1_pocket_labels.npy")):
                self.pdbs.append((pdb, d))
        self.precom_dir = precom_dir

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb, d = self.pdbs[idx]
        input_feat = np.load(os.path.join(d, "p1_input_feat.npy"))
        rho = np.load(os.path.join(d, "p1_rho_wrt_center.npy"))
        theta = np.load(os.path.join(d, "p1_theta_wrt_center.npy"))
        mask = np.expand_dims(np.load(os.path.join(d, "p1_mask.npy")), -1)
        pocket_labels = np.load(os.path.join(d, "p1_pocket_labels.npy"))
        return input_feat, rho, theta, mask, pocket_labels, pdb
