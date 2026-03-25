import os
import numpy as np
import torch
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand import MaSIF_ligand
from masif_modules.read_ligand_tfrecords import LigandDataset
from sklearn.metrics import confusion_matrix

"""
masif_ligand_evaluate_test.py: Evaluate and test MaSIF-ligand.
Freyr Sverrisson - LPDI STI EPFL 2019 (PyTorch port 2024)
Released under an Apache License 2.0
"""

params = masif_opts["ligand"]
precom_dir = params["masif_precomputation_dir"]

test_pdbs = np.load("lists/test_pdbs_sequence.npy").astype(str)
testing_data = LigandDataset(test_pdbs, precom_dir)

model_dir = params["model_dir"]
output_model = os.path.join(model_dir, "model.pt")

test_set_out_dir = params["test_set_out_dir"]
os.makedirs(test_set_out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaSIF_ligand(
    n_ligands=params["n_classes"],
    max_rho=params["max_distance"],
    feat_mask=params["feat_mask"],
)
model.load_state_dict(torch.load(output_model, map_location=device, weights_only=True))
model.to(device)
model.eval()

n_samples_per_pocket = 100

for data_element in testing_data:
    input_feat, rho, theta, mask, labels, pdb = data_element
    n_ligands = labels.shape[1]

    pdb_logits_softmax = []
    pdb_labels = []

    for ligand in range(n_ligands):
        pocket_points = np.where(labels[:, ligand] != 0)[0]
        label = int(np.max(labels[:, ligand])) - 1
        if pocket_points.shape[0] < 32:
            continue
        pdb_labels.append(label)

        samples_logits = []
        for _ in range(n_samples_per_pocket):
            sample = np.random.choice(pocket_points, 32, replace=False)
            feat_t  = torch.tensor(input_feat[sample], dtype=torch.float32).to(device)
            rho_t   = torch.tensor(np.expand_dims(rho, -1)[sample], dtype=torch.float32).to(device)
            theta_t = torch.tensor(np.expand_dims(theta, -1)[sample], dtype=torch.float32).to(device)
            mask_t  = torch.tensor(mask[sample], dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = torch.softmax(model(feat_t, rho_t, theta_t, mask_t), dim=1)
            samples_logits.append(logits.cpu().numpy())
        pdb_logits_softmax.append(samples_logits)

    np.save(os.path.join(test_set_out_dir, f"{pdb}_labels.npy"), pdb_labels)
    np.save(os.path.join(test_set_out_dir, f"{pdb}_logits.npy"), pdb_logits_softmax)
    print(f"Saved results for {pdb}")
