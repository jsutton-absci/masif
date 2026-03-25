import os
import numpy as np
import importlib
import sys
import torch
import torch.nn.functional as F
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand import MaSIF_ligand
from masif_modules.read_ligand_tfrecords import LigandDataset
from sklearn.metrics import confusion_matrix

"""
masif_ligand_train.py: Train MaSIF-ligand.
Freyr Sverrisson - LPDI STI EPFL 2019 (PyTorch port 2024)
Released under an Apache License 2.0
"""

params = masif_opts["ligand"]
precom_dir = params["masif_precomputation_dir"]

train_pdbs = np.load("lists/train_pdbs_sequence.npy").astype(str)
val_pdbs   = np.load("lists/val_pdbs_sequence.npy").astype(str)
test_pdbs  = np.load("lists/test_pdbs_sequence.npy").astype(str)

training_data   = LigandDataset(train_pdbs, precom_dir)
validation_data = LigandDataset(val_pdbs,   precom_dir)
testing_data    = LigandDataset(test_pdbs,  precom_dir)

out_dir = params["model_dir"]
os.makedirs(out_dir, exist_ok=True)
output_model = os.path.join(out_dir, "model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaSIF_ligand(
    n_ligands=params["n_classes"],
    max_rho=params["max_distance"],
    feat_mask=params["feat_mask"],
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def run_epoch(dataset, train=True):
    losses, ytrue, ypred = [], [], []
    for data_element in dataset:
        input_feat, rho, theta, mask, labels, pdb = data_element
        n_ligands = labels.shape[1]
        random_ligand = np.random.randint(n_ligands)
        pocket_points = np.where(labels[:, random_ligand] != 0)[0]
        label = int(np.max(labels[:, random_ligand])) - 1
        if pocket_points.shape[0] < 32:
            continue

        sample = pocket_points[:32] if not train else \
            np.random.choice(pocket_points, 32, replace=False)

        pocket_labels_onehot = np.zeros(params["n_classes"], dtype=np.float32)
        pocket_labels_onehot[label] = 1.0

        feat_t  = torch.tensor(input_feat[sample], dtype=torch.float32).to(device)
        rho_t   = torch.tensor(np.expand_dims(rho, -1)[sample], dtype=torch.float32).to(device)
        theta_t = torch.tensor(np.expand_dims(theta, -1)[sample], dtype=torch.float32).to(device)
        mask_t  = torch.tensor(mask[sample], dtype=torch.float32).to(device)
        label_t = torch.tensor([label], dtype=torch.long).to(device)

        if train:
            model.train()
            optimizer.zero_grad()
            logits = model(feat_t, rho_t, theta_t, mask_t)
            loss = F.cross_entropy(logits, label_t)
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                logits = model(feat_t, rho_t, theta_t, mask_t)
                loss = F.cross_entropy(logits, label_t)

        losses.append(loss.item())
        ytrue.append(label)
        ypred.append(int(logits.argmax(dim=1).cpu()))

    return losses, ytrue, ypred


best_val_accuracy = 0.0
num_epochs = 100

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch} ===")

    tr_losses, tr_true, tr_pred = run_epoch(training_data, train=True)
    if tr_true:
        cm = confusion_matrix(tr_true, tr_pred)
        acc = float(np.sum(np.diag(cm))) / np.sum(cm)
        print(f"Train   loss={np.mean(tr_losses):.4f}  acc={acc:.4f}")
        print(cm)

    val_losses, val_true, val_pred = run_epoch(validation_data, train=False)
    if val_true:
        cm = confusion_matrix(val_true, val_pred)
        val_acc = float(np.sum(np.diag(cm))) / np.sum(cm)
        print(f"Val     loss={np.mean(val_losses):.4f}  acc={val_acc:.4f}")
        print(cm)
        if val_acc > best_val_accuracy:
            print("Saving model")
            torch.save(model.state_dict(), output_model)
            best_val_accuracy = val_acc

    test_losses, test_true, test_pred = run_epoch(testing_data, train=False)
    if test_true:
        cm = confusion_matrix(test_true, test_pred)
        test_acc = float(np.sum(np.diag(cm))) / np.sum(cm)
        print(f"Test    loss={np.mean(test_losses):.4f}  acc={test_acc:.4f}")
        print(cm)
