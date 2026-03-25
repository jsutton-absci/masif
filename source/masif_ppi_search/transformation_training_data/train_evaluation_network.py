import numpy as np
from pathlib import Path
import glob
import os
import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from score_nn import ScoreNN

"""
train_evaluation_network.py: Train a neural network to score protein complex
alignments (based on MaSIF).
Freyr Sverrisson and Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

np.random.seed(42)
torch.manual_seed(42)

data_dir = "transformation_data/"

with open("../lists/training.txt") as f:
    training_list = f.read().splitlines()

n_positives = 1   # number of correctly aligned examples per protein
n_negatives = 200 # number of incorrectly aligned examples per protein
max_rmsd = 5.0
max_npoints = 200
n_features = 3

data_list = glob.glob(data_dir + '*')
data_list = [
    d for d in data_list
    if os.path.exists(d + "/features.npy") and d.split("/")[-1] in training_list
]

all_features = np.empty(
    (len(data_list) * (n_positives + n_negatives), max_npoints, n_features)
)
all_labels = np.empty((len(data_list) * (n_positives + n_negatives), 1))
n_samples = 0

for i, d in enumerate(data_list):
    if i % 100 == 0:
        print(i, "Feature array size (MB)", all_features.nbytes * 1e-6)

    source_patch_rmsds = np.load(d + "/source_patch_rmsds.npy")
    positive_alignments = np.where(source_patch_rmsds < max_rmsd)[0]
    negative_alignments = np.where(source_patch_rmsds >= max_rmsd)[0]

    if len(positive_alignments) == 0:
        continue
    if len(negative_alignments) < n_negatives:
        continue

    chosen_positives = np.random.choice(positive_alignments, n_positives, replace=False)
    chosen_negatives = np.random.choice(negative_alignments, n_negatives, replace=False)
    chosen_alignments = np.concatenate([chosen_positives, chosen_negatives])

    try:
        features = np.load(d + "/features.npy", encoding="latin1", allow_pickle=True)
    except Exception:
        continue

    features = features[chosen_alignments]
    features_trimmed = np.zeros((len(chosen_alignments), max_npoints, n_features))
    for j, f in enumerate(features):
        if f.shape[0] <= max_npoints:
            features_trimmed[j, :f.shape[0], :f.shape[1]] = f
        else:
            selected_rows = np.random.choice(f.shape[0], max_npoints, replace=False)
            features_trimmed[j, :, :f.shape[1]] = f[selected_rows]

    labels = (source_patch_rmsds[chosen_alignments] < max_rmsd).astype(int).reshape(-1, 1)
    all_features[n_samples:n_samples + len(chosen_alignments)] = features_trimmed
    all_labels[n_samples:n_samples + len(chosen_alignments)] = labels
    n_samples += len(chosen_alignments)

all_features = all_features[:n_samples]
all_labels = all_labels[:n_samples].ravel().astype(int)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    all_features, all_labels, test_size=0.1, random_state=42, shuffle=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ScoreNN()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor(
        [1.0 / n_negatives, 1.0 / n_positives], dtype=torch.float32
    ).to(device)
)

batch_size = 32
n_epochs = 50
best_val_loss = float('inf')
os.makedirs('models/nn_score', exist_ok=True)

for epoch in range(n_epochs):
    model.train()
    perm = np.random.permutation(len(X_train))
    epoch_losses = []
    for start in range(0, len(X_train), batch_size):
        idx = perm[start:start + batch_size]
        xb = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
        yb = torch.tensor(y_train[idx], dtype=torch.long).to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        yv = torch.tensor(y_val, dtype=torch.long).to(device)
        val_loss = loss_fn(model(xv), yv).item()

    print(f"Epoch {epoch+1}/{n_epochs}  train_loss={np.mean(epoch_losses):.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/nn_score/trained_model.pt')
        print("  → saved best model")
