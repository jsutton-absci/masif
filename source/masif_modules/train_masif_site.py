"""
train_masif_site.py: Training and inference functions for MaSIF-site.

Ported from the TensorFlow 1.x implementation to PyTorch.
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn import metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_input_feat(input_feat, mask):
    """Remove feature channels disabled by feat_mask."""
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    """Pad neighbour index lists to a uniform length.

    Indices shorter than max_verts are padded with the centre-vertex index
    (i.e. they self-loop) so the shape is [n_verts, max_verts].
    """
    padded = np.zeros((len(indices), max_verts), dtype=int)
    for k, idx in enumerate(indices):
        l = len(idx)
        padded[k, :l] = idx
        padded[k, l:] = k   # pad with self-index (maps to centre vertex)
    return padded


def _to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Inference (used at prediction time)
# ---------------------------------------------------------------------------

def run_masif_site(params, model, rho_wrt_center, theta_wrt_center,
                   input_feat, mask, indices, device=None):
    """Run MaSIF-site inference on one protein.

    Args:
        params:           masif_opts['site'] dict.
        model:            MaSIF_site instance (already loaded).
        rho_wrt_center:   [N, max_shape_size] numpy array.
        theta_wrt_center: [N, max_shape_size] numpy array.
        input_feat:       [N, max_shape_size, n_feat] numpy array.
        mask:             [N, max_shape_size] numpy array.
        indices:          list of N variable-length neighbour-index lists.
        device:           torch.device (auto-detected if None).

    Returns:
        scores: [N] numpy array of interface probabilities.
    """
    if device is None:
        device = next(model.parameters()).device

    indices = pad_indices(indices, mask.shape[1])
    mask_3d = mask[:, :, np.newaxis]

    with torch.no_grad():
        rho_t   = _to_tensor(rho_wrt_center[:, :, np.newaxis], device)
        theta_t = _to_tensor(theta_wrt_center[:, :, np.newaxis], device)
        feat_t  = _to_tensor(input_feat, device)
        mask_t  = _to_tensor(mask_3d, device)
        idx_t   = _to_tensor(indices, device, dtype=torch.long)

        model.eval()
        logits = model(feat_t, rho_t, theta_t, mask_t, idx_t)
        scores = torch.sigmoid(logits)[:, 0].cpu().numpy()

    return scores


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_masif_site(
    model,
    params,
    batch_size=100,
    num_iterations=100,
    device=None,
    learning_rate=1e-3,
):
    """Train MaSIF-site.

    Args:
        model:          MaSIF_site instance.
        params:         masif_opts['site'] dict with keys:
                        model_dir, training_list, testing_list,
                        masif_precomputation_dir, feat_mask, n_conv_layers.
        batch_size:     max patches per gradient step.
        num_iterations: number of full dataset passes (epochs).
        device:         torch.device (auto-detect if None).
        learning_rate:  Adam learning rate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    out_dir = params["model_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logfile = open(os.path.join(out_dir, "log.txt"), "w")
    for key, val in params.items():
        logfile.write(f"{key}: {val}\n")

    training_list = set(
        line.rstrip() for line in open(params["training_list"])
    )
    testing_list = set(
        line.rstrip() for line in open(params["testing_list"])
    )

    data_dirs = os.listdir(params["masif_precomputation_dir"])
    np.random.shuffle(data_dirs)
    n_val    = max(1, len(data_dirs) // 10)
    val_dirs = set(data_dirs[-n_val:])

    best_val_auc = 0.0

    for epoch in range(num_iterations):
        list_train_auc, list_train_loss = [], []
        list_val_auc = []
        list_test_auc = []
        all_test_labels, all_test_scores = [], []
        count = 0
        tic = time.time()

        logfile.write(f"Starting epoch {epoch}\n")
        print(f"Starting epoch {epoch}")

        # ---- Training / validation pass ----
        for ppi_pair_id in data_dirs:
            mydir = os.path.join(params["masif_precomputation_dir"], ppi_pair_id, "")
            fields = ppi_pair_id.split("_")
            pdbid   = fields[0]
            chain1  = fields[1]
            chain2  = fields[2] if len(fields) > 2 else ""

            pids = []
            if pdbid + "_" + chain1 in training_list or pdbid + "_" + chain1 in val_dirs:
                pids.append(("p1", chain1))
            if chain2 and (pdbid + "_" + chain2 in training_list or pdbid + "_" + chain2 in val_dirs):
                pids.append(("p2", chain2))

            for pid, chain in pids:
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except FileNotFoundError:
                    continue

                if len(iface_labels) > 8000:
                    continue
                pos_sum = np.sum(iface_labels)
                if pos_sum > 0.75 * len(iface_labels) or pos_sum < 30:
                    continue

                count += 1
                rho   = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta = np.load(mydir + pid + "_theta_wrt_center.npy")
                feat  = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    feat = mask_input_feat(feat, params["feat_mask"])
                mask_arr = np.load(mydir + pid + "_mask.npy")
                indices  = np.load(mydir + pid + "_list_indices.npy",
                                   encoding="latin1", allow_pickle=True)

                mask_3d  = mask_arr[:, :, np.newaxis]
                indices  = pad_indices(indices, mask_arr.shape[1])

                # One-hot labels
                labels_2d = np.zeros((len(iface_labels), 2))
                labels_2d[iface_labels == 1, 0] = 1
                labels_2d[iface_labels == 0, 1] = 1

                pos_idx_np = np.where(iface_labels == 1)[0]
                neg_idx_np = np.where(iface_labels == 0)[0]
                np.random.shuffle(neg_idx_np)
                np.random.shuffle(pos_idx_np)

                if params["n_conv_layers"] == 1:
                    n = min(len(pos_idx_np), len(neg_idx_np), batch_size // 2)
                    subset = np.concatenate([neg_idx_np[:n], pos_idx_np[:n]])
                    rho   = rho[subset];   theta = theta[subset]
                    feat  = feat[subset];  mask_3d = mask_3d[subset]
                    labels_2d = labels_2d[subset]
                    indices   = indices[subset]
                    pos_idx_np = np.arange(n)
                    neg_idx_np = np.arange(n, 2 * n)
                else:
                    neg_idx_np = neg_idx_np[:len(pos_idx_np)]

                # Tensors
                rho_t    = _to_tensor(rho[:, :, np.newaxis], device)
                theta_t  = _to_tensor(theta[:, :, np.newaxis], device)
                feat_t   = _to_tensor(feat, device)
                mask_t   = _to_tensor(mask_3d, device)
                idx_t    = _to_tensor(indices, device, dtype=torch.long)
                labels_t = _to_tensor(labels_2d, device)
                pos_t    = _to_tensor(pos_idx_np, device, dtype=torch.long)
                neg_t    = _to_tensor(neg_idx_np, device, dtype=torch.long)

                is_val = ppi_pair_id in val_dirs

                if is_val:
                    logfile.write(f"Validating on {ppi_pair_id} {pid}\n")
                    model.eval()
                    with torch.no_grad():
                        logits = model(feat_t, rho_t, theta_t, mask_t, idx_t)
                        loss, scores, el = model.compute_loss(logits, labels_t, pos_t, neg_t)
                    auc = metrics.roc_auc_score(el[:, 0].detach().cpu().numpy(), scores.detach().cpu().numpy())
                    list_val_auc.append(auc)
                else:
                    logfile.write(f"Training on {ppi_pair_id} {pid}\n")
                    model.train()
                    optimizer.zero_grad()
                    logits = model(feat_t, rho_t, theta_t, mask_t, idx_t)
                    loss, scores, el = model.compute_loss(logits, labels_t, pos_t, neg_t)
                    loss.backward()
                    optimizer.step()
                    auc = metrics.roc_auc_score(el[:, 0].detach().cpu().numpy(), scores.detach().cpu().numpy())
                    list_train_auc.append(auc)
                    list_train_loss.append(loss.item())

                logfile.flush()

        # ---- Test pass ----
        for ppi_pair_id in data_dirs:
            mydir = os.path.join(params["masif_precomputation_dir"], ppi_pair_id, "")
            fields = ppi_pair_id.split("_")
            pdbid  = fields[0]
            chain1 = fields[1]
            chain2 = fields[2] if len(fields) > 2 else ""

            pids = []
            if pdbid + "_" + chain1 in testing_list:
                pids.append(("p1", chain1))
            if chain2 and pdbid + "_" + chain2 in testing_list:
                pids.append(("p2", chain2))

            for pid, _ in pids:
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except FileNotFoundError:
                    continue
                if len(iface_labels) > 20000:
                    continue
                pos_sum = np.sum(iface_labels)
                if pos_sum > 0.75 * len(iface_labels) or pos_sum < 30:
                    continue

                rho   = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta = np.load(mydir + pid + "_theta_wrt_center.npy")
                feat  = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    feat = mask_input_feat(feat, params["feat_mask"])
                mask_arr = np.load(mydir + pid + "_mask.npy")
                indices  = np.load(mydir + pid + "_list_indices.npy",
                                   encoding="latin1", allow_pickle=True)
                indices = pad_indices(indices, mask_arr.shape[1])

                scores = run_masif_site(
                    params, model,
                    rho, theta, feat, mask_arr, indices,
                    device=device,
                )
                auc = metrics.roc_auc_score(iface_labels, scores)
                list_test_auc.append(auc)
                all_test_labels.append(iface_labels)
                all_test_scores.append(scores)

        flat_test_labels = np.concatenate(all_test_labels) if all_test_labels else np.array([])
        flat_test_scores = np.concatenate(all_test_scores) if all_test_scores else np.array([])

        outstr = f"Epoch ran on {count} proteins\n"
        outstr += (f"Per protein AUC mean (training): {np.mean(list_train_auc):.4f}; "
                   f"median: {np.median(list_train_auc):.4f}\n")
        outstr += (f"Per protein AUC mean (validation): {np.mean(list_val_auc):.4f}; "
                   f"median: {np.median(list_val_auc):.4f}\n")
        outstr += (f"Per protein AUC mean (test): {np.mean(list_test_auc):.4f}; "
                   f"median: {np.median(list_test_auc):.4f}\n")
        if len(flat_test_labels) > 0:
            outstr += f"Testing AUC (all points): {metrics.roc_auc_score(flat_test_labels, flat_test_scores):.4f}\n"
        outstr += f"Epoch took {time.time() - tic:.2f}s\n"

        logfile.write(outstr + "\n")
        print(outstr)

        val_mean = np.mean(list_val_auc) if list_val_auc else 0.0
        if val_mean > best_val_auc:
            best_val_auc = val_mean
            logfile.write(">>> Saving model.\n")
            print(">>> Saving model.")
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            np.save(os.path.join(out_dir, "test_labels.npy"), all_test_labels)
            np.save(os.path.join(out_dir, "test_scores.npy"), all_test_scores)

    logfile.close()
