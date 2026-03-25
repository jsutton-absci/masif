"""
train_ppi_search.py: Training and inference functions for MaSIF-search.

Ported from the TensorFlow 1.x implementation to PyTorch.
"""

import os
import time
import math
import numpy as np
import torch
import torch.optim as optim
from sklearn import metrics


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def construct_batch(
    binder_rho, binder_theta, binder_feat, binder_mask,
    c_pos_idx,
    pos_rho, pos_theta, pos_feat, pos_mask,
    c_neg_idx,
    neg_rho, neg_theta, neg_feat, neg_mask,
):
    """Build a training batch of shape [4n, V, ...].

    The batch is structured as:
        [0  :  n]   positives
        [n  : 2n]   binders (features & theta flipped)
        [2n : 3n]   negatives
        [3n : 4n]   neg_2 (copy of flipped binders)

    Feature flipping: negate all chemical features except the last one
    (hydrophobicity), and negate theta to model the complementary surface.
    """
    def _expand(arr, idx):
        return np.expand_dims(arr[idx], 2)  # [n, V] → [n, V, 1]

    def _flip_feat(feat):
        f = -feat.copy()
        # Do not negate hydrophobicity (last column when 3 or 5 features)
        if f.shape[2] in (3, 5):
            f[:, :, -1] = -f[:, :, -1]
        return f

    # Positive
    br_b = _expand(binder_rho,   c_pos_idx)
    bt_b = _expand(binder_theta, c_pos_idx)
    bf_b = binder_feat[c_pos_idx].copy()
    bm_b = binder_mask[c_pos_idx].copy()

    br_p = _expand(pos_rho,   c_pos_idx)
    bt_p = _expand(pos_theta, c_pos_idx)
    bf_p = pos_feat[c_pos_idx].copy()
    bm_p = pos_mask[c_pos_idx].copy()

    # Flip binder
    bf_b = _flip_feat(bf_b)
    bt_b = 2 * math.pi - bt_b

    # Negative
    br_n = _expand(neg_rho,   c_neg_idx)
    bt_n = _expand(neg_theta, c_neg_idx)
    bf_n = neg_feat[c_neg_idx].copy()
    bm_n = neg_mask[c_neg_idx].copy()

    # neg_2 is a copy of the flipped binder
    br_n2, bt_n2, bf_n2, bm_n2 = br_b.copy(), bt_b.copy(), bf_b.copy(), bm_b.copy()

    batch_rho   = np.concatenate([br_p, br_b, br_n, br_n2])
    batch_theta = np.concatenate([bt_p, bt_b, bt_n, bt_n2])
    batch_feat  = np.concatenate([bf_p, bf_b, bf_n, bf_n2])
    batch_mask  = np.expand_dims(
        np.concatenate([bm_p, bm_b, bm_n, bm_n2]), 2
    )
    return batch_rho, batch_theta, batch_feat, batch_mask


def construct_batch_val_test(c_idx, rho, theta, feat, mask, flip=False):
    """Build a validation / test batch for descriptor computation."""
    batch_rho   = np.expand_dims(rho[c_idx],   2)
    batch_theta = np.expand_dims(theta[c_idx], 2)
    batch_feat  = feat[c_idx].copy()
    batch_mask  = np.expand_dims(mask[c_idx],  2)

    if flip:
        batch_feat = -batch_feat
        batch_theta = 2 * math.pi - batch_theta
        if batch_feat.shape[2] in (3, 5):
            batch_feat[:, :, -1] = -batch_feat[:, :, -1]

    return batch_rho, batch_theta, batch_feat, batch_mask


def compute_dists(d1, d2):
    return np.sqrt(np.sum((d1 - d2) ** 2, axis=1))


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def compute_val_test_desc(model, idx, rho, theta, feat, mask,
                          batch_size=100, flip=False, device=None):
    """Compute surface descriptors for a set of patches.

    Returns:
        all_descs: [len(idx), n_gauss] numpy array.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_descs = []
    num_batches = math.ceil(len(idx) / batch_size)

    for k in range(num_batches):
        c_idx = idx[k * batch_size : (k + 1) * batch_size]
        br, bt, bf, bm = construct_batch_val_test(c_idx, rho, theta, feat, mask, flip=flip)

        with torch.no_grad():
            rho_t  = torch.tensor(br, dtype=torch.float32, device=device)
            theta_t = torch.tensor(bt, dtype=torch.float32, device=device)
            feat_t = torch.tensor(bf, dtype=torch.float32, device=device)
            mask_t = torch.tensor(bm, dtype=torch.float32, device=device)
            desc = model(feat_t, rho_t, theta_t, mask_t).cpu().numpy()

        if desc.ndim == 1:
            desc = desc[np.newaxis, :]
        all_descs.append(desc)

    return np.concatenate(all_descs, axis=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ppi_search(
    model,
    params,
    binder_rho, binder_theta, binder_feat, binder_mask,
    pos_training_idx, pos_val_idx, pos_test_idx,
    pos_rho, pos_theta, pos_feat, pos_mask,
    neg_training_idx, neg_val_idx, neg_test_idx,
    neg_rho, neg_theta, neg_feat, neg_mask,
    num_iterations=1_000_000,
    num_iter_test=1000,
    batch_size=32,
    batch_size_val_test=1000,
    device=None,
    learning_rate=1e-3,
):
    """Train MaSIF-search.

    Batch structure: [positives | binders | negatives | neg_2],
    each block of size batch_size // 4.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    out_dir = params["model_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logfile = open(os.path.join(out_dir, "log.txt"), "w")
    logfile.write(f"Training positives : {len(pos_training_idx)}\n")
    logfile.write(f"Validation positives: {len(pos_val_idx)}\n")
    logfile.write(f"Test positives      : {len(pos_test_idx)}\n")
    logfile.write(f"Training negatives : {len(neg_training_idx)}\n")
    logfile.write(f"Iterations          : {num_iterations}\n")

    pos_train_cp = pos_training_idx.copy()
    neg_train_cp = neg_training_idx.copy()

    best_val_auc = 0.0
    list_train_loss = []
    iter_pos_score, iter_neg_score = [], []
    tic = time.time()

    for it in range(num_iterations):
        np.random.shuffle(pos_train_cp)
        np.random.shuffle(neg_train_cp)

        n_sub = batch_size // 4
        c_pos = pos_train_cp[:n_sub]
        c_neg = neg_train_cp[:n_sub]

        br, bt, bf, bm = construct_batch(
            binder_rho, binder_theta, binder_feat, binder_mask, c_pos,
            pos_rho, pos_theta, pos_feat, pos_mask, c_neg,
            neg_rho, neg_theta, neg_feat, neg_mask,
        )

        rho_t  = torch.tensor(br, dtype=torch.float32, device=device)
        theta_t = torch.tensor(bt, dtype=torch.float32, device=device)
        feat_t = torch.tensor(bf, dtype=torch.float32, device=device)
        mask_t = torch.tensor(bm, dtype=torch.float32, device=device)

        if it == 0:
            # No gradient step on first iteration (matches original code)
            model.eval()
            with torch.no_grad():
                desc = model(feat_t, rho_t, theta_t, mask_t)
                loss, pos_d, neg_d = model.compute_loss(desc)
            training_loss = 0.0
        else:
            model.train()
            optimizer.zero_grad()
            desc = model(feat_t, rho_t, theta_t, mask_t)
            loss, pos_d, neg_d = model.compute_loss(desc)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()

        iter_pos_score = np.concatenate([pos_d.cpu().numpy(), iter_pos_score])
        iter_neg_score = np.concatenate([neg_d.cpu().numpy(), iter_neg_score])
        list_train_loss.append(training_loss)

        # ---- Periodic evaluation ----
        if it % num_iter_test == 0:
            roc_approx = 1 - compute_roc_auc(iter_pos_score, iter_neg_score)
            logfile.write(f"Iter {it}: train_loss={np.mean(list_train_loss):.4f}  "
                          f"approx_train_AUC={roc_approx:.4f}\n")
            print(f"Iter {it}: approx train AUC={roc_approx:.4f}")

            iter_pos_score, iter_neg_score = [], []
            list_train_loss = []

            t_eval = time.time()

            pos_desc    = compute_val_test_desc(model, pos_val_idx, pos_rho, pos_theta, pos_feat, pos_mask,
                                                batch_size=batch_size_val_test, device=device)
            binder_desc = compute_val_test_desc(model, pos_val_idx, binder_rho, binder_theta, binder_feat, binder_mask,
                                                batch_size=batch_size_val_test, flip=True, device=device)
            neg_desc    = compute_val_test_desc(model, neg_val_idx, neg_rho, neg_theta, neg_feat, neg_mask,
                                                batch_size=batch_size_val_test, device=device)
            neg_desc_2  = binder_desc.copy()
            np.random.shuffle(neg_desc)

            pos_dists = compute_dists(pos_desc, binder_desc)
            neg_dists = compute_dists(neg_desc, neg_desc_2)
            val_auc   = 1 - compute_roc_auc(pos_dists, neg_dists)
            logfile.write(f"Iter {it} val AUC={val_auc:.4f} ({time.time()-t_eval:.1f}s)\n")
            print(f"Iter {it} val AUC={val_auc:.4f}")

            t_test = time.time()
            pos_desc    = compute_val_test_desc(model, pos_test_idx, pos_rho, pos_theta, pos_feat, pos_mask,
                                                batch_size=batch_size_val_test, device=device)
            binder_desc = compute_val_test_desc(model, pos_test_idx, binder_rho, binder_theta, binder_feat, binder_mask,
                                                batch_size=batch_size_val_test, flip=True, device=device)
            neg_desc    = compute_val_test_desc(model, neg_test_idx, neg_rho, neg_theta, neg_feat, neg_mask,
                                                batch_size=batch_size_val_test, device=device)
            neg_desc_2  = binder_desc.copy()
            np.random.shuffle(neg_desc)
            pos_dists = compute_dists(pos_desc, binder_desc)
            neg_dists = compute_dists(neg_desc, neg_desc_2)
            test_auc  = 1 - compute_roc_auc(pos_dists, neg_dists)
            logfile.write(f"Iter {it} test AUC={test_auc:.4f} ({time.time()-t_test:.1f}s)\n")
            print(f"Iter {it} test AUC={test_auc:.4f}")
            logfile.flush()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                logfile.write(">>> Saving model.\n")
                print(">>> Saving model.")
                torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
                np.save(os.path.join(out_dir, "pos_dists.npy"),     pos_dists)
                np.save(os.path.join(out_dir, "neg_dists.npy"),     neg_dists)
                np.save(os.path.join(out_dir, "pos_desc.npy"),      pos_desc)
                np.save(os.path.join(out_dir, "binder_desc.npy"),   binder_desc)
                np.save(os.path.join(out_dir, "neg_desc.npy"),      neg_desc)
                np.save(os.path.join(out_dir, "neg_desc_2.npy"),    neg_desc_2)
                np.save(os.path.join(out_dir, "pos_test_idx.npy"),  pos_test_idx)
                np.save(os.path.join(out_dir, "neg_test_idx.npy"),  neg_test_idx)

            tic = time.time()

    logfile.close()
