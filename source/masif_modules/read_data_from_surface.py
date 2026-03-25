"""
read_data_from_surface.py: Decompose a MaSIF PLY surface into geodesic
patches and compute per-vertex input features.

Updated 2024: replaced PyMesh with plyfile + python-igl for curvature.
"""

import numpy as np
from scipy.spatial import cKDTree

from plyfile import PlyData
from geometry.compute_polar_coordinates import compute_polar_coordinates


# ---------------------------------------------------------------------------
# Simple mesh container (duck-type compatible with the geometry module)
# ---------------------------------------------------------------------------

class _Mesh:
    """Minimal mesh object consumed by compute_polar_coordinates."""
    def __init__(self, vertices, faces, normals):
        self.vertices = vertices
        self.faces    = faces
        self.normals  = normals


# ---------------------------------------------------------------------------
# Curvature helpers
# ---------------------------------------------------------------------------

def _compute_curvatures(vertices, faces):
    """Return mean (H) and Gaussian (K) curvature per vertex.

    Uses python-igl's principal_curvature which is numerically robust.
    Falls back to a cotangent-Laplacian approximation when igl is not
    installed.
    """
    try:
        import igl
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces,    dtype=np.int32)
        pd1, pd2, k1, k2, *_ = igl.principal_curvature(v, f)  # libigl 2.6+ returns 5 values
        H = (k1 + k2) / 2.0
        K = k1 * k2
    except ImportError:
        # Simple approximation via discrete Laplace–Beltrami (less accurate)
        import trimesh
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        r = max(np.linalg.norm(vertices.max(0) - vertices.min(0)) * 0.02, 1.0)
        H = trimesh.curvature.discrete_mean_curvature_measure(tm, tm.vertices, r)
        K = trimesh.curvature.discrete_gaussian_curvature_measure(tm, tm.vertices, r)
    return H, K


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def normalize_electrostatics(in_elec):
    """Clip to ±3 and remap to [-1, 1]."""
    elec = np.clip(in_elec, -3, 3)
    elec = (elec + 3) / 6.0 * 2.0 - 1.0
    return elec


def _mean_normal_center_patch(D, n, r):
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_n = np.mean(c_normal, axis=0, keepdims=True).T
    mean_n = mean_n / np.linalg.norm(mean_n)
    return np.squeeze(mean_n)


def compute_ddc(patch_v, patch_n, patch_cp, patch_rho):
    """Distance-dependent curvature (Yin et al. PNAS 2009)."""
    n = patch_n
    r = patch_v
    i = patch_cp
    ni = _mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    sf = np.linalg.norm(r + n - (ni + r[i]), axis=1) - dij
    sf = np.sign(sf)
    dij[dij == 0] = 1e-8
    kij = np.linalg.norm(n - ni, axis=1) / dij * sf
    kij[kij > 0.7]  = 0
    kij[kij < -0.7] = 0
    return kij


# ---------------------------------------------------------------------------
# Main surface reader
# ---------------------------------------------------------------------------

def read_data_from_surface(ply_fn: str, params: dict):
    """Load a PLY surface file and compute patch features.

    Args:
        ply_fn:  path to the ASCII PLY file written by MaSIF's triangulation
                 pipeline (contains vertex_nx/ny/nz, vertex_charge, etc.).
        params:  masif_opts dict for the application, must contain keys:
                 'max_distance', 'max_shape_size'.

    Returns:
        input_feat:    [N, max_shape_size, 5]
        rho:           [N, max_shape_size]
        theta:         [N, max_shape_size]
        mask:          [N, max_shape_size]
        neigh_indices: list of N lists
        iface_labels:  [N]
        vertices:      [N, 3]
    """
    plydata = PlyData.read(ply_fn)
    v = plydata["vertex"]
    prop_names = {p.name for p in v.properties}

    vertices = np.stack([
        np.array(v["x"], dtype=np.float64),
        np.array(v["y"], dtype=np.float64),
        np.array(v["z"], dtype=np.float64),
    ], axis=1)

    face_data = plydata["face"]["vertex_indices"]
    faces = np.vstack([np.asarray(f, dtype=np.int32) for f in face_data])

    normals = np.stack([
        np.array(v["vertex_nx"], dtype=np.float64),
        np.array(v["vertex_ny"], dtype=np.float64),
        np.array(v["vertex_nz"], dtype=np.float64),
    ], axis=1)

    mesh = _Mesh(vertices, faces, normals)

    # Geodesic polar coordinates
    rho, theta, neigh_indices, mask = compute_polar_coordinates(
        mesh,
        radius=params["max_distance"],
        max_vertices=params["max_shape_size"],
    )

    # Principal curvature → shape index
    H, K = _compute_curvatures(vertices, faces)
    elem = H ** 2 - K
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    denom = k1 - k2
    denom[denom == 0] = 1e-8
    si = np.arctan((k1 + k2) / denom) * (2 / np.pi)

    # Chemical features
    charge = normalize_electrostatics(
        np.array(v["vertex_charge"], dtype=np.float64)
        if "vertex_charge" in prop_names else np.zeros(len(vertices))
    )
    hbond = (
        np.array(v["vertex_hbond"], dtype=np.float64)
        if "vertex_hbond" in prop_names else np.zeros(len(vertices))
    )
    hphob = (
        np.array(v["vertex_hphob"], dtype=np.float64) / 4.5
        if "vertex_hphob" in prop_names else np.zeros(len(vertices))
    )
    iface_labels = (
        np.array(v["vertex_iface"], dtype=np.float64)
        if "vertex_iface" in prop_names else np.zeros(len(vertices))
    )

    n = len(vertices)
    input_feat = np.zeros((n, params["max_shape_size"], 5), dtype=np.float32)

    for vix in range(n):
        neigh_vix = np.array(neigh_indices[vix])

        patch_v  = vertices[neigh_vix]
        patch_n  = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0]
        mask_pos = np.where(mask[vix] == 1.0)[0]
        patch_rho = rho[vix][mask_pos]
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)

        m = len(neigh_vix)
        input_feat[vix, :m, 0] = si[neigh_vix]
        input_feat[vix, :m, 1] = ddc
        input_feat[vix, :m, 2] = hbond[neigh_vix]
        input_feat[vix, :m, 3] = charge[neigh_vix]
        input_feat[vix, :m, 4] = hphob[neigh_vix]

    return (
        input_feat,
        rho.astype(np.float32),
        theta.astype(np.float32),
        mask.astype(np.float32),
        neigh_indices,
        iface_labels,
        np.copy(vertices),
    )


# ---------------------------------------------------------------------------
# Shape complementarity
# ---------------------------------------------------------------------------

def compute_shape_complementarity(
    ply_fn1, ply_fn2, neigh1, neigh2, rho1, rho2, mask1, mask2, params
):
    """Compute shape complementarity between all interface patch pairs.

    Returns:
        v1_sc, v2_sc: arrays of shape [2, N, 10] with the 25th- and
        50th-percentile shape complementarity per vertex per radial ring.
    """
    def _load(fn):
        plydata = PlyData.read(fn)
        v = plydata["vertex"]
        verts = np.stack([
            np.array(v["x"], dtype=np.float64),
            np.array(v["y"], dtype=np.float64),
            np.array(v["z"], dtype=np.float64),
        ], axis=1)
        norms = np.stack([
            np.array(v["vertex_nx"], dtype=np.float64),
            np.array(v["vertex_ny"], dtype=np.float64),
            np.array(v["vertex_nz"], dtype=np.float64),
        ], axis=1)
        return verts, norms

    v1, n1 = _load(ply_fn1)
    v2, n2 = _load(ply_fn2)

    w           = params["sc_w"]
    int_cutoff  = params["sc_interaction_cutoff"]
    radius      = params["sc_radius"]
    num_rings   = 10
    scales      = np.append(np.arange(0, radius, radius / 10), radius)

    v1_sc = np.zeros((2, len(v1), 10))
    v2_sc = np.zeros((2, len(v2), 10))

    kdt = cKDTree(v2)
    d, nn_v1_to_v2 = kdt.query(v1)
    iface_v1 = np.where(d < int_cutoff)[0]

    for cv1_ix in iface_v1:
        patch_idx1 = np.where(mask1[cv1_ix] == 1)[0]
        neigh_cv1  = np.array(neigh1[cv1_ix])[patch_idx1]
        cv2_ix     = nn_v1_to_v2[cv1_ix]
        patch_idx2 = np.where(mask2[cv2_ix] == 1)[0]
        neigh_cv2  = np.array(neigh2[cv2_ix])[patch_idx2]

        pv1, pv2 = v1[neigh_cv1], v2[neigh_cv2]
        pn1, pn2 = n1[neigh_cv1], n2[neigh_cv2]

        kdt1 = cKDTree(pv1)
        kdt2 = cKDTree(pv2)
        d_v2_to_v1, nn_v2_to_v1 = kdt1.query(pv2)
        d_v1_to_v2, nn_v1_to_v2b = kdt2.query(pv1)

        comp1 = np.array([np.dot(pn1[x], -pn2[nn_v1_to_v2b[x]])
                          for x in range(len(pn1))])
        comp1 *= np.exp(-w * d_v1_to_v2 ** 2)

        comp2 = np.array([np.dot(pn2[x], -pn1[nn_v2_to_v1[x]])
                          for x in range(len(pn2))])
        comp2 *= np.exp(-w * d_v2_to_v1 ** 2)

        prho1 = np.array(rho1[cv1_ix])[patch_idx1]
        prho2 = np.array(rho2[cv2_ix])[patch_idx2]

        for ring in range(num_rings):
            for comp, prho, sc, cvi in [
                (comp1, prho1, v1_sc, cv1_ix),
                (comp2, prho2, v2_sc, cv2_ix),
            ]:
                mem = np.where((prho >= scales[ring]) & (prho < scales[ring + 1]))[0]
                sc[0, cvi, ring] = np.percentile(comp[mem], 25) if len(mem) > 0 else 0.0
                sc[1, cvi, ring] = np.percentile(comp[mem], 50) if len(mem) > 0 else 0.0

    return v1_sc, v2_sc
