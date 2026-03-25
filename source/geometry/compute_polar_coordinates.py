"""
compute_polar_coordinates.py: Compute the polar coordinates of all patches.

Pablo Gainza - LPDI STI EPFL 2019
Updated 2024: replaced PyMesh with plain numpy/networkx; fixed
time.clock() (removed in Python 3.8) → time.perf_counter().
"""

import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.manifold import MDS
import networkx as nx


def compute_polar_coordinates(mesh, do_fast=True, radius=12, max_vertices=200):
    """Compute polar coordinates for every patch in the mesh.

    Args:
        mesh:        object with attributes
                       .vertices  [N, 3] float
                       .faces     [F, 3] int
                       .normals   [N, 3] float  (vertex normals)
        do_fast:     use the fast approximate MDS (recommended).
        radius:      geodesic radius of patches (Å).
        max_vertices: maximum number of vertices per patch.

    Returns:
        rho:          [N, max_vertices]  geodesic radial distances (0-padded).
        theta:        [N, max_vertices]  angular coordinates (0-padded).
        neigh_indices: list of N lists, each containing the vertex indices
                       of patch members (sorted by geodesic distance).
        mask:         [N, max_vertices]  1 where a vertex is valid, 0 elsewhere.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.normals

    # Build weighted graph (edge weight = 3-D Euclidean distance)
    G = nx.Graph()
    n = len(vertices)
    G.add_nodes_from(np.arange(n))

    f = np.asarray(faces, dtype=int)
    rowi = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]])
    rowj = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    edgew = scipy.linalg.norm(vertices[rowi] - vertices[rowj], axis=1)
    G.add_weighted_edges_from(np.stack([rowi, rowj, edgew]).T)

    t0 = time.perf_counter()
    cutoff = radius if do_fast else radius * 2
    dists_iter = nx.all_pairs_dijkstra_path_length(G, cutoff=cutoff)
    d2 = {k: v for k, v in dists_iter}
    print("Dijkstra took {:.2f}s".format(time.perf_counter() - t0))

    D = _dict_to_sparse(d2)

    idx = {}  # faces per vertex
    for face_ix, face in enumerate(faces):
        for vi in face:
            idx.setdefault(int(vi), []).append(face_ix)

    i_diag = np.arange(D.shape[0])
    D[i_diag, i_diag] = 1e-8  # avoid exact zeros on diagonal

    t1 = time.perf_counter()
    if do_fast:
        theta_all = _compute_theta_all_fast(D, vertices, faces, normals, idx, radius)
    else:
        theta_all = _compute_theta_all(D, vertices, faces, normals, idx, radius)
    print("MDS took {:.2f}s".format(time.perf_counter() - t1))

    rho_out   = np.zeros((n, max_vertices))
    theta_out = np.zeros((n, max_vertices))
    mask_out  = np.zeros((n, max_vertices))
    neigh_indices = []

    for i in range(n):
        dists_i = d2[i]
        sorted_dists_i = sorted(dists_i.items(), key=lambda kv: kv[1])
        neigh = [int(x[0]) for x in sorted_dists_i[:max_vertices]]
        neigh_indices.append(neigh)
        rho_out[i, :len(neigh)]   = np.squeeze(np.asarray(D[i, neigh].todense()))
        theta_out[i, :len(neigh)] = np.squeeze(theta_all[i][neigh])
        mask_out[i, :len(neigh)]  = 1

    theta_out[theta_out < 0] += 2 * np.pi
    return rho_out, theta_out, neigh_indices, mask_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_sparse(mydict):
    data, row, col = [], [], []
    for r, cols in mydict.items():
        for c, v in cols.items():
            data.append(v)
            row.append(int(r))
            col.append(int(c))
    coo = coo_matrix((data, (row, col)))
    return csr_matrix(coo)


def _compute_thetas(plane, vix, verts, faces, normals, neighbors, idx):
    """Compute angular coordinates relative to a reference direction.

    Args:
        plane:     [len(neighbors), 2]  MDS 2-D embedding of patch vertices.
        vix:       index of the patch centre in the full mesh.
        verts:     [N, 3] mesh vertex positions.
        faces:     [F, 3] mesh faces.
        normals:   [N, 3] vertex normals.
        neighbors: array of vertex indices in the patch.
        idx:       dict mapping vertex index → list of face indices.

    Returns:
        thetas: [N]  angle for each mesh vertex (0 outside the patch).
    """
    neighbors = np.asarray(neighbors)
    plane_center_ix = np.where(neighbors == vix)[0][0]
    thetas = np.zeros(len(verts))

    plane = plane - plane[plane_center_ix]

    # Find a triangle whose three vertices are all in the patch
    valid = False
    for face_ix in idx[vix]:
        tt = faces[face_ix]
        if all(v in neighbors for v in tt):
            valid = True
            break
    assert valid, "No fully-contained triangle found for vertex {}".format(vix)

    normal_tt = normals[tt].mean(axis=0)
    neigh_tt = [x for x in tt if x != vix]
    v1ix, v2ix = neigh_tt[0], neigh_tt[1]
    v1ix_plane = np.where(neighbors == v1ix)[0][0]
    v2ix_plane = np.where(neighbors == v2ix)[0][0]

    norm_plane = np.sqrt((plane ** 2).sum(axis=1))
    norm_plane[plane_center_ix] = 1.0
    vecs = plane / norm_plane[:, None]
    vecs[plane_center_ix] = [0, 0]
    vecs = np.column_stack([vecs, np.zeros(len(vecs))])

    ref_vec = vecs[v1ix_plane]
    cross   = np.cross(vecs, ref_vec)
    term1   = np.arctan2(np.sqrt((cross ** 2).sum(axis=1)),
                         vecs @ ref_vec)
    normal_plane = np.array([0.0, 0.0, 1.0])
    theta = term1 * np.sign(vecs @ np.cross(normal_plane, ref_vec))

    v0 = verts[vix]
    v1 = verts[v1ix] - v0;  v1 /= np.linalg.norm(v1)
    v2 = verts[v2ix] - v0;  v2 /= np.linalg.norm(v2)
    angle_v1_v2 = np.arctan2(
        np.linalg.norm(np.cross(v2, v1)), np.dot(v2, v1)
    ) * np.sign(np.dot(v2, np.cross(normal_tt, v1)))

    if np.sign(angle_v1_v2) != np.sign(theta[v2ix_plane]):
        theta = -theta

    theta[theta == 0] = np.finfo(float).eps
    thetas[neighbors] = theta
    return thetas


def _call_mds(mds_obj, pair_dist):
    return mds_obj.fit_transform(pair_dist)


def _compute_theta_all(D, vertices, faces, normals, idx, radius):
    mymds = MDS(n_components=2, n_init=1, max_iter=50,
                dissimilarity="precomputed", n_jobs=10)
    all_theta = []
    for i in range(D.shape[0]):
        if i % 100 == 0:
            print(i)
        neigh = D[i].nonzero()
        ii = np.where(D[i][neigh] < radius)[1]
        neigh_i = neigh[1][ii]
        pair_dist_i = np.asarray(D[neigh_i, :][:, neigh_i].todense())
        plane_i = _call_mds(mymds, pair_dist_i)
        theta = _compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)
        all_theta.append(theta)
    return all_theta


def _compute_theta_all_fast(D, vertices, faces, normals, idx, radius):
    """Approximate MDS using only the inner half-radius; propagate angles
    to the outer ring by nearest-inner-neighbour assignment."""
    mymds = MDS(n_components=2, n_init=1, eps=0.1, max_iter=50,
                dissimilarity="precomputed", n_jobs=1)
    all_theta = []
    t0 = time.perf_counter()
    only_mds = 0.0

    for i in range(D.shape[0]):
        neigh = D[i].nonzero()
        ii = np.where(D[i][neigh] < radius / 2)[1]
        neigh_i = neigh[1][ii]
        pair_dist_i = np.asarray(D[neigh_i, :][:, neigh_i].todense())

        tic = time.perf_counter()
        plane_i = _call_mds(mymds, pair_dist_i)
        only_mds += time.perf_counter() - tic

        theta = _compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)

        # Propagate angles to outer-ring vertices (radius/2 … radius)
        kk = np.where(D[i][neigh] >= radius / 2)[1]
        neigh_k = neigh[1][kk]
        dist_kk = np.asarray(D[neigh_k, :][:, neigh_i].todense())
        dist_kk[dist_kk == 0] = float("inf")
        closest = neigh_i[np.squeeze(np.argmin(dist_kk, axis=1))]
        theta[neigh_k] = theta[closest]

        all_theta.append(theta)

    print("Only MDS time: {:.2f}s".format(only_mds))
    print("Full loop time: {:.2f}s".format(time.perf_counter() - t0))
    return all_theta
