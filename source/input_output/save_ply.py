"""
save_ply.py: Save a MaSIF surface mesh to a PLY file.

Replaces the previous PyMesh-based implementation with plyfile.
"""

import numpy as np
from plyfile import PlyData, PlyElement


def save_ply(
    filename: str,
    vertices: np.ndarray,
    faces=None,
    normals: np.ndarray = None,
    charges: np.ndarray = None,
    vertex_cb: np.ndarray = None,
    hbond: np.ndarray = None,
    hphob: np.ndarray = None,
    iface: np.ndarray = None,
    normalize_charges: bool = False,
):
    """Save a protein surface mesh to an ASCII PLY file.

    Args:
        filename:          output path.
        vertices:          [N, 3] float array of vertex coordinates.
        faces:             [F, 3] int array of face vertex indices (optional).
        normals:           [N, 3] vertex normals (stored as vertex_nx/ny/nz).
        charges:           [N]    electrostatic charges (vertex_charge).
        vertex_cb:         [N]    CB-atom distances (vertex_cb).
        hbond:             [N]    hydrogen-bond potential (vertex_hbond).
        hphob:             [N]    hydrophobicity (vertex_hphob).
        iface:             [N]    interface labels / prediction scores.
        normalize_charges: if True, divide charges by 10 before saving.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    n = len(vertices)

    # Build vertex dtype dynamically
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    cols = [vertices[:, 0], vertices[:, 1], vertices[:, 2]]

    if normals is not None:
        n_arr = np.asarray(normals, dtype=np.float32)
        dtype += [("vertex_nx", "f4"), ("vertex_ny", "f4"), ("vertex_nz", "f4")]
        cols += [n_arr[:, 0], n_arr[:, 1], n_arr[:, 2]]

    if charges is not None:
        c = np.asarray(charges, dtype=np.float32)
        if normalize_charges:
            c = c / 10.0
        dtype.append(("vertex_charge", "f4"))
        cols.append(c)

    if hbond is not None:
        dtype.append(("vertex_hbond", "f4"))
        cols.append(np.asarray(hbond, dtype=np.float32))

    if vertex_cb is not None:
        dtype.append(("vertex_cb", "f4"))
        cols.append(np.asarray(vertex_cb, dtype=np.float32))

    if hphob is not None:
        dtype.append(("vertex_hphob", "f4"))
        cols.append(np.asarray(hphob, dtype=np.float32))

    if iface is not None:
        dtype.append(("vertex_iface", "f4"))
        cols.append(np.asarray(iface, dtype=np.float32))

    vertex_data = np.array(list(zip(*cols)), dtype=dtype)
    vertex_el = PlyElement.describe(vertex_data, "vertex")

    elements = [vertex_el]

    if faces is not None and len(faces) > 0:
        faces_arr = np.asarray(faces, dtype=np.int32)
        face_data = np.array(
            [(row,) for row in faces_arr],
            dtype=[("vertex_indices", "O")],
        )
        # plyfile stores variable-length lists as objects
        face_data_proper = np.array(
            [([int(f[0]), int(f[1]), int(f[2])],) for f in faces_arr],
            dtype=[("vertex_indices", "O")],
        )
        face_el = PlyElement.describe(face_data_proper, "face")
        elements.append(face_el)

    PlyData(elements, text=True).write(filename)
