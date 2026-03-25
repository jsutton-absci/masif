"""
read_ply.py: Read a PLY surface file written by MaSIF.

Replaces the previous PyMesh-based implementation with plyfile, which is
a pure-Python library with no compiled dependencies.
"""

import numpy as np
from plyfile import PlyData


def read_ply(filename: str):
    """Read a MaSIF PLY surface file.

    Args:
        filename: path to an ASCII or binary PLY file.

    Returns:
        Tuple (vertices, faces, normals, charge, vertex_cb, hbond, hphob)
        where each item is a numpy array.  Missing attributes are returned
        as zero arrays of the appropriate length.
    """
    plydata = PlyData.read(filename)
    v = plydata["vertex"]
    n = len(v)

    vertices = np.stack([
        np.array(v["x"], dtype=np.float64),
        np.array(v["y"], dtype=np.float64),
        np.array(v["z"], dtype=np.float64),
    ], axis=1)

    face_data = plydata["face"]["vertex_indices"]
    faces = np.vstack([np.asarray(f, dtype=np.int32) for f in face_data])

    prop_names = {p.name for p in v.properties}

    if "vertex_nx" in prop_names:
        normals = np.stack([
            np.array(v["vertex_nx"], dtype=np.float64),
            np.array(v["vertex_ny"], dtype=np.float64),
            np.array(v["vertex_nz"], dtype=np.float64),
        ], axis=1)
    else:
        normals = None

    charge     = np.array(v["vertex_charge"], dtype=np.float64) if "vertex_charge" in prop_names else np.zeros(n)
    vertex_cb  = np.array(v["vertex_cb"],     dtype=np.float64) if "vertex_cb"     in prop_names else np.zeros(n)
    hbond      = np.array(v["vertex_hbond"],  dtype=np.float64) if "vertex_hbond"  in prop_names else np.zeros(n)
    hphob      = np.array(v["vertex_hphob"],  dtype=np.float64) if "vertex_hphob"  in prop_names else np.zeros(n)

    return vertices, faces, normals, charge, vertex_cb, hbond, hphob
