import numpy as np
import pymeshlab

"""
fixmesh.py: Regularize a protein surface mesh using pymeshlab.
Based on the original PyMesh-based implementation.
"""


def fix_mesh(vertices, faces, resolution):
    """
    Regularize a mesh to a target edge length.

    Args:
        vertices: (N, 3) float array
        faces: (M, 3) int array
        resolution: target edge length (float)

    Returns:
        (vertices, faces): regularized mesh as numpy arrays
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=vertices.astype(np.float64),
        face_matrix=faces.astype(np.int32),
    ))

    # Remove degenerate geometry before remeshing
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()

    # Isotropic remeshing to target edge length
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5,
        targetlen=pymeshlab.PureValue(resolution),
    )

    # Final cleanup
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()

    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()
