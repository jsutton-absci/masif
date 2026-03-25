#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains of interest.
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

# Compute MSMS of surface w/hydrogens, 
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+".pdb",\
        protonate=True)
except Exception as e:
    print(f"MSMS failed for {out_filename1}: {e}")
    sys.exit(1)

# Initialize feature arrays to None; populated below if the corresponding flag is enabled.
vertex_hbond = None
vertex_hphobicity = None
vertex_charges = None

# Compute "charged" vertices
if masif_opts['use_hbond']:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)

# For each surface residue, assign the hydrophobicity of its amino acid.
if masif_opts['use_hphob']:
    vertex_hphobicity = computeHydrophobicity(names1)

# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
reg_vertices, reg_faces = fix_mesh(vertices2, faces2, masif_opts['mesh_res'])

# Compute the normals
vertex_normal = compute_normal(reg_vertices, reg_faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)

if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(reg_vertices, vertices1,\
        vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(reg_vertices, vertices1,\
        vertex_hphobicity, masif_opts)

if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(reg_vertices, out_filename1+".pdb", out_filename1)

iface = np.zeros(len(reg_vertices))
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    # Large complexes (>~40K atoms) cause MSMS to segfault; fall back to iface=zeros.
    try:
        v3, f3, _, _, _ = computeMSMS(pdb_filename,\
            protonate=True)
        # Find the vertices that are in the iface.
        # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
        kdt = KDTree(v3)
        d, r = kdt.query(reg_vertices)
        d = np.square(d) # Square d, because this is how it was in the pyflann version.
        assert(len(d) == len(reg_vertices))
        iface_v = np.where(d >= 2.0)[0]
        iface[iface_v] = 1.0
    except Exception as e:
        print(f"Warning: MSMS failed on full complex ({e}); iface labels set to zero.")
    # Convert to ply and save.
    save_ply(out_filename1+".ply", reg_vertices,\
                        reg_faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface)

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", reg_vertices,\
                        reg_faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
