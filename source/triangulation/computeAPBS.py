import os
import re
import numpy
from subprocess import Popen, PIPE

from default_config.global_vars import apbs_bin, pdb2pqr_bin

"""
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0

Requires pdb2pqr >= 3.x and APBS >= 3.x.
"""

def _run_pdb2pqr(pdb2pqr_bin, pdbname, filename_base, directory):
    """Run pdb2pqr and return (returncode, stderr text)."""
    args = [
        pdb2pqr_bin,
        "--ff", "PARSE",
        "--whitespace",
        "--noopt",
        "--apbs-input", filename_base + ".in",
        pdbname,
        filename_base + ".pqr",
    ]
    p = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    _, stderr = p.communicate()
    return p.returncode, stderr.decode()


def _strip_incomplete_residues(pdb_path, out_path, stderr_text):
    """Remove residues flagged by pdb2pqr as having missing atoms.

    pdb2pqr 3.7.x crashes in rebuild_tetrahedral when it tries to add
    hydrogens to heavy atoms it just rebuilt from scratch.  Stripping those
    residues before a second pdb2pqr pass avoids the crash.
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    bad = set()
    for m in re.finditer(r"Missing atom \S+ in residue \S+ (\S+) (\d+)", stderr_text):
        bad.add((m.group(1), int(m.group(2))))

    if not bad:
        return False

    class _StripSelect(Select):
        def accept_residue(self, r):
            return (r.get_parent().id, r.get_id()[1]) not in bad

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    io_obj = PDBIO()
    io_obj.set_structure(struct)
    io_obj.save(out_path, _StripSelect())
    return True


def computeAPBS(vertices, pdb_file, tmp_file_base):
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """
    fields = tmp_file_base.split("/")[0:-1]
    directory = "/".join(fields) + "/"
    filename_base = tmp_file_base.split("/")[-1]
    pdbname = pdb_file.split("/")[-1]

    rc, stderr = _run_pdb2pqr(pdb2pqr_bin, pdbname, filename_base, directory)

    if rc != 0:
        # pdb2pqr 3.7.x crashes when rebuilding side chains for residues that
        # have missing heavy atoms.  Strip those residues and retry once.
        pdb_path = os.path.join(directory, pdbname)
        clean_pdbname = filename_base + "_clean.pdb"
        clean_path = os.path.join(directory, clean_pdbname)
        stripped = _strip_incomplete_residues(pdb_path, clean_path, stderr)
        if not stripped:
            raise RuntimeError(f"pdb2pqr failed and no incomplete residues found: {stderr[-300:]}")
        clean_base = filename_base + "_clean"
        rc2, stderr2 = _run_pdb2pqr(pdb2pqr_bin, clean_pdbname, clean_base, directory)
        if rc2 != 0:
            raise RuntimeError(f"pdb2pqr failed even after stripping incomplete residues: {stderr2[-300:]}")
        # Swap names so the rest of the function finds the right files
        filename_base = clean_base
        print(f"Warning: pdb2pqr ran on cleaned PDB (incomplete residues stripped).")

    args = [apbs_bin, filename_base + ".in"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    # Interpolate the APBS potential at each vertex using gridData + scipy.
    # This replaces the legacy `multivalue` binary (removed in APBS 3.x).
    # pdb2pqr v3 names the APBS output write-stem after the .pqr file, so
    # APBS produces "<filename_base>.pqr.dx" rather than "<filename_base>.dx".
    from gridData import Grid
    from scipy.interpolate import RegularGridInterpolator
    dx_path = os.path.join(directory, filename_base + ".pqr.dx")
    grid = Grid(dx_path)
    interp = RegularGridInterpolator(
        grid.midpoints, grid.grid,
        method='linear', bounds_error=False, fill_value=0.0,
    )
    charges = interp(vertices).astype(numpy.float64)

    remove_fn = os.path.join(directory, filename_base)
    os.remove(remove_fn + ".pqr")
    os.remove(remove_fn + ".pqr.dx")
    os.remove(remove_fn + '.in')
    # pdb2pqr v2 generated a pickle file; v3 does not — remove if present
    legacy_pickle = remove_fn + '-input.p'
    if os.path.exists(legacy_pickle):
        os.remove(legacy_pickle)
    # Clean up the stripped PDB if it was created
    clean_pdb = os.path.join(directory, filename_base.replace("_clean", "") + "_clean.pdb")
    if os.path.exists(clean_pdb):
        os.remove(clean_pdb)

    return charges
