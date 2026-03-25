# global_vars.py: Global variables used by MaSIF -- mainly pointing to environment variables of programs used by MaSIF.
# Pablo Gainza - LPDI STI EPFL 2018-2019
# Released under an Apache License 2.0

import os
import sys
epsilon = 1.0e-6

msms_bin = os.environ.get('MSMS_BIN', 'msms')
reduce_bin = os.environ.get('REDUCE_BIN', 'reduce')
pdb2pqr_bin = os.environ.get('PDB2PQR_BIN', 'pdb2pqr')
apbs_bin = os.environ.get('APBS_BIN', 'apbs')
  
# multivalue_bin: no longer required — APBS DX interpolation is handled in Python
# via gridData (see triangulation/computeAPBS.py)
multivalue_bin = os.environ.get('MULTIVALUE_BIN', '')


class NoSolutionError(Exception):
    pass
