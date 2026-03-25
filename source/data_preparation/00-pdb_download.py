#!/usr/bin/python
import sys
import os
import urllib.request

from default_config.masif_opts import masif_opts
# Local includes
from input_output.protonate import protonate

if len(sys.argv) <= 1:
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

if not os.path.exists(masif_opts['raw_pdb_dir']):
    os.makedirs(masif_opts['raw_pdb_dir'])

if not os.path.exists(masif_opts['tmp_dir']):
    os.mkdir(masif_opts['tmp_dir'])

in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]

# Download PDB via HTTPS (RCSB REST API; does not require FTP access)
pdb_filename = os.path.join(masif_opts['tmp_dir'], pdb_id + '.pdb')
url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
print(f'Downloading {url}')
urllib.request.urlretrieve(url, pdb_filename)

##### Protonate with reduce, if hydrogens included.
# - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
protonated_file = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

