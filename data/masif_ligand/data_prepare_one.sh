masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
export PYTHONPATH=$PYTHONPATH:$masif_source
PDB_ID=$(echo $1| cut -d"_" -f1)
CHAIN1=$(echo $1| cut -d"_" -f2)
CHAIN2=$(echo $1| cut -d"_" -f3)
python3 $masif_source/data_preparation/00-pdb_download.py $1
python3 $masif_source/data_preparation/00b-generate_assembly.py $1
python3 $masif_source/data_preparation/00c-save_ligand_coords.py $1
python3 $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1 masif_ligand
python3 $masif_source/data_preparation/04-masif_precompute.py masif_ligand $1
