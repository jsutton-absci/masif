# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MaSIF (Molecular Surface Interaction Fingerprints) is a geometric deep learning framework for analyzing protein molecular surfaces to predict biomolecular interactions. The core idea: proteins are represented as triangulated surfaces with chemical/geometric features per vertex, and geodesic patches extracted from those surfaces are fed into convolutional neural networks.

**Three applications:**
- **MaSIF-site** — predicts which surface patches are protein-protein interaction (PPI) sites
- **MaSIF-ligand** — predicts ligand binding pockets on protein surfaces
- **MaSIF-search** — ultrafast scanning of surfaces to predict PPI complex structures via surface fingerprints

## Installation

```bash
pip install -r requirements.txt
```

External tools required (must be on PATH or configured in `source/default_config/global_vars.py`):
- `reduce` v3.23 — adds protons to structures
- `MSMS` v2.6.1 — computes molecular surfaces
- `APBS` ≥ 3.0 + `pdb2pqr` ≥ 3.0 — electrostatics (**v3 CLI**: `--ff PARSE`, explicit `.pqr` output arg, `--apbs-input filename.in`)
- `dssp` or `mkdssp` — secondary structure (MaSIF-peptide only); code auto-detects which binary is present

Key Python dependencies: **PyTorch ≥ 2.5**, plyfile ≥ 1.0, trimesh ≥ 4.4, pymeshlab ≥ 2024.0, libigl ≥ 2.5 (optional but recommended for accurate curvature), networkx, BioPython, scikit-learn, open3d ≥ 0.19.

`libigl` (`pip install libigl`) is strongly recommended — it provides accurate principal curvatures used for the shape-index feature. Without it, the code falls back to trimesh's discrete curvature approximation.

## Running the Pipeline

Each application has its own `data/masif_<app>/` directory with shell scripts. The typical workflow for **MaSIF-site**:

```bash
cd data/masif_site/
./data_prepare_one.sh 4ZQK_A          # Prepare single protein (~2 min)
./data_prepare_one.sh 1AKJ_AB_DE      # Multiple chains
./predict_site.sh 4ZQK_A              # Run inference
./color_site.sh 4ZQK_A                # Generate colored PLY for visualization
```

For full training data preparation (parallel):
```bash
cd data/masif_site/
bash batch_prepare.sh --workers 8 --list lists/all_to_prepare.txt
# Progress: tail -f logs/batch_prepare.log
# Then train:
bash train_nn.sh nn_models.all_feat_3l.custom_params
```

For **MaSIF-search**:
```bash
cd data/masif_ppi_search/
./cache_nn.sh nn_models.sc05.custom_params
./train.sh nn_models.sc05.custom_params
./compute_descriptors.sh lists/testing.txt
./second_stage.sh                      # RANSAC alignment
```

For **MaSIF-ligand** (SLURM cluster):
```bash
sbatch prepare_data.slurm
sbatch make_tfrecord.slurm
sbatch train_model.slurm
sbatch evaluate_test.slurm
```

## Data Preparation Pipeline

Sequential scripts in `source/data_preparation/`:

1. `00-pdb_download.py` — fetch PDB structures
2. `00b-generate_assembly.py` — biological assemblies (ligand only)
3. `01-pdb_extract_and_triangulate.py` — extract chains, protonate (`reduce`), compute surface (`MSMS`), assign features
4. `04-masif_precompute.py` — decompose surfaces into geodesic patches, save as `.npy` files

Step 3 dominates runtime (~2 min/protein): MSMS surface generation, APBS electrostatics (~1 min), MDS angular/radial coordinate computation (~18 sec).

## Architecture

```
source/
├── default_config/
│   ├── masif_opts.py       # All hyperparameters and paths — edit this to customize
│   ├── chemistry.py        # Atomic radii, residue hydrophobicity tables
│   └── global_vars.py      # Paths to external tools (MSMS, APBS, reduce)
├── triangulation/          # Surface feature computation wrappers
│   ├── computeMSMS.py      # MSMS surface generation
│   ├── computeAPBS.py      # Electrostatics via APBS
│   ├── computeCharges.py   # H-bond donors/acceptors
│   ├── computeHydrophobicity.py
│   └── fixmesh.py          # Mesh regularization
├── input_output/           # PDB parsing, protonation, PLY file I/O (plyfile)
├── masif_modules/          # Neural network models and training
│   ├── masif_layers.py     # Shared GaussianBasisConvLayer (PyTorch nn.Module)
│   ├── MaSIF_site.py       # PyTorch model for PPI site prediction
│   ├── MaSIF_ligand.py     # PyTorch model for ligand pocket prediction
│   ├── MaSIF_ppi_search.py # PyTorch model for surface descriptor generation
│   ├── train_masif_site.py # Training loop + run_masif_site() inference helper
│   ├── train_ppi_search.py # Training loop + descriptor computation helpers
│   └── read_data_from_surface.py  # PLY → numpy features (uses plyfile + igl)
└── geometry/               # Geodesic patch extraction, MDS coordinates
```

### Key design: GaussianBasisConvLayer

All three models share a custom `GaussianBasisConvLayer` (`masif_layers.py`). It maps per-vertex (rho, theta, features, mask) to a descriptor via learnable Gaussian kernels placed on a polar coordinate grid. `n_rotations` rotational copies are max-pooled to achieve approximate rotation invariance. Layer 1 applies one instance per feature channel (separate Gaussian parameters per feature); deeper layers apply one instance to all features jointly.

**Data flow:** PDB → extract chains → protonate → MSMS triangulated surface → assign per-vertex features (shape index, hydrophobicity, H-bond potential, APBS electrostatics) → geodesic patches (9Å for site, 12Å for search/ligand) → neural network → per-vertex scores or surface descriptors.

## Configuration

`source/default_config/masif_opts.py` controls everything per-application:
- Feature flags: `use_hbond`, `use_hphob`, `use_apbs`, `use_iface`
- Patch radius: `masif_opts['ppi_search']['max_distance']`
- Mesh resolution: `masif_opts['mesh_res']` (default 1.0 Å)
- Directory layout for raw/precomputed data

Each `data/masif_<app>/` directory has its own `masif_opts.py` that overrides or extends the defaults.

## Key Data Formats

- **Input**: PDB files
- **Intermediate**: ASCII `.ply` files (triangulated surfaces with custom vertex properties: `vertex_nx/ny/nz`, `vertex_charge`, `vertex_hbond`, `vertex_hphob`, `vertex_iface`), `.npy` patch arrays
- **Output**: Colored `.ply` files for visualization in PyMOL (plugin in `source/masif_pymol_plugin/`)

## Model checkpoints

Models are saved as `model.pt` (PyTorch `state_dict`) in the application's `model_dir`. To load:
```python
model.load_state_dict(torch.load("path/to/model.pt", map_location="cpu", weights_only=True))
```

## Migration notes (traps to avoid)

This codebase was migrated from TF 1.9 + PyMesh → PyTorch + plyfile/pymeshlab. Key API choices that are easy to get wrong:

- **plyfile 1.1+**: Use `PlyElement.describe(data, name)` — the constructor `PlyElement(name, data)` requires `(name, properties, count)` and is not for direct use with structured arrays.
- **pymeshlab 2024+**: `pymeshlab.AbsoluteValue` was renamed to `pymeshlab.PureValue`. The `threshold` kwarg was removed from `meshing_remove_duplicate_vertices`.
- **libigl 2.6+**: `igl.principal_curvature(v, f)` returns **5** values `(pd1, pd2, k1, k2, extra)` — unpack as `pd1, pd2, k1, k2, *_ = igl.principal_curvature(v, f)`.
- **multivalue replacement**: The `multivalue` binary (APBS 1.x) no longer ships with APBS 3.x. Use `gridData.Grid` + `scipy.interpolate.RegularGridInterpolator` to evaluate `.pqr.dx` files at vertex coordinates.
- **APBS 3.x DX output name**: pdb2pqr v3 sets the APBS write stem to `<base>.pqr`, so APBS produces `<base>.pqr.dx` (not `<base>.dx`).
- **reduce binary**: Not on system PATH in conda envs. Set `REDUCE_BIN` env var or use full path to `$CONDA_PREFIX/bin/reduce`.
- **fix_mesh**: Signature is `fix_mesh(vertices, faces, resolution) → (vertices, faces)` — callers unpack a tuple, not a mesh object.
- **ScoreNN**: The `predict()` method runs inference. Do not call `.eval(features)` — that's `nn.Module.eval()` with no args.
- **open3d**: Always import via `from geometry.open3d_import import *`. Never `from open3d import *` directly — the registration API moved to `o3d.pipelines.registration` in v0.13.
- **LigandDataset**: Replaces TFRecords. Lives in `masif_modules/read_ligand_tfrecords.py`. Reads per-protein `.npy` files from `precomputation_dir/<pdb>/`.
- **torch.load**: Always pass `weights_only=True` to suppress FutureWarning in PyTorch ≥ 2.4.

## Environment setup (verified working)

```bash
conda create -n masif python=3.11
conda activate masif
pip install -r requirements.txt
pip install pdb2pqr gridDataFormats
conda install -c conda-forge apbs
conda install -c bioconda reduce
# MSMS: download arm64 binary from https://ccsb.scripps.edu/msms/downloads/
# Place at $CONDA_PREFIX/bin/msms

export MSMS_BIN=$CONDA_PREFIX/bin/msms
export PDB2PQR_BIN=$CONDA_PREFIX/bin/pdb2pqr
export APBS_BIN=$CONDA_PREFIX/bin/apbs
export REDUCE_BIN=$CONDA_PREFIX/bin/reduce
```

## Reproducing Paper Results

The current repo uses a Python-only implementation (refactored Feb 2020 from MATLAB), now migrated to PyTorch (replacing TensorFlow 1.9) and plyfile (replacing PyMesh). To reproduce exact paper numbers, see the separate `masif_paper` repository linked in the README.

Benchmarks against other methods are in `comparison/`.
