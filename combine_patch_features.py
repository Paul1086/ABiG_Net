import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def load_mat_features(mat_file):
    mat = loadmat(mat_file)
    if "graph_feats" in mat:
        feats = mat["graph_feats"]
    elif "graph_features" in mat:
        feats = mat["graph_features"]
    else:
        raise KeyError(f"No graph feature variable found in {mat_file}")
    return np.asarray(feats, dtype=float).reshape(-1)

def combine_one(mat_file, cell_file, overwrite=False):
    mat_file = Path(mat_file)
    cell_file = Path(cell_file)
    stem = mat_file.stem.replace("vordelmstnn_graph_features_", "")
    out_file = mat_file.parent / f"combined_features_{stem}.npy"
    if out_file.exists() and not overwrite:
        print(f"Already exists: {out_file.name}")
        return
    vdmn_feats = load_mat_features(mat_file)
    cell_feats = np.load(cell_file).astype(float).reshape(-1)
    if vdmn_feats.size != 51:
        print(f"Check size: {mat_file.name} has {vdmn_feats.size} features, expected 51")
    if cell_feats.size != 18:
        print(f"Check size: {cell_file.name} has {cell_feats.size} features, expected 18")
    combined = np.concatenate([vdmn_feats, cell_feats])
    np.save(out_file, combined)
    print(f"Saved {out_file.name} ({combined.size} features)")

def combine_folder(root_dir, overwrite=False):
    root_dir = Path(root_dir)
    mat_files = sorted(root_dir.rglob("vordelmstnn_graph_features_patch_*.mat"))
    if len(mat_files) == 0:
        print(f"No vordelmstnn feature files found in {root_dir}")
        return
    print(f"Found {len(mat_files)} vordelmstnn feature files")
    for mat_file in mat_files:
        stem = mat_file.stem.replace("vordelmstnn_graph_features_", "")
        cell_file = mat_file.parent / f"cell_graph_features_{stem}.npy"
        if not cell_file.exists():
            print(f"Missing: {cell_file.name}")
            continue
        combine_one(mat_file, cell_file, overwrite=overwrite)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", default=None)
    parser.add_argument("--cell", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.root_dir is not None:
        combine_folder(args.root_dir, overwrite=args.overwrite)
        return
    if args.mat is None or args.cell is None:
        raise ValueError("Use --root-dir or both --mat and --cell")
    combine_one(args.mat, args.cell, overwrite=args.overwrite)

if __name__ == "__main__":
    main()