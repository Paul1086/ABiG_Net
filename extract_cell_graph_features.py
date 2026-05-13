# Table 4 features
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

# All features in Table 4 of our paper: see the paper for description 
FEATURE_NAMES = [
    "clustering_coefficient", "average_degree", "number_connected_components", "giant_connected_component_ratio",
    "number_vertices", "number_edges", "average_eccentricity", "radius", "diameter", "number_central_points", "percent_central_points", "closeness_average",
    "laplacian_energy", "laplacian_trace", "upper_slope", "lower_slope", "largest_adjacency_eigenvalue", "adjacency_energy",
]


def load_props(props_file):
    props = np.load(props_file)
    if props.ndim != 2 or props.shape[1] < 4:
        raise ValueError("props file should have at least 4 columns.")
    # props[:, 1] = centroid-0 / y
    # props[:, 2] = centroid-1 / x
    y = props[:, 1]
    x = props[:, 2]
    valid = np.isfinite(x) & np.isfinite(y)
    pts = np.column_stack([x[valid], y[valid]]).astype(float)

    # Use the remaining nuclear measurements for similarity
    # We exclude label and centroid columns
    node_feats = props[valid, 3:].astype(float)
    good_feat = np.all(np.isfinite(node_feats), axis=1)
    pts = pts[good_feat]
    node_feats = node_feats[good_feat]
    return pts, node_feats

def build_graph(pts, node_feats, radius, theta_sim):
    G = nx.Graph()
    G.add_nodes_from(range(len(pts)))
    if len(pts) < 2:
        return G
    tree = cKDTree(pts)
    spatial_pairs = list(tree.query_pairs(radius))
    if len(spatial_pairs) == 0:
        return G
    sim = cosine_similarity(node_feats)
    edges = []
    for i, j in spatial_pairs:
        if sim[i, j] >= theta_sim:
            edges.append((i, j))
    G.add_edges_from(edges)
    return G


def graph_to_sparse(G):
    try:
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    except Exception:
        A = nx.to_scipy_sparse_matrix(G, format="csr", dtype=float)
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    return A


def get_slope(eig, low, high):
    eig = np.sort(np.asarray(eig, dtype=float))
    idx = np.where((eig >= low) & (eig < high))[0]
    if len(idx) < 2:
        return np.nan
    x = np.arange(len(idx), dtype=float)
    y = eig[idx]
    X = np.vstack([np.ones_like(x), x]).T
    _, m = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(m)

def distance_features(G):
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return np.nan, np.nan, np.nan, 0, np.nan, np.nan
    A = graph_to_sparse(G)
    dist = csgraph.shortest_path(A, directed=False, unweighted=True)
    finite = np.isfinite(dist)
    dist_only = np.where(finite, dist, np.nan)
    # feat 07
    ecc = np.nanmax(dist_only, axis=1)
    ecc = np.nan_to_num(ecc, nan=np.inf)
    average_eccentricity = float(np.mean(ecc))
    # feat 08
    radius = float(np.min(ecc))
    # feat 09
    diameter = float(np.max(ecc))
    # feat 10
    number_central_points = int(np.sum(ecc == radius))
    # feat 11
    percent_central_points = number_central_points / n_nodes * 100.0
    # feat 12
    closeness = nx.closeness_centrality(G)
    closeness_average = float(np.mean(list(closeness.values()))) if closeness else np.nan
    return (
        average_eccentricity, radius,
        diameter, number_central_points,
        percent_central_points, closeness_average,
    )


def spectral_features(G):
    A = graph_to_sparse(G)
    if A.shape[0] == 0:
        return [np.nan] * 6
    A_dense = A.toarray()
    adj_eig = np.linalg.eigvalsh(A_dense)
    # feat 17
    largest_adjacency_eigenvalue = float(np.max(adj_eig)) if adj_eig.size else np.nan
    # feat 18
    adjacency_energy = float(np.sum(np.abs(adj_eig)))
    L = csgraph.laplacian(A, normed=False).toarray()
    lap_eig = np.linalg.eigvalsh(L)
    # feat 14
    laplacian_trace = float(np.trace(L))
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else np.nan
    # feat 13
    laplacian_energy = float(np.sum(np.abs(lap_eig - avg_degree)))
    L_norm = csgraph.laplacian(A, normed=True).toarray()
    norm_eig = np.linalg.eigvalsh(L_norm)
    # feat 16
    lower_slope = get_slope(norm_eig, 0.0, 1.0)
    # feat 15
    upper_slope = get_slope(norm_eig, 1.0, 2.0)
    return [
        laplacian_energy, laplacian_trace, upper_slope,
        lower_slope, largest_adjacency_eigenvalue,
        adjacency_energy,
    ]


def cell_graph_features(G):
    # feat 5
    n_nodes = G.number_of_nodes()
    # feat 6
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    # feat 1
    clustering_coefficient = nx.average_clustering(G)
    # feat 2
    average_degree = float(np.mean(degrees))

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    # feat 3
    number_connected_components = len(components)
    # feat 4
    if len(components) > 0:
        giant_connected_component_ratio = len(components[0]) / n_nodes
    else:
        giant_connected_component_ratio = np.nan

    (
        average_eccentricity, radius, diameter,
        number_central_points, percent_central_points, closeness_average,
    ) = distance_features(G)

    (
        laplacian_energy, laplacian_trace, upper_slope, lower_slope,
        largest_adjacency_eigenvalue, adjacency_energy,
    ) = spectral_features(G)

    feats = [
        clustering_coefficient, average_degree,
        number_connected_components, giant_connected_component_ratio,
        n_nodes, n_edges, average_eccentricity, radius, diameter,
        number_central_points, percent_central_points, closeness_average,
        laplacian_energy, laplacian_trace, upper_slope, lower_slope,
        largest_adjacency_eigenvalue, adjacency_energy,
    ]
    return np.array(feats, dtype=float)

def process_one_file(
    props_file, radius=64, theta_sim=0.70, min_nuclei=64, overwrite=False,
):
    props_file = Path(props_file)
    out_dir = props_file.parent

    stem = props_file.stem.replace("props_", "")
    out_file = out_dir / f"cell_graph_features_{stem}.npy"
    name_file = out_dir / "cell_graph_feature_names.json"

    if out_file.exists() and not overwrite:
        print(f"Already exists: {out_file.name}")
        return

    pts, node_feats = load_props(props_file)

    print(f"Loaded {props_file.name}")
    print(f"Nuclei found: {pts.shape[0]}")

    if pts.shape[0] <= min_nuclei:
        print("Too few nuclei. Skipping.")
        return
    G = build_graph(
        pts=pts,
        node_feats=node_feats,
        radius=radius,
        theta_sim=theta_sim,
    )
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    feats = cell_graph_features(G)

    np.save(out_file, feats)
    with open(name_file, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Saved: {out_file}")
    print(f"Feature vector size: {feats.shape}")


def process_root_dir(
    root_dir, radius=64, theta_sim=0.70, min_nuclei=64, overwrite=False,
):
    root_dir = Path(root_dir)
    img_folders = sorted([
        p for p in root_dir.iterdir()
        if p.is_dir() and p.name.startswith("img_")
    ])

    if len(img_folders) == 0:
        print(f"No img_* folders found in {root_dir}")
        return

    print(f"Found {len(img_folders)} img folders.")

    for img_folder in img_folders:
        props_files = sorted(img_folder.glob("props_*.npy"))

        print(f"\n{img_folder.name}: found {len(props_files)} props files")

        for props_file in props_files:
            process_one_file(
                props_file=props_file, radius=radius,
                theta_sim=theta_sim, min_nuclei=min_nuclei,
                overwrite=overwrite,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--props", default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--radius", type=float, default=64)
    parser.add_argument("--theta-sim", type=float, default=0.70)
    parser.add_argument("--min-nuclei", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.props is None and args.folder is None and args.root_dir is None:
        raise ValueError("Use --props, --folder, or --root-dir.")
    if args.props is not None:
        process_one_file(
            props_file=args.props,
            radius=args.radius,
            theta_sim=args.theta_sim,
            min_nuclei=args.min_nuclei,
            overwrite=args.overwrite,
        )
    if args.folder is not None:
        folder = Path(args.folder)
        props_files = sorted(folder.glob("props_*.npy"))
        print(f"{folder.name}: found {len(props_files)} props files")
        for props_file in props_files:
            process_one_file(
                props_file=props_file,
                radius=args.radius,
                theta_sim=args.theta_sim,
                min_nuclei=args.min_nuclei,
                overwrite=args.overwrite,
            )
    if args.root_dir is not None:
        process_root_dir(
            root_dir=args.root_dir,
            radius=args.radius,
            theta_sim=args.theta_sim,
            min_nuclei=args.min_nuclei,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()