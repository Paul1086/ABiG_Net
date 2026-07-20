#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import csgraph
from scipy.stats import kruskal, mannwhitneyu
from torch_geometric.data import Data, InMemoryDataset


@dataclass
class Config:
    dataset_root: Path
    raw_csv_dir: Path
    checkpoint_path: Path
    output_dir: Path
    test_indices: List[int]
    remove_test_positions: List[int]
    seed: int = 101
    input_dim: int = 69
    num_classes: int = 3
    generator_hidden_dim1: int = 32
    generator_hidden_dim2: int = 16
    adjacency_mode: str = "deterministic"
    mc_samples: int = 10
    edge_threshold: float = 0.50
    pairwise_row_chunk_size: int = 64
    save_individual_matrices: bool = True
    save_combined_matrix_pickles: bool = True
    show_plots: bool = False
    save_plots: bool = True
    metrics_per_plot_page: int = 6


CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Low grade",
    2: "High grade",
}
CLASS_ORDER: Tuple[str, ...] = ("Normal", "Low grade", "High grade")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Safe Loading

def safe_torch_load(path: str | Path, map_location: str | torch.device):
    path = str(path)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)



# Dataset

class CRCGraphDataset(InMemoryDataset):
    def __init__(self, root: str, raw_csv_dir: str):
        self.source_csv_dir = Path(raw_csv_dir)
        super().__init__(root=root)

        loaded = safe_torch_load(self.processed_paths[0], map_location="cpu")

        if not isinstance(loaded, (tuple, list)):
            raise TypeError(
                f"Expected processed dataset to contain tuple/list, got {type(loaded)}"
            )

        if len(loaded) == 2:
            self.data, self.slices = loaded

        elif len(loaded) >= 3:
            self.data, self.slices = loaded[0], loaded[1]

        else:
            raise ValueError(
                f"Unexpected processed dataset format with {len(loaded)} items."
            )

    @property
    def raw_dir(self) -> str:
        return str(self.source_csv_dir)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "all_image_sampled_node_feats_final.csv",
            "each_line_labels_sampled.csv",
            "all_image_edge_index_transpose.csv",
            "graph_label.csv",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        missing = [
            name for name in self.raw_file_names
            if not (self.source_csv_dir / name).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Processed data.pt was not found and raw CSV files are missing: "
                + ", ".join(missing)
            )

        node_attrs = pd.read_csv(
            self.source_csv_dir / "all_image_sampled_node_feats_final.csv",
            sep=",",
            header=None,
        )
        node_attrs.index += 1

        graph_idx = pd.read_csv(
            self.source_csv_dir / "each_line_labels_sampled.csv",
            sep=",",
            names=["idx"],
            dtype=int,
        )
        graph_idx.index += 1

        edge_table = pd.read_csv(
            self.source_csv_dir / "all_image_edge_index_transpose.csv",
            sep=",",
            names=["source", "target"],
            dtype=int,
        )
        edge_table.index += 1

        graph_labels = pd.read_csv(
            self.source_csv_dir / "graph_label.csv",
            sep=",",
            names=["label"],
            dtype=int,
        )
        graph_labels.index += 1

        data_list: List[Data] = []

        for graph_id in graph_idx["idx"].unique():
            node_ids = graph_idx.index[graph_idx["idx"] == graph_id]
            x_np = node_attrs.loc[node_ids].to_numpy(dtype=np.float32)

            graph_edges = edge_table.loc[
                edge_table["source"].isin(node_ids)
                & edge_table["target"].isin(node_ids)
            ]

            if graph_edges.empty:
                local_edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                node_id_to_local = {
                    int(node_id): local_id
                    for local_id, node_id in enumerate(node_ids)
                }
                edge_np = graph_edges[["source", "target"]].to_numpy(dtype=np.int64)
                local_np = np.asarray(
                    [
                        [node_id_to_local[int(source_id)], node_id_to_local[int(target_id)]]
                        for source_id, target_id in edge_np
                    ],
                    dtype=np.int64,
                ).T
                local_edge_index = torch.as_tensor(local_np, dtype=torch.long)

            label = int(graph_labels.loc[graph_id, "label"])

            data_list.append(
                Data(
                    x=torch.as_tensor(x_np, dtype=torch.float32),
                    edge_index=local_edge_index,
                    y=torch.tensor([label], dtype=torch.long),
                )
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



# Model Definitions

def sample_binary_gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    eps = 1e-10

    uniform = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
    gumbel_noise = -torch.log(-torch.log(uniform))

    values = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        hard_values = (values > 0.5).to(values.dtype)
        values = (hard_values - values).detach() + values

    return values


class AdjacencyGenerator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        tau: float = 1.0,
        hard: bool = False,
        prior_pi: float = 0.2,
        train_gumbel_samples: int = 5,
    ):
        super().__init__()

        self.tau = tau
        self.hard = hard
        self.prior_pi = prior_pi
        self.train_gumbel_samples = train_gumbel_samples

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def pairwise_logits(
        self,
        x: torch.Tensor,
        row_chunk_size: int | None = None,
    ) -> torch.Tensor:
        n = x.size(0)

        if n == 0:
            return x.new_zeros((0, 0))

        if row_chunk_size is None or row_chunk_size >= n:
            xi = x.unsqueeze(1).expand(n, n, -1)
            xj = x.unsqueeze(0).expand(n, n, -1)
            pairs = torch.cat([xi, xj], dim=-1)

            return self.mlp(pairs.reshape(n * n, -1)).reshape(n, n)

        rows: List[torch.Tensor] = []
        xj_all = x.unsqueeze(0)

        for start in range(0, n, row_chunk_size):
            end = min(start + row_chunk_size, n)
            row_count = end - start

            xi = x[start:end].unsqueeze(1).expand(row_count, n, -1)
            xj = xj_all.expand(row_count, n, -1)

            pairs = torch.cat([xi, xj], dim=-1)

            logits_chunk = self.mlp(
                pairs.reshape(row_count * n, -1)
            ).reshape(row_count, n)

            rows.append(logits_chunk)

        return torch.cat(rows, dim=0)

    def deterministic_adjacency(
        self,
        x: torch.Tensor,
        row_chunk_size: int | None = None,
        add_self_loops: bool = True,
    ) -> torch.Tensor:
        logits = self.pairwise_logits(x, row_chunk_size=row_chunk_size)

        adjacency = torch.sigmoid(logits)
        adjacency = 0.5 * (adjacency + adjacency.T)

        if add_self_loops and adjacency.numel() > 0:
            adjacency.fill_diagonal_(1.0)

        return adjacency

    def forward(
        self,
        x: torch.Tensor,
        return_kl: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        logits = self.pairwise_logits(x)

        adjacency = torch.zeros_like(logits)

        for _ in range(self.train_gumbel_samples):
            adjacency += sample_binary_gumbel_sigmoid(
                logits,
                tau=self.tau,
                hard=self.hard,
            )

        adjacency /= float(self.train_gumbel_samples)
        adjacency = 0.5 * (adjacency + adjacency.T)

        if adjacency.numel() > 0:
            adjacency.fill_diagonal_(1.0)

        if return_kl:
            return adjacency, self.kl_divergence(logits)

        return adjacency

    def kl_divergence(self, logits: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        probabilities = torch.sigmoid(logits)
        prior = self.prior_pi

        kl = (
            probabilities * torch.log((probabilities + eps) / (prior + eps))
            + (1.0 - probabilities)
            * torch.log((1.0 - probabilities + eps) / (1.0 - prior + eps))
        )

        return kl.mean()


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=1).clamp_min(1e-12)
        degree_inv_sqrt = degree.pow(-0.5)

        adjacency_norm = (
            degree_inv_sqrt[:, None]
            * adjacency
            * degree_inv_sqrt[None, :]
        )

        return adjacency_norm @ (x @ self.weight) + self.bias


class DenseGCN(nn.Module):
    def __init__(self, in_dim: int = 69, out_dim: int = 3):
        super().__init__()

        self.gcn1 = DenseGCNLayer(in_dim, 64)
        self.norm1 = nn.LayerNorm(64)

        self.gcn2 = DenseGCNLayer(64, 512)
        self.norm2 = nn.LayerNorm(512)

        self.gcn3 = DenseGCNLayer(512, 1024)
        self.norm3 = nn.LayerNorm(1024)

        self.classifier1 = nn.Linear(1024, 256)
        self.classifier2 = nn.Linear(256, 32)
        self.classifier3 = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.norm1(self.gcn1(x, adjacency)))
        h2 = F.relu(self.norm2(self.gcn2(h1, adjacency)))
        h3 = F.relu(self.norm3(self.gcn3(h2, adjacency)))

        graph_embedding = h3.mean(dim=0, keepdim=True)

        hidden = F.relu(self.classifier1(graph_embedding))
        hidden = F.relu(self.classifier2(hidden))

        return self.classifier3(hidden)



# Loading And Validation

def load_models(
    cfg: Config,
    device: torch.device,
) -> Tuple[DenseGCN, AdjacencyGenerator]:
    checkpoint_path = Path(cfg.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    gcn = DenseGCN(in_dim=cfg.input_dim, out_dim=cfg.num_classes)

    generator = AdjacencyGenerator(
        in_dim=cfg.input_dim,
        hidden_dim1=cfg.generator_hidden_dim1,
        hidden_dim2=cfg.generator_hidden_dim2,
    )

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")

    required_keys = {"gcn_state", "generator_state"}
    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing_keys)}. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )

    gcn.load_state_dict(checkpoint["gcn_state"], strict=True)
    generator.load_state_dict(checkpoint["generator_state"], strict=True)

    gcn.to(device).eval()
    generator.to(device).eval()

    return gcn, generator


def validate_dataset(dataset: CRCGraphDataset, cfg: Config) -> None:
    if not cfg.test_indices:
        raise ValueError("test_indices is empty.")

    invalid_indices = [
        index for index in cfg.test_indices
        if index < 0 or index >= len(dataset)
    ]

    if invalid_indices:
        raise IndexError(
            f"These test indices are outside dataset range 0..{len(dataset) - 1}: "
            f"{invalid_indices}"
        )

    bad_feature_shapes = []

    for dataset_index in cfg.test_indices:
        graph = dataset[dataset_index]

        if graph.x.ndim != 2 or graph.x.size(1) != cfg.input_dim:
            bad_feature_shapes.append((dataset_index, tuple(graph.x.shape)))

    if bad_feature_shapes:
        raise ValueError(
            f"Expected every graph to have {cfg.input_dim} node features. "
            f"Mismatches: {bad_feature_shapes[:10]}"
        )



# Learned Adjacency Extraction

@torch.no_grad()
def obtain_learned_adjacency(
    generator: AdjacencyGenerator,
    x: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    mode = cfg.adjacency_mode.strip().lower()

    if mode == "deterministic":
        return generator.deterministic_adjacency(
            x,
            row_chunk_size=cfg.pairwise_row_chunk_size,
            add_self_loops=True,
        )

    if mode == "mc_gumbel":
        if cfg.mc_samples <= 0:
            raise ValueError("mc_samples must be positive for mc_gumbel mode.")

        adjacency_sum = torch.zeros(
            (x.size(0), x.size(0)),
            dtype=x.dtype,
            device=x.device,
        )

        for _ in range(cfg.mc_samples):
            adjacency_sum += generator(x)

        return adjacency_sum / float(cfg.mc_samples)

    raise ValueError(
        "adjacency_mode must be either 'deterministic' or 'mc_gumbel', "
        f"got {cfg.adjacency_mode!r}."
    )


def clean_and_binarize_adjacency(
    adjacency_soft: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = np.asarray(adjacency_soft, dtype=np.float64).copy()

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adjacency.shape}")

    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = 0.5 * (adjacency + adjacency.T)

    np.fill_diagonal(adjacency, 0.0)

    binary = (adjacency >= threshold).astype(np.uint8)
    binary = np.maximum(binary, binary.T)

    np.fill_diagonal(binary, 0)

    return adjacency, binary



# Extended Graph Feature Calculation

def _slope_for_eigen_segment(sorted_values: np.ndarray) -> float:
    values = np.asarray(sorted_values, dtype=np.float64)

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=np.float64)
    slope, intercept = np.polyfit(x, values, deg=1)

    return float(slope)


def _distance_features(adjacency_binary: np.ndarray) -> Dict[str, float | int]:
    n = adjacency_binary.shape[0]

    if n == 0:
        return {
            "avg_eccentricity": 0.0,
            "diameter": 0.0,
            "radius": 0.0,
            "avg_eccentricity_90": 0.0,
            "diameter_90": 0.0,
            "radius_90": 0.0,
            "avg_shortest_path_reachable": 0.0,
            "global_efficiency": 0.0,
            "num_central_points": 0,
            "percent_central_points": 0.0,
        }

    sparse_adj = sparse.csr_matrix(adjacency_binary)

    distances = csgraph.shortest_path(
        sparse_adj,
        directed=False,
        unweighted=True,
    )

    eccentricities = []
    eccentricities_90 = []
    node_mean_shortest_paths = []

    for i in range(n):
        row = distances[i]
        finite_values_no_self = row[np.isfinite(row) & (row > 0)]

        if len(finite_values_no_self) == 0:
            eccentricities.append(0.0)
            eccentricities_90.append(0.0)
            node_mean_shortest_paths.append(0.0)
        else:
            eccentricities.append(float(np.max(finite_values_no_self)))
            eccentricities_90.append(
                float(np.percentile(finite_values_no_self, 90))
            )
            node_mean_shortest_paths.append(
                float(np.mean(finite_values_no_self))
            )

    eccentricities = np.asarray(eccentricities, dtype=np.float64)
    eccentricities_90 = np.asarray(eccentricities_90, dtype=np.float64)
    node_mean_shortest_paths = np.asarray(
        node_mean_shortest_paths,
        dtype=np.float64,
    )

    avg_eccentricity = float(np.mean(eccentricities))
    diameter = float(np.max(eccentricities))
    radius = float(np.min(eccentricities))

    avg_eccentricity_90 = float(np.mean(eccentricities_90))
    diameter_90 = float(np.max(eccentricities_90))
    radius_90 = float(np.min(eccentricities_90))

    avg_shortest_path_reachable = float(
        np.mean(node_mean_shortest_paths)
    )

    if n > 1:
        pair_distances = distances[np.triu_indices(n, k=1)]
        inverse_distances = np.zeros_like(
            pair_distances,
            dtype=np.float64,
        )
        valid_pairs = np.isfinite(pair_distances) & (pair_distances > 0)
        inverse_distances[valid_pairs] = (
            1.0 / pair_distances[valid_pairs]
        )
        global_efficiency = float(np.mean(inverse_distances))
    else:
        global_efficiency = 0.0

    num_central_points = int(
        np.sum(np.isclose(eccentricities, radius))
    )
    percent_central_points = float(
        num_central_points / n
    ) if n > 0 else 0.0

    return {
        "avg_eccentricity": avg_eccentricity,
        "diameter": diameter,
        "radius": radius,
        "avg_eccentricity_90": avg_eccentricity_90,
        "diameter_90": diameter_90,
        "radius_90": radius_90,
        "avg_shortest_path_reachable": avg_shortest_path_reachable,
        "global_efficiency": global_efficiency,
        "num_central_points": num_central_points,
        "percent_central_points": percent_central_points,
    }


def _spectral_features(adjacency_binary: np.ndarray) -> Dict[str, float | int]:
    A = np.asarray(adjacency_binary, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return {
            "largest_eigenvalue_adjacency": 0.0,
            "second_largest_eigenvalue_adjacency": 0.0,
            "trace_adjacency": 0.0,
            "energy_adjacency_squared_sum": 0.0,
            "algebraic_connectivity": 0.0,
            "num_laplacian_eigenvalues_zero": 0,
            "laplacian_lower_slope_0_to_1": 0.0,
            "num_laplacian_eigenvalues_one": 0,
            "laplacian_upper_slope_1_to_2": 0.0,
            "num_laplacian_eigenvalues_two": 0,
            "trace_laplacian": 0.0,
            "energy_laplacian_squared_sum": 0.0,
        }

    adjacency_eigenvalues = np.linalg.eigvalsh(A)
    adjacency_eigenvalues_sorted = np.sort(
        adjacency_eigenvalues
    )[::-1]

    largest_adj = float(
        adjacency_eigenvalues_sorted[0]
    ) if n >= 1 else 0.0

    second_largest_adj = float(
        adjacency_eigenvalues_sorted[1]
    ) if n >= 2 else 0.0

    trace_adjacency = float(np.trace(A))
    energy_adjacency_squared_sum = float(
        np.sum(adjacency_eigenvalues ** 2)
    )

    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A

    inv_sqrt_degree = np.zeros_like(
        degrees,
        dtype=np.float64,
    )
    nonzero_degree_mask = degrees > 0
    inv_sqrt_degree[nonzero_degree_mask] = (
        1.0 / np.sqrt(degrees[nonzero_degree_mask])
    )

    L_norm = (
        inv_sqrt_degree[:, None]
        * L
        * inv_sqrt_degree[None, :]
    )

    laplacian_eigenvalues = np.sort(
        np.real(np.linalg.eigvalsh(L_norm))
    )

    algebraic_connectivity = (
        float(laplacian_eigenvalues[1])
        if n >= 2
        else 0.0
    )

    tol = 1e-6

    num_zero = int(
        np.sum(
            np.isclose(
                laplacian_eigenvalues,
                0.0,
                atol=tol,
            )
        )
    )

    num_one = int(
        np.sum(
            np.isclose(
                laplacian_eigenvalues,
                1.0,
                atol=tol,
            )
        )
    )

    num_two = int(
        np.sum(
            np.isclose(
                laplacian_eigenvalues,
                2.0,
                atol=tol,
            )
        )
    )

    lower_segment = laplacian_eigenvalues[
        (laplacian_eigenvalues > tol)
        & (laplacian_eigenvalues < 1.0 - tol)
    ]

    upper_segment = laplacian_eigenvalues[
        (laplacian_eigenvalues > 1.0 + tol)
        & (laplacian_eigenvalues < 2.0 - tol)
    ]

    lower_slope = _slope_for_eigen_segment(
        lower_segment
    )
    upper_slope = _slope_for_eigen_segment(
        upper_segment
    )

    trace_laplacian = float(
        np.sum(laplacian_eigenvalues)
    )
    energy_laplacian_squared_sum = float(
        np.sum(laplacian_eigenvalues ** 2)
    )

    return {
        "largest_eigenvalue_adjacency": largest_adj,
        "second_largest_eigenvalue_adjacency": second_largest_adj,
        "trace_adjacency": trace_adjacency,
        "energy_adjacency_squared_sum": energy_adjacency_squared_sum,
        "algebraic_connectivity": algebraic_connectivity,
        "num_laplacian_eigenvalues_zero": num_zero,
        "laplacian_lower_slope_0_to_1": lower_slope,
        "num_laplacian_eigenvalues_one": num_one,
        "laplacian_upper_slope_1_to_2": upper_slope,
        "num_laplacian_eigenvalues_two": num_two,
        "trace_laplacian": trace_laplacian,
        "energy_laplacian_squared_sum": energy_laplacian_squared_sum,
    }


def _additional_topology_features(
    graph: nx.Graph,
    adjacency_soft_no_selfloops: np.ndarray,
    degrees: np.ndarray,
    clustering_values: np.ndarray,
    num_components: int,
) -> Dict[str, float]:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return {
            "degree_assortativity": 0.0,
            "cycle_rank_normalized": 0.0,
            "bridge_ratio": 0.0,
            "articulation_point_ratio": 0.0,
            "max_core_number_normalized": 0.0,
            "mean_core_number_normalized": 0.0,
            "triangle_participation_ratio": 0.0,
            "community_modularity": 0.0,
            "degree_entropy": 0.0,
            "soft_edge_weight_entropy": 0.0,
        }

    if n_edges > 0:
        assortativity = nx.degree_assortativity_coefficient(
            graph
        )
        degree_assortativity = (
            float(assortativity)
            if np.isfinite(assortativity)
            else 0.0
        )

        num_bridges = sum(
            1 for _ in nx.bridges(graph)
        )
        num_articulation_points = len(
            list(nx.articulation_points(graph))
        )

        core_numbers = np.asarray(
            list(nx.core_number(graph).values()),
            dtype=np.float64,
        )

        try:
            if hasattr(
                nx.community,
                "louvain_communities",
            ):
                communities = (
                    nx.community.louvain_communities(
                        graph,
                        seed=101,
                    )
                )
            else:
                communities = (
                    nx.community.greedy_modularity_communities(
                        graph
                    )
                )

            community_modularity = float(
                nx.community.modularity(
                    graph,
                    communities,
                )
            )
        except Exception:
            community_modularity = 0.0
    else:
        degree_assortativity = 0.0
        num_bridges = 0
        num_articulation_points = 0
        core_numbers = np.zeros(
            n_nodes,
            dtype=np.float64,
        )
        community_modularity = 0.0

    cycle_rank = max(
        n_edges - n_nodes + num_components,
        0,
    )

    possible_edges = n_nodes * (n_nodes - 1) // 2
    maximum_cycle_rank = max(
        possible_edges - n_nodes + 1,
        1,
    )

    cycle_rank_normalized = (
        float(cycle_rank / maximum_cycle_rank)
        if n_nodes >= 3
        else 0.0
    )

    bridge_ratio = (
        float(num_bridges / n_edges)
        if n_edges > 0
        else 0.0
    )

    articulation_point_ratio = float(
        num_articulation_points / n_nodes
    )

    degree_denominator = max(
        n_nodes - 1,
        1,
    )

    max_core_number_normalized = float(
        np.max(core_numbers) / degree_denominator
    )

    mean_core_number_normalized = float(
        np.mean(core_numbers) / degree_denominator
    )

    triangle_participation_ratio = float(
        np.mean(clustering_values > 0)
    )

    _, degree_counts = np.unique(
        degrees,
        return_counts=True,
    )
    degree_probabilities = (
        degree_counts.astype(np.float64) / n_nodes
    )

    degree_entropy_raw = float(
        -np.sum(
            degree_probabilities
            * np.log(
                np.clip(
                    degree_probabilities,
                    1e-12,
                    1.0,
                )
            )
        )
    )

    degree_entropy = (
        float(degree_entropy_raw / np.log(n_nodes))
        if n_nodes > 1
        else 0.0
    )

    if n_nodes > 1:
        offdiag_weights = adjacency_soft_no_selfloops[
            np.triu_indices(n_nodes, k=1)
        ]

        probabilities = np.clip(
            offdiag_weights,
            1e-12,
            1.0 - 1e-12,
        )

        soft_edge_weight_entropy = float(
            np.mean(
                -(
                    probabilities * np.log2(probabilities)
                    + (1.0 - probabilities)
                    * np.log2(1.0 - probabilities)
                )
            )
        )
    else:
        soft_edge_weight_entropy = 0.0

    return {
        "degree_assortativity": degree_assortativity,
        "cycle_rank_normalized": cycle_rank_normalized,
        "bridge_ratio": bridge_ratio,
        "articulation_point_ratio": articulation_point_ratio,
        "max_core_number_normalized": max_core_number_normalized,
        "mean_core_number_normalized": mean_core_number_normalized,
        "triangle_participation_ratio": triangle_participation_ratio,
        "community_modularity": community_modularity,
        "degree_entropy": degree_entropy,
        "soft_edge_weight_entropy": soft_edge_weight_entropy,
    }


def compute_graph_features(
    adjacency_soft_no_selfloops: np.ndarray,
    adjacency_binary: np.ndarray,
) -> Dict[str, float | int]:
    graph = nx.from_numpy_array(adjacency_binary)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    possible_undirected_edges = (
        n_nodes * (n_nodes - 1) // 2
    )

    degrees = np.fromiter(
        (
            degree
            for _, degree in graph.degree()
        ),
        dtype=np.float64,
        count=n_nodes,
    )

    weighted_degrees = (
        adjacency_soft_no_selfloops.sum(axis=1)
    )

    clustering_values = np.asarray(
        list(nx.clustering(graph).values()),
        dtype=np.float64,
    )

    component_sizes = (
        sorted(
            (
                len(component)
                for component
                in nx.connected_components(graph)
            ),
            reverse=True,
        )
        if n_nodes > 0
        else []
    )

    giant_component_size = (
        component_sizes[0]
        if component_sizes
        else 0
    )

    second_largest_component_size = (
        component_sizes[1]
        if len(component_sizes) > 1
        else 0
    )

    num_components = len(component_sizes)
    isolated_nodes = (
        int(np.count_nonzero(degrees == 0))
        if n_nodes
        else 0
    )
    end_points = (
        int(np.count_nonzero(degrees == 1))
        if n_nodes
        else 0
    )
    nontrivial_components = int(
        sum(
            size > 1
            for size in component_sizes
        )
    )

    if possible_undirected_edges > 0:
        upper_triangle = np.triu_indices(
            n_nodes,
            k=1,
        )
        offdiag_weights = (
            adjacency_soft_no_selfloops[
                upper_triangle
            ]
        )

        edge_density = (
            n_edges
            / possible_undirected_edges
        )

        weighted_edge_sum = float(
            offdiag_weights.sum()
        )

        weighted_edge_density = (
            weighted_edge_sum
            / possible_undirected_edges
        )

        soft_mean = float(
            np.mean(offdiag_weights)
        )
        soft_median = float(
            np.median(offdiag_weights)
        )
        soft_std = float(
            np.std(offdiag_weights)
        )
        soft_min = float(
            np.min(offdiag_weights)
        )
        soft_max = float(
            np.max(offdiag_weights)
        )
    else:
        edge_density = 0.0
        weighted_edge_sum = 0.0
        weighted_edge_density = 0.0
        soft_mean = 0.0
        soft_median = 0.0
        soft_std = 0.0
        soft_min = 0.0
        soft_max = 0.0

    avg_degree = (
        float(degrees.mean())
        if n_nodes
        else 0.0
    )

    avg_degree_normalized = (
        avg_degree / (n_nodes - 1)
        if n_nodes > 1
        else 0.0
    )

    median_degree = (
        float(np.median(degrees))
        if n_nodes
        else 0.0
    )

    median_degree_normalized = (
        median_degree / (n_nodes - 1)
        if n_nodes > 1
        else 0.0
    )

    std_degree = (
        float(np.std(degrees))
        if n_nodes
        else 0.0
    )

    std_degree_normalized = (
        std_degree / (n_nodes - 1)
        if n_nodes > 1
        else 0.0
    )

    max_degree = (
        float(np.max(degrees))
        if n_nodes
        else 0.0
    )

    max_degree_normalized = (
        max_degree / (n_nodes - 1)
        if n_nodes > 1
        else 0.0
    )

    if n_nodes > 0:
        avg_clustering = float(
            np.mean(clustering_values)
        )
        median_clustering = float(
            np.median(clustering_values)
        )
        std_clustering = float(
            np.std(clustering_values)
        )
    else:
        avg_clustering = 0.0
        median_clustering = 0.0
        std_clustering = 0.0

    base_features = {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "possible_undirected_edges": int(
            possible_undirected_edges
        ),

        "edge_density": float(edge_density),

        "avg_degree": avg_degree,
        "avg_degree_normalized": float(
            avg_degree_normalized
        ),

        "median_degree": median_degree,
        "median_degree_normalized": float(
            median_degree_normalized
        ),

        "std_degree": std_degree,
        "std_degree_normalized": float(
            std_degree_normalized
        ),

        "max_degree": max_degree,
        "max_degree_normalized": float(
            max_degree_normalized
        ),

        "min_degree": (
            float(np.min(degrees))
            if n_nodes
            else 0.0
        ),

        "mean_weighted_degree": (
            float(weighted_degrees.mean())
            if n_nodes
            else 0.0
        ),

        "weighted_edge_sum": weighted_edge_sum,
        "weighted_edge_density": float(
            weighted_edge_density
        ),

        "clustering_coefficient_mean": avg_clustering,
        "clustering_coefficient_median": (
            median_clustering
        ),
        "clustering_coefficient_std": std_clustering,

        "avg_clustering": avg_clustering,
        "transitivity": (
            float(nx.transitivity(graph))
            if n_nodes
            else 0.0
        ),

        "giant_component_size": int(
            giant_component_size
        ),
        "giant_component_ratio": (
            float(
                giant_component_size / n_nodes
            )
            if n_nodes
            else 0.0
        ),

        "num_components": int(num_components),
        "nontrivial_components": int(
            nontrivial_components
        ),

        "second_largest_component_size": int(
            second_largest_component_size
        ),
        "second_largest_component_ratio": (
            float(
                second_largest_component_size
                / n_nodes
            )
            if n_nodes
            else 0.0
        ),

        "isolated_nodes": int(isolated_nodes),
        "isolated_node_ratio": (
            float(isolated_nodes / n_nodes)
            if n_nodes
            else 0.0
        ),
        "percentage_isolated_points": (
            float(isolated_nodes / n_nodes)
            if n_nodes
            else 0.0
        ),

        "end_points": int(end_points),
        "percentage_end_points": (
            float(end_points / n_nodes)
            if n_nodes
            else 0.0
        ),

        "soft_mean_offdiag_weight": soft_mean,
        "soft_median_offdiag_weight": soft_median,
        "soft_std_offdiag_weight": soft_std,
        "soft_min_offdiag_weight": soft_min,
        "soft_max_offdiag_weight": soft_max,
    }

    topology_features = _additional_topology_features(
        graph=graph,
        adjacency_soft_no_selfloops=(
            adjacency_soft_no_selfloops
        ),
        degrees=degrees,
        clustering_values=clustering_values,
        num_components=num_components,
    )

    distance_features = _distance_features(
        adjacency_binary
    )
    spectral_features = _spectral_features(
        adjacency_binary
    )

    base_features.update(topology_features)
    base_features.update(distance_features)
    base_features.update(spectral_features)

    return base_features



# Inference Over Test Set

@torch.no_grad()
def extract_features_from_test_set(
    dataset: CRCGraphDataset,
    gcn: DenseGCN,
    generator: AdjacencyGenerator,
    cfg: Config,
    device: torch.device,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    remove_positions = set(cfg.remove_test_positions)

    unknown_positions = sorted(
        position for position in remove_positions
        if position < 0 or position >= len(cfg.test_indices)
    )

    if unknown_positions:
        raise IndexError(
            "remove_test_positions contains positions outside test set: "
            f"{unknown_positions}"
        )

    rows: List[Dict[str, float | int | str | bool]] = []
    soft_adjacencies: List[np.ndarray] = []
    binary_adjacencies: List[np.ndarray] = []

    retained_index = 0

    for test_position, dataset_index in enumerate(cfg.test_indices):
        if test_position in remove_positions:
            print(
                f"Skipping test position {test_position:03d} "
                f"(dataset index {dataset_index:03d})"
            )
            continue

        graph = dataset[dataset_index]
        x = graph.x.to(device=device, dtype=torch.float32)
        true_label = int(graph.y.view(-1)[0].item())

        if x.size(0) == 0:
            raise ValueError(f"Dataset index {dataset_index} contains zero nodes.")

        if x.size(1) != cfg.input_dim:
            raise ValueError(
                f"Dataset index {dataset_index} has x shape {tuple(x.shape)}, "
                f"but model expects {cfg.input_dim} features."
            )

        adjacency_model = obtain_learned_adjacency(generator, x, cfg)

        logits = gcn(x, adjacency_model)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_label = int(torch.argmax(probabilities).item())

        adjacency_np = adjacency_model.detach().cpu().numpy()

        adjacency_clean, adjacency_binary = clean_and_binarize_adjacency(
            adjacency_np,
            threshold=cfg.edge_threshold,
        )

        features = compute_graph_features(
            adjacency_soft_no_selfloops=adjacency_clean,
            adjacency_binary=adjacency_binary,
        )

        class_name = CLASS_NAMES.get(true_label, f"Class {true_label}")

        row: Dict[str, float | int | str | bool] = {
            "retained_index": retained_index,
            "test_position": test_position,
            "dataset_index": dataset_index,
            "true_label": true_label,
            "class": class_name,
            "pred_label": predicted_label,
            "pred_class": CLASS_NAMES.get(
                predicted_label,
                f"Class {predicted_label}",
            ),
            "correct": bool(predicted_label == true_label),
            "edge_threshold": float(cfg.edge_threshold),
            "adjacency_mode": cfg.adjacency_mode,
        }

        for class_index in range(cfg.num_classes):
            row[f"prob_class_{class_index}"] = float(
                probabilities[class_index].item()
            )

        row.update(features)

        rows.append(row)
        soft_adjacencies.append(adjacency_clean.astype(np.float32))
        binary_adjacencies.append(adjacency_binary.astype(np.uint8))

        print(
            f"[{retained_index + 1:03d}] dataset={dataset_index:03d} | "
            f"class={class_name:<10} | "
            f"nodes={features['n_nodes']:4d} | "
            f"edges={features['n_edges']:7d} | "
            f"density={features['edge_density']:.4f} | "
            f"GCR={features['giant_component_ratio']:.4f} | "
            f"eff={features['global_efficiency']:.4f} | "
            f"true={true_label} pred={predicted_label}"
        )

        retained_index += 1

    if not rows:
        raise RuntimeError("No test graphs retained after exclusion.")

    features_df = pd.DataFrame(rows)

    features_df["class"] = pd.Categorical(
        features_df["class"],
        categories=CLASS_ORDER,
        ordered=True,
    )

    return features_df, soft_adjacencies, binary_adjacencies



# Feature Lists

GRAPH_FEATURE_COLUMNS: Tuple[str, ...] = (
    # Basic
    "n_nodes",
    "n_edges",
    "possible_undirected_edges",

    # Existing useful metrics
    "edge_density",
    "avg_degree",
    "avg_degree_normalized",
    "median_degree",
    "median_degree_normalized",
    "std_degree",
    "std_degree_normalized",
    "max_degree",
    "max_degree_normalized",
    "min_degree",

    # Weighted soft adjacency
    "mean_weighted_degree",
    "weighted_edge_sum",
    "weighted_edge_density",
    "soft_mean_offdiag_weight",
    "soft_median_offdiag_weight",
    "soft_std_offdiag_weight",
    "soft_min_offdiag_weight",
    "soft_max_offdiag_weight",
    "soft_edge_weight_entropy",

    # Additional topology
    "degree_assortativity",
    "cycle_rank_normalized",
    "bridge_ratio",
    "articulation_point_ratio",
    "max_core_number_normalized",
    "mean_core_number_normalized",
    "triangle_participation_ratio",
    "community_modularity",
    "degree_entropy",

    # Connectedness / cliquishness
    "clustering_coefficient_mean",
    "clustering_coefficient_median",
    "clustering_coefficient_std",
    "avg_clustering",
    "transitivity",
    "giant_component_size",
    "giant_component_ratio",
    "num_components",
    "nontrivial_components",
    "second_largest_component_size",
    "second_largest_component_ratio",
    "isolated_nodes",
    "isolated_node_ratio",
    "percentage_isolated_points",
    "end_points",
    "percentage_end_points",

    # Distance / shortest path
    "avg_eccentricity",
    "diameter",
    "radius",
    "avg_eccentricity_90",
    "diameter_90",
    "radius_90",
    "avg_shortest_path_reachable",
    "global_efficiency",
    "num_central_points",
    "percent_central_points",

    # Spectral
    "largest_eigenvalue_adjacency",
    "second_largest_eigenvalue_adjacency",
    "trace_adjacency",
    "energy_adjacency_squared_sum",
    "algebraic_connectivity",
    "num_laplacian_eigenvalues_zero",
    "laplacian_lower_slope_0_to_1",
    "num_laplacian_eigenvalues_one",
    "laplacian_upper_slope_1_to_2",
    "num_laplacian_eigenvalues_two",
    "trace_laplacian",
    "energy_laplacian_squared_sum",
)


PLOT_LABELS: Dict[str, str] = {
    "n_nodes": "Number of Nodes",
    "n_edges": "Number of Edges",
    "possible_undirected_edges": "Possible Undirected Edges",

    "edge_density": "Edge Density",
    "avg_degree": "Average Degree",
    "avg_degree_normalized": "Average Degree Normalized",
    "median_degree": "Median Degree",
    "median_degree_normalized": "Median Degree Normalized",
    "std_degree": "Degree Standard Deviation",
    "std_degree_normalized": "Degree Standard Deviation Normalized",
    "max_degree": "Maximum Degree",
    "max_degree_normalized": "Maximum Degree Normalized",
    "min_degree": "Minimum Degree",

    "mean_weighted_degree": "Mean Weighted Degree",
    "weighted_edge_sum": "Weighted Edge Sum",
    "weighted_edge_density": "Weighted Edge Density",
    "soft_mean_offdiag_weight": "Mean Off-Diagonal Edge Weight",
    "soft_median_offdiag_weight": "Median Off-Diagonal Edge Weight",
    "soft_std_offdiag_weight": "Std Off-Diagonal Edge Weight",
    "soft_min_offdiag_weight": "Minimum Off-Diagonal Edge Weight",
    "soft_max_offdiag_weight": "Maximum Off-Diagonal Edge Weight",
    "soft_edge_weight_entropy": "Soft Edge-Weight Entropy",

    "degree_assortativity": "Degree Assortativity",
    "cycle_rank_normalized": "Normalized Cycle Rank",
    "bridge_ratio": "Bridge Ratio",
    "articulation_point_ratio": "Articulation-Point Ratio",
    "max_core_number_normalized": "Maximum Core Number Normalized",
    "mean_core_number_normalized": "Mean Core Number Normalized",
    "triangle_participation_ratio": "Triangle Participation Ratio",
    "community_modularity": "Community Modularity",
    "degree_entropy": "Degree Entropy",

    "clustering_coefficient_mean": "Mean Clustering Coefficient",
    "clustering_coefficient_median": "Median Clustering Coefficient",
    "clustering_coefficient_std": "Std Clustering Coefficient",
    "avg_clustering": "Average Clustering",
    "transitivity": "Transitivity",

    "giant_component_size": "Giant Component Size",
    "giant_component_ratio": "Giant Component Ratio",
    "num_components": "Number of Connected Components",
    "nontrivial_components": "Nontrivial Components",
    "second_largest_component_size": "Second Largest Component Size",
    "second_largest_component_ratio": "Second Largest Component Ratio",
    "isolated_nodes": "Isolated Nodes",
    "isolated_node_ratio": "Isolated Node Ratio",
    "percentage_isolated_points": "Percentage of Isolated Points",
    "end_points": "End Points",
    "percentage_end_points": "Percentage of End Points",

    "avg_eccentricity": "Average Eccentricity",
    "diameter": "Diameter",
    "radius": "Radius",
    "avg_eccentricity_90": "90% Average Eccentricity",
    "diameter_90": "90% Diameter",
    "radius_90": "90% Radius",
    "avg_shortest_path_reachable": "Average Reachable Shortest Path",
    "global_efficiency": "Global Efficiency",
    "num_central_points": "Number of Central Points",
    "percent_central_points": "Percent Central Points",

    "largest_eigenvalue_adjacency": "Largest Adjacency Eigenvalue",
    "second_largest_eigenvalue_adjacency": "Second Largest Adjacency Eigenvalue",
    "trace_adjacency": "Trace of Adjacency",
    "energy_adjacency_squared_sum": "Adjacency Energy",
    "algebraic_connectivity": "Algebraic Connectivity",
    "num_laplacian_eigenvalues_zero": "Laplacian Eigenvalues at 0",
    "laplacian_lower_slope_0_to_1": "Laplacian Lower Slope",
    "num_laplacian_eigenvalues_one": "Laplacian Eigenvalues at 1",
    "laplacian_upper_slope_1_to_2": "Laplacian Upper Slope",
    "num_laplacian_eigenvalues_two": "Laplacian Eigenvalues at 2",
    "trace_laplacian": "Trace of Laplacian",
    "energy_laplacian_squared_sum": "Laplacian Energy",
}



# Summary And Tests

def make_class_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    available_features = [
        column for column in GRAPH_FEATURE_COLUMNS
        if column in features_df.columns
    ]

    return (
        features_df.groupby("class", observed=False)[available_features]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .round(6)
    )


def run_kruskal_wallis_tests(features_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in GRAPH_FEATURE_COLUMNS:
        if feature not in features_df.columns:
            continue

        groups = [
            features_df.loc[
                features_df["class"] == class_name,
                feature,
            ].dropna().to_numpy(dtype=float)
            for class_name in CLASS_ORDER
        ]

        if any(len(group) == 0 for group in groups):
            statistic, p_value = np.nan, np.nan

        else:
            try:
                statistic, p_value = kruskal(*groups)
            except ValueError:
                statistic, p_value = 0.0, 1.0

        rows.append(
            {
                "feature": feature,
                "kruskal_H": float(statistic),
                "p_value": float(p_value),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("p_value", na_position="last")
        .reset_index(drop=True)
    )


def run_pairwise_mannwhitney_tests(features_df: pd.DataFrame) -> pd.DataFrame:
    class_pairs = (
        ("Normal", "Low grade"),
        ("Normal", "High grade"),
        ("Low grade", "High grade"),
    )

    rows = []

    for feature in GRAPH_FEATURE_COLUMNS:
        if feature not in features_df.columns:
            continue

        for class_1, class_2 in class_pairs:
            values_1 = features_df.loc[
                features_df["class"] == class_1,
                feature,
            ].dropna().to_numpy(dtype=float)

            values_2 = features_df.loc[
                features_df["class"] == class_2,
                feature,
            ].dropna().to_numpy(dtype=float)

            if len(values_1) == 0 or len(values_2) == 0:
                statistic, p_value = np.nan, np.nan

            else:
                statistic, p_value = mannwhitneyu(
                    values_1,
                    values_2,
                    alternative="two-sided",
                )

            rows.append(
                {
                    "feature": feature,
                    "class_1": class_1,
                    "class_2": class_2,
                    "mannwhitney_U": float(statistic),
                    "p_value": float(p_value),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["feature", "p_value"], na_position="last")
        .reset_index(drop=True)
    )



# Plotting

def plot_feature_boxplots(
    features_df: pd.DataFrame,
    metrics: Sequence[str],
    output_dir: Path,
    metrics_per_page: int = 6,
    show_plots: bool = True,
    save_plots: bool = True,
) -> None:
    if not show_plots and not save_plots:
        return

    plot_dir = output_dir / "extended_boxplots"

    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    available_metrics = [
        metric for metric in metrics
        if metric in features_df.columns
    ]

    missing_metrics = [
        metric for metric in metrics
        if metric not in features_df.columns
    ]

    print("\nAvailable metrics:", len(available_metrics))
    print("Missing metrics:", missing_metrics)

    for page_start in range(0, len(available_metrics), metrics_per_page):
        current_metrics = available_metrics[page_start:page_start + metrics_per_page]

        n_cols = 3
        n_rows = math.ceil(len(current_metrics) / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.3 * n_cols, 4.3 * n_rows),
        )

        axes = np.asarray(axes).reshape(-1)

        for ax, metric in zip(axes, current_metrics):
            grouped_values = [
                features_df.loc[
                    features_df["class"] == class_name,
                    metric,
                ].dropna().to_numpy(dtype=float)
                for class_name in CLASS_ORDER
            ]

            ax.boxplot(
                grouped_values,
                labels=CLASS_ORDER,
                showfliers=True,
            )

            label = PLOT_LABELS.get(metric, metric.replace("_", " ").title())

            ax.set_title(label)
            ax.set_xlabel("Histology class")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

        for ax in axes[len(current_metrics):]:
            ax.axis("off")

        fig.tight_layout()

        page_number = page_start // metrics_per_page + 1

        if save_plots:
            fig.savefig(
                plot_dir / f"extended_graph_features_page_{page_number:02d}.png",
                dpi=300,
                bbox_inches="tight",
            )

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    if save_plots:
        print("\nSaved extended boxplots to:")
        print(plot_dir)


def save_example_heatmaps(
    features_df: pd.DataFrame,
    soft_adjacencies: Sequence[np.ndarray],
    output_dir: Path,
) -> None:
    heatmap_dir = output_dir / "example_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_ORDER:
        class_rows = features_df.index[features_df["class"] == class_name]

        if len(class_rows) == 0:
            continue

        row_index = int(class_rows[0])
        adjacency = soft_adjacencies[row_index]
        dataset_index = int(features_df.loc[row_index, "dataset_index"])

        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(adjacency, aspect="auto")

        fig.colorbar(image, ax=ax, label="Learned edge probability")

        ax.set_title(
            f"{class_name}: learned adjacency, dataset index {dataset_index}"
        )
        ax.set_xlabel("Patch node")
        ax.set_ylabel("Patch node")

        fig.tight_layout()

        safe_class_name = class_name.lower().replace(" ", "_")

        fig.savefig(
            heatmap_dir / f"{safe_class_name}_dataset_{dataset_index:03d}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)



# Save Outputs

def save_outputs(
    cfg: Config,
    features_df: pd.DataFrame,
    soft_adjacencies: Sequence[np.ndarray],
    binary_adjacencies: Sequence[np.ndarray],
    class_summary: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(
        output_dir / "extended_learned_graph_features_per_image.csv",
        index=False,
    )

    class_summary.to_csv(
        output_dir / "extended_learned_graph_features_class_summary.csv",
    )

    kruskal_df.to_csv(
        output_dir / "extended_kruskal_wallis_tests.csv",
        index=False,
    )

    pairwise_df.to_csv(
        output_dir / "extended_pairwise_mannwhitney_tests.csv",
        index=False,
    )

    run_summary = {
        "num_retained_graphs": int(len(features_df)),
        "num_correct": int(features_df["correct"].sum()),
        "accuracy": float(features_df["correct"].mean()),
        "class_counts": {
            str(key): int(value)
            for key, value in features_df["class"]
            .value_counts(sort=False)
            .items()
        },
        "config": asdict(cfg),
    }

    with open(output_dir / "extended_run_summary.json", "w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2, default=str)

    with open(
        output_dir / "removed_test_positions.txt",
        "w",
        encoding="utf-8",
    ) as file:
        for position in sorted(cfg.remove_test_positions):
            file.write(f"{position}\n")

    if cfg.save_combined_matrix_pickles:
        with open(output_dir / "soft_adjacencies_no_selfloops.pkl", "wb") as file:
            pickle.dump(list(soft_adjacencies), file)

        with open(output_dir / "binary_adjacencies.pkl", "wb") as file:
            pickle.dump(list(binary_adjacencies), file)

    if cfg.save_individual_matrices:
        matrix_dir = output_dir / "individual_matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)

        for row_position, row in features_df.reset_index(drop=True).iterrows():
            dataset_index = int(row["dataset_index"])
            test_position = int(row["test_position"])
            class_name = str(row["class"]).lower().replace(" ", "_")

            filename = (
                f"test_{test_position:03d}_dataset_{dataset_index:03d}_"
                f"{class_name}.npz"
            )

            np.savez_compressed(
                matrix_dir / filename,
                adjacency_soft_no_selfloops=soft_adjacencies[row_position],
                adjacency_binary=binary_adjacencies[row_position],
                test_position=test_position,
                dataset_index=dataset_index,
                true_label=int(row["true_label"]),
                pred_label=int(row["pred_label"]),
                edge_threshold=float(cfg.edge_threshold),
            )

    plot_feature_boxplots(
        features_df=features_df,
        metrics=GRAPH_FEATURE_COLUMNS,
        output_dir=output_dir,
        metrics_per_page=cfg.metrics_per_plot_page,
        show_plots=cfg.show_plots,
        save_plots=cfg.save_plots,
    )

    if cfg.save_plots:
        save_example_heatmaps(features_df, soft_adjacencies, output_dir)

    print("\nSaved all extended outputs to:")
    print(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract graph features from a trained ABiG-Net checkpoint."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="PyG dataset root containing processed/data.pt.",
    )
    parser.add_argument(
        "--raw-csv-dir",
        type=Path,
        default=None,
        help="Directory containing the raw CSV files. Defaults to dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint containing gcn_state and generator_state.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV files, matrices, and plots.",
    )
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument(
        "--test-stop",
        type=int,
        default=100,
        help="Exclusive upper bound for dataset indices.",
    )
    parser.add_argument(
        "--exclude-test-positions",
        type=int,
        nargs="*",
        default=[],
        help="Positions within the selected test-index list to skip.",
    )
    parser.add_argument("--input-dim", type=int, default=69)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--generator-hidden-dim1", type=int, default=32)
    parser.add_argument("--generator-hidden-dim2", type=int, default=16)
    parser.add_argument(
        "--adjacency-mode",
        choices=("deterministic", "mc_gumbel"),
        default="deterministic",
    )
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--edge-threshold", type=float, default=0.50)
    parser.add_argument("--pairwise-row-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--metrics-per-plot-page", type=int, default=6)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots while the script runs.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not save boxplots or heatmaps.",
    )
    parser.add_argument(
        "--skip-individual-matrices",
        action="store_true",
        help="Do not save one NPZ file per graph.",
    )
    parser.add_argument(
        "--skip-matrix-pickles",
        action="store_true",
        help="Do not save combined adjacency pickle files.",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_config(args: argparse.Namespace) -> Config:
    if args.test_start < 0:
        raise ValueError("test-start must be non-negative.")
    if args.test_stop <= args.test_start:
        raise ValueError("test-stop must be greater than test-start.")
    if not 0.0 <= args.edge_threshold <= 1.0:
        raise ValueError("edge-threshold must be between 0 and 1.")
    if args.pairwise_row_chunk_size <= 0:
        raise ValueError("pairwise-row-chunk-size must be positive.")

    raw_csv_dir = args.raw_csv_dir or args.dataset_root

    return Config(
        dataset_root=args.dataset_root,
        raw_csv_dir=raw_csv_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        test_indices=list(range(args.test_start, args.test_stop)),
        remove_test_positions=list(args.exclude_test_positions),
        seed=args.seed,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        generator_hidden_dim1=args.generator_hidden_dim1,
        generator_hidden_dim2=args.generator_hidden_dim2,
        adjacency_mode=args.adjacency_mode,
        mc_samples=args.mc_samples,
        edge_threshold=args.edge_threshold,
        pairwise_row_chunk_size=args.pairwise_row_chunk_size,
        save_individual_matrices=not args.skip_individual_matrices,
        save_combined_matrix_pickles=not args.skip_matrix_pickles,
        show_plots=args.show_plots,
        save_plots=not args.skip_plots,
        metrics_per_plot_page=args.metrics_per_plot_page,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)
    device = select_device(args.device)

    print(f"Using device: {device}")
    print(f"Adjacency mode: {cfg.adjacency_mode}")
    print(f"Edge threshold: {cfg.edge_threshold}")

    dataset = CRCGraphDataset(
        root=str(cfg.dataset_root),
        raw_csv_dir=str(cfg.raw_csv_dir),
    )
    print(f"Loaded {len(dataset)} graphs.")
    validate_dataset(dataset, cfg)

    gcn, generator = load_models(cfg, device)
    print("Checkpoint loaded successfully.")

    features_df, soft_adjacencies, binary_adjacencies = extract_features_from_test_set(
        dataset=dataset,
        gcn=gcn,
        generator=generator,
        cfg=cfg,
        device=device,
    )

    class_summary = make_class_summary(features_df)
    kruskal_df = run_kruskal_wallis_tests(features_df)
    pairwise_df = run_pairwise_mannwhitney_tests(features_df)

    print("\n" + "=" * 72)
    print("GRAPH FEATURE SUMMARY")
    print("=" * 72)
    print(f"Retained test graphs: {len(features_df)}")
    print(f"Classification accuracy: {features_df['correct'].mean():.4f}")
    print("Class counts:")
    print(features_df["class"].value_counts(sort=False))

    selected_columns = [
        "edge_density",
        "median_degree",
        "std_degree",
        "soft_mean_offdiag_weight",
        "giant_component_ratio",
        "isolated_node_ratio",
        "num_components",
        "avg_eccentricity",
        "diameter",
        "largest_eigenvalue_adjacency",
        "energy_laplacian_squared_sum",
        "degree_assortativity",
        "global_efficiency",
        "algebraic_connectivity",
        "cycle_rank_normalized",
        "bridge_ratio",
        "articulation_point_ratio",
        "max_core_number_normalized",
        "mean_core_number_normalized",
        "triangle_participation_ratio",
        "community_modularity",
        "degree_entropy",
        "soft_edge_weight_entropy",
    ]
    selected_columns = [
        column for column in selected_columns if column in features_df.columns
    ]

    print("\nSelected feature means by class:")
    print(
        features_df.groupby("class", observed=False)[selected_columns]
        .mean()
        .round(4)
    )

    save_outputs(
        cfg=cfg,
        features_df=features_df,
        soft_adjacencies=soft_adjacencies,
        binary_adjacencies=binary_adjacencies,
        class_summary=class_summary,
        kruskal_df=kruskal_df,
        pairwise_df=pairwise_df,
    )


if __name__ == "__main__":
    main()
