from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests


CLASS_ORDER = ("Normal", "Low grade", "High grade")

ORIGINAL_33_METRICS = (
    "soft_std_offdiag_weight",
    "percentage_end_points",
    "soft_median_offdiag_weight",
    "second_largest_component_ratio",
    "edge_density",
    "avg_degree_normalized",
    "median_degree_normalized",
    "transitivity",
    "isolated_node_ratio",
    "giant_component_ratio",
    "weighted_edge_density",
    "soft_mean_offdiag_weight",
    "num_components",
    "num_laplacian_eigenvalues_zero",
    "max_degree_normalized",
    "laplacian_lower_slope_0_to_1",
    "laplacian_upper_slope_1_to_2",
    "clustering_coefficient_mean",
    "num_laplacian_eigenvalues_one",
    "avg_eccentricity_90",
    "soft_max_offdiag_weight",
    "radius",
    "radius_90",
    "avg_shortest_path_reachable",
    "percent_central_points",
    "clustering_coefficient_median",
    "diameter_90",
    "std_degree_normalized",
    "min_degree",
    "diameter",
    "clustering_coefficient_std",
    "num_laplacian_eigenvalues_two",
    "avg_eccentricity",
)

NEW_12_TOPOLOGY_METRICS = (
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
)

REMOVED_METRICS = {
    "avg_degree_normalized": "exact duplicate of edge_density",
    "soft_mean_offdiag_weight": "exact duplicate of weighted_edge_density",
    "num_laplacian_eigenvalues_zero": "same graph property as num_components",
    "bridge_ratio": "excluded from the curated analysis",
    "second_largest_component_ratio": "excluded from the curated analysis",
    "soft_max_offdiag_weight": "excluded from the curated analysis",
    "max_degree_normalized": "excluded from the curated analysis",
    "soft_median_offdiag_weight": "excluded from the curated analysis",
}

ALL_45_METRICS = ORIGINAL_33_METRICS + NEW_12_TOPOLOGY_METRICS
ANALYSIS_METRICS = tuple(
    metric for metric in ALL_45_METRICS if metric not in REMOVED_METRICS
)

if len(ALL_45_METRICS) != 45:
    raise RuntimeError(f"Expected 45 initial metrics, found {len(ALL_45_METRICS)}.")
if len(ANALYSIS_METRICS) != 37:
    raise RuntimeError(f"Expected 37 retained metrics, found {len(ANALYSIS_METRICS)}.")
if len(set(ANALYSIS_METRICS)) != len(ANALYSIS_METRICS):
    raise RuntimeError("Duplicate metric names remain in the curated feature list.")

PLOT_LABELS = {
    "soft_std_offdiag_weight": "Std Off-Diagonal Edge Weight",
    "percentage_end_points": "Percentage of End Points",
    "edge_density": "Edge Density",
    "median_degree_normalized": "Median Degree Normalized",
    "transitivity": "Transitivity",
    "isolated_node_ratio": "Isolated Node Ratio",
    "giant_component_ratio": "Giant Component Ratio",
    "weighted_edge_density": "Weighted Edge Density",
    "num_components": "Number of Connected Components",
    "laplacian_lower_slope_0_to_1": "Laplacian Lower Slope",
    "laplacian_upper_slope_1_to_2": "Laplacian Upper Slope",
    "clustering_coefficient_mean": "Mean Clustering Coefficient",
    "num_laplacian_eigenvalues_one": "Laplacian Eigenvalues at 1",
    "avg_eccentricity_90": "90% Average Eccentricity",
    "radius": "Radius",
    "radius_90": "90% Radius",
    "avg_shortest_path_reachable": "Average Reachable Shortest Path",
    "percent_central_points": "Percent Central Points",
    "clustering_coefficient_median": "Median Clustering Coefficient",
    "diameter_90": "90% Diameter",
    "std_degree_normalized": "Degree Standard Deviation Normalized",
    "min_degree": "Minimum Degree",
    "diameter": "Diameter",
    "clustering_coefficient_std": "Std Clustering Coefficient",
    "num_laplacian_eigenvalues_two": "Laplacian Eigenvalues at 2",
    "avg_eccentricity": "Average Eccentricity",
    "degree_assortativity": "Degree Assortativity",
    "global_efficiency": "Global Efficiency",
    "algebraic_connectivity": "Algebraic Connectivity",
    "cycle_rank_normalized": "Normalized Cycle Rank",
    "articulation_point_ratio": "Articulation-Point Ratio",
    "max_core_number_normalized": "Maximum Core Number Normalized",
    "mean_core_number_normalized": "Mean Core Number Normalized",
    "triangle_participation_ratio": "Triangle Participation Ratio",
    "community_modularity": "Community Modularity",
    "degree_entropy": "Degree Entropy",
    "soft_edge_weight_entropy": "Soft Edge-Weight Entropy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Kruskal-Wallis tests and FDR correction to graph features."
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="CSV produced by extract_learned_graph_features.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--metrics-per-page", type=int, default=6)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_feature_table(feature_file: Path) -> pd.DataFrame:
    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature CSV was not found: {feature_file}\n"
            "Run the graph-feature extraction script first."
        )

    dataframe = pd.read_csv(feature_file)
    required_columns = ["class", *ANALYSIS_METRICS]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        formatted = "\n".join(f"  - {column}" for column in missing_columns)
        raise KeyError(f"The CSV is missing required columns:\n{formatted}")

    dataframe["class"] = dataframe["class"].astype(str).str.strip()
    unexpected_classes = sorted(
        set(dataframe["class"].dropna().unique()) - set(CLASS_ORDER)
    )
    if unexpected_classes:
        formatted = "\n".join(f"  - {name}" for name in unexpected_classes)
        raise ValueError(f"Unexpected class labels:\n{formatted}")

    dataframe["class"] = pd.Categorical(
        dataframe["class"], categories=CLASS_ORDER, ordered=True
    )
    for metric in ANALYSIS_METRICS:
        dataframe[metric] = pd.to_numeric(
            dataframe[metric], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    return dataframe


def calculate_kruskal_fdr(
    dataframe: pd.DataFrame,
    fdr_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for metric in ANALYSIS_METRICS:
        class_values = {
            class_name: dataframe.loc[
                dataframe["class"] == class_name, metric
            ].dropna().to_numpy(dtype=float)
            for class_name in CLASS_ORDER
        }
        groups = [class_values[class_name] for class_name in CLASS_ORDER]

        if any(len(group) == 0 for group in groups):
            h_statistic = np.nan
            raw_p_value = np.nan
        else:
            combined_values = np.concatenate(groups)
            if np.allclose(combined_values, combined_values[0]):
                h_statistic = 0.0
                raw_p_value = 1.0
            else:
                h_statistic, raw_p_value = kruskal(*groups, nan_policy="omit")

        rows.append(
            {
                "metric": metric,
                "metric_label": PLOT_LABELS.get(
                    metric, metric.replace("_", " ").title()
                ),
                "n_normal": len(class_values["Normal"]),
                "n_low_grade": len(class_values["Low grade"]),
                "n_high_grade": len(class_values["High grade"]),
                "kruskal_H": float(h_statistic),
                "kruskal_p_value": float(raw_p_value),
            }
        )

    results = pd.DataFrame(rows)
    results["kruskal_fdr_p_value"] = np.nan
    valid_mask = results["kruskal_p_value"].notna()

    if valid_mask.any():
        adjusted = multipletests(
            results.loc[valid_mask, "kruskal_p_value"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
        results.loc[valid_mask, "kruskal_fdr_p_value"] = adjusted

    results["significant_after_fdr"] = (
        results["kruskal_fdr_p_value"] < fdr_threshold
    )
    results = results.sort_values(
        ["kruskal_fdr_p_value", "kruskal_p_value"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    return results


def plot_metric_pages(
    dataframe: pd.DataFrame,
    results: pd.DataFrame,
    metrics: Sequence[str],
    plot_dir: Path,
    filename_prefix: str,
    metrics_per_page: int,
    show_plots: bool,
) -> None:
    if not metrics:
        print(f"No metrics available for {filename_prefix}.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    result_lookup = results.set_index("metric")

    for page_start in range(0, len(metrics), metrics_per_page):
        current_metrics = metrics[page_start : page_start + metrics_per_page]
        n_columns = 3
        n_rows = math.ceil(len(current_metrics) / n_columns)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(5.4 * n_columns, 4.8 * n_rows),
        )
        axes = np.asarray(axes).reshape(-1)

        for ax, metric in zip(axes, current_metrics):
            grouped_values = [
                dataframe.loc[
                    dataframe["class"] == class_name, metric
                ].dropna().to_numpy(dtype=float)
                for class_name in CLASS_ORDER
            ]
            try:
                ax.boxplot(
                    grouped_values,
                    tick_labels=CLASS_ORDER,
                    showfliers=True,
                )
            except TypeError:
                ax.boxplot(
                    grouped_values,
                    labels=CLASS_ORDER,
                    showfliers=True,
                )

            result_row = result_lookup.loc[metric]
            raw_p = float(result_row["kruskal_p_value"])
            fdr_p = float(result_row["kruskal_fdr_p_value"])
            significance = (
                "significant"
                if bool(result_row["significant_after_fdr"])
                else "not significant"
            )
            label = PLOT_LABELS.get(metric, metric.replace("_", " ").title())

            ax.set_title(
                f"{label}\nraw p={raw_p:.4g}, FDR p={fdr_p:.4g} "
                f"({significance})",
                fontsize=9.5,
            )
            ax.set_xlabel("Histology class")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

        for ax in axes[len(current_metrics) :]:
            ax.axis("off")

        fig.tight_layout()
        page_number = page_start // metrics_per_page + 1
        fig.savefig(
            plot_dir / f"{filename_prefix}_page_{page_number:02d}.png",
            dpi=300,
            bbox_inches="tight",
        )

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved {filename_prefix} plots to: {plot_dir}")


def save_results(
    output_dir: Path,
    results: pd.DataFrame,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    compact_results = results[
        [
            "rank",
            "metric",
            "kruskal_p_value",
            "kruskal_fdr_p_value",
            "significant_after_fdr",
        ]
    ].copy()
    significant_results = results.loc[
        results["significant_after_fdr"]
    ].copy()

    compact_results.to_csv(
        output_dir / "full_37_features_true_false.csv", index=False
    )
    results.to_csv(
        output_dir / "all_37_kruskal_fdr_detailed_results.csv", index=False
    )
    significant_results.to_csv(
        output_dir / "significant_37_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {"removed_metric": metric, "reason": reason}
            for metric, reason in REMOVED_METRICS.items()
        ]
    ).to_csv(output_dir / "removed_metrics.csv", index=False)

    return significant_results["metric"].tolist()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fdr_threshold < 1.0:
        raise ValueError("fdr-threshold must be between 0 and 1.")
    if args.metrics_per_page <= 0:
        raise ValueError("metrics-per-page must be positive.")

    features = load_feature_table(args.feature_file)
    results = calculate_kruskal_fdr(features, args.fdr_threshold)
    significant_metrics = save_results(args.output_dir, results)

    compact_results = results[
        [
            "rank",
            "metric",
            "kruskal_p_value",
            "kruskal_fdr_p_value",
            "significant_after_fdr",
        ]
    ]

    print("=" * 96)
    print("KRUSKAL-WALLIS TESTS WITH BENJAMINI-HOCHBERG FDR CORRECTION")
    print("=" * 96)
    print(
        compact_results.round(
            {"kruskal_p_value": 6, "kruskal_fdr_p_value": 6}
        ).to_string(index=False)
    )
    print(f"\nFeatures tested: {len(compact_results)}")
    print(f"Significant after FDR correction: {len(significant_metrics)}")
    print("\nClass counts:")
    print(features["class"].value_counts(sort=False))

    if significant_metrics:
        print("\nSignificant features:")
        for number, metric in enumerate(significant_metrics, start=1):
            print(f"{number}. {metric}")
    else:
        print("\nNo feature remained significant after FDR correction.")

    if not args.skip_plots:
        plot_metric_pages(
            dataframe=features,
            results=results,
            metrics=ANALYSIS_METRICS,
            plot_dir=args.output_dir / "all_37_feature_boxplots",
            filename_prefix="all_37_features",
            metrics_per_page=args.metrics_per_page,
            show_plots=args.show_plots,
        )
        plot_metric_pages(
            dataframe=features,
            results=results,
            metrics=significant_metrics,
            plot_dir=args.output_dir / "significant_feature_boxplots",
            filename_prefix="significant_features",
            metrics_per_page=args.metrics_per_page,
            show_plots=args.show_plots,
        )

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
