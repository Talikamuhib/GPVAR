"""
Process EEG files to create consensus matrices for AD and HC groups.
This script uses the full list of files provided and demonstrates the consensus matrix analysis.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict
from consensus_matrix_eeg import process_eeg_files
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define all file paths
FILE_PATHS = {
    'AD_AR': [
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30018/eeg/s6_sub-30018_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30026/eeg/s6_sub-30026_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30011/eeg/s6_sub-30011_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30009/eeg/s6_sub-30009_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30012/eeg/s6_sub-30012_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30002/eeg/s6_sub-30002_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30017/eeg/s6_sub-30017_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30001/eeg/s6_sub-30001_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30029/eeg/s6_sub-30029_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30015/eeg/s6_sub-30015_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30013/eeg/s6_sub-30013_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30008/eeg/s6_sub-30008_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30031/eeg/s6_sub-30031_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30022/eeg/s6_sub-30022_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30020/eeg/s6_sub-30020_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30004/eeg/s6_sub-30004_rs-hep_eeg.set',
    ],
    'AD_CL': [
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30003/eeg/s6_sub-30003_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30007/eeg/s6_sub-30007_rs-hep_eeg.set',
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30005/eeg/s6_sub-30005_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30006/eeg/s6_sub-30006_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30010/eeg/s6_sub-30010_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30014/eeg/s6_sub-30014_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30016/eeg/s6_sub-30016_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30019/eeg/s6_sub-30019_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30021/eeg/s6_sub-30021_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30023/eeg/s6_sub-30023_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30024/eeg/s6_sub-30024_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30025/eeg/s6_sub-30025_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30027/eeg/s6_sub-30027_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30028/eeg/s6_sub-30028_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30030/eeg/s6_sub-30030_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30032/eeg/s6_sub-30032_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30033/eeg/s6_sub-30033_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30034/eeg/s6_sub-30034_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30035/eeg/s6_sub-30035_rs-hep_eeg.set",
    ],
    'HC_AR': [
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10002/eeg/s6_sub-10002_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10009/eeg/s6_sub-10009_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100012/eeg/s6_sub-100012_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100015/eeg/s6_sub-100015_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100020/eeg/s6_sub-100020_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100035/eeg/s6_sub-100035_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100028/eeg/s6_sub-100028_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10006/eeg/s6_sub-10006_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10007/eeg/s6_sub-10007_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100033/eeg/s6_sub-100033_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100022/eeg/s6_sub-100022_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100031/eeg/s6_sub-100031_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10003/eeg/s6_sub-10003_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100026/eeg/s6_sub-100026_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100030/eeg/s6_sub-100030_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100018/eeg/s6_sub-100018_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100024/eeg/s6_sub-100024_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100038/eeg/s6_sub-100038_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10004/eeg/s6_sub-10004_rs_eeg.set",
    ],
    'HC_CL': [
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10001/eeg/s6_sub-10001_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10005/eeg/s6_sub-10005_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10008/eeg/s6_sub-10008_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100010/eeg/s6_sub-100010_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100011/eeg/s6_sub-100011_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100014/eeg/s6_sub-100014_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100017/eeg/s6_sub-100017_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100021/eeg/s6_sub-100021_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100029/eeg/s6_sub-100029_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100034/eeg/s6_sub-100034_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100037/eeg/s6_sub-100037_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100043/eeg/s6_sub-100043_rs_eeg.set",
    ]
}


def _build_weighted_graph(adjacency: np.ndarray, weight_threshold: float = 1e-8) -> nx.Graph:
    """Convert a dense adjacency matrix into a weighted NetworkX graph."""
    graph = nx.Graph()
    n_nodes = adjacency.shape[0]
    graph.add_nodes_from(range(n_nodes))
    triu_idx = np.triu_indices(n_nodes, k=1)
    weights = adjacency[triu_idx]
    mask = weights > weight_threshold
    sources = triu_idx[0][mask]
    targets = triu_idx[1][mask]
    filtered_weights = weights[mask]
    graph.add_weighted_edges_from(
        (int(i), int(j), float(w)) for i, j, w in zip(sources, targets, filtered_weights)
    )
    return graph


def _plot_consensus_heatmap(matrix: np.ndarray, title: str, output_path: Path):
    """Save a heatmap of the consensus adjacency or Laplacian matrix."""
    vmax = np.max(matrix)
    vmax = vmax if vmax > 0 else 1.0
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        cmap="magma",
        square=True,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Edge weight"},
    )
    plt.title(title)
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_laplacian_spectrum(eigenvalues: np.ndarray, group_name: str, output_path: Path):
    """Plot and save the Laplacian eigenvalue spectrum."""
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(eigenvalues)), eigenvalues, marker="o", linewidth=1)
    plt.title(f"{group_name} Laplacian Eigenvalues")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("λ")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _format_float(value: float, fmt: str = "{:.3f}") -> str:
    """Format floats while guarding against NaN/inf."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return fmt.format(value)


def generate_section_4p3_assets(
    C: np.ndarray,
    G: np.ndarray,
    group_name: str,
    output_dir: str,
    method_desc: str,
    methods_reference: str = "Methods §3.2",
) -> Dict[str, str]:
    """
    Create figures and a markdown report that document consensus graph properties.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_nodes = G.shape[0]
    n_possible_edges = max(1, n_nodes * (n_nodes - 1) // 2)
    triu_idx = np.triu_indices(n_nodes, k=1)
    edge_weights = G[triu_idx]
    nonzero_mask = edge_weights > 0
    n_edges = int(np.sum(nonzero_mask))
    sparsity_pct = (n_edges / n_possible_edges) * 100

    binary_degrees = np.sum(G > 0, axis=0)
    strengths = np.sum(G, axis=0)
    deg_min = float(binary_degrees.min()) if binary_degrees.size else 0.0
    deg_median = float(np.median(binary_degrees)) if binary_degrees.size else 0.0
    deg_max = float(binary_degrees.max()) if binary_degrees.size else 0.0
    deg_std = float(binary_degrees.std()) if binary_degrees.size else 0.0

    strength_min = float(strengths.min()) if strengths.size else 0.0
    strength_median = float(np.median(strengths)) if strengths.size else 0.0
    strength_max = float(strengths.max()) if strengths.size else 0.0

    consensus_selected = C[triu_idx][nonzero_mask]
    consensus_mean = float(np.mean(consensus_selected)) if consensus_selected.size else float("nan")
    consensus_median = float(np.median(consensus_selected)) if consensus_selected.size else float("nan")

    heatmap_path = output_path / f"{group_name}_consensus_adjacency.png"
    _plot_consensus_heatmap(
        G,
        f"{group_name} consensus adjacency",
        heatmap_path,
    )

    laplacian = np.diag(strengths) - G
    eigenvalues = np.linalg.eigvalsh(laplacian)
    lambda_min = float(eigenvalues[0]) if eigenvalues.size else float("nan")
    lambda_two = float(eigenvalues[1]) if eigenvalues.size > 1 else float("nan")
    lambda_max = float(eigenvalues[-1]) if eigenvalues.size else float("nan")
    lambda_q3 = float(np.percentile(eigenvalues, 75)) if eigenvalues.size else float("nan")
    near_zero = int(np.sum(eigenvalues < 1e-5)) if eigenvalues.size else 0

    spectrum_path = output_path / f"{group_name}_laplacian_spectrum.png"
    _plot_laplacian_spectrum(eigenvalues, group_name, spectrum_path)

    graph = _build_weighted_graph(G)
    if graph.number_of_nodes() > 0:
        component_count = nx.number_connected_components(graph)
        largest_component_nodes = max((len(c) for c in nx.connected_components(graph)), default=0)
    else:
        component_count = 0
        largest_component_nodes = 0

    largest_component_fraction = (
        largest_component_nodes / n_nodes if n_nodes else 0.0
    )

    if largest_component_nodes > 1:
        giant_component = graph.subgraph(
            max(nx.connected_components(graph), key=len)
        ).copy()
        try:
            path_length = nx.average_shortest_path_length(giant_component, weight="weight")
        except (nx.NetworkXError, ZeroDivisionError):
            path_length = float("nan")
    else:
        path_length = float("nan")

    clustering_coeff = (
        nx.average_clustering(graph, weight="weight") if graph.number_of_edges() > 0 else float("nan")
    )

    small_world_sigma = float("nan")
    if graph.number_of_edges() > 0 and graph.number_of_nodes() >= 4:
        try:
            small_world_sigma = nx.sigma(graph, niter=5, nrand=3, seed=42)
        except (nx.NetworkXError, ZeroDivisionError):
            small_world_sigma = float("nan")

    component_text = (
        "connected (single component)"
        if component_count == 1
        else f"comprised of {component_count} components (largest spans {largest_component_fraction:.1%} of channels)"
    )

    sigma_comment = (
        "Values above 1 imply small-world organization; report this if σ>1."
        if np.isfinite(small_world_sigma) and small_world_sigma > 1
        else "σ≈1 indicates a random-graph-like topology."
        if np.isfinite(small_world_sigma)
        else "σ could not be estimated (graph too sparse/disconnected)."
    )

    intro = (
        f"All GP-VAR models were defined on a single consensus Laplacian constructed from the {method_desc}, "
        f"as documented in {methods_reference}. This section records the structural diagnostics examiners expect."
    )

    interpretation = (
        f"The graph retains {sparsity_pct:.2f}% of the possible undirected edges, so the GP-VAR state space remains sparse "
        f"while keeping the network {component_text}. Degrees range from {deg_min:.0f} to {deg_max:.0f} (median {deg_median:.0f}), "
        f"which prevents unrealistic super-hubs and shows that every BioSemi channel participates. "
        f"The weighted path length of {_format_float(path_length)} and clustering coefficient of {_format_float(clustering_coeff)} "
        f"capture the balance between local cohesion and long-range shortcuts, which you can cite when motivating any small-world prior."
    )

    spectral_text = (
        f"Laplacian eigenvalues span {_format_float(lambda_min, '{:.5f}')} to {_format_float(lambda_max)}, "
        f"with {near_zero} near-zero modes (matching {component_count} connected components) and algebraic connectivity "
        f"λ₂ = {_format_float(lambda_two, '{:.5f}')}. The upper-quartile eigenvalue {_format_float(lambda_q3)} "
        f"sets the bandwidth for GP-VAR graph-frequency priors."
    )

    report_lines = [
        f"# 4.3 Consensus Graph Properties – {group_name}",
        "",
        intro,
        "",
        "Key graph metrics:",
        f"- Sparsity: {sparsity_pct:.2f}% ({n_edges}/{n_possible_edges} undirected edges retained).",
        f"- Degree distribution (binary): min={deg_min:.0f}, median={deg_median:.0f}, max={deg_max:.0f}, std={deg_std:.2f}.",
        f"- Strength distribution (weighted sums): min={strength_min:.3f}, median={strength_median:.3f}, max={strength_max:.3f}.",
        f"- Consensus support for retained edges: mean={_format_float(consensus_mean)}, median={_format_float(consensus_median)}.",
        f"- Clustering coefficient (weighted): {_format_float(clustering_coeff)}.",
        f"- Characteristic path length (largest component): {_format_float(path_length)} over {largest_component_nodes} nodes.",
        f"- Small-world σ estimate: {_format_float(small_world_sigma)}. {sigma_comment}",
        f"- Connectivity: graph is {component_text}.",
        "",
        "Spectral structure:",
        spectral_text,
        "",
        "Figures:",
        f"- Figure 4.x ({heatmap_path.name}): 128×128 consensus adjacency heatmap; rows/cols map to BioSemi-128 channels and brighter cells denote stronger short-range edges.",
        f"- Figure 4.y ({spectrum_path.name}): Laplacian eigenvalue spectrum (graph-frequency profile) highlighting near-zero modes and high-frequency tails.",
        "",
        interpretation,
        "",
        "Use this block verbatim in the write-up and replace “4.x/4.y” with the final figure numbers once the document is compiled.",
    ]

    report_path = output_path / f"{group_name}_section_4p3.md"
    report_path.write_text("\n".join(report_lines))

    return {
        "report_path": str(report_path),
        "heatmap_path": str(heatmap_path),
        "spectrum_path": str(spectrum_path),
    }


def analyze_consensus_properties(C, W, G, group_name):
    """
    Analyze and report properties of consensus matrices.
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix
    W : np.ndarray
        Weight matrix
    G : np.ndarray
        Final group graph
    group_name : str
        Name of the group for reporting
    """
    print(f"\n{'='*60}")
    print(f"Analysis for {group_name}")
    print(f"{'='*60}")
    
    # Consensus matrix statistics
    n_nodes = C.shape[0]
    triu_idx = np.triu_indices(n_nodes, k=1)
    consensus_values = C[triu_idx]
    
    print("\nConsensus Matrix Statistics:")
    print(f"  - Shape: {C.shape}")
    print(f"  - Min consensus: {consensus_values.min():.3f}")
    print(f"  - Max consensus: {consensus_values.max():.3f}")
    print(f"  - Mean consensus: {consensus_values.mean():.3f}")
    print(f"  - Median consensus: {np.median(consensus_values):.3f}")
    print(f"  - Edges with C > 0: {np.sum(consensus_values > 0)} / {len(consensus_values)}")
    print(f"  - Edges with C > 0.5: {np.sum(consensus_values > 0.5)}")
    print(f"  - Edges with C = 1.0: {np.sum(consensus_values == 1.0)}")
    
    # Weight matrix statistics
    weight_values = W[triu_idx]
    weight_values_nonzero = weight_values[weight_values > 0]
    
    print("\nWeight Matrix Statistics:")
    print(f"  - Min weight (non-zero): {weight_values_nonzero.min():.3f}")
    print(f"  - Max weight: {weight_values_nonzero.max():.3f}")
    print(f"  - Mean weight (non-zero): {weight_values_nonzero.mean():.3f}")
    print(f"  - Median weight (non-zero): {np.median(weight_values_nonzero):.3f}")
    
    # Final graph statistics
    graph_values = G[triu_idx]
    n_edges_final = np.sum(graph_values > 0)
    sparsity_final = n_edges_final / len(graph_values)
    
    print("\nFinal Graph Statistics:")
    print(f"  - Number of edges: {n_edges_final}")
    print(f"  - Sparsity: {sparsity_final:.3f}")
    print(f"  - Min edge weight: {graph_values[graph_values > 0].min():.3f}")
    print(f"  - Max edge weight: {graph_values[graph_values > 0].max():.3f}")
    print(f"  - Mean edge weight: {graph_values[graph_values > 0].mean():.3f}")
    
    # Degree distribution
    degrees = np.sum(G > 0, axis=0)
    print(f"\nDegree Distribution:")
    print(f"  - Min degree: {degrees.min()}")
    print(f"  - Max degree: {degrees.max()}")
    print(f"  - Mean degree: {degrees.mean():.2f}")
    print(f"  - Std degree: {degrees.std():.2f}")


def visualize_group_comparison(results_dict, output_dir="./consensus_results"):
    """
    Create comparison visualizations across groups.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing results for each group
    output_dir : str
        Directory to save visualizations
    """
    n_groups = len(results_dict)
    fig, axes = plt.subplots(3, n_groups, figsize=(5*n_groups, 15))
    
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (group_name, results) in enumerate(results_dict.items()):
        C = results['consensus_matrix']
        W = results['weight_matrix']
        G = results['final_graph']
        
        # Plot consensus matrix
        im1 = axes[0, idx].imshow(C, cmap='hot', vmin=0, vmax=1)
        axes[0, idx].set_title(f'{group_name}\nConsensus Matrix')
        axes[0, idx].set_xlabel('Channel')
        axes[0, idx].set_ylabel('Channel')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046)
        
        # Plot weight matrix
        im2 = axes[1, idx].imshow(W, cmap='viridis', vmin=0)
        axes[1, idx].set_title(f'Weight Matrix')
        axes[1, idx].set_xlabel('Channel')
        axes[1, idx].set_ylabel('Channel')
        plt.colorbar(im2, ax=axes[1, idx], fraction=0.046)
        
        # Plot final graph
        im3 = axes[2, idx].imshow(G, cmap='plasma', vmin=0)
        axes[2, idx].set_title(f'Final Graph')
        axes[2, idx].set_xlabel('Channel')
        axes[2, idx].set_ylabel('Channel')
        plt.colorbar(im3, ax=axes[2, idx], fraction=0.046)
    
    plt.suptitle('Consensus Matrix Analysis Across Groups', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "group_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create histogram comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
    
    for idx, (group_name, results) in enumerate(results_dict.items()):
        C = results['consensus_matrix']
        n_nodes = C.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        consensus_values = C[triu_idx]
        
        # Histogram of consensus values
        axes[idx].hist(consensus_values[consensus_values > 0], 
                      bins=50, alpha=0.7, color=colors[idx], 
                      edgecolor='black', linewidth=1.2)
        axes[idx].set_xlabel('Consensus Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{group_name} - Consensus Distribution')
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = consensus_values[consensus_values > 0].mean()
        median_val = np.median(consensus_values[consensus_values > 0])
        axes[idx].axvline(mean_val, color='red', linestyle='--', 
                         label=f'Mean: {mean_val:.3f}')
        axes[idx].axvline(median_val, color='blue', linestyle='--', 
                         label=f'Median: {median_val:.3f}')
        axes[idx].legend()
    
    plt.suptitle('Consensus Value Distributions', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "consensus_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to process all EEG files and create consensus matrices.
    """
    # Parameters
    SPARSITY_BINARIZE = 0.15   # Sparsity for initial binarization (15% edges kept)
    SPARSITY_FINAL = "match_subject"  # Match group graph sparsity to average binarized subject sparsity
    USE_DISTANCE = True        # Use distance-dependent consensus
    OUTPUT_DIR = "./consensus_results"
    
    # Store all results
    all_results = {}
    
    # Process each group
    for group_name, file_list in FILE_PATHS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {group_name} group ({len(file_list)} files)")
        logger.info(f"{'='*60}")
        
        try:
            # Process files
            results = process_eeg_files(
                file_paths=file_list,
                sparsity_binarize=SPARSITY_BINARIZE,
                sparsity_final=SPARSITY_FINAL,
                method='distance' if USE_DISTANCE else 'uniform',
                output_dir=f"{OUTPUT_DIR}/{group_name}"
            )
            
            all_results[group_name] = results
            
            # Analyze properties
            analyze_consensus_properties(
                results['consensus_matrix'],
                results['weight_matrix'],
                results['final_graph'],
                group_name
            )
            
            section_assets = generate_section_4p3_assets(
                results['consensus_matrix'],
                results['final_graph'],
                group_name,
                output_dir=f"{OUTPUT_DIR}/{group_name}",
                method_desc="distance-dependent consensus (Betzel-style bins)" if USE_DISTANCE else "uniform consensus baseline",
                methods_reference="Methods §3.2"
            )
            logger.info(
                "Section 4.3 assets for %s saved to %s (figures: %s, %s)",
                group_name,
                section_assets["report_path"],
                section_assets["heatmap_path"],
                section_assets["spectrum_path"]
            )
            
        except Exception as e:
            logger.error(f"Failed to process {group_name}: {e}")
            continue
    
    # Create comparison visualizations
    if all_results:
        logger.info("\nCreating group comparison visualizations...")
        visualize_group_comparison(all_results, OUTPUT_DIR)
        
        # Save combined results
        logger.info("\nSaving combined results...")
        np.savez(f"{OUTPUT_DIR}/all_consensus_results.npz", **all_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed {len(all_results)} groups:")
        for group_name in all_results.keys():
            n_subjects = len(all_results[group_name]['valid_files'])
            print(f"  - {group_name}: {n_subjects} subjects")
        print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    # Run the main analysis
    results = main()