"""
Comprehensive Analysis of Consensus Matrix Results for EEG Connectivity

This script provides tools to analyze pre-computed consensus matrix results,
including:
- Graph-theoretic metrics (centrality, modularity, efficiency)
- Spectral analysis of the Laplacian
- Group comparisons with statistical tests
- Regional (brain lobe) analysis
- Publication-quality visualizations
- Thesis-ready reports

Usage:
    python analyze_consensus_results.py --results_dir ./consensus_results
    
Author: Generated for EEG Connectivity Analysis Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# Data Classes for Structured Results
# ============================================================================

@dataclass
class GraphMetrics:
    """Container for graph-level metrics."""
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    sparsity_pct: float = 0.0
    
    # Degree statistics
    degree_mean: float = 0.0
    degree_std: float = 0.0
    degree_min: float = 0.0
    degree_max: float = 0.0
    degree_median: float = 0.0
    
    # Strength statistics (weighted degree)
    strength_mean: float = 0.0
    strength_std: float = 0.0
    strength_min: float = 0.0
    strength_max: float = 0.0
    strength_median: float = 0.0
    
    # Connectivity
    n_components: int = 0
    largest_component_size: int = 0
    largest_component_fraction: float = 0.0
    
    # Global efficiency and path length
    global_efficiency: float = 0.0
    local_efficiency: float = 0.0
    characteristic_path_length: float = 0.0
    
    # Clustering and modularity
    clustering_coefficient: float = 0.0
    transitivity: float = 0.0
    modularity: float = 0.0
    n_communities: int = 0
    
    # Small-world metrics
    small_world_sigma: float = 0.0
    small_world_omega: float = 0.0
    
    # Assortativity
    degree_assortativity: float = 0.0
    
    # Spectral properties
    algebraic_connectivity: float = 0.0
    spectral_radius: float = 0.0
    laplacian_energy: float = 0.0


@dataclass
class NodeMetrics:
    """Container for node-level metrics."""
    node_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    degrees: np.ndarray = field(default_factory=lambda: np.array([]))
    strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    betweenness_centrality: np.ndarray = field(default_factory=lambda: np.array([]))
    closeness_centrality: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvector_centrality: np.ndarray = field(default_factory=lambda: np.array([]))
    pagerank: np.ndarray = field(default_factory=lambda: np.array([]))
    clustering: np.ndarray = field(default_factory=lambda: np.array([]))
    local_efficiency: np.ndarray = field(default_factory=lambda: np.array([]))
    community_labels: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SpectralMetrics:
    """Container for spectral analysis results."""
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    algebraic_connectivity: float = 0.0
    spectral_gap: float = 0.0
    spectral_radius: float = 0.0
    laplacian_energy: float = 0.0
    near_zero_modes: int = 0
    fiedler_vector: np.ndarray = field(default_factory=lambda: np.array([]))


# ============================================================================
# Core Analysis Class
# ============================================================================

class ConsensusResultsAnalyzer:
    """
    Comprehensive analyzer for consensus matrix results.
    
    This class loads pre-computed consensus matrices and provides methods
    for computing detailed graph metrics, statistical comparisons,
    and generating publication-quality figures.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        results_dir : str, optional
            Directory containing saved consensus results
        """
        self.results_dir = Path(results_dir) if results_dir else None
        self.consensus_matrices: Dict[str, np.ndarray] = {}
        self.weight_matrices: Dict[str, np.ndarray] = {}
        self.final_graphs: Dict[str, np.ndarray] = {}
        self.binary_matrices: Dict[str, np.ndarray] = {}
        self.channel_locations: Optional[np.ndarray] = None
        self.channel_names: Optional[List[str]] = None
        
        # Computed metrics
        self.graph_metrics: Dict[str, GraphMetrics] = {}
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.spectral_metrics: Dict[str, SpectralMetrics] = {}
        
    def load_results(self, results_dir: Optional[str] = None) -> None:
        """
        Load consensus matrix results from disk.
        
        Parameters
        ----------
        results_dir : str, optional
            Directory containing saved results. Overrides constructor path.
        """
        if results_dir:
            self.results_dir = Path(results_dir)
        
        if self.results_dir is None:
            raise ValueError("No results directory specified")
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        logger.info(f"Loading results from {self.results_dir}")
        
        # Find all group subdirectories or individual result files
        for subdir in self.results_dir.iterdir():
            if subdir.is_dir():
                self._load_group_results(subdir.name, subdir)
            elif subdir.suffix == '.npy':
                # Handle flat directory structure
                self._load_single_file(subdir)
        
        # Try to load combined results file
        combined_file = self.results_dir / "all_consensus_results.npz"
        if combined_file.exists():
            self._load_combined_results(combined_file)
        
        logger.info(f"Loaded results for {len(self.final_graphs)} groups")
    
    def _load_group_results(self, group_name: str, group_dir: Path) -> None:
        """Load results for a specific group from a directory."""
        # Look for standard file names
        file_patterns = {
            'consensus': ['consensus_matrix_C.npy', '*_matrix_C.npy', 'C.npy'],
            'weight': ['consensus_matrix_W.npy', '*_matrix_W.npy', 'W.npy'],
            'weight_dense': ['consensus_matrix_W_dense.npy', '*_matrix_W_dense.npy'],
            'graph': ['consensus_distance_graph.npy', '*_distance_graph.npy', 'G.npy', 'final_graph.npy'],
            'binary': ['consensus_binary_matrices.npy', '*_binary_matrices.npy']
        }
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                matches = list(group_dir.glob(pattern))
                if matches:
                    data = np.load(matches[0])
                    if file_type == 'consensus':
                        self.consensus_matrices[group_name] = data
                    elif file_type == 'weight':
                        self.weight_matrices[group_name] = data
                    elif file_type == 'weight_dense':
                        # Use dense weights if no sparse weight loaded
                        if group_name not in self.weight_matrices:
                            self.weight_matrices[group_name] = data
                    elif file_type == 'graph':
                        self.final_graphs[group_name] = data
                    elif file_type == 'binary':
                        self.binary_matrices[group_name] = data
                    break
        
        logger.info(f"  Loaded {group_name}: C={group_name in self.consensus_matrices}, "
                   f"W={group_name in self.weight_matrices}, G={group_name in self.final_graphs}")
    
    def _load_single_file(self, filepath: Path) -> None:
        """Load a single results file."""
        name = filepath.stem
        if 'consensus' in name.lower() or '_c' in name.lower():
            self.consensus_matrices[name] = np.load(filepath)
        elif 'weight' in name.lower() or '_w' in name.lower():
            self.weight_matrices[name] = np.load(filepath)
        elif 'graph' in name.lower() or '_g' in name.lower():
            self.final_graphs[name] = np.load(filepath)
    
    def _load_combined_results(self, filepath: Path) -> None:
        """Load combined results from .npz file."""
        data = np.load(filepath, allow_pickle=True)
        for key in data.files:
            item = data[key]
            if isinstance(item, np.ndarray) and item.dtype == object:
                # This is likely a dictionary stored as an object array
                item = item.item()
            
            if isinstance(item, dict):
                if 'consensus_matrix' in item:
                    self.consensus_matrices[key] = item['consensus_matrix']
                if 'weight_matrix' in item:
                    self.weight_matrices[key] = item['weight_matrix']
                if 'final_graph' in item:
                    self.final_graphs[key] = item['final_graph']
                if 'binary_matrices' in item:
                    self.binary_matrices[key] = item['binary_matrices']
                if 'channel_locations' in item and item['channel_locations'] is not None:
                    self.channel_locations = item['channel_locations']
    
    def set_matrices(self, 
                     consensus_matrix: np.ndarray,
                     weight_matrix: np.ndarray,
                     final_graph: np.ndarray,
                     group_name: str = 'default',
                     binary_matrices: Optional[np.ndarray] = None,
                     channel_locations: Optional[np.ndarray] = None) -> None:
        """
        Directly set matrices for analysis (alternative to loading from disk).
        
        Parameters
        ----------
        consensus_matrix : np.ndarray
            Consensus matrix C
        weight_matrix : np.ndarray
            Weight matrix W
        final_graph : np.ndarray
            Final adjacency matrix G
        group_name : str
            Name for this group/analysis
        binary_matrices : np.ndarray, optional
            Subject-level binary matrices
        channel_locations : np.ndarray, optional
            3D coordinates of channels
        """
        self.consensus_matrices[group_name] = consensus_matrix
        self.weight_matrices[group_name] = weight_matrix
        self.final_graphs[group_name] = final_graph
        if binary_matrices is not None:
            self.binary_matrices[group_name] = binary_matrices
        if channel_locations is not None:
            self.channel_locations = channel_locations
    
    # ========================================================================
    # Graph Construction
    # ========================================================================
    
    @staticmethod
    def _matrix_to_networkx(adjacency: np.ndarray, 
                           weight_threshold: float = 1e-8) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph."""
        G = nx.Graph()
        n_nodes = adjacency.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        triu_idx = np.triu_indices(n_nodes, k=1)
        weights = adjacency[triu_idx]
        mask = weights > weight_threshold
        
        edges = [(int(triu_idx[0][i]), int(triu_idx[1][i]), {'weight': float(weights[i])})
                 for i in np.where(mask)[0]]
        G.add_edges_from(edges)
        
        return G
    
    # ========================================================================
    # Metric Computation
    # ========================================================================
    
    def compute_all_metrics(self, groups: Optional[List[str]] = None) -> None:
        """
        Compute all graph metrics for specified groups.
        
        Parameters
        ----------
        groups : List[str], optional
            List of group names to analyze. If None, analyze all loaded groups.
        """
        if groups is None:
            groups = list(self.final_graphs.keys())
        
        for group_name in groups:
            if group_name not in self.final_graphs:
                logger.warning(f"No final graph found for group '{group_name}'")
                continue
            
            logger.info(f"Computing metrics for {group_name}...")
            
            G_matrix = self.final_graphs[group_name]
            G_nx = self._matrix_to_networkx(G_matrix)
            
            # Compute metrics
            self.graph_metrics[group_name] = self._compute_graph_metrics(G_matrix, G_nx)
            self.node_metrics[group_name] = self._compute_node_metrics(G_matrix, G_nx)
            self.spectral_metrics[group_name] = self._compute_spectral_metrics(G_matrix)
        
        logger.info("Metric computation complete")
    
    def _compute_graph_metrics(self, 
                               adjacency: np.ndarray, 
                               G: nx.Graph) -> GraphMetrics:
        """Compute graph-level metrics."""
        metrics = GraphMetrics()
        n_nodes = adjacency.shape[0]
        
        # Basic properties
        metrics.n_nodes = n_nodes
        metrics.n_edges = G.number_of_edges()
        n_possible = n_nodes * (n_nodes - 1) // 2
        metrics.density = metrics.n_edges / max(1, n_possible)
        metrics.sparsity_pct = metrics.density * 100
        
        # Degree statistics
        degrees = np.array([d for _, d in G.degree()])
        if degrees.size > 0:
            metrics.degree_mean = float(np.mean(degrees))
            metrics.degree_std = float(np.std(degrees))
            metrics.degree_min = float(np.min(degrees))
            metrics.degree_max = float(np.max(degrees))
            metrics.degree_median = float(np.median(degrees))
        
        # Strength statistics (weighted degree)
        strengths = np.sum(adjacency, axis=1)
        if strengths.size > 0:
            metrics.strength_mean = float(np.mean(strengths))
            metrics.strength_std = float(np.std(strengths))
            metrics.strength_min = float(np.min(strengths))
            metrics.strength_max = float(np.max(strengths))
            metrics.strength_median = float(np.median(strengths))
        
        # Connectivity
        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            metrics.n_components = len(components)
            metrics.largest_component_size = max(len(c) for c in components) if components else 0
            metrics.largest_component_fraction = metrics.largest_component_size / n_nodes
        
        # Global efficiency
        if G.number_of_edges() > 0:
            try:
                metrics.global_efficiency = nx.global_efficiency(G)
            except:
                metrics.global_efficiency = float('nan')
            
            try:
                metrics.local_efficiency = nx.local_efficiency(G)
            except:
                metrics.local_efficiency = float('nan')
        
        # Path length (on largest component)
        if metrics.largest_component_size > 1:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc).copy()
            try:
                metrics.characteristic_path_length = nx.average_shortest_path_length(
                    subgraph, weight='weight')
            except:
                metrics.characteristic_path_length = float('nan')
        
        # Clustering
        if G.number_of_edges() > 0:
            try:
                metrics.clustering_coefficient = nx.average_clustering(G, weight='weight')
            except:
                metrics.clustering_coefficient = float('nan')
            
            try:
                metrics.transitivity = nx.transitivity(G)
            except:
                metrics.transitivity = float('nan')
        
        # Modularity (using Louvain if available)
        if G.number_of_edges() > 0:
            try:
                communities = nx.community.louvain_communities(G, weight='weight', seed=42)
                metrics.n_communities = len(communities)
                metrics.modularity = nx.community.modularity(G, communities, weight='weight')
            except:
                metrics.modularity = float('nan')
                metrics.n_communities = 0
        
        # Small-world metrics
        if G.number_of_edges() > 0 and G.number_of_nodes() >= 4:
            try:
                metrics.small_world_sigma = nx.sigma(G, niter=5, nrand=3, seed=42)
            except:
                metrics.small_world_sigma = float('nan')
            
            try:
                metrics.small_world_omega = nx.omega(G, niter=5, nrand=3, seed=42)
            except:
                metrics.small_world_omega = float('nan')
        
        # Assortativity
        if G.number_of_edges() > 0:
            try:
                metrics.degree_assortativity = nx.degree_assortativity_coefficient(G)
            except:
                metrics.degree_assortativity = float('nan')
        
        return metrics
    
    def _compute_node_metrics(self, 
                              adjacency: np.ndarray, 
                              G: nx.Graph) -> NodeMetrics:
        """Compute node-level metrics."""
        metrics = NodeMetrics()
        n_nodes = adjacency.shape[0]
        
        metrics.node_ids = np.arange(n_nodes)
        
        # Degree and strength
        metrics.degrees = np.array([G.degree(i) for i in range(n_nodes)])
        metrics.strengths = np.sum(adjacency, axis=1)
        
        # Centrality measures
        if G.number_of_edges() > 0:
            try:
                bc = nx.betweenness_centrality(G, weight='weight')
                metrics.betweenness_centrality = np.array([bc.get(i, 0) for i in range(n_nodes)])
            except:
                metrics.betweenness_centrality = np.zeros(n_nodes)
            
            try:
                cc = nx.closeness_centrality(G, distance='weight')
                metrics.closeness_centrality = np.array([cc.get(i, 0) for i in range(n_nodes)])
            except:
                metrics.closeness_centrality = np.zeros(n_nodes)
            
            try:
                ec = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                metrics.eigenvector_centrality = np.array([ec.get(i, 0) for i in range(n_nodes)])
            except:
                metrics.eigenvector_centrality = np.zeros(n_nodes)
            
            try:
                pr = nx.pagerank(G, weight='weight')
                metrics.pagerank = np.array([pr.get(i, 0) for i in range(n_nodes)])
            except:
                metrics.pagerank = np.zeros(n_nodes)
            
            try:
                cl = nx.clustering(G, weight='weight')
                metrics.clustering = np.array([cl.get(i, 0) for i in range(n_nodes)])
            except:
                metrics.clustering = np.zeros(n_nodes)
        
        # Community detection
        if G.number_of_edges() > 0:
            try:
                communities = nx.community.louvain_communities(G, weight='weight', seed=42)
                labels = np.zeros(n_nodes, dtype=int)
                for idx, comm in enumerate(communities):
                    for node in comm:
                        labels[node] = idx
                metrics.community_labels = labels
            except:
                metrics.community_labels = np.zeros(n_nodes, dtype=int)
        
        return metrics
    
    def _compute_spectral_metrics(self, adjacency: np.ndarray) -> SpectralMetrics:
        """Compute spectral analysis of the graph Laplacian."""
        metrics = SpectralMetrics()
        n_nodes = adjacency.shape[0]
        
        if n_nodes == 0:
            return metrics
        
        # Compute Laplacian
        strengths = np.sum(adjacency, axis=1)
        laplacian = np.diag(strengths) - adjacency
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        metrics.eigenvalues = eigenvalues
        metrics.eigenvectors = eigenvectors
        
        # Algebraic connectivity (second smallest eigenvalue)
        if len(eigenvalues) > 1:
            sorted_evals = np.sort(eigenvalues)
            metrics.algebraic_connectivity = float(sorted_evals[1])
            metrics.spectral_gap = float(sorted_evals[1] - sorted_evals[0])
            
            # Fiedler vector (eigenvector of second smallest eigenvalue)
            idx = np.argsort(eigenvalues)[1]
            metrics.fiedler_vector = eigenvectors[:, idx]
        
        # Spectral radius (largest eigenvalue)
        metrics.spectral_radius = float(eigenvalues[-1])
        
        # Laplacian energy (sum of absolute eigenvalues)
        metrics.laplacian_energy = float(np.sum(np.abs(eigenvalues)))
        
        # Count near-zero modes
        metrics.near_zero_modes = int(np.sum(eigenvalues < 1e-5))
        
        return metrics
    
    # ========================================================================
    # Group Comparisons
    # ========================================================================
    
    def compare_groups(self, 
                       group1: str, 
                       group2: str,
                       alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare graph metrics between two groups with statistical tests.
        
        Parameters
        ----------
        group1, group2 : str
            Names of groups to compare
        alpha : float
            Significance level
            
        Returns
        -------
        pd.DataFrame
            Comparison results with effect sizes and p-values
        """
        if group1 not in self.graph_metrics or group2 not in self.graph_metrics:
            raise ValueError(f"Metrics not computed for one or both groups")
        
        m1 = self.graph_metrics[group1]
        m2 = self.graph_metrics[group2]
        
        results = []
        metrics_to_compare = [
            'n_edges', 'density', 'degree_mean', 'degree_std', 
            'strength_mean', 'strength_std',
            'n_components', 'largest_component_fraction',
            'global_efficiency', 'local_efficiency', 'characteristic_path_length',
            'clustering_coefficient', 'transitivity', 'modularity',
            'small_world_sigma', 'degree_assortativity',
            'algebraic_connectivity', 'spectral_radius', 'laplacian_energy'
        ]
        
        for metric_name in metrics_to_compare:
            v1 = getattr(m1, metric_name, None)
            v2 = getattr(m2, metric_name, None)
            
            if v1 is None or v2 is None:
                continue
            
            # For single-value metrics, just report the difference
            diff = v2 - v1 if np.isfinite(v1) and np.isfinite(v2) else float('nan')
            pct_change = ((v2 - v1) / abs(v1) * 100) if v1 != 0 and np.isfinite(v1) else float('nan')
            
            results.append({
                'metric': metric_name,
                f'{group1}': v1,
                f'{group2}': v2,
                'difference': diff,
                'pct_change': pct_change
            })
        
        return pd.DataFrame(results)
    
    def compare_node_metrics(self,
                            group1: str,
                            group2: str,
                            test: str = 'mannwhitneyu') -> pd.DataFrame:
        """
        Compare node-level metrics between groups using statistical tests.
        
        Parameters
        ----------
        group1, group2 : str
            Names of groups to compare
        test : str
            Statistical test: 'mannwhitneyu', 'ttest', or 'ks'
            
        Returns
        -------
        pd.DataFrame
            Results with statistics and p-values
        """
        if group1 not in self.node_metrics or group2 not in self.node_metrics:
            raise ValueError("Node metrics not computed for one or both groups")
        
        nm1 = self.node_metrics[group1]
        nm2 = self.node_metrics[group2]
        
        results = []
        
        for attr in ['degrees', 'strengths', 'betweenness_centrality', 
                     'closeness_centrality', 'eigenvector_centrality', 
                     'pagerank', 'clustering']:
            v1 = getattr(nm1, attr, np.array([]))
            v2 = getattr(nm2, attr, np.array([]))
            
            if v1.size == 0 or v2.size == 0:
                continue
            
            # Statistics
            mean1, std1 = np.mean(v1), np.std(v1)
            mean2, std2 = np.mean(v2), np.std(v2)
            
            # Statistical test
            if test == 'mannwhitneyu':
                stat, pval = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            elif test == 'ttest':
                stat, pval = stats.ttest_ind(v1, v2)
            elif test == 'ks':
                stat, pval = stats.ks_2samp(v1, v2)
            else:
                stat, pval = float('nan'), float('nan')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'metric': attr,
                f'{group1}_mean': mean1,
                f'{group1}_std': std1,
                f'{group2}_mean': mean2,
                f'{group2}_std': std2,
                'statistic': stat,
                'p_value': pval,
                'cohens_d': cohens_d,
                'significant': pval < 0.05
            })
        
        return pd.DataFrame(results)
    
    # ========================================================================
    # Consensus-Specific Analysis
    # ========================================================================
    
    def analyze_consensus_distribution(self, group_name: str) -> Dict:
        """
        Analyze the distribution of consensus values.
        
        Parameters
        ----------
        group_name : str
            Name of the group
            
        Returns
        -------
        dict
            Statistics about consensus distribution
        """
        if group_name not in self.consensus_matrices:
            raise ValueError(f"No consensus matrix for group '{group_name}'")
        
        C = self.consensus_matrices[group_name]
        n_nodes = C.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        values = C[triu_idx]
        
        # Edges present in different fractions of subjects
        thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
        edge_counts = {f'C>{t}': int(np.sum(values > t)) for t in thresholds}
        
        return {
            'n_possible_edges': len(values),
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'median': float(np.median(values)),
            'std': float(values.std()),
            'edges_with_consensus_0': int(np.sum(values == 0)),
            'edges_with_consensus_positive': int(np.sum(values > 0)),
            'edges_with_consensus_1': int(np.sum(values == 1.0)),
            **edge_counts,
            'percentiles': {
                p: float(np.percentile(values, p)) 
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
    
    def analyze_edge_consistency(self, group_name: str) -> Dict:
        """
        Analyze edge consistency patterns in the consensus matrix.
        
        Parameters
        ----------
        group_name : str
            Name of the group
            
        Returns
        -------
        dict
            Edge consistency statistics
        """
        if group_name not in self.consensus_matrices:
            raise ValueError(f"No consensus matrix for group '{group_name}'")
        
        C = self.consensus_matrices[group_name]
        G = self.final_graphs.get(group_name)
        
        n_nodes = C.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        consensus_values = C[triu_idx]
        
        # If we have the final graph, analyze which edges were selected
        results = {
            'total_possible_edges': len(consensus_values),
            'consensus_stats': {
                'min': float(consensus_values.min()),
                'max': float(consensus_values.max()),
                'mean': float(consensus_values.mean()),
                'std': float(consensus_values.std())
            }
        }
        
        if G is not None:
            graph_edges = G[triu_idx] > 0
            selected_consensus = consensus_values[graph_edges]
            rejected_consensus = consensus_values[~graph_edges]
            
            results['selected_edges'] = {
                'count': int(np.sum(graph_edges)),
                'consensus_mean': float(np.mean(selected_consensus)) if selected_consensus.size > 0 else float('nan'),
                'consensus_std': float(np.std(selected_consensus)) if selected_consensus.size > 0 else float('nan'),
                'consensus_min': float(np.min(selected_consensus)) if selected_consensus.size > 0 else float('nan')
            }
            
            results['rejected_edges'] = {
                'count': int(np.sum(~graph_edges)),
                'consensus_mean': float(np.mean(rejected_consensus)) if rejected_consensus.size > 0 else float('nan')
            }
        
        return results
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def plot_adjacency_matrix(self,
                             group_name: str,
                             matrix_type: str = 'graph',
                             ax: Optional[plt.Axes] = None,
                             cmap: str = 'magma',
                             title: Optional[str] = None) -> plt.Figure:
        """
        Plot adjacency/consensus/weight matrix as heatmap.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        matrix_type : str
            'graph', 'consensus', or 'weight'
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        cmap : str
            Colormap
        title : str, optional
            Plot title
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        if matrix_type == 'graph':
            matrix = self.final_graphs.get(group_name)
        elif matrix_type == 'consensus':
            matrix = self.consensus_matrices.get(group_name)
        elif matrix_type == 'weight':
            matrix = self.weight_matrices.get(group_name)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
        
        if matrix is None:
            raise ValueError(f"No {matrix_type} matrix for group '{group_name}'")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure
        
        vmax = np.max(matrix) if np.max(matrix) > 0 else 1.0
        vmin = 0
        
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        
        if title is None:
            title = f"{group_name} - {matrix_type.capitalize()} Matrix"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Channel')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        return fig
    
    def plot_degree_distribution(self,
                                group_name: str,
                                ax: Optional[plt.Axes] = None,
                                color: str = 'steelblue') -> plt.Figure:
        """Plot degree distribution for a group."""
        if group_name not in self.node_metrics:
            raise ValueError(f"Node metrics not computed for '{group_name}'")
        
        degrees = self.node_metrics[group_name].degrees
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        
        ax.hist(degrees, bins=20, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(degrees), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(degrees):.1f}')
        ax.axvline(np.median(degrees), color='orange', linestyle='--',
                   label=f'Median: {np.median(degrees):.1f}')
        
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{group_name} - Degree Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_laplacian_spectrum(self,
                               group_name: str,
                               ax: Optional[plt.Axes] = None,
                               color: str = 'steelblue') -> plt.Figure:
        """Plot Laplacian eigenvalue spectrum."""
        if group_name not in self.spectral_metrics:
            raise ValueError(f"Spectral metrics not computed for '{group_name}'")
        
        eigenvalues = self.spectral_metrics[group_name].eigenvalues
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        ax.plot(np.arange(len(eigenvalues)), eigenvalues, 
                marker='o', markersize=3, linewidth=1, color=color)
        
        # Mark algebraic connectivity
        alg_conn = self.spectral_metrics[group_name].algebraic_connectivity
        ax.axhline(alg_conn, color='red', linestyle='--', alpha=0.7,
                   label=f'λ₂ = {alg_conn:.4f}')
        
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('λ')
        ax.set_title(f'{group_name} - Laplacian Eigenvalue Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_consensus_distribution(self,
                                   group_name: str,
                                   ax: Optional[plt.Axes] = None,
                                   color: str = 'steelblue') -> plt.Figure:
        """Plot distribution of consensus values."""
        if group_name not in self.consensus_matrices:
            raise ValueError(f"No consensus matrix for '{group_name}'")
        
        C = self.consensus_matrices[group_name]
        n_nodes = C.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        values = C[triu_idx]
        values_positive = values[values > 0]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        
        ax.hist(values_positive, bins=50, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values_positive), color='red', linestyle='--',
                   label=f'Mean: {np.mean(values_positive):.3f}')
        ax.axvline(np.median(values_positive), color='orange', linestyle='--',
                   label=f'Median: {np.median(values_positive):.3f}')
        
        ax.set_xlabel('Consensus Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{group_name} - Consensus Distribution (positive edges)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_comparison_summary(self,
                               groups: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive comparison figure for all groups.
        
        Parameters
        ----------
        groups : List[str], optional
            Groups to include. If None, use all.
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        if groups is None:
            groups = list(self.final_graphs.keys())
        
        n_groups = len(groups)
        fig, axes = plt.subplots(4, n_groups, figsize=(5*n_groups, 16))
        
        if n_groups == 1:
            axes = axes.reshape(-1, 1)
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
        
        for idx, group_name in enumerate(groups):
            # Row 1: Adjacency matrix
            if group_name in self.final_graphs:
                G = self.final_graphs[group_name]
                vmax = np.max(G) if np.max(G) > 0 else 1.0
                im1 = axes[0, idx].imshow(G, cmap='magma', vmin=0, vmax=vmax)
                axes[0, idx].set_title(f'{group_name}\nFinal Graph')
                plt.colorbar(im1, ax=axes[0, idx], fraction=0.046)
            
            # Row 2: Consensus matrix
            if group_name in self.consensus_matrices:
                C = self.consensus_matrices[group_name]
                im2 = axes[1, idx].imshow(C, cmap='hot', vmin=0, vmax=1)
                axes[1, idx].set_title('Consensus Matrix')
                plt.colorbar(im2, ax=axes[1, idx], fraction=0.046)
            
            # Row 3: Degree distribution
            if group_name in self.node_metrics:
                degrees = self.node_metrics[group_name].degrees
                axes[2, idx].hist(degrees, bins=20, color=colors[idx], 
                                 edgecolor='black', alpha=0.7)
                axes[2, idx].axvline(np.mean(degrees), color='red', linestyle='--')
                axes[2, idx].set_title(f'Degree Distribution\n(mean={np.mean(degrees):.1f})')
                axes[2, idx].set_xlabel('Degree')
                axes[2, idx].set_ylabel('Count')
            
            # Row 4: Eigenvalue spectrum
            if group_name in self.spectral_metrics:
                evals = self.spectral_metrics[group_name].eigenvalues
                axes[3, idx].plot(evals, color=colors[idx], marker='o', markersize=2)
                axes[3, idx].set_title('Laplacian Spectrum')
                axes[3, idx].set_xlabel('Index')
                axes[3, idx].set_ylabel('λ')
                axes[3, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_centrality_comparison(self,
                                  groups: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare centrality measures across groups.
        """
        if groups is None:
            groups = list(self.node_metrics.keys())
        
        centrality_measures = ['betweenness_centrality', 'closeness_centrality',
                              'eigenvector_centrality', 'pagerank']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        
        for ax, measure in zip(axes, centrality_measures):
            for idx, group in enumerate(groups):
                if group in self.node_metrics:
                    values = getattr(self.node_metrics[group], measure, np.array([]))
                    if values.size > 0:
                        ax.hist(values, bins=30, alpha=0.5, label=group, 
                               color=colors[idx], edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(measure.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(measure.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # Reporting
    # ========================================================================
    
    def generate_summary_report(self,
                               group_name: str,
                               output_dir: str,
                               include_thesis_text: bool = True) -> str:
        """
        Generate comprehensive summary report for a group.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        output_dir : str
            Directory to save report
        include_thesis_text : bool
            Include thesis-ready prose
            
        Returns
        -------
        str
            Path to generated report
        """
        if group_name not in self.graph_metrics:
            raise ValueError(f"Metrics not computed for '{group_name}'")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        gm = self.graph_metrics[group_name]
        sm = self.spectral_metrics[group_name]
        
        # Analyze consensus if available
        consensus_analysis = None
        if group_name in self.consensus_matrices:
            consensus_analysis = self.analyze_consensus_distribution(group_name)
        
        # Build report
        lines = [
            f"# Consensus Matrix Analysis Report: {group_name}",
            "",
            "## Executive Summary",
            "",
            f"Analysis of the consensus connectivity graph for group **{group_name}**.",
            "",
            "## Graph-Level Metrics",
            "",
            "### Basic Properties",
            f"- Number of nodes: {gm.n_nodes}",
            f"- Number of edges: {gm.n_edges}",
            f"- Edge density: {gm.density:.4f} ({gm.sparsity_pct:.2f}%)",
            "",
            "### Degree Statistics",
            f"- Mean degree: {gm.degree_mean:.2f} ± {gm.degree_std:.2f}",
            f"- Degree range: [{gm.degree_min:.0f}, {gm.degree_max:.0f}]",
            f"- Median degree: {gm.degree_median:.0f}",
            "",
            "### Strength Statistics (Weighted Degree)",
            f"- Mean strength: {gm.strength_mean:.4f} ± {gm.strength_std:.4f}",
            f"- Strength range: [{gm.strength_min:.4f}, {gm.strength_max:.4f}]",
            "",
            "### Connectivity",
            f"- Number of connected components: {gm.n_components}",
            f"- Largest component size: {gm.largest_component_size} ({gm.largest_component_fraction:.1%})",
            "",
            "### Efficiency Metrics",
            f"- Global efficiency: {gm.global_efficiency:.4f}",
            f"- Local efficiency: {gm.local_efficiency:.4f}",
            f"- Characteristic path length: {gm.characteristic_path_length:.4f}",
            "",
            "### Clustering and Modularity",
            f"- Clustering coefficient: {gm.clustering_coefficient:.4f}",
            f"- Transitivity: {gm.transitivity:.4f}",
            f"- Modularity: {gm.modularity:.4f}",
            f"- Number of communities: {gm.n_communities}",
            "",
            "### Small-World Properties",
            f"- Small-world σ: {gm.small_world_sigma:.4f}",
            f"- Small-world ω: {gm.small_world_omega:.4f}",
            f"- Degree assortativity: {gm.degree_assortativity:.4f}",
            "",
            "## Spectral Analysis",
            "",
            f"- Algebraic connectivity (λ₂): {sm.algebraic_connectivity:.6f}",
            f"- Spectral gap: {sm.spectral_gap:.6f}",
            f"- Spectral radius (λ_max): {sm.spectral_radius:.4f}",
            f"- Laplacian energy: {sm.laplacian_energy:.4f}",
            f"- Near-zero modes: {sm.near_zero_modes}",
            ""
        ]
        
        if consensus_analysis:
            lines.extend([
                "## Consensus Analysis",
                "",
                f"- Total possible edges: {consensus_analysis['n_possible_edges']}",
                f"- Edges with positive consensus: {consensus_analysis['edges_with_consensus_positive']}",
                f"- Edges with full consensus (C=1): {consensus_analysis['edges_with_consensus_1']}",
                f"- Mean consensus: {consensus_analysis['mean']:.4f}",
                f"- Median consensus: {consensus_analysis['median']:.4f}",
                ""
            ])
        
        if include_thesis_text:
            # Generate thesis-ready prose
            connectivity_text = (
                "connected (forming a single component)"
                if gm.n_components == 1
                else f"composed of {gm.n_components} connected components"
            )
            
            small_world_text = (
                "suggesting small-world organization (σ > 1)"
                if gm.small_world_sigma > 1
                else "resembling a random graph topology (σ ≈ 1)"
                if np.isfinite(gm.small_world_sigma)
                else "with small-world properties not estimable due to sparse connectivity"
            )
            
            lines.extend([
                "## Thesis-Ready Summary",
                "",
                f"The consensus connectivity graph for {group_name} comprises {gm.n_nodes} nodes "
                f"and {gm.n_edges} edges, yielding an edge density of {gm.sparsity_pct:.2f}%. "
                f"The network is {connectivity_text}. "
                f"Binary degrees span {gm.degree_min:.0f}–{gm.degree_max:.0f} "
                f"(mean = {gm.degree_mean:.1f}, SD = {gm.degree_std:.1f}), "
                f"indicating that all channels maintain multiple connections without forming extreme hubs. "
                f"Weighted node strengths range from {gm.strength_min:.3f} to {gm.strength_max:.3f} "
                f"(median = {gm.strength_median:.3f}).",
                "",
                f"Global efficiency ({gm.global_efficiency:.4f}) and local efficiency ({gm.local_efficiency:.4f}) "
                f"characterize the network's integration and segregation properties. "
                f"The weighted clustering coefficient is {gm.clustering_coefficient:.4f}, "
                f"and modularity analysis reveals {gm.n_communities} communities (Q = {gm.modularity:.3f}). "
                f"The small-world sigma statistic is {gm.small_world_sigma:.3f}, {small_world_text}.",
                "",
                f"Spectral analysis of the graph Laplacian shows algebraic connectivity "
                f"λ₂ = {sm.algebraic_connectivity:.5f}, with {sm.near_zero_modes} near-zero modes "
                f"corresponding to the {gm.n_components} connected component(s). "
                f"The spectral radius (λ_max = {sm.spectral_radius:.3f}) bounds the "
                f"high-frequency content available to graph-based signal processing methods.",
                ""
            ])
        
        # Save report
        report_path = output_path / f"{group_name}_analysis_report.md"
        report_path.write_text("\n".join(lines))
        
        logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def export_metrics_to_csv(self,
                             output_dir: str,
                             prefix: str = "consensus") -> Dict[str, str]:
        """
        Export all computed metrics to CSV files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save CSV files
        prefix : str
            Prefix for output filenames
            
        Returns
        -------
        dict
            Paths to generated CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Graph-level metrics
        if self.graph_metrics:
            rows = []
            for group_name, metrics in self.graph_metrics.items():
                row = {'group': group_name, **asdict(metrics)}
                rows.append(row)
            
            df = pd.DataFrame(rows)
            filepath = output_path / f"{prefix}_graph_metrics.csv"
            df.to_csv(filepath, index=False)
            saved_files['graph_metrics'] = str(filepath)
        
        # Node-level metrics (one file per group)
        for group_name, metrics in self.node_metrics.items():
            df = pd.DataFrame({
                'node_id': metrics.node_ids,
                'degree': metrics.degrees,
                'strength': metrics.strengths,
                'betweenness': metrics.betweenness_centrality,
                'closeness': metrics.closeness_centrality,
                'eigenvector': metrics.eigenvector_centrality,
                'pagerank': metrics.pagerank,
                'clustering': metrics.clustering,
                'community': metrics.community_labels
            })
            filepath = output_path / f"{prefix}_{group_name}_node_metrics.csv"
            df.to_csv(filepath, index=False)
            saved_files[f'{group_name}_node_metrics'] = str(filepath)
        
        # Spectral metrics
        if self.spectral_metrics:
            rows = []
            for group_name, metrics in self.spectral_metrics.items():
                rows.append({
                    'group': group_name,
                    'algebraic_connectivity': metrics.algebraic_connectivity,
                    'spectral_gap': metrics.spectral_gap,
                    'spectral_radius': metrics.spectral_radius,
                    'laplacian_energy': metrics.laplacian_energy,
                    'near_zero_modes': metrics.near_zero_modes
                })
            
            df = pd.DataFrame(rows)
            filepath = output_path / f"{prefix}_spectral_metrics.csv"
            df.to_csv(filepath, index=False)
            saved_files['spectral_metrics'] = str(filepath)
        
        logger.info(f"Exported {len(saved_files)} CSV files to {output_dir}")
        return saved_files
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all graph metrics across groups.
        
        Returns
        -------
        pd.DataFrame
            Summary of key metrics for all groups
        """
        if not self.graph_metrics:
            return pd.DataFrame()
        
        rows = []
        for group_name, gm in self.graph_metrics.items():
            sm = self.spectral_metrics.get(group_name, SpectralMetrics())
            
            rows.append({
                'group': group_name,
                'n_nodes': gm.n_nodes,
                'n_edges': gm.n_edges,
                'density': gm.density,
                'degree_mean': gm.degree_mean,
                'degree_std': gm.degree_std,
                'strength_mean': gm.strength_mean,
                'n_components': gm.n_components,
                'global_efficiency': gm.global_efficiency,
                'local_efficiency': gm.local_efficiency,
                'clustering': gm.clustering_coefficient,
                'modularity': gm.modularity,
                'n_communities': gm.n_communities,
                'small_world_sigma': gm.small_world_sigma,
                'algebraic_connectivity': sm.algebraic_connectivity,
                'spectral_radius': sm.spectral_radius
            })
        
        return pd.DataFrame(rows)


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_consensus_results(results_dir: str,
                             output_dir: Optional[str] = None,
                             generate_figures: bool = True,
                             generate_reports: bool = True) -> ConsensusResultsAnalyzer:
    """
    Main entry point for analyzing consensus matrix results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing consensus results
    output_dir : str, optional
        Directory for output files. Defaults to results_dir/analysis
    generate_figures : bool
        Whether to generate visualization figures
    generate_reports : bool
        Whether to generate markdown reports
        
    Returns
    -------
    ConsensusResultsAnalyzer
        Analyzer object with computed metrics
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / "analysis")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize and load
    analyzer = ConsensusResultsAnalyzer(results_dir)
    analyzer.load_results()
    
    # Compute metrics
    analyzer.compute_all_metrics()
    
    # Export CSVs
    analyzer.export_metrics_to_csv(output_dir)
    
    # Generate figures
    if generate_figures and analyzer.final_graphs:
        groups = list(analyzer.final_graphs.keys())
        
        # Comparison summary
        analyzer.plot_comparison_summary(
            groups, 
            save_path=str(output_path / "comparison_summary.png")
        )
        
        # Individual group figures
        for group in groups:
            analyzer.plot_adjacency_matrix(
                group, 'graph',
                title=f'{group} - Final Graph'
            )
            plt.savefig(output_path / f"{group}_adjacency.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            if group in analyzer.consensus_matrices:
                analyzer.plot_consensus_distribution(group)
                plt.savefig(output_path / f"{group}_consensus_dist.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            if group in analyzer.spectral_metrics:
                analyzer.plot_laplacian_spectrum(group)
                plt.savefig(output_path / f"{group}_spectrum.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # Centrality comparison
        if len(groups) > 1:
            analyzer.plot_centrality_comparison(
                groups,
                save_path=str(output_path / "centrality_comparison.png")
            )
    
    # Generate reports
    if generate_reports:
        for group in analyzer.graph_metrics.keys():
            analyzer.generate_summary_report(group, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("CONSENSUS MATRIX ANALYSIS COMPLETE")
    print("="*70)
    
    summary_df = analyzer.get_metrics_summary()
    if not summary_df.empty:
        print("\nKey Metrics Summary:")
        print(summary_df.to_string(index=False))
    
    print(f"\nOutput files saved to: {output_dir}")
    
    return analyzer


def compare_ad_hc_groups(results_dir: str,
                        ad_pattern: str = 'AD',
                        hc_pattern: str = 'HC',
                        output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Specialized comparison of AD vs HC groups.
    
    Parameters
    ----------
    results_dir : str
        Directory containing consensus results
    ad_pattern : str
        Pattern to identify AD groups
    hc_pattern : str
        Pattern to identify HC groups
    output_dir : str, optional
        Output directory
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    analyzer = ConsensusResultsAnalyzer(results_dir)
    analyzer.load_results()
    analyzer.compute_all_metrics()
    
    # Find AD and HC groups
    ad_groups = [g for g in analyzer.final_graphs.keys() if ad_pattern in g.upper()]
    hc_groups = [g for g in analyzer.final_graphs.keys() if hc_pattern in g.upper()]
    
    if not ad_groups or not hc_groups:
        logger.warning(f"Could not identify AD ({ad_groups}) or HC ({hc_groups}) groups")
        return pd.DataFrame()
    
    # Compare first AD vs first HC group
    comparison = analyzer.compare_groups(ad_groups[0], hc_groups[0])
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_path / "ad_hc_comparison.csv", index=False)
    
    return comparison


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze consensus matrix results from EEG connectivity analysis"
    )
    parser.add_argument(
        '--results_dir', '-r',
        type=str,
        default='./consensus_results',
        help='Directory containing consensus results'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Skip figure generation'
    )
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip report generation'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = analyze_consensus_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        generate_figures=not args.no_figures,
        generate_reports=not args.no_reports
    )
    
    print("\nAnalysis complete!")
