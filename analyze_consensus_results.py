"""
Comprehensive Analysis of Consensus Matrix Results for EEG Connectivity

This script provides tools to analyze pre-computed consensus matrix results,
including:
- Graph-theoretic metrics (centrality, modularity, efficiency)
- Spectral analysis of the Laplacian
- Group comparisons with statistical tests
- Sparsity analysis (critical for AD research)
- Regional (brain lobe) analysis
- Publication-quality visualizations
- Thesis-ready reports

================================================================================
WHY SPARSITY MATTERS FOR ALZHEIMER'S DISEASE (AD) EEG ANALYSIS
================================================================================

Sparse connectivity matrices are crucial for AD research for several reasons:

1. DISCONNECTION SYNDROME HYPOTHESIS
   - AD is fundamentally a "disconnection syndrome" characterized by progressive
     loss of synaptic connections and white matter integrity
   - Sparse networks directly model this pathophysiology by retaining only the
     most robust, reproducible connections
   - Dense networks would obscure the disconnection pattern central to AD

2. NOISE REDUCTION & VOLUME CONDUCTION
   - EEG signals suffer from volume conduction artifacts that create spurious
     correlations between nearby electrodes
   - Proportional thresholding (keeping top κ% of edges) removes weak/noisy
     connections that likely reflect artifacts rather than true neural coupling
   - This is especially important when comparing AD (with reduced signal power)
     to healthy controls

3. IDENTIFYING PATHOLOGICAL CHANGES
   - AD patients show specific patterns of network disruption:
     * Reduced long-range connectivity (especially frontoparietal)
     * Preserved or increased local clustering (compensatory mechanism)
     * Hub vulnerability (highly connected regions fail first)
   - Sparse networks make these patterns statistically detectable
   - Dense networks dilute group differences with noise

4. GRAPH-THEORETICAL VALIDITY
   - Many graph metrics (small-worldness, modularity, efficiency) are only
     meaningful on sparse networks
   - Dense networks trivially approach complete graphs where all nodes connect
   - Standard practice: 10-30% edge density for brain network analysis

5. COMPUTATIONAL TRACTABILITY
   - GP-VAR and other graph-signal processing methods scale with edge count
   - Sparse Laplacians enable efficient spectral decomposition
   - Clinical translation requires computationally feasible methods

6. REPRODUCIBILITY & RELIABILITY
   - Consensus approach: only edges present in multiple subjects survive
   - This naturally produces sparse networks representing consistent connectivity
   - Edges in the final graph are statistically reliable across the cohort

7. CLINICAL INTERPRETABILITY
   - Sparse networks can be visualized and interpreted by clinicians
   - Dense "hairball" networks provide no actionable insights
   - Sparse hubs and modules map to known anatomical/functional systems

RECOMMENDED SPARSITY APPROACH FOR AD RESEARCH:
- Individual subject binarization: 10-20% (sparsity_binarize=0.15)
- Final consensus graph: USE NATURAL SPARSITY (sparsity_final=None or "match_subject")
  
  IMPORTANT: Do NOT artificially cut off at a fixed percentage (e.g., 10%)!
  The consensus process naturally produces sparse networks by:
  1. Only retaining edges present across multiple subjects
  2. Distance-dependent selection preserves anatomically meaningful connections
  3. The resulting "natural" sparsity reflects true group-level connectivity
  
  Artificial thresholding can:
  - Remove clinically meaningful weak connections
  - Introduce arbitrary bias in group comparisons
  - Distort the graph frequency spectrum used in GP-VAR analysis

8. GRAPH FREQUENCY SPECTRUM (Critical for GP-VAR Analysis)
   - The consensus Laplacian's eigenvalues define the graph frequency spectrum
   - GP-VAR models use this spectrum for spectral filtering on graphs
   - Artificially sparse networks distort the spectrum
   - Natural sparsity preserves the true frequency content of brain networks
   - The algebraic connectivity (λ₂) determines network integration
   - Higher eigenvalues capture local, high-frequency variation

REFERENCES:
- Delbeuck et al. (2003) "Alzheimer's disease as a disconnection syndrome"
- Stam (2014) "Modern network science of neurological disorders"
- Fornito et al. (2015) "The connectomics of brain disorders"
- Tijms et al. (2013) "AD as a network disorder"

================================================================================

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


@dataclass
class SparsityMetrics:
    """
    Container for sparsity-related metrics.
    
    These metrics are particularly important for AD research because:
    - AD is a disconnection syndrome - sparsity captures the degree of disconnection
    - Comparing sparsity between AD and HC reveals pathological connectivity loss
    - Edge consistency measures show which connections are reliably present/absent
    """
    # Basic sparsity measures
    n_possible_edges: int = 0
    n_actual_edges: int = 0
    edge_density: float = 0.0  # n_actual / n_possible (0 to 1)
    sparsity_percent: float = 0.0  # edge_density * 100
    
    # Consensus-based sparsity (across subjects)
    edges_full_consensus: int = 0  # C = 1.0 (present in ALL subjects)
    edges_majority_consensus: int = 0  # C > 0.5 (present in >50% subjects)
    edges_any_consensus: int = 0  # C > 0 (present in at least one subject)
    edges_zero_consensus: int = 0  # C = 0 (never present)
    
    # Consistency measures
    mean_consensus: float = 0.0  # Average C value for all possible edges
    mean_consensus_positive: float = 0.0  # Average C for edges with C > 0
    consensus_variance: float = 0.0  # Variance in consensus values
    
    # Distance-dependent sparsity (important for AD - long-range disconnection)
    short_range_density: float = 0.0  # Density for nearby electrodes
    medium_range_density: float = 0.0  # Density for medium-distance electrodes  
    long_range_density: float = 0.0  # Density for far electrodes
    
    # Weight distribution in sparse network
    mean_edge_weight: float = 0.0
    median_edge_weight: float = 0.0
    weight_std: float = 0.0
    weight_cv: float = 0.0  # Coefficient of variation
    
    # Hub-related sparsity (AD shows hub vulnerability)
    hub_edge_fraction: float = 0.0  # Fraction of edges connected to hubs
    peripheral_edge_fraction: float = 0.0  # Fraction of edges between low-degree nodes
    
    # Thresholding summary
    threshold_used: float = 0.0  # Weight threshold that achieves this sparsity
    edges_above_threshold: int = 0


@dataclass
class GraphFrequencyMetrics:
    """
    Container for graph frequency spectrum analysis.
    
    CRITICAL FOR GP-VAR AND GRAPH SIGNAL PROCESSING:
    
    The consensus Laplacian L = D - A defines the graph frequency domain:
    - Eigenvalues λ₀ ≤ λ₁ ≤ ... ≤ λₙ₋₁ are the graph frequencies
    - Eigenvectors form the Graph Fourier Transform (GFT) basis
    - Low frequencies (small λ) = smooth signals across connected nodes
    - High frequencies (large λ) = rapid variation between neighbors
    
    For AD Analysis:
    - AD disrupts low-frequency (global) integration
    - Local high-frequency processing may be preserved/compensatory
    - The spectrum shape reflects network organization
    - GP-VAR uses this spectrum for time-varying graph filtering
    """
    # Eigenvalue distribution
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    n_eigenvalues: int = 0
    
    # Key spectral landmarks
    lambda_min: float = 0.0  # Should be ~0 for connected graph
    lambda_2: float = 0.0  # Algebraic connectivity (Fiedler value)
    lambda_max: float = 0.0  # Spectral radius
    lambda_median: float = 0.0
    lambda_mean: float = 0.0
    
    # Spectral distribution
    spectral_gap: float = 0.0  # λ₂ - λ₁ (connectivity measure)
    spectral_range: float = 0.0  # λ_max - λ_min
    spectral_variance: float = 0.0
    
    # Frequency band analysis (like EEG bands but for graph)
    low_freq_energy: float = 0.0  # Energy in lowest 25% of spectrum
    mid_freq_energy: float = 0.0  # Energy in middle 50%
    high_freq_energy: float = 0.0  # Energy in highest 25%
    low_high_ratio: float = 0.0  # Balance between global and local
    
    # Connectivity indicators
    n_zero_eigenvalues: int = 0  # Number of connected components
    algebraic_connectivity: float = 0.0  # λ₂ - key for network integration
    
    # Graph filter properties (for GP-VAR)
    effective_bandwidth: float = 0.0  # Useful frequency range
    spectral_entropy: float = 0.0  # Uniformity of spectrum
    
    # Eigenvector properties
    fiedler_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    fiedler_bipartition: np.ndarray = field(default_factory=lambda: np.array([]))  # Binary partition


@dataclass  
class ADConnectivityMetrics:
    """
    Metrics specifically relevant to Alzheimer's Disease connectivity analysis.
    
    These capture the pathophysiological signatures of AD:
    - Disconnection (reduced connectivity)
    - Hub vulnerability (loss of high-degree nodes)
    - Altered integration/segregation balance
    - Slowing of neural dynamics (reflected in spectral changes)
    """
    # Disconnection measures
    global_connectivity_strength: float = 0.0  # Sum of all edge weights
    mean_connection_strength: float = 0.0  # Average edge weight
    connectivity_loss_index: float = 0.0  # Compared to theoretical maximum
    
    # Hub vulnerability (AD preferentially affects hubs)
    hub_strength_ratio: float = 0.0  # Hub strength / peripheral strength
    hub_degree_ratio: float = 0.0  # Hub degree / peripheral degree
    n_hubs: int = 0  # Number of hub nodes (degree > mean + 1 SD)
    hub_node_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Integration vs Segregation (AD alters this balance)
    integration_index: float = 0.0  # Global efficiency
    segregation_index: float = 0.0  # Modularity * clustering
    integration_segregation_ratio: float = 0.0
    
    # Long-range connectivity (specifically affected in AD)
    long_range_strength: float = 0.0
    short_range_strength: float = 0.0
    long_short_ratio: float = 0.0  # AD typically shows reduced ratio
    
    # Frontoparietal connectivity (key AD-affected network)
    frontoparietal_strength: float = 0.0  # If electrode regions known
    
    # Network resilience (AD networks are more vulnerable)
    robustness_random: float = 0.0  # Resilience to random node removal
    robustness_targeted: float = 0.0  # Resilience to hub removal (lower in AD)
    
    # Rich-club organization (often disrupted in AD)
    rich_club_coefficient: float = 0.0
    normalized_rich_club: float = 0.0


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
        self.sparsity_metrics: Dict[str, SparsityMetrics] = {}
        self.ad_metrics: Dict[str, ADConnectivityMetrics] = {}
        self.graph_frequency_metrics: Dict[str, GraphFrequencyMetrics] = {}
        
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
            
            # Compute sparsity metrics (critical for AD analysis)
            C_matrix = self.consensus_matrices.get(group_name)
            self.sparsity_metrics[group_name] = self._compute_sparsity_metrics(
                G_matrix, C_matrix, self.channel_locations
            )
            
            # Compute AD-specific metrics
            self.ad_metrics[group_name] = self._compute_ad_metrics(
                G_matrix, G_nx, self.channel_locations
            )
            
            # Compute graph frequency spectrum metrics (CRITICAL for GP-VAR)
            self.graph_frequency_metrics[group_name] = self._compute_graph_frequency_metrics(G_matrix)
        
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
    
    def _compute_sparsity_metrics(self,
                                  adjacency: np.ndarray,
                                  consensus: Optional[np.ndarray] = None,
                                  channel_locations: Optional[np.ndarray] = None) -> SparsityMetrics:
        """
        Compute comprehensive sparsity metrics.
        
        CLINICAL RELEVANCE FOR AD:
        - Edge density directly measures network sparsity/disconnection
        - Consensus-based metrics show reliability of connections across subjects
        - Distance-dependent sparsity reveals long-range disconnection (AD hallmark)
        - Hub edge fractions indicate hub vulnerability
        
        Parameters
        ----------
        adjacency : np.ndarray
            Final graph adjacency matrix
        consensus : np.ndarray, optional
            Consensus matrix C (fraction of subjects with each edge)
        channel_locations : np.ndarray, optional
            3D electrode coordinates for distance calculations
            
        Returns
        -------
        SparsityMetrics
            Comprehensive sparsity analysis
        """
        metrics = SparsityMetrics()
        n_nodes = adjacency.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        
        # Basic sparsity measures
        metrics.n_possible_edges = len(triu_idx[0])
        edge_weights = adjacency[triu_idx]
        edge_mask = edge_weights > 0
        metrics.n_actual_edges = int(np.sum(edge_mask))
        metrics.edge_density = metrics.n_actual_edges / max(1, metrics.n_possible_edges)
        metrics.sparsity_percent = metrics.edge_density * 100
        
        # Weight distribution for existing edges
        positive_weights = edge_weights[edge_mask]
        if positive_weights.size > 0:
            metrics.mean_edge_weight = float(np.mean(positive_weights))
            metrics.median_edge_weight = float(np.median(positive_weights))
            metrics.weight_std = float(np.std(positive_weights))
            metrics.weight_cv = metrics.weight_std / metrics.mean_edge_weight if metrics.mean_edge_weight > 0 else 0
        
        # Find threshold that achieves this sparsity
        if positive_weights.size > 0:
            metrics.threshold_used = float(np.min(positive_weights))
            metrics.edges_above_threshold = int(np.sum(edge_weights >= metrics.threshold_used))
        
        # Consensus-based sparsity (if consensus matrix available)
        if consensus is not None:
            consensus_values = consensus[triu_idx]
            metrics.edges_full_consensus = int(np.sum(consensus_values >= 0.999))
            metrics.edges_majority_consensus = int(np.sum(consensus_values > 0.5))
            metrics.edges_any_consensus = int(np.sum(consensus_values > 0))
            metrics.edges_zero_consensus = int(np.sum(consensus_values == 0))
            
            metrics.mean_consensus = float(np.mean(consensus_values))
            positive_consensus = consensus_values[consensus_values > 0]
            if positive_consensus.size > 0:
                metrics.mean_consensus_positive = float(np.mean(positive_consensus))
            metrics.consensus_variance = float(np.var(consensus_values))
        
        # Distance-dependent sparsity (critical for AD - shows long-range disconnection)
        if channel_locations is not None and channel_locations.shape[0] == n_nodes:
            distances = squareform(pdist(channel_locations))
            edge_distances = distances[triu_idx]
            
            # Define distance bins (thirds of distance range)
            d_min, d_max = edge_distances.min(), edge_distances.max()
            d_range = d_max - d_min
            d_short = d_min + d_range / 3
            d_medium = d_min + 2 * d_range / 3
            
            # Short-range: nearby electrodes
            short_mask = edge_distances <= d_short
            n_short = np.sum(short_mask)
            short_edges = np.sum(edge_mask & short_mask)
            metrics.short_range_density = short_edges / max(1, n_short)
            
            # Medium-range
            medium_mask = (edge_distances > d_short) & (edge_distances <= d_medium)
            n_medium = np.sum(medium_mask)
            medium_edges = np.sum(edge_mask & medium_mask)
            metrics.medium_range_density = medium_edges / max(1, n_medium)
            
            # Long-range: distant electrodes (often reduced in AD)
            long_mask = edge_distances > d_medium
            n_long = np.sum(long_mask)
            long_edges = np.sum(edge_mask & long_mask)
            metrics.long_range_density = long_edges / max(1, n_long)
        
        # Hub-related sparsity (AD shows hub vulnerability)
        degrees = np.sum(adjacency > 0, axis=0)
        if degrees.size > 0:
            degree_threshold = np.mean(degrees) + np.std(degrees)
            hub_nodes = degrees >= degree_threshold
            peripheral_nodes = ~hub_nodes
            
            # Count edges involving hubs vs peripheral nodes
            n_hub_edges = 0
            n_peripheral_edges = 0
            
            for i, j in zip(triu_idx[0][edge_mask], triu_idx[1][edge_mask]):
                if hub_nodes[i] or hub_nodes[j]:
                    n_hub_edges += 1
                if peripheral_nodes[i] and peripheral_nodes[j]:
                    n_peripheral_edges += 1
            
            total_edges = max(1, metrics.n_actual_edges)
            metrics.hub_edge_fraction = n_hub_edges / total_edges
            metrics.peripheral_edge_fraction = n_peripheral_edges / total_edges
        
        return metrics
    
    def _compute_ad_metrics(self,
                           adjacency: np.ndarray,
                           G: nx.Graph,
                           channel_locations: Optional[np.ndarray] = None) -> ADConnectivityMetrics:
        """
        Compute AD-specific connectivity metrics.
        
        These metrics capture pathophysiological signatures of Alzheimer's Disease:
        - Disconnection syndrome (global connectivity reduction)
        - Hub vulnerability (hubs fail before peripheral nodes)
        - Altered integration/segregation balance
        - Long-range disconnection (especially frontoparietal)
        
        Parameters
        ----------
        adjacency : np.ndarray
            Final graph adjacency matrix
        G : nx.Graph
            NetworkX graph representation
        channel_locations : np.ndarray, optional
            3D electrode coordinates
            
        Returns
        -------
        ADConnectivityMetrics
            AD-relevant network measures
        """
        metrics = ADConnectivityMetrics()
        n_nodes = adjacency.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        
        # === DISCONNECTION MEASURES ===
        # Total connectivity strength (AD shows global reduction)
        edge_weights = adjacency[triu_idx]
        metrics.global_connectivity_strength = float(np.sum(edge_weights))
        
        # Mean connection strength (per existing edge)
        positive_weights = edge_weights[edge_weights > 0]
        if positive_weights.size > 0:
            metrics.mean_connection_strength = float(np.mean(positive_weights))
        
        # Connectivity loss index (compared to fully connected network)
        # Higher values = more disconnection (worse in AD)
        max_possible_strength = len(triu_idx[0])  # If all edges had weight 1
        metrics.connectivity_loss_index = 1 - (metrics.global_connectivity_strength / max_possible_strength)
        
        # === HUB VULNERABILITY ===
        # AD preferentially affects highly connected hub regions
        degrees = np.array([G.degree(i) for i in range(n_nodes)])
        strengths = np.sum(adjacency, axis=1)
        
        if degrees.size > 0 and np.std(degrees) > 0:
            # Define hubs as nodes with degree > mean + 1 SD
            degree_threshold = np.mean(degrees) + np.std(degrees)
            hub_mask = degrees >= degree_threshold
            metrics.n_hubs = int(np.sum(hub_mask))
            metrics.hub_node_indices = np.where(hub_mask)[0]
            
            if metrics.n_hubs > 0 and np.sum(~hub_mask) > 0:
                hub_strength = np.mean(strengths[hub_mask])
                peripheral_strength = np.mean(strengths[~hub_mask])
                hub_degree = np.mean(degrees[hub_mask])
                peripheral_degree = np.mean(degrees[~hub_mask])
                
                metrics.hub_strength_ratio = hub_strength / max(peripheral_strength, 1e-10)
                metrics.hub_degree_ratio = hub_degree / max(peripheral_degree, 1e-10)
        
        # === INTEGRATION VS SEGREGATION ===
        # AD disrupts the balance between integration (global efficiency)
        # and segregation (clustering/modularity)
        try:
            metrics.integration_index = nx.global_efficiency(G)
        except:
            metrics.integration_index = 0.0
        
        try:
            clustering = nx.average_clustering(G, weight='weight')
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
            modularity = nx.community.modularity(G, communities, weight='weight')
            metrics.segregation_index = clustering * modularity
        except:
            metrics.segregation_index = 0.0
        
        if metrics.segregation_index > 0:
            metrics.integration_segregation_ratio = metrics.integration_index / metrics.segregation_index
        
        # === LONG-RANGE CONNECTIVITY ===
        # AD specifically affects long-range (especially frontoparietal) connections
        if channel_locations is not None and channel_locations.shape[0] == n_nodes:
            distances = squareform(pdist(channel_locations))
            edge_distances = distances[triu_idx]
            
            # Median split for short vs long range
            median_distance = np.median(edge_distances)
            
            for idx, (i, j) in enumerate(zip(triu_idx[0], triu_idx[1])):
                weight = adjacency[i, j]
                if weight > 0:
                    if edge_distances[idx] > median_distance:
                        metrics.long_range_strength += weight
                    else:
                        metrics.short_range_strength += weight
            
            if metrics.short_range_strength > 0:
                metrics.long_short_ratio = metrics.long_range_strength / metrics.short_range_strength
        
        # === NETWORK RESILIENCE ===
        # AD networks are less resilient to perturbation
        if G.number_of_edges() > 0 and G.number_of_nodes() > 5:
            # Robustness to random node removal
            try:
                metrics.robustness_random = self._compute_robustness(G, targeted=False)
            except:
                metrics.robustness_random = float('nan')
            
            # Robustness to targeted (hub) removal - typically lower in AD
            try:
                metrics.robustness_targeted = self._compute_robustness(G, targeted=True)
            except:
                metrics.robustness_targeted = float('nan')
        
        # === RICH-CLUB ORGANIZATION ===
        # AD often shows disrupted rich-club (hub-to-hub connections)
        if G.number_of_edges() > 0:
            try:
                rc = nx.rich_club_coefficient(G, normalized=False)
                if rc:
                    # Use coefficient at median degree
                    k_values = sorted(rc.keys())
                    if k_values:
                        k_median = k_values[len(k_values)//2]
                        metrics.rich_club_coefficient = rc.get(k_median, 0.0)
            except:
                metrics.rich_club_coefficient = float('nan')
        
        return metrics
    
    def _compute_robustness(self, G: nx.Graph, targeted: bool = False, 
                           n_iterations: int = 5) -> float:
        """
        Compute network robustness to node removal.
        
        Parameters
        ----------
        G : nx.Graph
            Network graph
        targeted : bool
            If True, remove highest-degree nodes first (hub attack)
            If False, remove random nodes
        n_iterations : int
            Number of iterations for random removal
            
        Returns
        -------
        float
            Robustness score (area under largest component curve)
        """
        n_nodes = G.number_of_nodes()
        if n_nodes < 3:
            return 0.0
        
        robustness_scores = []
        
        for _ in range(n_iterations if not targeted else 1):
            G_copy = G.copy()
            nodes = list(G_copy.nodes())
            lcc_sizes = [1.0]  # Initial: 100% in largest component
            
            # Remove nodes one by one
            n_to_remove = max(1, n_nodes // 2)  # Remove up to 50% of nodes
            
            for _ in range(n_to_remove):
                if G_copy.number_of_nodes() == 0:
                    break
                
                if targeted:
                    # Remove highest-degree node
                    degrees = dict(G_copy.degree())
                    node_to_remove = max(degrees, key=degrees.get)
                else:
                    # Remove random node
                    node_to_remove = np.random.choice(list(G_copy.nodes()))
                
                G_copy.remove_node(node_to_remove)
                
                if G_copy.number_of_nodes() > 0:
                    components = list(nx.connected_components(G_copy))
                    largest_cc = max(len(c) for c in components) if components else 0
                    lcc_sizes.append(largest_cc / n_nodes)
                else:
                    lcc_sizes.append(0.0)
            
            # Area under curve (higher = more robust)
            robustness_scores.append(np.trapz(lcc_sizes) / len(lcc_sizes))
        
        return float(np.mean(robustness_scores))
    
    def _compute_graph_frequency_metrics(self, adjacency: np.ndarray) -> GraphFrequencyMetrics:
        """
        Compute graph frequency spectrum analysis.
        
        CRITICAL FOR GP-VAR AND GRAPH SIGNAL PROCESSING:
        
        The consensus matrix defines a graph Laplacian whose eigenvalues form
        the graph frequency spectrum. This is analogous to temporal frequencies
        in traditional signal processing, but defined over the graph structure.
        
        For AD Analysis:
        - Low graph frequencies represent global, smooth patterns across the brain
        - High graph frequencies capture local, rapid variations between neighbors
        - AD typically shows disrupted low-frequency (integration) with preserved
          high-frequency (local) processing
        - The spectrum shape is used by GP-VAR for graph-based filtering
        
        IMPORTANT: Natural sparsity preserves the true spectrum!
        Artificial thresholding distorts these frequencies.
        
        Parameters
        ----------
        adjacency : np.ndarray
            Consensus adjacency matrix (use NATURAL sparsity, not artificial cutoff)
            
        Returns
        -------
        GraphFrequencyMetrics
            Complete graph frequency analysis
        """
        metrics = GraphFrequencyMetrics()
        n_nodes = adjacency.shape[0]
        
        if n_nodes == 0:
            return metrics
        
        # Compute graph Laplacian: L = D - A
        # D = diagonal degree matrix, A = adjacency
        strengths = np.sum(adjacency, axis=1)
        laplacian = np.diag(strengths) - adjacency
        
        # Eigendecomposition (eigenvalues are graph frequencies)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Sort eigenvalues (should already be sorted, but ensure)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        metrics.eigenvalues = eigenvalues
        metrics.n_eigenvalues = len(eigenvalues)
        
        # Key spectral landmarks
        metrics.lambda_min = float(eigenvalues[0])
        metrics.lambda_max = float(eigenvalues[-1])
        metrics.lambda_mean = float(np.mean(eigenvalues))
        metrics.lambda_median = float(np.median(eigenvalues))
        
        if len(eigenvalues) > 1:
            metrics.lambda_2 = float(eigenvalues[1])  # Algebraic connectivity
            metrics.algebraic_connectivity = metrics.lambda_2
            metrics.spectral_gap = float(eigenvalues[1] - eigenvalues[0])
        
        metrics.spectral_range = metrics.lambda_max - metrics.lambda_min
        metrics.spectral_variance = float(np.var(eigenvalues))
        
        # Count zero eigenvalues (indicates connected components)
        metrics.n_zero_eigenvalues = int(np.sum(eigenvalues < 1e-10))
        
        # Frequency band analysis (analogous to EEG frequency bands)
        # Divide spectrum into low (global), mid, and high (local) frequencies
        n_eig = len(eigenvalues)
        low_idx = n_eig // 4
        high_idx = 3 * n_eig // 4
        
        # Energy in each band (sum of squared eigenvalues)
        low_eigs = eigenvalues[:low_idx] if low_idx > 0 else eigenvalues[:1]
        mid_eigs = eigenvalues[low_idx:high_idx] if high_idx > low_idx else eigenvalues
        high_eigs = eigenvalues[high_idx:] if high_idx < n_eig else eigenvalues[-1:]
        
        metrics.low_freq_energy = float(np.sum(low_eigs ** 2))
        metrics.mid_freq_energy = float(np.sum(mid_eigs ** 2))
        metrics.high_freq_energy = float(np.sum(high_eigs ** 2))
        
        if metrics.high_freq_energy > 0:
            metrics.low_high_ratio = metrics.low_freq_energy / metrics.high_freq_energy
        
        # Effective bandwidth (range containing 90% of spectral energy)
        total_energy = np.sum(eigenvalues ** 2)
        if total_energy > 0:
            cumulative_energy = np.cumsum(eigenvalues ** 2) / total_energy
            # Find 5% and 95% points
            idx_5 = np.searchsorted(cumulative_energy, 0.05)
            idx_95 = np.searchsorted(cumulative_energy, 0.95)
            if idx_95 < len(eigenvalues) and idx_5 < len(eigenvalues):
                metrics.effective_bandwidth = float(eigenvalues[idx_95] - eigenvalues[idx_5])
        
        # Spectral entropy (uniformity of spectrum - higher = more uniform)
        positive_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(positive_eigs) > 0:
            # Normalize to probability distribution
            p = positive_eigs / np.sum(positive_eigs)
            # Shannon entropy
            metrics.spectral_entropy = float(-np.sum(p * np.log(p + 1e-10)))
        
        # Fiedler vector (eigenvector of λ₂ - used for graph partitioning)
        if len(eigenvalues) > 1:
            metrics.fiedler_vector = eigenvectors[:, 1]
            # Binary partition based on Fiedler vector sign
            metrics.fiedler_bipartition = (metrics.fiedler_vector >= 0).astype(int)
        
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
    
    def plot_graph_frequency_spectrum(self,
                                      group_name: str,
                                      ax: Optional[plt.Axes] = None,
                                      show_bands: bool = True,
                                      color: str = 'steelblue') -> plt.Figure:
        """
        Plot the graph frequency spectrum (Laplacian eigenvalues).
        
        This is CRITICAL for understanding GP-VAR behavior on the consensus graph.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        ax : plt.Axes, optional
            Axes to plot on
        show_bands : bool
            If True, show low/mid/high frequency bands
        color : str
            Line color
            
        Returns
        -------
        plt.Figure
        """
        if group_name not in self.graph_frequency_metrics:
            raise ValueError(f"Graph frequency metrics not computed for '{group_name}'")
        
        gf = self.graph_frequency_metrics[group_name]
        eigenvalues = gf.eigenvalues
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
        
        n_eig = len(eigenvalues)
        x = np.arange(n_eig)
        
        # Main spectrum plot
        ax.plot(x, eigenvalues, color=color, marker='o', markersize=3, 
                linewidth=1.5, label='Graph Frequencies (λ)')
        
        # Mark key eigenvalues
        ax.axhline(gf.lambda_2, color='red', linestyle='--', alpha=0.7,
                   label=f'λ₂ (Algebraic Connectivity) = {gf.lambda_2:.4f}')
        ax.axhline(gf.lambda_max, color='orange', linestyle=':', alpha=0.7,
                   label=f'λ_max (Spectral Radius) = {gf.lambda_max:.2f}')
        
        if show_bands:
            # Show frequency bands
            low_idx = n_eig // 4
            high_idx = 3 * n_eig // 4
            
            ax.axvspan(0, low_idx, alpha=0.1, color='blue', 
                      label='Low freq (global)')
            ax.axvspan(low_idx, high_idx, alpha=0.1, color='green',
                      label='Mid freq')
            ax.axvspan(high_idx, n_eig, alpha=0.1, color='red',
                      label='High freq (local)')
        
        ax.set_xlabel('Eigenvalue Index', fontsize=12)
        ax.set_ylabel('λ (Graph Frequency)', fontsize=12)
        ax.set_title(f'{group_name} - Graph Frequency Spectrum\n'
                    f'(Laplacian Eigenvalues for GP-VAR)', fontsize=14)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        sp = self.sparsity_metrics.get(group_name, SparsityMetrics())
        textstr = (f'Natural Sparsity: {sp.sparsity_percent:.1f}%\n'
                  f'Spectral Entropy: {gf.spectral_entropy:.3f}\n'
                  f'Low/High Ratio: {gf.low_high_ratio:.3f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        return fig
    
    def plot_frequency_band_comparison(self,
                                       groups: Optional[List[str]] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare graph frequency band energy across groups.
        
        Useful for AD analysis: AD often shows reduced low-frequency (global)
        energy with preserved/increased high-frequency (local) energy.
        """
        if groups is None:
            groups = list(self.graph_frequency_metrics.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Bar chart of frequency band energies
        x = np.arange(len(groups))
        width = 0.25
        
        low_energies = [self.graph_frequency_metrics[g].low_freq_energy for g in groups]
        mid_energies = [self.graph_frequency_metrics[g].mid_freq_energy for g in groups]
        high_energies = [self.graph_frequency_metrics[g].high_freq_energy for g in groups]
        
        axes[0].bar(x - width, low_energies, width, label='Low Freq (Global)', color='blue', alpha=0.7)
        axes[0].bar(x, mid_energies, width, label='Mid Freq', color='green', alpha=0.7)
        axes[0].bar(x + width, high_energies, width, label='High Freq (Local)', color='red', alpha=0.7)
        
        axes[0].set_xlabel('Group')
        axes[0].set_ylabel('Frequency Band Energy')
        axes[0].set_title('Graph Frequency Band Energy by Group')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(groups)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: Low/High ratio (key AD metric)
        ratios = [self.graph_frequency_metrics[g].low_high_ratio for g in groups]
        colors = ['red' if 'AD' in g.upper() else 'blue' for g in groups]
        
        axes[1].bar(x, ratios, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Group')
        axes[1].set_ylabel('Low/High Frequency Ratio')
        axes[1].set_title('Low/High Frequency Ratio\n(AD typically shows lower ratio)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(groups)
        axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Equal balance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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
        
        # Get sparsity, AD, and graph frequency metrics if available
        sp = self.sparsity_metrics.get(group_name, SparsityMetrics())
        ad = self.ad_metrics.get(group_name, ADConnectivityMetrics())
        gf = self.graph_frequency_metrics.get(group_name, GraphFrequencyMetrics())
        
        # Add sparsity section (CRITICAL FOR AD RESEARCH)
        lines.extend([
            "## Sparsity Analysis (Critical for AD Research)",
            "",
            "### Why Sparsity Matters for Alzheimer's Disease",
            "",
            "Sparse connectivity matrices are essential for AD research because:",
            "1. **Disconnection Hypothesis**: AD is fundamentally a disconnection syndrome",
            "2. **Noise Reduction**: Removes spurious correlations from volume conduction",
            "3. **Hub Vulnerability**: Sparse networks reveal hub disruption patterns",
            "4. **Clinical Interpretability**: Sparse networks can be visualized and interpreted",
            "",
            "### Sparsity Metrics",
            f"- Possible edges: {sp.n_possible_edges}",
            f"- Actual edges retained: {sp.n_actual_edges}",
            f"- **Edge density: {sp.sparsity_percent:.2f}%** (target: 5-15% for AD analysis)",
            f"- Mean edge weight: {sp.mean_edge_weight:.4f}",
            f"- Median edge weight: {sp.median_edge_weight:.4f}",
            f"- Weight coefficient of variation: {sp.weight_cv:.4f}",
            "",
            "### Consensus-Based Edge Reliability",
            f"- Edges with full consensus (C=1): {sp.edges_full_consensus} (present in ALL subjects)",
            f"- Edges with majority consensus (C>0.5): {sp.edges_majority_consensus}",
            f"- Edges with any presence (C>0): {sp.edges_any_consensus}",
            f"- Never-present edges (C=0): {sp.edges_zero_consensus}",
            f"- Mean consensus for positive edges: {sp.mean_consensus_positive:.4f}",
            "",
            "### Distance-Dependent Sparsity (AD shows long-range disconnection)",
            f"- Short-range edge density: {sp.short_range_density:.4f}",
            f"- Medium-range edge density: {sp.medium_range_density:.4f}",
            f"- **Long-range edge density: {sp.long_range_density:.4f}** (reduced in AD)",
            "",
            "### Hub-Related Sparsity (AD shows hub vulnerability)",
            f"- Fraction of edges connected to hubs: {sp.hub_edge_fraction:.4f}",
            f"- Fraction of peripheral-only edges: {sp.peripheral_edge_fraction:.4f}",
            ""
        ])
        
        # Add AD-specific metrics section
        lines.extend([
            "## Alzheimer's Disease Connectivity Metrics",
            "",
            "### Disconnection Measures",
            f"- Global connectivity strength: {ad.global_connectivity_strength:.4f}",
            f"- Mean connection strength: {ad.mean_connection_strength:.4f}",
            f"- **Connectivity loss index: {ad.connectivity_loss_index:.4f}** (higher = more disconnection)",
            "",
            "### Hub Vulnerability (AD preferentially affects hubs)",
            f"- Number of hub nodes: {ad.n_hubs}",
            f"- Hub strength ratio (hub/peripheral): {ad.hub_strength_ratio:.4f}",
            f"- Hub degree ratio: {ad.hub_degree_ratio:.4f}",
            "",
            "### Integration vs Segregation Balance",
            f"- Integration index (global efficiency): {ad.integration_index:.4f}",
            f"- Segregation index (clustering × modularity): {ad.segregation_index:.4f}",
            f"- **Integration/Segregation ratio: {ad.integration_segregation_ratio:.4f}**",
            "",
            "### Long-Range Connectivity (specifically affected in AD)",
            f"- Long-range connection strength: {ad.long_range_strength:.4f}",
            f"- Short-range connection strength: {ad.short_range_strength:.4f}",
            f"- **Long/Short ratio: {ad.long_short_ratio:.4f}** (typically reduced in AD)",
            "",
            "### Network Resilience",
            f"- Robustness to random attack: {ad.robustness_random:.4f}",
            f"- **Robustness to targeted (hub) attack: {ad.robustness_targeted:.4f}** (lower in AD)",
            "",
            "### Rich-Club Organization",
            f"- Rich-club coefficient: {ad.rich_club_coefficient:.4f}",
            ""
        ])
        
        # Add Graph Frequency Spectrum section (CRITICAL for GP-VAR)
        lines.extend([
            "## Graph Frequency Spectrum (Critical for GP-VAR Analysis)",
            "",
            "### Why Graph Frequencies Matter",
            "",
            "The consensus Laplacian's eigenvalues define the **graph frequency spectrum**:",
            "- Eigenvalues λ₀ ≤ λ₁ ≤ ... ≤ λₙ₋₁ are the graph frequencies",
            "- **Low frequencies** (small λ) = smooth, global signals across connected nodes",
            "- **High frequencies** (large λ) = rapid, local variation between neighbors",
            "- GP-VAR uses this spectrum for graph-based temporal filtering",
            "",
            "### Spectral Landmarks",
            f"- Number of eigenvalues: {gf.n_eigenvalues}",
            f"- λ_min: {gf.lambda_min:.6f} (should be ~0 for connected graph)",
            f"- **λ₂ (Algebraic connectivity): {gf.lambda_2:.6f}** (key for network integration)",
            f"- λ_median: {gf.lambda_median:.4f}",
            f"- λ_mean: {gf.lambda_mean:.4f}",
            f"- λ_max (Spectral radius): {gf.lambda_max:.4f}",
            "",
            "### Spectral Distribution",
            f"- Spectral gap (λ₂ - λ₁): {gf.spectral_gap:.6f}",
            f"- Spectral range: {gf.spectral_range:.4f}",
            f"- Spectral variance: {gf.spectral_variance:.4f}",
            f"- Near-zero eigenvalues: {gf.n_zero_eigenvalues} (= number of components)",
            "",
            "### Graph Frequency Band Analysis",
            f"- Low-frequency energy (bottom 25%): {gf.low_freq_energy:.4f}",
            f"- Mid-frequency energy (middle 50%): {gf.mid_freq_energy:.4f}",
            f"- High-frequency energy (top 25%): {gf.high_freq_energy:.4f}",
            f"- **Low/High ratio: {gf.low_high_ratio:.4f}** (AD shows altered ratio)",
            "",
            "### GP-VAR Filter Properties",
            f"- Effective bandwidth (90% energy): {gf.effective_bandwidth:.4f}",
            f"- Spectral entropy: {gf.spectral_entropy:.4f} (uniformity of spectrum)",
            "",
            "### Interpretation for AD",
            "- Lower algebraic connectivity (λ₂) indicates weaker global integration",
            "- AD typically shows reduced low-frequency energy (disrupted global processing)",
            "- Preserved high-frequency energy may reflect compensatory local processing",
            "- Natural sparsity preserves the true spectrum - DO NOT use artificial thresholds!",
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
            
            # Sparsity interpretation for AD
            sparsity_interpretation = (
                "within the recommended range (5-15%) for detecting AD-related network changes"
                if 5 <= sp.sparsity_percent <= 15
                else "below the typical range, which may indicate very conservative thresholding"
                if sp.sparsity_percent < 5
                else "above the typical range, which may include noise-related connections"
            )
            
            long_range_interpretation = (
                "potentially indicating preserved long-range connectivity"
                if sp.long_range_density >= sp.short_range_density * 0.7
                else "suggesting reduced long-range connectivity consistent with disconnection patterns"
            )
            
            lines.extend([
                "## Thesis-Ready Summary",
                "",
                "### Network Structure",
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
                "### Efficiency and Small-World Properties",
                "",
                f"Global efficiency ({gm.global_efficiency:.4f}) and local efficiency ({gm.local_efficiency:.4f}) "
                f"characterize the network's integration and segregation properties. "
                f"The weighted clustering coefficient is {gm.clustering_coefficient:.4f}, "
                f"and modularity analysis reveals {gm.n_communities} communities (Q = {gm.modularity:.3f}). "
                f"The small-world sigma statistic is {gm.small_world_sigma:.3f}, {small_world_text}.",
                "",
                "### Spectral Properties",
                "",
                f"Spectral analysis of the graph Laplacian shows algebraic connectivity "
                f"λ₂ = {sm.algebraic_connectivity:.5f}, with {sm.near_zero_modes} near-zero modes "
                f"corresponding to the {gm.n_components} connected component(s). "
                f"The spectral radius (λ_max = {sm.spectral_radius:.3f}) bounds the "
                f"high-frequency content available to graph-based signal processing methods.",
                "",
                "### Clinical Relevance for Alzheimer's Disease",
                "",
                f"The network sparsity of {sp.sparsity_percent:.2f}% is {sparsity_interpretation}. "
                f"The consensus approach retained edges present in {sp.mean_consensus_positive:.1%} of subjects on average, "
                f"ensuring that the final graph represents robust, reproducible connectivity patterns rather than noise.",
                "",
                f"Distance-dependent analysis reveals short-range density of {sp.short_range_density:.4f} versus "
                f"long-range density of {sp.long_range_density:.4f}, {long_range_interpretation}. "
                f"This is particularly relevant as AD is characterized by preferential loss of long-range connections, "
                f"especially in frontoparietal circuits.",
                "",
                f"The network contains {ad.n_hubs} hub nodes (degree > mean + 1 SD), with hub-to-peripheral "
                f"strength ratio of {ad.hub_strength_ratio:.2f}. Hub vulnerability analysis shows robustness of "
                f"{ad.robustness_targeted:.3f} to targeted attack, compared to {ad.robustness_random:.3f} for random attack. "
                f"{'Lower targeted robustness suggests the network may be vulnerable to hub failure, consistent with AD pathophysiology.' if ad.robustness_targeted < ad.robustness_random else 'Similar robustness values suggest a distributed network architecture.'}",
                "",
                f"The integration/segregation ratio of {ad.integration_segregation_ratio:.3f} quantifies the balance "
                f"between global information integration and local modular processing. AD typically disrupts this balance, "
                f"showing reduced integration while potentially preserving (or even increasing) local clustering as a "
                f"compensatory mechanism.",
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
        
        # Sparsity metrics (critical for AD analysis)
        if self.sparsity_metrics:
            rows = []
            for group_name, metrics in self.sparsity_metrics.items():
                rows.append({
                    'group': group_name,
                    'n_possible_edges': metrics.n_possible_edges,
                    'n_actual_edges': metrics.n_actual_edges,
                    'edge_density': metrics.edge_density,
                    'sparsity_percent': metrics.sparsity_percent,
                    'edges_full_consensus': metrics.edges_full_consensus,
                    'edges_majority_consensus': metrics.edges_majority_consensus,
                    'mean_consensus': metrics.mean_consensus,
                    'mean_consensus_positive': metrics.mean_consensus_positive,
                    'short_range_density': metrics.short_range_density,
                    'medium_range_density': metrics.medium_range_density,
                    'long_range_density': metrics.long_range_density,
                    'mean_edge_weight': metrics.mean_edge_weight,
                    'weight_cv': metrics.weight_cv,
                    'hub_edge_fraction': metrics.hub_edge_fraction,
                    'peripheral_edge_fraction': metrics.peripheral_edge_fraction
                })
            
            df = pd.DataFrame(rows)
            filepath = output_path / f"{prefix}_sparsity_metrics.csv"
            df.to_csv(filepath, index=False)
            saved_files['sparsity_metrics'] = str(filepath)
        
        # AD-specific metrics
        if self.ad_metrics:
            rows = []
            for group_name, metrics in self.ad_metrics.items():
                rows.append({
                    'group': group_name,
                    'global_connectivity_strength': metrics.global_connectivity_strength,
                    'mean_connection_strength': metrics.mean_connection_strength,
                    'connectivity_loss_index': metrics.connectivity_loss_index,
                    'n_hubs': metrics.n_hubs,
                    'hub_strength_ratio': metrics.hub_strength_ratio,
                    'hub_degree_ratio': metrics.hub_degree_ratio,
                    'integration_index': metrics.integration_index,
                    'segregation_index': metrics.segregation_index,
                    'integration_segregation_ratio': metrics.integration_segregation_ratio,
                    'long_range_strength': metrics.long_range_strength,
                    'short_range_strength': metrics.short_range_strength,
                    'long_short_ratio': metrics.long_short_ratio,
                    'robustness_random': metrics.robustness_random,
                    'robustness_targeted': metrics.robustness_targeted,
                    'rich_club_coefficient': metrics.rich_club_coefficient
                })
            
            df = pd.DataFrame(rows)
            filepath = output_path / f"{prefix}_ad_connectivity_metrics.csv"
            df.to_csv(filepath, index=False)
            saved_files['ad_metrics'] = str(filepath)
        
        # Graph frequency spectrum metrics (CRITICAL for GP-VAR)
        if self.graph_frequency_metrics:
            rows = []
            for group_name, metrics in self.graph_frequency_metrics.items():
                rows.append({
                    'group': group_name,
                    'n_eigenvalues': metrics.n_eigenvalues,
                    'lambda_min': metrics.lambda_min,
                    'lambda_2_algebraic_connectivity': metrics.lambda_2,
                    'lambda_median': metrics.lambda_median,
                    'lambda_mean': metrics.lambda_mean,
                    'lambda_max_spectral_radius': metrics.lambda_max,
                    'spectral_gap': metrics.spectral_gap,
                    'spectral_range': metrics.spectral_range,
                    'spectral_variance': metrics.spectral_variance,
                    'n_zero_eigenvalues': metrics.n_zero_eigenvalues,
                    'low_freq_energy': metrics.low_freq_energy,
                    'mid_freq_energy': metrics.mid_freq_energy,
                    'high_freq_energy': metrics.high_freq_energy,
                    'low_high_ratio': metrics.low_high_ratio,
                    'effective_bandwidth': metrics.effective_bandwidth,
                    'spectral_entropy': metrics.spectral_entropy
                })
            
            df = pd.DataFrame(rows)
            filepath = output_path / f"{prefix}_graph_frequency_metrics.csv"
            df.to_csv(filepath, index=False)
            saved_files['graph_frequency_metrics'] = str(filepath)
        
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
            sp = self.sparsity_metrics.get(group_name, SparsityMetrics())
            gf = self.graph_frequency_metrics.get(group_name, GraphFrequencyMetrics())
            ad = self.ad_metrics.get(group_name, ADConnectivityMetrics())
            
            rows.append({
                'group': group_name,
                'n_nodes': gm.n_nodes,
                'n_edges': gm.n_edges,
                # SPARSITY (critical for AD)
                'sparsity_percent': sp.sparsity_percent,
                'long_range_density': sp.long_range_density,
                # Graph metrics
                'degree_mean': gm.degree_mean,
                'strength_mean': gm.strength_mean,
                'n_components': gm.n_components,
                'global_efficiency': gm.global_efficiency,
                'clustering': gm.clustering_coefficient,
                'modularity': gm.modularity,
                'small_world_sigma': gm.small_world_sigma,
                # Graph frequency (for GP-VAR)
                'lambda_2_algebraic_connectivity': gf.lambda_2,
                'spectral_radius': gf.lambda_max,
                'low_high_freq_ratio': gf.low_high_ratio,
                'spectral_entropy': gf.spectral_entropy,
                # AD-specific
                'connectivity_loss_index': ad.connectivity_loss_index,
                'hub_strength_ratio': ad.hub_strength_ratio,
                'robustness_targeted': ad.robustness_targeted
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
        
        # Graph frequency spectrum (CRITICAL for GP-VAR)
        for group in groups:
            if group in analyzer.graph_frequency_metrics:
                analyzer.plot_graph_frequency_spectrum(group)
                plt.savefig(output_path / f"{group}_graph_frequency_spectrum.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Graph frequency band comparison (AD vs HC)
        if len(groups) > 1 and analyzer.graph_frequency_metrics:
            analyzer.plot_frequency_band_comparison(
                groups,
                save_path=str(output_path / "graph_frequency_band_comparison.png")
            )
            plt.close()
    
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
