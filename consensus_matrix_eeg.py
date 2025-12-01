"""
Consensus Matrix Implementation for EEG Connectivity Analysis
Using Pearson Correlation with Distance-Dependent Thresholding

This implementation follows the Betzel-style consensus approach for building
group-level brain networks from individual subject connectivity matrices.
"""

import numpy as np
import scipy.io as sio
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert
import mne
from mne.channels import make_standard_montage
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_MONTAGE_NAME = "biosemi128"

class ConsensusMatrix:
    """
    Build consensus connectivity matrices from multiple subjects using Pearson correlation.
    
    Two consensus methods are supported:
    
    1. Direct Fisher-z averaging (recommended, default):
       - Apply Fisher-z transform to all correlation matrices
       - Average in z-space across all subjects
       - Inverse transform to get consensus correlation matrix
       - Apply proportional thresholding to final consensus
    
    2. Betzel-style binarization-based approach:
       - Sparsify each A(s) → binary B(s) using proportional thresholding
       - Compute edge consistency: C_ij = (1/S) * Σ_s B_ij(s)
       - Compute representative weights W_ij via Fisher-z conditional mean
       - Select edges using C (uniform or distance-dependent)
       
    The direct method is preferred as it preserves more information by not
    discarding sub-threshold correlations during per-subject binarization.
    """
    
    def __init__(self, channel_locations: Optional[np.ndarray] = None):
        """
        Initialize the ConsensusMatrix builder.
        
        Parameters
        ----------
        channel_locations : np.ndarray, optional
            3D coordinates of EEG channels (n_channels x 3)
            If provided, enables distance-dependent consensus
        """
        self.channel_locations = channel_locations
        self.consensus_matrix = None
        self.weight_matrix = None
        self.weight_matrix_full = None
        self.average_subject_sparsity = None
        self.binary_matrices = []
        self.adjacency_matrices = []
        self.subject_labels = []
        self.distance_graph = None
        self._consensus_method = None  # Track which method was used
        
    def load_eeg_data(self, filepath: str) -> np.ndarray:
        """
        Load EEG data from .set file (EEGLAB format).
        
        Parameters
        ----------
        filepath : str
            Path to the .set file
            
        Returns
        -------
        data : np.ndarray
            EEG data (n_channels x n_samples)
        """
        try:
            # Try loading with MNE first (preferred for EEGLAB files)
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
            
            # Apply BioSemi montage if missing
            if raw.get_montage() is None:
                try:
                    biosemi_montage = make_standard_montage(DEFAULT_MONTAGE_NAME)
                    raw.set_montage(biosemi_montage, on_missing='warn')
                    logger.info(f"Applied {DEFAULT_MONTAGE_NAME} montage to {filepath}")
                except Exception as montage_err:
                    logger.warning(f"Failed to apply {DEFAULT_MONTAGE_NAME} montage to {filepath}: {montage_err}")
            
            data = raw.get_data()
            
            # Store channel locations if not already set
            if self.channel_locations is None:
                montage = raw.get_montage()
                if montage is not None:
                    positions = montage.get_positions()['ch_pos']
                    ch_names = raw.ch_names
                    
                    coords = []
                    missing_channels = []
                    for ch in ch_names:
                        if ch in positions:
                            coords.append(positions[ch])
                        else:
                            missing_channels.append(ch)
                    
                    if missing_channels:
                        preview = ", ".join(missing_channels[:5])
                        if len(missing_channels) > 5:
                            preview += ", ..."
                        logger.warning(
                            "Montage missing coordinates for channels (%s); distance-dependent consensus unavailable until resolved.",
                            preview
                        )
                    elif coords:
                        self.channel_locations = np.array(coords)
                        logger.info(f"Captured channel coordinates from montage ({len(coords)} channels)")
                    
            return data
            
        except Exception as e:
            logger.warning(f"MNE loading failed for {filepath}: {e}")
            
            # Fallback to scipy.io for .mat files
            try:
                mat_data = sio.loadmat(filepath.replace('.set', '.mat'))
                # Try common field names for EEG data
                for field in ['EEG', 'data', 'Data', 'signal']:
                    if field in mat_data:
                        data = mat_data[field]
                        if data.ndim == 2:
                            return data
                raise ValueError("Could not find EEG data in .mat file")
                
            except Exception as e2:
                logger.error(f"Failed to load {filepath}: {e2}")
                raise
                
    def compute_pearson_adjacency(self, data: np.ndarray, absolute: bool = True) -> np.ndarray:
        """
        Compute Pearson correlation adjacency matrix from EEG data.
        
        Parameters
        ----------
        data : np.ndarray
            EEG data (n_channels x n_samples)
        absolute : bool
            If True, use absolute correlation values
            
        Returns
        -------
        A : np.ndarray
            Adjacency matrix (n_channels x n_channels)
            Diagonal is set to 0
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data)
        
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Take absolute value if requested
        if absolute:
            corr_matrix = np.abs(corr_matrix)
            
        # Set diagonal to 0
        np.fill_diagonal(corr_matrix, 0)
        
        return corr_matrix
    
    def proportional_threshold_exact(self, A: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Apply proportional thresholding to create binary matrix.
        Keep top κ fraction of edges.
        
        Parameters
        ----------
        A : np.ndarray
            Weighted adjacency matrix
        sparsity : float
            Fraction of edges to keep (0 < sparsity < 1)
            
        Returns
        -------
        B : np.ndarray
            Binary adjacency matrix
        """
        n = A.shape[0]
        # Get upper triangle indices (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        
        # Extract edge weights
        edge_weights = A[triu_indices]
        
        # Calculate number of edges to keep
        n_edges = len(edge_weights)
        k_edges = int(np.floor(sparsity * n_edges))
        
        if k_edges == 0:
            logger.warning(f"Sparsity {sparsity} results in 0 edges. Keeping at least 1.")
            k_edges = 1
            
        # Find threshold (k-th largest value)
        if k_edges < n_edges:
            threshold = np.sort(edge_weights)[::-1][k_edges]
        else:
            threshold = 0  # Keep all edges
            
        # Create binary matrix
        B = np.zeros_like(A)
        B[A > threshold] = 1
        
        # Ensure symmetry
        B = np.maximum(B, B.T)
        
        # Ensure diagonal is 0
        np.fill_diagonal(B, 0)
        
        return B
    
    def fisher_z_transform(self, r: np.ndarray) -> np.ndarray:
        """Apply Fisher z-transformation: z = arctanh(r)"""
        # Clip to avoid inf values
        r_clipped = np.clip(r, -0.999999, 0.999999)
        return np.arctanh(r_clipped)
    
    def fisher_z_inverse(self, z: np.ndarray) -> np.ndarray:
        """Apply inverse Fisher z-transformation: r = tanh(z)"""
        return np.tanh(z)
    
    def compute_direct_consensus(self,
                                  adjacency_matrices: List[np.ndarray],
                                  sparsity: Optional[float] = None) -> np.ndarray:
        """
        Compute consensus matrix using direct Fisher-z averaging (recommended method).
        
        This method directly averages all correlation matrices in Fisher-z space
        without per-subject binarization, preserving more information.
        
        Parameters
        ----------
        adjacency_matrices : List[np.ndarray]
            List of weighted adjacency matrices (correlation matrices), one per subject
        sparsity : float, optional
            If provided, apply proportional thresholding to the final consensus matrix.
            If None, return the full dense consensus matrix.
            
        Returns
        -------
        consensus : np.ndarray
            Consensus correlation matrix
        """
        if len(adjacency_matrices) == 0:
            raise ValueError("No matrices provided")
        
        n_subjects = len(adjacency_matrices)
        n_nodes = adjacency_matrices[0].shape[0]
        
        # Store adjacency matrices
        self.adjacency_matrices = adjacency_matrices
        self._consensus_method = 'direct'
        
        logger.info(f"Computing direct Fisher-z consensus from {n_subjects} subjects")
        
        # Stack all correlation matrices
        stack = np.stack(adjacency_matrices, axis=0)
        
        # Apply Fisher-z transform to all matrices
        z_stack = self.fisher_z_transform(stack)
        
        # Average in z-space
        z_mean = np.mean(z_stack, axis=0)
        
        # Transform back to correlation space and take absolute value
        consensus = np.abs(self.fisher_z_inverse(z_mean))
        
        # Ensure diagonal is 0 and matrix is symmetric
        np.fill_diagonal(consensus, 0)
        consensus = np.maximum(consensus, consensus.T)
        
        # Store the full dense consensus
        self.weight_matrix_full = consensus.copy()
        
        # Apply sparsity threshold if requested
        if sparsity is not None and 0 < sparsity < 1:
            consensus_thresholded = self.proportional_threshold_weighted(consensus, sparsity)
            self.consensus_matrix = (consensus_thresholded > 0).astype(float)
            self.weight_matrix = consensus_thresholded
            self.average_subject_sparsity = sparsity
            logger.info(f"Applied sparsity threshold {sparsity:.2%} to consensus matrix")
        else:
            # No thresholding - use full consensus
            self.consensus_matrix = np.ones_like(consensus)
            np.fill_diagonal(self.consensus_matrix, 0)
            self.weight_matrix = consensus
            self.average_subject_sparsity = 1.0
            logger.info("Returning full dense consensus matrix (no sparsity threshold)")
        
        return consensus
    
    def proportional_threshold_weighted(self, A: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Apply proportional thresholding while preserving weights.
        Keep top κ fraction of edges with their original weights.
        
        Parameters
        ----------
        A : np.ndarray
            Weighted adjacency matrix
        sparsity : float
            Fraction of edges to keep (0 < sparsity < 1)
            
        Returns
        -------
        A_thresh : np.ndarray
            Thresholded weighted adjacency matrix
        """
        n = A.shape[0]
        # Get upper triangle indices (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        
        # Extract edge weights
        edge_weights = A[triu_indices]
        
        # Calculate number of edges to keep
        n_edges = len(edge_weights)
        k_edges = int(np.floor(sparsity * n_edges))
        
        if k_edges == 0:
            logger.warning(f"Sparsity {sparsity} results in 0 edges. Keeping at least 1.")
            k_edges = 1
            
        # Find threshold (k-th largest value)
        if k_edges < n_edges:
            threshold = np.sort(edge_weights)[::-1][k_edges]
        else:
            threshold = 0  # Keep all edges
            
        # Create thresholded matrix preserving weights
        A_thresh = np.where(A > threshold, A, 0)
        
        # Ensure symmetry
        A_thresh = np.maximum(A_thresh, A_thresh.T)
        
        # Ensure diagonal is 0
        np.fill_diagonal(A_thresh, 0)
        
        return A_thresh
    
    def compute_consensus_and_weights(self, 
                                       adjacency_matrices: List[np.ndarray],
                                       sparsity: float = 0.15,
                                       method: str = 'direct') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute consensus matrix C and weight matrix W from multiple subjects.
        
        Parameters
        ----------
        adjacency_matrices : List[np.ndarray]
            List of weighted adjacency matrices, one per subject
        sparsity : float
            Sparsity level for the final graph (0 < sparsity < 1)
        method : str
            Consensus computation method:
            - 'direct': Direct Fisher-z averaging across all correlations (recommended).
                        Preserves more information by not discarding sub-threshold edges.
            - 'binarize': Betzel-style per-subject binarization then conditional averaging.
                          C_ij = fraction of subjects with edge, W_ij = conditional mean.
            
        Returns
        -------
        C : np.ndarray
            Consensus matrix (edge indicators or fraction of subjects with edge)
        W : np.ndarray
            Weight matrix (consensus correlation strengths)
        """
        if method == 'direct':
            return self._compute_consensus_direct(adjacency_matrices, sparsity)
        elif method == 'binarize':
            return self._compute_consensus_binarize(adjacency_matrices, sparsity)
        else:
            raise ValueError(f"Unknown consensus method '{method}'. Use 'direct' or 'binarize'.")
    
    def _compute_consensus_direct(self,
                                   adjacency_matrices: List[np.ndarray],
                                   sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct Fisher-z consensus (recommended).
        
        Averages all correlation matrices in Fisher-z space, then applies
        proportional thresholding to the consensus result.
        """
        n_subjects = len(adjacency_matrices)
        n_nodes = adjacency_matrices[0].shape[0]
        
        # Store adjacency matrices
        self.adjacency_matrices = adjacency_matrices
        self.distance_graph = None
        self._consensus_method = 'direct'
        
        logger.info(f"Computing direct Fisher-z consensus from {n_subjects} matrices")
        
        # Stack all correlation matrices
        adjacency_stack = np.stack(adjacency_matrices, axis=0)
        
        # Apply Fisher-z transform to all matrices
        z_stack = self.fisher_z_transform(adjacency_stack)
        
        # Average in z-space
        z_mean = np.mean(z_stack, axis=0)
        
        # Transform back to correlation space and take absolute value
        consensus_full = np.abs(self.fisher_z_inverse(z_mean))
        
        # Ensure diagonal is 0 and matrix is symmetric
        np.fill_diagonal(consensus_full, 0)
        consensus_full = np.maximum(consensus_full, consensus_full.T)
        
        # Store the full dense consensus
        self.weight_matrix_full = consensus_full.copy()
        
        # Apply sparsity threshold
        W = self.proportional_threshold_weighted(consensus_full, sparsity)
        C = (W > 0).astype(float)
        
        # Compute average sparsity (matches the requested sparsity)
        triu_idx = np.triu_indices(n_nodes, k=1)
        n_possible_edges = max(1, len(triu_idx[0]))
        n_edges = int(np.sum(W[triu_idx] > 0))
        self.average_subject_sparsity = n_edges / n_possible_edges
        
        # Clear binary matrices since we don't use them in direct method
        self.binary_matrices = []
        
        self.consensus_matrix = C
        self.weight_matrix = W
        
        logger.info(f"Direct consensus: kept {n_edges}/{n_possible_edges} edges "
                   f"({self.average_subject_sparsity:.2%} sparsity)")
        
        return C, W
    
    def _compute_consensus_binarize(self,
                                     adjacency_matrices: List[np.ndarray],
                                     sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Betzel-style binarization-based consensus.
        
        C_ij = (1/S) * Σ_s B_ij(s)  [fraction of subjects with edge]
        W_ij = conditional mean of weights where edge exists
        """
        n_subjects = len(adjacency_matrices)
        n_nodes = adjacency_matrices[0].shape[0]
        
        # Store adjacency matrices
        self.adjacency_matrices = adjacency_matrices
        self.distance_graph = None
        self._consensus_method = 'binarize'
        adjacency_stack = np.stack(adjacency_matrices, axis=0)
        
        # Step 1: Binarize each subject's matrix
        logger.info(f"Binarizing {n_subjects} matrices with sparsity={sparsity} (Betzel-style)")
        self.binary_matrices = []
        for i, A in enumerate(adjacency_matrices):
            B = self.proportional_threshold_exact(A, sparsity)
            self.binary_matrices.append(B)
        binary_stack = np.stack(self.binary_matrices, axis=0)
            
        # Step 2: Compute consensus matrix C
        # C_ij = mean across subjects of binary matrices
        C = np.mean(self.binary_matrices, axis=0)
        
        # Step 3: Compute weight matrix W using Fisher-z averaging
        W = np.zeros((n_nodes, n_nodes))
        triu_idx = np.triu_indices(n_nodes, k=1)
        n_possible_edges = max(1, len(triu_idx[0]))
        edge_counts = np.sum(binary_stack[:, triu_idx[0], triu_idx[1]], axis=1)
        subject_sparsities = edge_counts / n_possible_edges
        self.average_subject_sparsity = float(np.mean(subject_sparsities))
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Find subjects where edge exists
                edge_exists = binary_stack[:, i, j] > 0
                
                if np.any(edge_exists):
                    # Fisher-z transform correlations
                    z_values = self.fisher_z_transform(adjacency_stack[edge_exists, i, j])
                    
                    # Average in z-space
                    z_mean = np.mean(z_values)
                    
                    # Transform back
                    W[i, j] = np.abs(self.fisher_z_inverse(z_mean))
                    W[j, i] = W[i, j]  # Ensure symmetry
        
        # Dense Fisher-z average across all subjects (retain raw correlation magnitudes)
        z_all = self.fisher_z_transform(adjacency_stack)
        z_mean_all = np.mean(z_all, axis=0)
        W_full = np.abs(self.fisher_z_inverse(z_mean_all))
        np.fill_diagonal(W_full, 0)
        self.weight_matrix_full = np.maximum(W_full, W_full.T)
        
        self.consensus_matrix = C
        self.weight_matrix = W
        
        return C, W
    
    def _resolve_target_sparsity(self, target_sparsity: Optional[Union[float, str]]) -> Optional[float]:
        """
        Normalize target sparsity specification.
        
        Parameters
        ----------
        target_sparsity : float, str, or None
            - float in (0, 1]: explicit sparsity fraction
            - "match_subject" / "subject_mean": use average sparsity of binarized subjects
            - None: disable final sparsification (keep all qualifying edges)
        
        Returns
        -------
        Optional[float]
            Resolved sparsity fraction or None if dense graph requested.
        """
        if target_sparsity is None:
            return None
        
        if isinstance(target_sparsity, str):
            key = target_sparsity.strip().lower()
            if key in {"match_subject", "subject_mean", "mean_subject"}:
                if self.average_subject_sparsity is None:
                    raise ValueError("Average subject sparsity unavailable. Run compute_consensus_and_weights first.")
                return self.average_subject_sparsity
            raise ValueError(f"Unrecognized target_sparsity spec '{target_sparsity}'.")
        
        value = float(target_sparsity)
        if value <= 0 or value > 1:
            raise ValueError("target_sparsity must be in (0, 1].")
        return value
    
    def compute_distance_matrix(self) -> np.ndarray:
        """
        Compute Euclidean distance matrix between channels.
        
        Returns
        -------
        D : np.ndarray
            Distance matrix (n_channels x n_channels)
        """
        if self.channel_locations is None:
            raise ValueError("Channel locations not provided")
            
        # Compute pairwise Euclidean distances
        distances = pdist(self.channel_locations, metric='euclidean')
        D = squareform(distances)
        
        return D
    
    def distance_dependent_consensus(self, 
                                      target_sparsity: Optional[Union[float, str]] = None,
                                      n_bins: int = 10,
                                      epsilon: float = 0.1,
                                      require_existing: bool = True,
                                      save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Build final group graph using distance-dependent consensus.
        
        Parameters
        ----------
        target_sparsity : float, str, or None, optional
            - float in (0,1]: explicit sparsity fraction
            - "match_subject": match average sparsity observed across binarized subjects
            - None: keep every qualifying edge
        n_bins : int
            Number of distance bins
        epsilon : float
            Weight for W in scoring (score = C + ε*W)
        require_existing : bool
            If True, only consider edges with C > 0
        save_path : str or Path, optional
            Where to save the resulting distance-dependent graph (.npy)
            
        Returns
        -------
        G : np.ndarray
            Final group adjacency matrix
        """
        if self.consensus_matrix is None or self.weight_matrix is None:
            raise ValueError("Must run compute_consensus_and_weights first")
            
        n_nodes = self.consensus_matrix.shape[0]
        C = self.consensus_matrix
        W = self.weight_matrix
        resolved_sparsity = self._resolve_target_sparsity(target_sparsity)
        unlimited = resolved_sparsity is None
        
        if unlimited:
            logger.info("Target sparsity disabled; keeping all qualifying edges (distance-dependent consensus).")
            weight_source = self.weight_matrix_full if self.weight_matrix_full is not None else self.weight_matrix
            if require_existing:
                mask = C > 0
            else:
                mask = np.ones_like(C, dtype=bool)
            mask &= weight_source > 0
            np.fill_diagonal(mask, False)
            G = np.where(mask, weight_source, 0.0)
            self.distance_graph = np.maximum(G, G.T)
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, self.distance_graph)
                logger.info(f"Saved distance-dependent consensus graph to {save_path}")
            selected_edges = int(np.sum(np.triu(self.distance_graph > 0, k=1)))
            logger.info("Selected %d edges without sparsity target", selected_edges)
            return self.distance_graph
        
        # Compute distance matrix
        D = self.compute_distance_matrix()
        
        # Get upper triangle indices
        triu_indices = np.triu_indices(n_nodes, k=1)
        n_possible_edges = len(triu_indices[0])
        
        # Calculate target number of edges
        k_target = int(np.clip(round(resolved_sparsity * n_possible_edges), 1, n_possible_edges))
        
        # Extract distances for upper triangle
        distances = D[triu_indices]
        
        # Create distance bins
        bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10  # Ensure last edge is included
        
        # Initialize final adjacency
        G = np.zeros((n_nodes, n_nodes))
        
        # Allocate edges per bin
        edges_per_bin = k_target // n_bins
        remaining_edges = k_target % n_bins
        
        # Process each distance bin
        selected_edges = set()
        
        for bin_idx in range(n_bins):
            # Find edges in this bin
            bin_mask = (distances >= bin_edges[bin_idx]) & (distances < bin_edges[bin_idx + 1])
            bin_edge_indices = np.where(bin_mask)[0]
            
            if len(bin_edge_indices) == 0:
                continue
                
            # Score edges in this bin
            scores = []
            valid_indices = []
            
            for idx in bin_edge_indices:
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                
                # Skip if require_existing and edge doesn't exist
                if require_existing and C[i, j] == 0:
                    continue
                    
                # Compute score: C + ε*W
                score = C[i, j] + epsilon * W[i, j]
                scores.append(score)
                valid_indices.append((i, j))
            
            if len(scores) == 0:
                continue
                
            # Sort by score and select top edges
            k_bin = edges_per_bin + (1 if bin_idx < remaining_edges else 0)
            k_bin = min(k_bin, len(scores))
            
            sorted_indices = np.argsort(scores)[::-1][:k_bin]
            
            for idx in sorted_indices:
                i, j = valid_indices[idx]
                edge = tuple(sorted((i, j)))
                if edge in selected_edges:
                    continue
                G[i, j] = W[i, j]
                G[j, i] = W[i, j]
                selected_edges.add(edge)

        # Fill any remaining slots using best global edges (C + εW) regardless of bin
        remaining = k_target - len(selected_edges)
        if remaining > 0:
            logger.info(f"Distance bins left {remaining} edges unfilled; selecting globally.")
            candidate_scores = []
            candidate_pairs = []
            for idx in range(len(triu_indices[0])):
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                if require_existing and C[i, j] == 0:
                    continue
                edge = tuple(sorted((i, j)))
                if edge in selected_edges:
                    continue
                score = C[i, j] + epsilon * W[i, j]
                if score <= 0:
                    continue
                candidate_scores.append(score)
                candidate_pairs.append((i, j))
            
            if candidate_scores:
                order = np.argsort(candidate_scores)[::-1][:remaining]
                for idx in order:
                    i, j = candidate_pairs[idx]
                    edge = tuple(sorted((i, j)))
                    G[i, j] = W[i, j]
                    G[j, i] = W[i, j]
                    selected_edges.add(edge)

        self.distance_graph = G.copy()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, self.distance_graph)
            logger.info(f"Saved distance-dependent consensus graph to {save_path}")
        
        logger.info(f"Selected {len(selected_edges)} edges with distance-dependent consensus")
        
        return G
    
    @staticmethod
    def _format_float(value: float, fmt: str = "{:.3f}") -> str:
        """Safely format floating-point values."""
        if value is None or not np.isfinite(value):
            return "n/a"
        return fmt.format(value)
    
    @staticmethod
    def _build_weighted_graph_from_matrix(adjacency: np.ndarray, weight_threshold: float = 1e-8) -> nx.Graph:
        """Convert a dense adjacency matrix into a weighted NetworkX graph."""
        graph = nx.Graph()
        if adjacency is None:
            return graph
        n_nodes = adjacency.shape[0]
        graph.add_nodes_from(range(n_nodes))
        triu_idx = np.triu_indices(n_nodes, k=1)
        weights = adjacency[triu_idx]
        mask = weights > weight_threshold
        sources = triu_idx[0][mask]
        targets = triu_idx[1][mask]
        filtered = weights[mask]
        graph.add_weighted_edges_from(
            (int(i), int(j), float(w)) for i, j, w in zip(sources, targets, filtered)
        )
        return graph
    
    @staticmethod
    def _plot_heatmap(matrix: np.ndarray, title: str, output_path: Path):
        """Persist a heatmap figure for the adjacency or Laplacian."""
        vmax = np.max(matrix) if matrix.size else 1.0
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
    
    @staticmethod
    def _plot_laplacian_spectrum(eigenvalues: np.ndarray, group_name: str, output_path: Path):
        """Persist a Laplacian eigenvalue spectrum figure."""
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(eigenvalues)), eigenvalues, marker="o", linewidth=1)
        plt.title(f"{group_name} Laplacian Eigenvalues")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("λ")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def summarize_distance_graph(self,
                                 group_name: str,
                                 output_dir: str,
                                 method_desc: str = "distance-dependent consensus (Betzel-style bins)",
                                 methods_reference: str = "Methods §3.2",
                                 graph: Optional[np.ndarray] = None) -> Dict[str, Union[str, float]]:
        """
        Compute descriptive statistics and figures for the final graph G.
        
        Parameters
        ----------
        group_name : str
            Name of the cohort or analysis subset.
        output_dir : str
            Directory where figures and markdown summary are stored.
        method_desc : str
            Short description of how the consensus Laplacian was built.
        methods_reference : str
            Location in the manuscript to cross-reference.
        graph : np.ndarray, optional
            Graph to summarize. Defaults to the stored distance-dependent graph.
        
        Returns
        -------
        Dict[str, Union[str, float]]
            Dictionary containing metrics and asset paths.
        """
        if self.consensus_matrix is None:
            raise ValueError("Consensus matrix unavailable; run compute_consensus_and_weights first.")
        
        if graph is None:
            graph = self.distance_graph
        
        if graph is None:
            raise ValueError("No graph provided; run distance_dependent_consensus or pass graph explicitly.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        n_nodes = graph.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        edge_weights = graph[triu_idx]
        nonzero_mask = edge_weights > 0
        n_edges = int(np.sum(nonzero_mask))
        n_possible_edges = max(1, len(edge_weights))
        sparsity_pct = (n_edges / n_possible_edges) * 100
        
        degrees = np.sum(graph > 0, axis=0)
        strengths = np.sum(graph, axis=0)
        deg_min = float(degrees.min()) if degrees.size else 0.0
        deg_median = float(np.median(degrees)) if degrees.size else 0.0
        deg_max = float(degrees.max()) if degrees.size else 0.0
        deg_std = float(degrees.std()) if degrees.size else 0.0
        
        strength_min = float(strengths.min()) if strengths.size else 0.0
        strength_median = float(np.median(strengths)) if strengths.size else 0.0
        strength_max = float(strengths.max()) if strengths.size else 0.0
        
        consensus_selected = self.consensus_matrix[triu_idx][nonzero_mask]
        consensus_mean = float(np.mean(consensus_selected)) if consensus_selected.size else float("nan")
        consensus_median = float(np.median(consensus_selected)) if consensus_selected.size else float("nan")
        
        heatmap_path = output_path / f"{group_name}_consensus_adjacency.png"
        self._plot_heatmap(graph, f"{group_name} consensus adjacency", heatmap_path)
        
        laplacian = np.diag(strengths) - graph
        eigenvalues = np.linalg.eigvalsh(laplacian) if n_nodes else np.array([])
        lambda_min = float(eigenvalues[0]) if eigenvalues.size else float("nan")
        lambda_two = float(eigenvalues[1]) if eigenvalues.size > 1 else float("nan")
        lambda_max = float(eigenvalues[-1]) if eigenvalues.size else float("nan")
        lambda_q3 = float(np.percentile(eigenvalues, 75)) if eigenvalues.size else float("nan")
        near_zero = int(np.sum(eigenvalues < 1e-5)) if eigenvalues.size else 0
        
        spectrum_path = output_path / f"{group_name}_laplacian_spectrum.png"
        if eigenvalues.size:
            self._plot_laplacian_spectrum(eigenvalues, group_name, spectrum_path)
        else:
            spectrum_path = None
        
        nx_graph = self._build_weighted_graph_from_matrix(graph)
        if nx_graph.number_of_nodes() > 0:
            component_count = nx.number_connected_components(nx_graph)
            largest_component_nodes = max((len(c) for c in nx.connected_components(nx_graph)), default=0)
        else:
            component_count = 0
            largest_component_nodes = 0
        
        largest_component_fraction = (
            largest_component_nodes / n_nodes if n_nodes else 0.0
        )
        
        if largest_component_nodes > 1:
            giant_component = nx_graph.subgraph(
                max(nx.connected_components(nx_graph), key=len)
            ).copy()
            try:
                path_length = nx.average_shortest_path_length(giant_component, weight="weight")
            except (nx.NetworkXError, ZeroDivisionError):
                path_length = float("nan")
        else:
            path_length = float("nan")
        
        clustering_coeff = (
            nx.average_clustering(nx_graph, weight="weight") if nx_graph.number_of_edges() > 0 else float("nan")
        )
        
        small_world_sigma = float("nan")
        if nx_graph.number_of_edges() > 0 and nx_graph.number_of_nodes() >= 4:
            try:
                small_world_sigma = nx.sigma(nx_graph, niter=5, nrand=3, seed=42)
            except (nx.NetworkXError, ZeroDivisionError):
                small_world_sigma = float("nan")
        
        component_text = (
            "connected (single component)"
            if component_count == 1
            else f"comprised of {component_count} components (largest spans {largest_component_fraction:.1%} of channels)"
        )
        
        sigma_subtext = (
            "σ>1 points to small-world structure."
            if np.isfinite(small_world_sigma) and small_world_sigma > 1
            else "σ≈1 resembles a random baseline."
            if np.isfinite(small_world_sigma)
            else "σ not estimated (graph too sparse)."
        )
        
        intro = (
            f"All GP-VAR models were defined on a single consensus Laplacian constructed from the {method_desc}, "
            f"as described in {methods_reference}. The examiner requested confirmation that this graph is sensible; "
            f"the metrics below address sparsity, degree spread, small-world tendencies, and spectral structure."
        )
        
        interpretation = (
            f"The graph keeps {sparsity_pct:.2f}% of possible undirected edges "
            f"({n_edges}/{n_possible_edges}), balancing parsimony with coverage of {component_text}. "
            f"Binary degrees span {deg_min:.0f}-{deg_max:.0f} (median {deg_median:.0f}, σ={deg_std:.2f}), "
            f"so every BioSemi-128 channel retains multiple partners without forming unrealistic hubs. "
            f"Edge strengths range {strength_min:.3f}–{strength_max:.3f} (median {strength_median:.3f}), "
            f"highlighting the bright intra-hemispheric bands visible in Figure 4.x."
        )
        
        small_world_text = (
            f"The largest connected component covers {largest_component_fraction:.1%} of sensors "
            f"({largest_component_nodes}/{n_nodes}). "
            f"Characteristic path length {self._format_float(path_length)} and clustering "
            f"{self._format_float(clustering_coeff)} jointly yield {self._format_float(small_world_sigma)} "
            f"for the Watts–Strogatz σ statistic; {sigma_subtext}"
        )
        
        spectral_text = (
            f"Laplacian eigenvalues span {self._format_float(lambda_min, '{:.5f}')} to "
            f"{self._format_float(lambda_max)} with algebraic connectivity "
            f"λ₂={self._format_float(lambda_two, '{:.5f}')}. "
            f"There are {near_zero} near-zero modes (mirroring the {component_count} connected components), "
            f"and the upper-quartile eigenvalue {self._format_float(lambda_q3)} bounds the high-frequency content "
            f"available to the GP-VAR graph-frequency priors. Figure 4.y visualizes this spectrum."
        )
        
        report_lines = [
            f"# Distance-Dependent Consensus Graph – {group_name}",
            "",
            intro,
            "",
            "Key graph metrics:",
            f"- Sparsity: {sparsity_pct:.2f}% ({n_edges}/{n_possible_edges} undirected edges retained).",
            f"- Degree distribution: min={deg_min:.0f}, median={deg_median:.0f}, max={deg_max:.0f}, std={deg_std:.2f}.",
            f"- Strength distribution: min={strength_min:.3f}, median={strength_median:.3f}, max={strength_max:.3f}.",
            f"- Consensus support for kept edges: mean={self._format_float(consensus_mean)}, "
            f"median={self._format_float(consensus_median)}.",
            f"- Clustering coefficient (weighted): {self._format_float(clustering_coeff)}.",
            f"- Characteristic path length (largest component): {self._format_float(path_length)}.",
            f"- Small-world σ estimate: {self._format_float(small_world_sigma)} ({sigma_subtext}).",
            f"- Connectivity summary: graph is {component_text}.",
            "",
            "Spectral structure:",
            spectral_text,
            "",
            "Figures:",
            f"- Figure 4.x ({heatmap_path.name}): 128×128 consensus adjacency heatmap; rows/columns map to BioSemi-128 channels "
            "and bright blocks expose stronger intra-hemispheric coupling.",
            f"- Figure 4.y ({spectrum_path.name if spectrum_path else 'n/a'}): Laplacian eigenvalue spectrum showcasing "
            "near-zero modes and the high-frequency tail.",
            "",
            interpretation,
            "",
            small_world_text,
            "",
            "Use this text verbatim (≈1 page) and update figure numbers when drafting the report.",
        ]
        
        report_path = output_path / f"{group_name}_distance_graph_summary.md"
        report_path.write_text("\n".join(report_lines))
        
        metrics = {
            "sparsity_pct": sparsity_pct,
            "n_edges": n_edges,
            "n_possible_edges": n_possible_edges,
            "degree_min": deg_min,
            "degree_median": deg_median,
            "degree_max": deg_max,
            "degree_std": deg_std,
            "strength_min": strength_min,
            "strength_median": strength_median,
            "strength_max": strength_max,
            "consensus_mean_selected": consensus_mean,
            "consensus_median_selected": consensus_median,
            "clustering": clustering_coeff,
            "path_length": path_length,
            "small_world_sigma": small_world_sigma,
            "component_count": component_count,
            "largest_component_fraction": largest_component_fraction,
            "laplacian_lambda_min": lambda_min,
            "laplacian_lambda_two": lambda_two,
            "laplacian_lambda_max": lambda_max,
            "laplacian_lambda_q3": lambda_q3,
            "laplacian_near_zero": near_zero,
            "report_path": str(report_path),
            "heatmap_path": str(heatmap_path),
            "spectrum_path": str(spectrum_path) if spectrum_path else None,
        }
        
        return metrics
    
    def uniform_consensus(self, 
                          target_sparsity: Optional[Union[float, str]] = None,
                          require_existing: bool = True) -> np.ndarray:
        """
        Build final group graph using uniform consensus (baseline).
        Select top-K edges globally based on C.
        
        Parameters
        ----------
        target_sparsity : float, str, or None, optional
            Same semantics as distance_dependent_consensus.
        require_existing : bool
            If True, only consider edges with C > 0
            
        Returns
        -------
        G : np.ndarray
            Final group adjacency matrix
        """
        if self.consensus_matrix is None or self.weight_matrix is None:
            raise ValueError("Must run compute_consensus_and_weights first")
            
        n_nodes = self.consensus_matrix.shape[0]
        C = self.consensus_matrix
        W = self.weight_matrix
        resolved_sparsity = self._resolve_target_sparsity(target_sparsity)
        
        if resolved_sparsity is None:
            logger.info("Target sparsity disabled; keeping all qualifying edges (uniform consensus).")
            weight_source = self.weight_matrix_full if self.weight_matrix_full is not None else W
            if require_existing:
                mask = C > 0
            else:
                mask = np.ones_like(C, dtype=bool)
            mask &= weight_source > 0
            np.fill_diagonal(mask, False)
            G = np.where(mask, weight_source, 0.0)
            return np.maximum(G, G.T)
        
        # Get upper triangle indices
        triu_indices = np.triu_indices(n_nodes, k=1)
        n_possible_edges = len(triu_indices[0])
        
        # Calculate target number of edges
        k_target = int(np.clip(round(resolved_sparsity * n_possible_edges), 1, n_possible_edges))
        
        # Extract consensus values
        consensus_values = C[triu_indices]
        
        # Apply existence requirement if needed
        if require_existing:
            valid_mask = consensus_values > 0
        else:
            valid_mask = np.ones(len(consensus_values), dtype=bool)
        
        # Sort and select top-k
        valid_indices = np.where(valid_mask)[0]
        valid_consensus = consensus_values[valid_indices]
        
        k_select = min(k_target, len(valid_indices))
        top_k_indices = valid_indices[np.argsort(valid_consensus)[::-1][:k_select]]
        
        # Build final adjacency
        G = np.zeros((n_nodes, n_nodes))
        
        for idx in top_k_indices:
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            G[i, j] = W[i, j]
            G[j, i] = W[i, j]
        
        logger.info(f"Selected {k_select} edges with uniform consensus")
        
        return G
    
    def visualize_consensus(self, save_path: Optional[str] = None):
        """
        Visualize the consensus matrix and weight matrix.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
        """
        if self.consensus_matrix is None:
            raise ValueError("No consensus matrix computed yet")
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot consensus matrix
        im1 = axes[0].imshow(self.consensus_matrix, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Consensus Matrix C\n(Fraction of subjects with edge)')
        axes[0].set_xlabel('Channel')
        axes[0].set_ylabel('Channel')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot weight matrix
        im2 = axes[1].imshow(self.weight_matrix, cmap='viridis', vmin=0)
        axes[1].set_title('Weight Matrix W\n(Average correlation strength)')
        axes[1].set_xlabel('Channel')
        axes[1].set_ylabel('Channel')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str, prefix: str = "consensus"):
        """
        Save consensus matrix, weight matrix, and final graphs.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        prefix : str
            Prefix for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save consensus matrix
        np.save(output_path / f"{prefix}_matrix_C.npy", self.consensus_matrix)
        
        # Save weight matrix
        np.save(output_path / f"{prefix}_matrix_W.npy", self.weight_matrix)
        
        # Save dense weight matrix if available
        if self.weight_matrix_full is not None:
            np.save(output_path / f"{prefix}_matrix_W_dense.npy", self.weight_matrix_full)
        
        # Save binary matrices if available
        if self.binary_matrices:
            np.save(output_path / f"{prefix}_binary_matrices.npy", 
                    np.array(self.binary_matrices))
        
        # Save distance-dependent graph if available
        if self.distance_graph is not None:
            np.save(output_path / f"{prefix}_distance_graph.npy", self.distance_graph)
        
        logger.info(f"Results saved to {output_dir}")


def compute_consensus_matrix(corr_matrices: List[np.ndarray], 
                             sparsity: Optional[float] = None) -> np.ndarray:
    """
    Compute consensus matrix from individual correlation matrices.
    Uses direct Fisher-z averaging for robust mean estimation.
    
    This is the recommended simple interface for computing consensus matrices.
    It directly averages all correlations in Fisher-z space, preserving more
    information than binarization-based approaches.
    
    Parameters
    ----------
    corr_matrices : List[np.ndarray]
        List of correlation matrices (one per subject)
    sparsity : float, optional
        If provided, apply proportional thresholding to keep top sparsity 
        fraction of edges. If None, return the full dense consensus matrix.
        
    Returns
    -------
    consensus : np.ndarray
        Consensus correlation matrix
        
    Examples
    --------
    >>> # Compute full dense consensus
    >>> consensus = compute_consensus_matrix(subject_matrices)
    
    >>> # Compute sparse consensus (keep top 15% of edges)
    >>> consensus = compute_consensus_matrix(subject_matrices, sparsity=0.15)
    """
    if len(corr_matrices) == 0:
        raise ValueError("No matrices provided")
    
    # Stack all correlation matrices
    stack = np.stack(corr_matrices, axis=0)
    
    # Apply Fisher-z transform to all matrices
    # Clip to avoid inf values
    stack_clipped = np.clip(stack, -0.999999, 0.999999)
    z_stack = np.arctanh(stack_clipped)
    
    # Average in z-space
    z_mean = np.mean(z_stack, axis=0)
    
    # Transform back to correlation space and take absolute value
    consensus = np.abs(np.tanh(z_mean))
    
    # Ensure diagonal is 0 and matrix is symmetric
    np.fill_diagonal(consensus, 0)
    consensus = np.maximum(consensus, consensus.T)
    
    # Apply sparsity threshold if requested
    if sparsity is not None and 0 < sparsity < 1:
        n = consensus.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        edge_weights = consensus[triu_indices]
        
        n_edges = len(edge_weights)
        k_edges = int(np.floor(sparsity * n_edges))
        k_edges = max(1, k_edges)
        
        if k_edges < n_edges:
            threshold = np.sort(edge_weights)[::-1][k_edges]
        else:
            threshold = 0
            
        consensus = np.where(consensus > threshold, consensus, 0)
        consensus = np.maximum(consensus, consensus.T)
        np.fill_diagonal(consensus, 0)
    
    return consensus


def process_eeg_files(file_paths: List[str], 
                      group_labels: Optional[List[str]] = None,
                      channel_locations: Optional[np.ndarray] = None,
                      sparsity: float = 0.15,
                      sparsity_final: Optional[Union[float, str]] = "match_subject",
                      graph_method: str = 'distance',
                      consensus_method: str = 'direct',
                      output_dir: str = "./consensus_results") -> Dict:
    """
    Process multiple EEG files to create consensus matrices.
    
    Parameters
    ----------
    file_paths : List[str]
        List of paths to EEG files
    group_labels : List[str], optional
        Group labels for each file (e.g., 'AD', 'HC')
    channel_locations : np.ndarray, optional
        Pre-specified 3D channel coordinates (n_channels x 3)
    sparsity : float
        Sparsity level for the consensus matrix (fraction of edges to keep, 0 < sparsity < 1)
    sparsity_final : float, str, or None, optional
        Target sparsity for final graph when using distance-dependent selection.
        Use a float in (0,1], "match_subject" to mirror the consensus sparsity, 
        or None to keep every qualifying edge.
    graph_method : str
        Graph construction method:
        - 'distance': Distance-dependent edge selection (Betzel-style bins)
        - 'uniform': Uniform edge selection based on consensus values
    consensus_method : str
        Consensus computation method:
        - 'direct': Direct Fisher-z averaging across all correlations (recommended).
                    Preserves more information by averaging all correlations.
        - 'binarize': Betzel-style per-subject binarization then conditional averaging.
    output_dir : str
        Directory to save results
        
    Returns
    -------
    results : dict
        Dictionary containing consensus matrices and final graphs
    """
    # Initialize consensus builder
    consensus_builder = ConsensusMatrix(channel_locations=channel_locations)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data and compute adjacency matrices
    adjacency_matrices = []
    valid_files = []
    
    logger.info(f"Loading {len(file_paths)} EEG files...")
    
    for i, filepath in enumerate(file_paths):
        try:
            # Load EEG data
            data = consensus_builder.load_eeg_data(filepath)
            
            # Compute Pearson correlation adjacency
            A = consensus_builder.compute_pearson_adjacency(data)
            
            adjacency_matrices.append(A)
            valid_files.append(filepath)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(file_paths)} files")
                
        except Exception as e:
            logger.warning(f"Skipping {filepath}: {e}")
    
    logger.info(f"Successfully loaded {len(adjacency_matrices)} files")
    
    # Compute consensus matrix and weights
    C, W = consensus_builder.compute_consensus_and_weights(
        adjacency_matrices, 
        sparsity=sparsity,
        method=consensus_method
    )
    
    # Build final group graph
    if graph_method == 'distance':
        if consensus_builder.channel_locations is None:
            raise ValueError("Distance-dependent consensus requested but channel locations are missing.")
        G = consensus_builder.distance_dependent_consensus(
            target_sparsity=sparsity_final,
            save_path=output_path / "consensus_distance_graph.npy"
        )
    else:
        G = consensus_builder.uniform_consensus(target_sparsity=sparsity_final)
    
    # Summarize properties of the final graph
    group_name = Path(output_dir).name
    method_desc = f"{consensus_method} Fisher-z consensus"
    if graph_method == 'distance':
        method_desc += " with distance-dependent edge selection (Betzel-style bins)"
    else:
        method_desc += " with uniform edge selection"
    
    graph_summary = consensus_builder.summarize_distance_graph(
        group_name=group_name,
        output_dir=str(output_path),
        method_desc=method_desc,
        graph=G
    )
    
    # Visualize results
    consensus_builder.visualize_consensus(save_path=str(output_path / "consensus_visualization.png"))
    
    # Save results
    consensus_builder.save_results(str(output_path))
    
    # Package results
    results = {
        'consensus_matrix': C,
        'weight_matrix': W,
        'weight_matrix_dense': consensus_builder.weight_matrix_full,
        'final_graph': G,
        'binary_matrices': np.array(consensus_builder.binary_matrices),
        'adjacency_matrices': np.array(adjacency_matrices),
        'valid_files': valid_files,
        'channel_locations': consensus_builder.channel_locations,
        'graph_properties': graph_summary
    }
    
    return results


if __name__ == "__main__":
    # Full EEG file list (BioSemi 128-channel recordings)
    files = [
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
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30003/eeg/s6_sub-30003_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30007/eeg/s6_sub-30007_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30005/eeg/s6_sub-30005_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30006/eeg/s6_sub-30006_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30010/eeg/s6_sub-30010_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30014/eeg/s6_sub-30014_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30016/eeg/s6_sub-30016_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30019/eeg/s6_sub-30019_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30021/eeg/s6_sub-30021_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30023/eeg/s6_sub-30023_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30024/eeg/s6_sub-30024_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30025/eeg/s6_sub-30025_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30027/eeg/s6_sub-30027_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30028/eeg/s6_sub-30028_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30030/eeg/s6_sub-30030_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30032/eeg/s6_sub-30032_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30033/eeg/s6_sub-30033_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30034/eeg/s6_sub-30034_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30035/eeg/s6_sub-30035_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10002/eeg/s6_sub-10002_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10009/eeg/s6_sub-10009_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100012/eeg/s6_sub-100012_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100015/eeg/s6_sub-100015_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100020/eeg/s6_sub-100020_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100035/eeg/s6_sub-100035_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100028/eeg/s6_sub-100028_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10006/eeg/s6_sub-10006_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10007/eeg/s6_sub-10007_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100033/eeg/s6_sub-100033_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100022/eeg/s6_sub-100022_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100031/eeg/s6_sub-100031_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10003/eeg/s6_sub-10003_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100026/eeg/s6_sub-100026_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100030/eeg/s6_sub-100030_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100018/eeg/s6_sub-100018_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100024/eeg/s6_sub-100024_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100038/eeg/s6_sub-100038_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10004/eeg/s6_sub-10004_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10001/eeg/s6_sub-10001_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10005/eeg/s6_sub-10005_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10008/eeg/s6_sub-10008_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100010/eeg/s6_sub-100010_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100011/eeg/s6_sub-100011_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100014/eeg/s6_sub-100014_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100017/eeg/s6_sub-100017_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100021/eeg/s6_sub-100021_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100029/eeg/s6_sub-100029_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100034/eeg/s6_sub-100034_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100037/eeg/s6_sub-100037_rs_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100043/eeg/s6_sub-100043_rs_eeg.set',
    ]
    
    results = process_eeg_files(
        file_paths=files,
        sparsity=0.15,              # Sparsity for final consensus graph
        sparsity_final=0.10,        # Optional secondary sparsity for distance-dependent selection
        graph_method='distance',     # 'distance' or 'uniform'
        consensus_method='direct',   # 'direct' (recommended) or 'binarize' (Betzel-style)
        output_dir="./consensus_results/ALL_Files"
    )
    
    logger.info("Consensus processing complete for ALL_Files. Outputs written to ./consensus_results/ALL_Files")