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
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsensusMatrix:
    """
    Build consensus connectivity matrices from multiple subjects using Pearson correlation.
    
    The process:
    1. Compute per-subject Pearson correlation matrices A(s)
    2. Sparsify each A(s) → binary B(s) using proportional thresholding
    3. Compute edge consistency: C_ij = (1/S) * Σ_s B_ij(s)
    4. Compute representative weights W_ij via Fisher-z conditional mean
    5. Select edges using C (uniform or distance-dependent)
    6. Assign weights W_ij to selected edges → final group adjacency G
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
        self.binary_matrices = []
        self.adjacency_matrices = []
        self.subject_labels = []
        
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
            data = raw.get_data()
            
            # Store channel locations if not already set
            if self.channel_locations is None:
                montage = raw.get_montage()
                if montage is not None:
                    positions = montage.get_positions()['ch_pos']
                    ch_names = raw.ch_names
                    self.channel_locations = np.array([positions[ch] for ch in ch_names if ch in positions])
                    
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
    
    def compute_consensus_and_weights(self, 
                                       adjacency_matrices: List[np.ndarray],
                                       sparsity: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute consensus matrix C and weight matrix W from multiple subjects.
        
        C_ij = (1/S) * Σ_s B_ij(s)  [fraction of subjects with edge]
        W_ij = conditional mean of weights where edge exists
        
        Parameters
        ----------
        adjacency_matrices : List[np.ndarray]
            List of weighted adjacency matrices, one per subject
        sparsity : float
            Sparsity level for binarization
            
        Returns
        -------
        C : np.ndarray
            Consensus matrix (fraction of subjects with each edge)
        W : np.ndarray
            Weight matrix (average weight where edge exists)
        """
        n_subjects = len(adjacency_matrices)
        n_nodes = adjacency_matrices[0].shape[0]
        
        # Store adjacency matrices
        self.adjacency_matrices = adjacency_matrices
        
        # Step 1: Binarize each subject's matrix
        logger.info(f"Binarizing {n_subjects} matrices with sparsity={sparsity}")
        self.binary_matrices = []
        for i, A in enumerate(adjacency_matrices):
            B = self.proportional_threshold_exact(A, sparsity)
            self.binary_matrices.append(B)
            
        # Step 2: Compute consensus matrix C
        # C_ij = mean across subjects of binary matrices
        C = np.mean(self.binary_matrices, axis=0)
        
        # Step 3: Compute weight matrix W using Fisher-z averaging
        W = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Find subjects where edge exists
                edge_exists = [self.binary_matrices[s][i, j] > 0 for s in range(n_subjects)]
                
                if any(edge_exists):
                    # Fisher-z transform correlations
                    z_values = []
                    for s in range(n_subjects):
                        if edge_exists[s]:
                            z = self.fisher_z_transform(adjacency_matrices[s][i, j])
                            z_values.append(z)
                    
                    # Average in z-space
                    z_mean = np.mean(z_values)
                    
                    # Transform back
                    W[i, j] = np.abs(self.fisher_z_inverse(z_mean))
                    W[j, i] = W[i, j]  # Ensure symmetry
        
        self.consensus_matrix = C
        self.weight_matrix = W
        
        return C, W
    
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
                                      target_sparsity: float = 0.10,
                                      n_bins: int = 10,
                                      epsilon: float = 0.1,
                                      require_existing: bool = True) -> np.ndarray:
        """
        Build final group graph using distance-dependent consensus.
        
        Parameters
        ----------
        target_sparsity : float
            Target sparsity of final graph
        n_bins : int
            Number of distance bins
        epsilon : float
            Weight for W in scoring (score = C + ε*W)
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
        
        # Compute distance matrix
        D = self.compute_distance_matrix()
        
        # Get upper triangle indices
        triu_indices = np.triu_indices(n_nodes, k=1)
        n_possible_edges = len(triu_indices[0])
        
        # Calculate target number of edges
        k_target = int(np.floor(target_sparsity * n_possible_edges))
        
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

        logger.info(f"Selected {len(selected_edges)} edges with distance-dependent consensus")
        
        return G
    
    def uniform_consensus(self, 
                          target_sparsity: float = 0.10,
                          require_existing: bool = True) -> np.ndarray:
        """
        Build final group graph using uniform consensus (baseline).
        Select top-K edges globally based on C.
        
        Parameters
        ----------
        target_sparsity : float
            Target sparsity of final graph
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
        
        # Get upper triangle indices
        triu_indices = np.triu_indices(n_nodes, k=1)
        n_possible_edges = len(triu_indices[0])
        
        # Calculate target number of edges
        k_target = int(np.floor(target_sparsity * n_possible_edges))
        
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
        
        # Save binary matrices if available
        if self.binary_matrices:
            np.save(output_path / f"{prefix}_binary_matrices.npy", 
                    np.array(self.binary_matrices))
        
        logger.info(f"Results saved to {output_dir}")


def process_eeg_files(file_paths: List[str], 
                      group_labels: Optional[List[str]] = None,
                      sparsity_binarize: float = 0.15,
                      sparsity_final: float = 0.10,
                      method: str = 'distance',
                      output_dir: str = "./consensus_results") -> Dict:
    """
    Process multiple EEG files to create consensus matrices.
    
    Parameters
    ----------
    file_paths : List[str]
        List of paths to EEG files
    group_labels : List[str], optional
        Group labels for each file (e.g., 'AD', 'HC')
    sparsity_binarize : float
        Sparsity for initial binarization
    sparsity_final : float
        Target sparsity for final graph
    method : str
        'distance' for distance-dependent or 'uniform' for baseline
    output_dir : str
        Directory to save results
        
    Returns
    -------
    results : dict
        Dictionary containing consensus matrices and final graphs
    """
    # Initialize consensus builder
    consensus_builder = ConsensusMatrix()
    
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
    C, W = consensus_builder.compute_consensus_and_weights(adjacency_matrices, 
                                                             sparsity=sparsity_binarize)
    
    # Build final group graph
    if method == 'distance':
        if consensus_builder.channel_locations is None:
            logger.warning("No channel locations available, falling back to uniform consensus")
            G = consensus_builder.uniform_consensus(target_sparsity=sparsity_final)
        else:
            G = consensus_builder.distance_dependent_consensus(target_sparsity=sparsity_final)
    else:
        G = consensus_builder.uniform_consensus(target_sparsity=sparsity_final)
    
    # Visualize results
    consensus_builder.visualize_consensus(save_path=f"{output_dir}/consensus_visualization.png")
    
    # Save results
    consensus_builder.save_results(output_dir)
    
    # Package results
    results = {
        'consensus_matrix': C,
        'weight_matrix': W,
        'final_graph': G,
        'binary_matrices': np.array(consensus_builder.binary_matrices),
        'adjacency_matrices': np.array(adjacency_matrices),
        'valid_files': valid_files,
        'channel_locations': consensus_builder.channel_locations
    }
    
    return results


if __name__ == "__main__":
    # Example usage with your file paths
    
    # File paths (truncated for example - you would use your full list)
    ad_ar_files = [
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30018/eeg/s6_sub-30018_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30026/eeg/s6_sub-30026_rs-hep_eeg.set',
        # ... add all AD AR files
    ]
    
    ad_cl_files = [
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30003/eeg/s6_sub-30003_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30007/eeg/s6_sub-30007_rs-hep_eeg.set',
        # ... add all AD CL files
    ]
    
    hc_ar_files = [
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10002/eeg/s6_sub-10002_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10009/eeg/s6_sub-10009_rs_eeg.set",
        # ... add all HC AR files
    ]
    
    hc_cl_files = [
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10001/eeg/s6_sub-10001_rs_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10005/eeg/s6_sub-10005_rs_eeg.set",
        # ... add all HC CL files
    ]
    
    # Process each group separately
    groups = {
        'AD_AR': ad_ar_files,
        'AD_CL': ad_cl_files,
        'HC_AR': hc_ar_files,
        'HC_CL': hc_cl_files
    }
    
    all_results = {}
    
    for group_name, file_list in groups.items():
        logger.info(f"\nProcessing {group_name} group...")
        
        results = process_eeg_files(
            file_paths=file_list,
            sparsity_binarize=0.15,  # Sparsity for binarization
            sparsity_final=0.10,      # Target sparsity for final graph
            method='distance',        # Use distance-dependent consensus
            output_dir=f"./consensus_results/{group_name}"
        )
        
        all_results[group_name] = results
        
    logger.info("\nAll groups processed successfully!")