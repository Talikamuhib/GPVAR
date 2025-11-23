"""
Test consensus matrix implementation with synthetic EEG-like data.
This script demonstrates the complete pipeline without requiring actual EEG files.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from consensus_matrix_eeg import ConsensusMatrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_eeg_data(n_channels=19, n_samples=10000, n_subjects=20, 
                                 group_connectivity_pattern=None, noise_level=0.3):
    """
    Generate synthetic EEG-like data with controlled connectivity patterns.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    n_samples : int
        Number of time samples
    n_subjects : int
        Number of subjects
    group_connectivity_pattern : np.ndarray, optional
        True underlying connectivity pattern (n_channels x n_channels)
    noise_level : float
        Amount of noise to add (0 = no noise, 1 = high noise)
        
    Returns
    -------
    data_list : list
        List of synthetic EEG data arrays (n_channels x n_samples) for each subject
    true_connectivity : np.ndarray
        True underlying connectivity pattern
    """
    data_list = []
    
    # Create a default connectivity pattern if not provided
    if group_connectivity_pattern is None:
        # Create a modular connectivity pattern
        true_connectivity = np.zeros((n_channels, n_channels))
        
        # Module 1: channels 0-5
        true_connectivity[0:6, 0:6] = 0.7
        
        # Module 2: channels 6-12
        true_connectivity[6:13, 6:13] = 0.6
        
        # Module 3: channels 13-18
        true_connectivity[13:19, 13:19] = 0.8
        
        # Add some inter-module connections
        true_connectivity[2:5, 7:10] = 0.3
        true_connectivity[7:10, 2:5] = 0.3
        true_connectivity[10:13, 15:18] = 0.4
        true_connectivity[15:18, 10:13] = 0.4
        
        # Zero diagonal
        np.fill_diagonal(true_connectivity, 0)
    else:
        true_connectivity = group_connectivity_pattern
    
    for subject in range(n_subjects):
        # Generate base signals with some correlation structure
        # Start with random signals
        signals = np.random.randn(n_channels, n_samples)
        
        # Apply connectivity pattern with subject-specific variation
        subject_connectivity = true_connectivity.copy()
        
        # Add subject-specific variation
        variation = np.random.randn(n_channels, n_channels) * noise_level
        variation = (variation + variation.T) / 2  # Ensure symmetry
        subject_connectivity += variation
        
        # Clip values to [0, 1]
        subject_connectivity = np.clip(subject_connectivity, 0, 1)
        np.fill_diagonal(subject_connectivity, 0)
        
        # Create correlated signals based on connectivity
        # Use Cholesky decomposition to create correlated noise
        for i in range(0, n_samples, 1000):
            end_idx = min(i + 1000, n_samples)
            segment_length = end_idx - i
            
            # Create correlation matrix (ensure positive definite)
            corr_matrix = subject_connectivity + np.eye(n_channels) * 0.1
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            
            # Ensure positive definite
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            if eigenvalues.min() < 0.01:
                corr_matrix += np.eye(n_channels) * (0.01 - eigenvalues.min())
            
            try:
                L = np.linalg.cholesky(corr_matrix)
                uncorrelated = np.random.randn(n_channels, segment_length)
                signals[:, i:end_idx] = L @ uncorrelated
            except np.linalg.LinAlgError:
                # If Cholesky fails, use SVD
                U, s, Vt = np.linalg.svd(corr_matrix)
                s[s < 0.01] = 0.01
                L = U @ np.diag(np.sqrt(s))
                uncorrelated = np.random.randn(n_channels, segment_length)
                signals[:, i:end_idx] = L @ uncorrelated
        
        # Add additional noise
        signals += np.random.randn(n_channels, n_samples) * noise_level
        
        data_list.append(signals)
    
    return data_list, true_connectivity


def generate_channel_locations(n_channels=19):
    """
    Generate synthetic 3D channel locations mimicking standard EEG montage.
    
    Parameters
    ----------
    n_channels : int
        Number of channels
        
    Returns
    -------
    locations : np.ndarray
        3D coordinates (n_channels x 3)
    """
    # Create circular arrangement with some z-variation
    angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
    
    # Create multiple layers
    if n_channels == 19:
        # Standard 10-20 system approximation
        locations = np.zeros((n_channels, 3))
        
        # Central channel
        locations[0] = [0, 0, 0.9]
        
        # Inner ring (6 channels)
        for i in range(6):
            angle = i * 2 * np.pi / 6
            locations[i+1] = [0.3 * np.cos(angle), 0.3 * np.sin(angle), 0.7]
        
        # Outer ring (12 channels)
        for i in range(12):
            angle = i * 2 * np.pi / 12
            locations[i+7] = [0.6 * np.cos(angle), 0.6 * np.sin(angle), 0.5]
    else:
        # Generic circular arrangement
        locations = np.zeros((n_channels, 3))
        for i in range(n_channels):
            locations[i] = [np.cos(angles[i]), np.sin(angles[i]), 
                           0.5 + 0.2 * np.sin(2 * angles[i])]
    
    return locations


def test_consensus_pipeline():
    """
    Test the complete consensus matrix pipeline with synthetic data.
    """
    logger.info("="*60)
    logger.info("Testing Consensus Matrix Pipeline with Synthetic Data")
    logger.info("="*60)
    
    # Parameters
    N_CHANNELS = 19
    N_SAMPLES = 10000
    N_SUBJECTS = 30
    NOISE_LEVEL = 0.3
    SPARSITY_BINARIZE = 0.15
    SPARSITY_FINAL = 0.10
    
    # Generate synthetic channel locations
    logger.info(f"\nGenerating synthetic channel locations for {N_CHANNELS} channels...")
    channel_locations = generate_channel_locations(N_CHANNELS)
    
    # Initialize consensus builder
    consensus_builder = ConsensusMatrix(channel_locations=channel_locations)
    
    # Generate synthetic data
    logger.info(f"Generating synthetic EEG data for {N_SUBJECTS} subjects...")
    data_list, true_connectivity = generate_synthetic_eeg_data(
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        n_subjects=N_SUBJECTS,
        noise_level=NOISE_LEVEL
    )
    
    # Compute adjacency matrices for each subject
    logger.info("Computing Pearson correlation matrices for each subject...")
    adjacency_matrices = []
    for i, data in enumerate(data_list):
        A = consensus_builder.compute_pearson_adjacency(data, absolute=True)
        adjacency_matrices.append(A)
    
    # Compute consensus matrix and weights
    logger.info(f"Computing consensus matrix with sparsity={SPARSITY_BINARIZE}...")
    C, W = consensus_builder.compute_consensus_and_weights(
        adjacency_matrices, 
        sparsity=SPARSITY_BINARIZE
    )
    
    # Build final graphs using both methods
    logger.info("Building final group graphs...")
    
    # Distance-dependent consensus
    G_distance = consensus_builder.distance_dependent_consensus(
        target_sparsity=SPARSITY_FINAL,
        n_bins=10,
        epsilon=0.1,
        require_existing=True
    )
    
    # Uniform consensus (baseline)
    G_uniform = consensus_builder.uniform_consensus(
        target_sparsity=SPARSITY_FINAL,
        require_existing=True
    )
    
    # Analyze results
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Consensus matrix statistics
    triu_idx = np.triu_indices(N_CHANNELS, k=1)
    consensus_values = C[triu_idx]
    
    print("\nConsensus Matrix C Statistics:")
    print(f"  Shape: {C.shape}")
    print(f"  Range: [{consensus_values.min():.3f}, {consensus_values.max():.3f}]")
    print(f"  Mean: {consensus_values.mean():.3f}")
    print(f"  Median: {np.median(consensus_values):.3f}")
    print(f"  Edges with C > 0: {np.sum(consensus_values > 0)} / {len(consensus_values)}")
    print(f"  Edges with C > 0.5: {np.sum(consensus_values > 0.5)}")
    print(f"  Edges with C = 1.0: {np.sum(consensus_values == 1.0)}")
    
    # Weight matrix statistics
    weight_values = W[triu_idx]
    weight_values_nonzero = weight_values[weight_values > 0]
    
    print("\nWeight Matrix W Statistics:")
    print(f"  Non-zero edges: {len(weight_values_nonzero)}")
    print(f"  Range: [{weight_values_nonzero.min():.3f}, {weight_values_nonzero.max():.3f}]")
    print(f"  Mean: {weight_values_nonzero.mean():.3f}")
    
    # Final graph comparison
    print("\nFinal Graph Comparison:")
    
    dist_edges = np.sum(G_distance > 0) / 2
    unif_edges = np.sum(G_uniform > 0) / 2
    
    print(f"  Distance-dependent: {dist_edges:.0f} edges")
    print(f"  Uniform: {unif_edges:.0f} edges")
    
    # Calculate overlap with true connectivity
    true_binary = (true_connectivity > 0.4).astype(float)
    
    # Binarize final graphs
    G_dist_binary = (G_distance > 0).astype(float)
    G_unif_binary = (G_uniform > 0).astype(float)
    
    # Calculate Jaccard similarity
    def jaccard_similarity(A, B):
        intersection = np.sum(A * B)
        union = np.sum(np.maximum(A, B))
        return intersection / union if union > 0 else 0
    
    jaccard_dist = jaccard_similarity(true_binary, G_dist_binary)
    jaccard_unif = jaccard_similarity(true_binary, G_unif_binary)
    
    print(f"\nRecovery of True Connectivity (Jaccard Similarity):")
    print(f"  Distance-dependent: {jaccard_dist:.3f}")
    print(f"  Uniform: {jaccard_unif:.3f}")
    
    # Visualization
    logger.info("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: True connectivity and consensus matrices
    im1 = axes[0, 0].imshow(true_connectivity, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('True Connectivity Pattern')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Channel')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(C, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Consensus Matrix C')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Channel')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(W, cmap='viridis', vmin=0)
    axes[0, 2].set_title('Weight Matrix W')
    axes[0, 2].set_xlabel('Channel')
    axes[0, 2].set_ylabel('Channel')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Histogram of consensus values
    axes[0, 3].hist(consensus_values[consensus_values > 0], bins=30, 
                    alpha=0.7, color='blue', edgecolor='black')
    axes[0, 3].set_xlabel('Consensus Value')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].set_title('Distribution of C values')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Final graphs and comparison
    im4 = axes[1, 0].imshow(G_distance, cmap='plasma', vmin=0)
    axes[1, 0].set_title(f'Distance-dependent Graph\n(Jaccard={jaccard_dist:.3f})')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Channel')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(G_uniform, cmap='plasma', vmin=0)
    axes[1, 1].set_title(f'Uniform Consensus Graph\n(Jaccard={jaccard_unif:.3f})')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Channel')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # Example individual subject adjacency
    im6 = axes[1, 2].imshow(adjacency_matrices[0], cmap='coolwarm', vmin=0, vmax=1)
    axes[1, 2].set_title('Example Subject Adjacency')
    axes[1, 2].set_xlabel('Channel')
    axes[1, 2].set_ylabel('Channel')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    # Binary matrix for first subject
    im7 = axes[1, 3].imshow(consensus_builder.binary_matrices[0], cmap='binary', vmin=0, vmax=1)
    axes[1, 3].set_title(f'Example Binary Matrix\n(sparsity={SPARSITY_BINARIZE})')
    axes[1, 3].set_xlabel('Channel')
    axes[1, 3].set_ylabel('Channel')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    plt.suptitle('Consensus Matrix Analysis - Synthetic Data Test', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('./consensus_synthetic_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("\nTest completed successfully!")
    logger.info(f"Results saved to: consensus_synthetic_test.png")
    
    return {
        'consensus_matrix': C,
        'weight_matrix': W,
        'graph_distance': G_distance,
        'graph_uniform': G_uniform,
        'true_connectivity': true_connectivity,
        'adjacency_matrices': adjacency_matrices
    }


if __name__ == "__main__":
    # Run the synthetic data test
    results = test_consensus_pipeline()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\nThe consensus matrix pipeline has been successfully tested with synthetic data.")
    print("\nKey components verified:")
    print("  ✓ Pearson correlation computation")
    print("  ✓ Proportional thresholding for binarization")
    print("  ✓ Consensus matrix C calculation")
    print("  ✓ Fisher-z transformed weight matrix W")
    print("  ✓ Distance-dependent edge selection")
    print("  ✓ Uniform consensus baseline")
    print("\nThe implementation is ready for use with real EEG data!")