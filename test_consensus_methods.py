#!/usr/bin/env python
"""
Test script to verify the consensus matrix implementations.
Compares the direct Fisher-z averaging method with the binarization-based approach.
"""

import numpy as np
import sys

# Add workspace to path
sys.path.insert(0, '/workspace')

from consensus_matrix_eeg import (
    ConsensusMatrix, 
    compute_consensus_matrix
)

def generate_synthetic_correlation_matrices(n_subjects: int = 10, 
                                            n_channels: int = 64,
                                            noise_level: float = 0.1,
                                            seed: int = 42) -> list:
    """
    Generate synthetic correlation matrices for testing.
    Creates a "ground truth" correlation structure with per-subject noise.
    """
    np.random.seed(seed)
    
    # Create a ground truth correlation pattern
    # Simulate block structure (like hemispheric connectivity)
    ground_truth = np.zeros((n_channels, n_channels))
    
    # Strong connections within blocks (first half, second half)
    block_size = n_channels // 2
    ground_truth[:block_size, :block_size] = 0.6 + 0.2 * np.random.randn(block_size, block_size) * 0.1
    ground_truth[block_size:, block_size:] = 0.6 + 0.2 * np.random.randn(block_size, block_size) * 0.1
    
    # Weaker connections between blocks
    ground_truth[:block_size, block_size:] = 0.3 + 0.1 * np.random.randn(block_size, block_size) * 0.1
    ground_truth[block_size:, :block_size] = ground_truth[:block_size, block_size:].T
    
    # Make symmetric and clip to valid range
    ground_truth = (ground_truth + ground_truth.T) / 2
    ground_truth = np.clip(ground_truth, 0, 0.99)
    np.fill_diagonal(ground_truth, 0)
    
    # Generate per-subject matrices with noise
    matrices = []
    for _ in range(n_subjects):
        noise = np.random.randn(n_channels, n_channels) * noise_level
        noise = (noise + noise.T) / 2  # Symmetric noise
        
        subject_matrix = ground_truth + noise
        subject_matrix = np.clip(subject_matrix, 0, 0.99)
        np.fill_diagonal(subject_matrix, 0)
        
        matrices.append(np.abs(subject_matrix))
    
    return matrices, ground_truth


def test_simple_consensus_function():
    """Test the simple compute_consensus_matrix function."""
    print("\n" + "="*60)
    print("Test 1: Simple compute_consensus_matrix function")
    print("="*60)
    
    matrices, ground_truth = generate_synthetic_correlation_matrices(
        n_subjects=20, n_channels=32, noise_level=0.1
    )
    
    # Test dense consensus
    consensus_dense = compute_consensus_matrix(matrices)
    print(f"Dense consensus shape: {consensus_dense.shape}")
    print(f"Dense consensus range: [{consensus_dense.min():.4f}, {consensus_dense.max():.4f}]")
    
    # Test sparse consensus
    consensus_sparse = compute_consensus_matrix(matrices, sparsity=0.15)
    n_edges = np.sum(consensus_sparse > 0) // 2  # Divide by 2 for symmetric
    n_possible = 32 * 31 // 2
    print(f"Sparse consensus edges: {n_edges}/{n_possible} ({n_edges/n_possible:.2%})")
    
    # Check correlation with ground truth
    triu_idx = np.triu_indices(32, k=1)
    corr_dense = np.corrcoef(consensus_dense[triu_idx], ground_truth[triu_idx])[0, 1]
    print(f"Correlation with ground truth (dense): {corr_dense:.4f}")
    
    assert consensus_dense.shape == (32, 32), "Wrong shape"
    assert np.allclose(consensus_dense, consensus_dense.T), "Not symmetric"
    assert np.allclose(np.diag(consensus_dense), 0), "Diagonal not zero"
    
    print("✓ Simple consensus function works correctly!")
    return True


def test_consensus_matrix_class_direct():
    """Test the ConsensusMatrix class with direct method."""
    print("\n" + "="*60)
    print("Test 2: ConsensusMatrix class - Direct method")
    print("="*60)
    
    matrices, ground_truth = generate_synthetic_correlation_matrices(
        n_subjects=20, n_channels=32, noise_level=0.1
    )
    
    builder = ConsensusMatrix()
    C, W = builder.compute_consensus_and_weights(matrices, sparsity=0.15, method='direct')
    
    print(f"Consensus matrix C shape: {C.shape}")
    print(f"Weight matrix W shape: {W.shape}")
    print(f"Method used: {builder._consensus_method}")
    
    n_edges = np.sum(W > 0) // 2
    n_possible = 32 * 31 // 2
    print(f"Edges in W: {n_edges}/{n_possible} ({n_edges/n_possible:.2%})")
    
    # Check correlation with ground truth
    triu_idx = np.triu_indices(32, k=1)
    if builder.weight_matrix_full is not None:
        corr_full = np.corrcoef(builder.weight_matrix_full[triu_idx], ground_truth[triu_idx])[0, 1]
        print(f"Correlation with ground truth (full): {corr_full:.4f}")
    
    assert builder._consensus_method == 'direct', "Wrong method recorded"
    assert W.shape == (32, 32), "Wrong shape"
    assert np.allclose(W, W.T), "Not symmetric"
    
    print("✓ Direct method works correctly!")
    return True


def test_consensus_matrix_class_binarize():
    """Test the ConsensusMatrix class with binarize method."""
    print("\n" + "="*60)
    print("Test 3: ConsensusMatrix class - Binarize method")
    print("="*60)
    
    matrices, ground_truth = generate_synthetic_correlation_matrices(
        n_subjects=20, n_channels=32, noise_level=0.1
    )
    
    builder = ConsensusMatrix()
    C, W = builder.compute_consensus_and_weights(matrices, sparsity=0.15, method='binarize')
    
    print(f"Consensus matrix C shape: {C.shape}")
    print(f"Weight matrix W shape: {W.shape}")
    print(f"Method used: {builder._consensus_method}")
    print(f"Binary matrices count: {len(builder.binary_matrices)}")
    
    # Check C values are fractions
    unique_c = np.unique(C[np.triu_indices(32, k=1)])
    print(f"Unique C values (sample): {unique_c[:5]}...")
    
    assert builder._consensus_method == 'binarize', "Wrong method recorded"
    assert len(builder.binary_matrices) == 20, "Wrong number of binary matrices"
    assert C.max() <= 1.0, "C values should be fractions"
    
    print("✓ Binarize method works correctly!")
    return True


def test_method_comparison():
    """Compare direct vs binarize methods."""
    print("\n" + "="*60)
    print("Test 4: Method comparison - Direct vs Binarize")
    print("="*60)
    
    matrices, ground_truth = generate_synthetic_correlation_matrices(
        n_subjects=30, n_channels=32, noise_level=0.15
    )
    
    # Direct method
    builder_direct = ConsensusMatrix()
    C_direct, W_direct = builder_direct.compute_consensus_and_weights(
        matrices, sparsity=0.15, method='direct'
    )
    
    # Binarize method
    builder_binarize = ConsensusMatrix()
    C_binarize, W_binarize = builder_binarize.compute_consensus_and_weights(
        matrices, sparsity=0.15, method='binarize'
    )
    
    # Compare with ground truth
    triu_idx = np.triu_indices(32, k=1)
    
    # Use full weight matrices for comparison
    W_full_direct = builder_direct.weight_matrix_full
    W_full_binarize = builder_binarize.weight_matrix_full
    
    corr_direct = np.corrcoef(W_full_direct[triu_idx], ground_truth[triu_idx])[0, 1]
    corr_binarize = np.corrcoef(W_full_binarize[triu_idx], ground_truth[triu_idx])[0, 1]
    
    print(f"Correlation with ground truth:")
    print(f"  Direct method:   {corr_direct:.4f}")
    print(f"  Binarize method: {corr_binarize:.4f}")
    
    # Direct method should typically recover ground truth slightly better
    # because it uses all information
    print(f"\nDirect method {'better' if corr_direct > corr_binarize else 'worse'} "
          f"than binarize by {abs(corr_direct - corr_binarize):.4f}")
    
    # Test that weight matrices have similar structure
    edge_corr = np.corrcoef(W_direct[triu_idx], W_binarize[triu_idx])[0, 1]
    print(f"Correlation between method outputs: {edge_corr:.4f}")
    
    print("✓ Method comparison completed!")
    return True


def test_direct_consensus_method():
    """Test the standalone compute_direct_consensus method."""
    print("\n" + "="*60)
    print("Test 5: compute_direct_consensus method")
    print("="*60)
    
    matrices, ground_truth = generate_synthetic_correlation_matrices(
        n_subjects=15, n_channels=32, noise_level=0.1
    )
    
    builder = ConsensusMatrix()
    
    # Test without sparsity
    consensus_dense = builder.compute_direct_consensus(matrices, sparsity=None)
    print(f"Dense consensus range: [{consensus_dense.min():.4f}, {consensus_dense.max():.4f}]")
    
    # Test with sparsity
    builder2 = ConsensusMatrix()
    consensus_sparse = builder2.compute_direct_consensus(matrices, sparsity=0.20)
    n_edges = np.sum(builder2.weight_matrix > 0) // 2
    n_possible = 32 * 31 // 2
    print(f"Sparse consensus edges: {n_edges}/{n_possible} ({n_edges/n_possible:.2%})")
    
    assert builder.weight_matrix_full is not None, "Full weight matrix should be stored"
    assert builder._consensus_method == 'direct', "Method should be recorded"
    
    print("✓ compute_direct_consensus works correctly!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Testing Consensus Matrix Implementations")
    print("#"*60)
    
    tests = [
        test_simple_consensus_function,
        test_consensus_matrix_class_direct,
        test_consensus_matrix_class_binarize,
        test_method_comparison,
        test_direct_consensus_method,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
