"""
Test Script for GP-VAR Analysis with Synthetic Data
====================================================

This script generates synthetic EEG-like data with known properties
to validate the LTI vs Time-Varying analysis pipeline.

Two scenarios:
1. TRUE LTI: Generate data with fixed dynamics
2. TRUE TV: Generate data with changing dynamics over time
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from main analysis
from lti_tv_gpvar_analysis import (
    GPVAR_SharedH,
    safe_zscore,
    split_into_windows,
    compute_tv_models,
    compare_transfer_functions,
    coeff_variation_stats,
    circular_shift_surrogate
)

# ============================================================================
# Synthetic Data Generation
# ============================================================================

def create_synthetic_laplacian(n_nodes: int = 19) -> np.ndarray:
    """Create a simple synthetic Laplacian matrix."""
    # Create a circular graph with some additional connections
    L = np.zeros((n_nodes, n_nodes))
    
    # Circular connections
    for i in range(n_nodes):
        L[i, (i+1) % n_nodes] = -0.5
        L[i, (i-1) % n_nodes] = -0.5
    
    # Add some cross connections
    for i in range(0, n_nodes, 3):
        if i + n_nodes // 2 < n_nodes:
            L[i, i + n_nodes // 2] = -0.2
            L[i + n_nodes // 2, i] = -0.2
    
    # Set diagonal (degree)
    np.fill_diagonal(L, -L.sum(axis=1))
    
    # Make symmetric and positive semi-definite
    L = (L + L.T) / 2
    eigvals = np.linalg.eigvalsh(L)
    if eigvals.min() < 0:
        L = L - eigvals.min() * np.eye(n_nodes) + 1e-6 * np.eye(n_nodes)
    
    return L


def generate_lti_data(L: np.ndarray, T: int = 3000, 
                      P: int = 3, K: int = 2, 
                      noise_std: float = 0.1) -> tuple:
    """Generate synthetic LTI GP-VAR data with fixed coefficients."""
    n = L.shape[0]
    np.random.seed(42)
    
    # Fixed coefficients for LTI system
    h_true = np.zeros(P * (K + 1))
    # Make a stable system
    h_true[0] = 0.3   # p=1, k=0
    h_true[1] = -0.1  # p=1, k=1
    h_true[2] = 0.05  # p=1, k=2
    h_true[3] = 0.2   # p=2, k=0
    h_true[4] = -0.05 # p=2, k=1
    h_true[5] = 0.02  # p=2, k=2
    h_true[6] = 0.1   # p=3, k=0
    h_true[7] = -0.02 # p=3, k=1
    h_true[8] = 0.01  # p=3, k=2
    
    # Precompute L powers
    L_powers = [np.eye(n)]
    for k in range(1, K+1):
        L_powers.append(L_powers[-1] @ L)
    
    # Generate data
    X = np.zeros((n, T))
    # Random initialization
    X[:, :P] = np.random.randn(n, P) * 0.5
    
    for t in range(P, T):
        x_t = np.zeros(n)
        
        # Apply GP-VAR dynamics
        idx = 0
        for p in range(1, P+1):
            for k in range(K+1):
                x_t += h_true[idx] * (L_powers[k] @ X[:, t-p])
                idx += 1
        
        # Add noise
        x_t += np.random.randn(n) * noise_std
        X[:, t] = x_t
    
    return X, h_true


def generate_tv_data(L: np.ndarray, T: int = 3000, 
                     P: int = 3, K: int = 2,
                     noise_std: float = 0.1,
                     n_changes: int = 2) -> tuple:
    """Generate synthetic TV GP-VAR data with time-varying coefficients."""
    n = L.shape[0]
    np.random.seed(42)
    
    # Create time-varying coefficients
    h_segments = []
    segment_length = T // (n_changes + 1)
    
    # Different coefficients for each segment
    base_h = np.array([0.3, -0.1, 0.05, 0.2, -0.05, 0.02, 0.1, -0.02, 0.01])
    
    for seg in range(n_changes + 1):
        # Modify coefficients for each segment
        h_seg = base_h.copy()
        h_seg[0] = 0.3 + 0.2 * np.sin(2 * np.pi * seg / (n_changes + 1))
        h_seg[3] = 0.2 - 0.1 * seg / n_changes
        h_segments.append(h_seg)
    
    # Precompute L powers
    L_powers = [np.eye(n)]
    for k in range(1, K+1):
        L_powers.append(L_powers[-1] @ L)
    
    # Generate data
    X = np.zeros((n, T))
    X[:, :P] = np.random.randn(n, P) * 0.5
    
    for t in range(P, T):
        # Determine which segment we're in
        seg_idx = min(t // segment_length, n_changes)
        h_current = h_segments[seg_idx]
        
        x_t = np.zeros(n)
        
        # Apply time-varying GP-VAR dynamics
        idx = 0
        for p in range(1, P+1):
            for k in range(K+1):
                x_t += h_current[idx] * (L_powers[k] @ X[:, t-p])
                idx += 1
        
        # Add noise
        x_t += np.random.randn(n) * noise_std
        X[:, t] = x_t
    
    # Create time vector of true h values
    h_true_time = np.zeros((T, P * (K + 1)))
    for t in range(T):
        seg_idx = min(t // segment_length, n_changes)
        h_true_time[t] = h_segments[seg_idx]
    
    return X, h_true_time


# ============================================================================
# Simplified Analysis Function
# ============================================================================

def analyze_synthetic_data(X: np.ndarray, L: np.ndarray, 
                          true_P: int, true_K: int,
                          scenario_name: str) -> dict:
    """Run simplified analysis on synthetic data."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {scenario_name}")
    print(f"{'='*60}")
    
    n, T = X.shape
    fs = 100.0  # Assumed sampling frequency
    
    # Standardize
    X_std = safe_zscore(X, X)
    
    # Fit LTI model
    print("Fitting LTI model...")
    lti_model = GPVAR_SharedH(P=true_P, K=true_K, L=L)
    lti_model.fit(X_std)
    lti_rho = lti_model.spectral_radius()
    print(f"  LTI spectral radius: {lti_rho:.3f}")
    
    # Fit TV models
    print("Fitting TV models (10s windows, 50% overlap)...")
    tv_results = compute_tv_models(X_std, L, true_P, true_K,
                                  window_length_sec=10.0, 
                                  overlap=0.5, fs=fs)
    print(f"  Fitted {len(tv_results)} windows")
    
    if len(tv_results) < 3:
        print("  ERROR: Too few windows")
        return None
    
    # Coefficient variation
    coeff_stats = coeff_variation_stats(tv_results)
    print(f"  Coefficient CV: {coeff_stats['global_coeff_cv']:.4f}")
    
    # Compare transfer functions
    comparison = compare_transfer_functions(lti_model, tv_results)
    print(f"  Global MSD: {comparison['global_msd']:.6f}")
    print(f"  Global variance: {comparison['global_variance']:.6f}")
    
    # Simple statistical test
    G_lti_flat = comparison['G_lti'].ravel()
    G_tv_flat = comparison['G_tv_all'].reshape(comparison['n_windows'], -1)
    
    G_tv_mean = G_tv_flat.mean(axis=0)
    G_tv_se = G_tv_flat.std(axis=0) / np.sqrt(comparison['n_windows'])
    ci_lower = G_tv_mean - 1.96 * G_tv_se
    ci_upper = G_tv_mean + 1.96 * G_tv_se
    
    outside_ci = ((G_lti_flat < ci_lower) | (G_lti_flat > ci_upper)).mean()
    print(f"  Outside CI: {outside_ci*100:.1f}%")
    
    # Decision
    is_tv = (comparison['global_msd'] > 1e-4) or (outside_ci > 0.10)
    print(f"\nDecision: {'TIME-VARYING' if is_tv else 'TIME-INVARIANT'}")
    
    return {
        'scenario': scenario_name,
        'lti_model': lti_model,
        'tv_results': tv_results,
        'coeff_stats': coeff_stats,
        'comparison': comparison,
        'outside_ci': outside_ci,
        'is_time_varying': is_tv
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_synthetic_results(results_lti: dict, results_tv: dict):
    """Plot comparison of LTI vs TV scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (results, title) in enumerate([(results_lti, "TRUE LTI"), 
                                            (results_tv, "TRUE TV")]):
        if results is None:
            continue
            
        comp = results['comparison']
        coeff = results['coeff_stats']
        
        # Plot 1: Transfer function (LTI)
        ax = axes[idx, 0]
        im = ax.imshow(comp['G_lti'], aspect='auto', origin='lower', cmap='hot')
        ax.set_title(f"{title}: LTI |G(ω,λ)|")
        ax.set_xlabel("λ (Graph Freq)")
        ax.set_ylabel("ω (Temporal Freq)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot 2: Variance across windows
        ax = axes[idx, 1]
        im = ax.imshow(comp['variance_across_windows'], aspect='auto', 
                      origin='lower', cmap='YlOrRd')
        ax.set_title(f"{title}: Variance Across Windows")
        ax.set_xlabel("λ (Graph Freq)")
        ax.set_ylabel("ω (Temporal Freq)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot 3: Coefficient variation
        ax = axes[idx, 2]
        H_windows = coeff['H_windows']
        for w_idx in range(min(5, H_windows.shape[0])):
            ax.plot(H_windows[w_idx, :], alpha=0.3, linewidth=1)
        ax.plot(coeff['h_mean'], 'k-', linewidth=2, label='Mean')
        ax.set_title(f"{title}: Coefficients (CV={coeff['global_coeff_cv']:.3f})")
        ax.set_xlabel("Coefficient Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add decision text
        decision = 'TIME-VARYING' if results['is_time_varying'] else 'TIME-INVARIANT'
        ax.text(0.5, 1.05, f"Decision: {decision}", 
                transform=ax.transAxes, ha='center', fontweight='bold',
                color='red' if results['is_time_varying'] else 'green')
    
    plt.suptitle("Synthetic Data Validation: LTI vs TV Detection", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("test_synthetic_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved test results to: test_synthetic_results.png")


# ============================================================================
# Main Test Function
# ============================================================================

def main():
    """Run validation tests with synthetic data."""
    print("="*60)
    print("GP-VAR SYNTHETIC DATA VALIDATION")
    print("="*60)
    
    # Create synthetic Laplacian
    print("\nCreating synthetic graph Laplacian...")
    n_nodes = 19
    L = create_synthetic_laplacian(n_nodes)
    print(f"  Nodes: {n_nodes}")
    print(f"  Eigenvalue range: [{np.linalg.eigvalsh(L).min():.3f}, "
          f"{np.linalg.eigvalsh(L).max():.3f}]")
    
    # Test parameters
    T = 3000  # 30 seconds at 100 Hz
    true_P = 3
    true_K = 2
    
    # Scenario 1: True LTI system
    print("\n" + "="*60)
    print("SCENARIO 1: TRUE LTI SYSTEM")
    print("="*60)
    print("Generating LTI data...")
    X_lti, h_true_lti = generate_lti_data(L, T=T, P=true_P, K=true_K)
    print(f"  Data shape: {X_lti.shape}")
    print(f"  True coefficients: {h_true_lti}")
    
    results_lti = analyze_synthetic_data(X_lti, L, true_P, true_K, "LTI System")
    
    # Scenario 2: True TV system
    print("\n" + "="*60)
    print("SCENARIO 2: TRUE TIME-VARYING SYSTEM")
    print("="*60)
    print("Generating TV data...")
    X_tv, h_true_tv = generate_tv_data(L, T=T, P=true_P, K=true_K, n_changes=2)
    print(f"  Data shape: {X_tv.shape}")
    print(f"  Coefficient changes: {h_true_tv.std(axis=0)}")
    
    results_tv = analyze_synthetic_data(X_tv, L, true_P, true_K, "TV System")
    
    # Visualize results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    plot_synthetic_results(results_lti, results_tv)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if results_lti and results_tv:
        lti_correct = not results_lti['is_time_varying']
        tv_correct = results_tv['is_time_varying']
        
        print(f"\nLTI Detection: {'✓ CORRECT' if lti_correct else '✗ INCORRECT'}")
        print(f"  - Expected: TIME-INVARIANT")
        print(f"  - Detected: {'TIME-VARYING' if results_lti['is_time_varying'] else 'TIME-INVARIANT'}")
        print(f"  - MSD: {results_lti['comparison']['global_msd']:.6f}")
        print(f"  - Outside CI: {results_lti['outside_ci']*100:.1f}%")
        
        print(f"\nTV Detection: {'✓ CORRECT' if tv_correct else '✗ INCORRECT'}")
        print(f"  - Expected: TIME-VARYING")
        print(f"  - Detected: {'TIME-VARYING' if results_tv['is_time_varying'] else 'TIME-INVARIANT'}")
        print(f"  - MSD: {results_tv['comparison']['global_msd']:.6f}")
        print(f"  - Outside CI: {results_tv['outside_ci']*100:.1f}%")
        
        if lti_correct and tv_correct:
            print("\n🎉 VALIDATION PASSED: Both scenarios correctly classified!")
        else:
            print("\n⚠️ VALIDATION ISSUES: Check parameters and thresholds")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    return results_lti, results_tv


if __name__ == "__main__":
    results = main()