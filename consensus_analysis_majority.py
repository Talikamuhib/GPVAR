#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                    MAJORITY CONSENSUS ANALYSIS
                    Best Practice for AD vs HC with GP-VAR
═══════════════════════════════════════════════════════════════════════════════════════

TECHNIQUE: Majority Consensus (C > 0.50)

WHY THIS IS BEST:
1. Biological validity: Edge present in >50% of subjects = population-level connection
2. Noise removal: Excludes rare edges that may be artifacts
3. GP-VAR stability: Consistent graph structure for eigenvalue analysis
4. Group comparison: Meaningful for comparing AD vs HC

METHODOLOGY:
1. Individual matrices → Binary (proportional threshold κ=15%)
2. Compute consensus C = mean(binary matrices)
3. Compute weights W = Fisher-z average of correlations
4. SELECTION: Keep edges where C > 0.50 (majority)
5. WEIGHTING: Use W for edge weights

Author: Consensus Analysis - Majority Method
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

OUTPUT_DIR = Path("./majority_consensus_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute absolute Pearson correlation matrix."""
    A = np.abs(np.corrcoef(data))
    A = np.nan_to_num(A, nan=0.0)
    np.fill_diagonal(A, 0)
    return A


def proportional_threshold(A: np.ndarray, sparsity: float = 0.15) -> np.ndarray:
    """Proportional thresholding to binary matrix (κ = sparsity)."""
    n = A.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    weights = A[triu_idx]
    k = max(1, int(sparsity * len(weights)))
    threshold = np.sort(weights)[-k] if k < len(weights) else 0
    B = (A > threshold).astype(float)
    B = np.maximum(B, B.T)
    np.fill_diagonal(B, 0)
    return B


def compute_consensus_and_weights(adjacency_matrices: np.ndarray, 
                                   binary_matrices: np.ndarray):
    """
    Compute consensus matrix C and weight matrix W.
    
    C[i,j] = fraction of subjects with edge (i,j)
    W[i,j] = Fisher-z average of correlations (only where edge exists)
    """
    n_subjects, n_channels, _ = adjacency_matrices.shape
    
    # Consensus
    C = np.mean(binary_matrices, axis=0)
    
    # Weights with Fisher-z
    W = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            mask = binary_matrices[:, i, j] > 0
            if np.any(mask):
                r = adjacency_matrices[mask, i, j]
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                W[i, j] = np.abs(np.tanh(np.mean(z)))
                W[j, i] = W[i, j]
    
    return C, W


def majority_consensus_selection(C: np.ndarray, W: np.ndarray, 
                                  threshold: float = 0.50):
    """
    Apply majority consensus selection.
    
    RULE: Keep edge (i,j) if C[i,j] > threshold
    
    Parameters
    ----------
    C : Consensus matrix
    W : Weight matrix  
    threshold : Consensus threshold (default 0.50 = majority)
    
    Returns
    -------
    G : Final graph with W as edge weights
    mask : Boolean mask of kept edges
    stats : Dictionary of statistics
    """
    n = C.shape[0]
    n_possible = n * (n - 1) // 2
    
    # Apply majority rule
    mask = C > threshold
    
    # Build final graph
    G = np.zeros_like(C)
    G[mask] = W[mask]
    
    # Statistics
    n_kept = np.sum(mask) // 2  # Divide by 2 for undirected
    
    # Consensus level breakdown
    triu = np.triu(mask, k=1)
    C_kept = C[triu]
    
    n_unanimous = np.sum(C_kept >= 0.99)
    n_strong = np.sum((C_kept >= 0.75) & (C_kept < 0.99))
    n_moderate = np.sum((C_kept >= 0.50) & (C_kept < 0.75))
    
    stats = {
        'n_possible': n_possible,
        'n_kept': n_kept,
        'sparsity': n_kept / n_possible,
        'threshold': threshold,
        'n_unanimous': n_unanimous,
        'n_strong': n_strong,
        'n_moderate': n_moderate,
        'mean_consensus': np.mean(C_kept) if len(C_kept) > 0 else 0,
        'mean_weight': np.mean(W[triu]) if np.sum(triu) > 0 else 0,
    }
    
    return G, mask, stats


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def plot_majority_consensus_analysis(C, W, G, stats, channel_locations, 
                                      n_ad, n_hc, save_path=None):
    """Create comprehensive visualization for majority consensus."""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)
    
    n_channels = C.shape[0]
    
    # ═══════ Row 1: Matrices ═══════
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(C, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_title('Consensus Matrix C', fontsize=11, fontweight='bold')
    ax1.axhline(y=-0.5, color='black', linewidth=2)
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='Fraction')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(W, cmap='viridis', vmin=0)
    ax2.set_title('Weight Matrix W', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='Correlation')
    
    ax3 = fig.add_subplot(gs[0, 2])
    C_thresh = np.where(C > 0.5, C, 0)
    im3 = ax3.imshow(C_thresh, cmap='YlOrRd', vmin=0, vmax=1)
    ax3.set_title('After Majority Rule (C > 0.5)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, label='Fraction')
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(G, cmap='hot', vmin=0)
    ax4.set_title(f'Final Graph G\n({stats["sparsity"]*100:.1f}% sparsity)', 
                 fontsize=11, fontweight='bold', color='green')
    plt.colorbar(im4, ax=ax4, fraction=0.046, label='Weight')
    
    # ═══════ Row 2: Analysis ═══════
    
    # Consensus distribution
    ax5 = fig.add_subplot(gs[1, 0])
    triu_idx = np.triu_indices(n_channels, k=1)
    all_consensus = C[triu_idx]
    kept_consensus = all_consensus[all_consensus > 0.5]
    
    ax5.hist(all_consensus, bins=50, alpha=0.5, color='gray', label='All edges')
    ax5.hist(kept_consensus, bins=50, alpha=0.7, color='green', label='Kept (C>0.5)')
    ax5.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax5.set_xlabel('Consensus C')
    ax5.set_ylabel('Frequency')
    ax5.set_title('CONSENSUS DISTRIBUTION', fontsize=11, fontweight='bold')
    ax5.legend()
    
    # Consensus levels
    ax6 = fig.add_subplot(gs[1, 1])
    levels = ['Unanimous\n(≥99%)', 'Strong\n(75-99%)', 'Moderate\n(50-75%)']
    counts = [stats['n_unanimous'], stats['n_strong'], stats['n_moderate']]
    colors = ['#2ecc71', '#27ae60', '#f1c40f']
    bars = ax6.bar(levels, counts, color=colors, edgecolor='black')
    ax6.set_ylabel('Number of Edges')
    ax6.set_title('EDGE QUALITY BREAKDOWN', fontsize=11, fontweight='bold')
    for bar, count in zip(bars, counts):
        pct = count / stats['n_kept'] * 100 if stats['n_kept'] > 0 else 0
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{count}\n({pct:.1f}%)', ha='center', fontsize=10)
    
    # Distance distribution
    ax7 = fig.add_subplot(gs[1, 2:4])
    D = squareform(pdist(channel_locations))
    distances_all = D[triu_idx]
    mask_kept = C[triu_idx] > 0.5
    distances_kept = distances_all[mask_kept]
    
    bins = np.linspace(0, np.max(distances_all), 20)
    ax7.hist(distances_all, bins=bins, alpha=0.4, color='gray', 
            label=f'All possible ({len(distances_all)})', density=True)
    ax7.hist(distances_kept, bins=bins, alpha=0.7, color='green',
            label=f'Kept ({len(distances_kept)})', density=True)
    ax7.set_xlabel('Distance')
    ax7.set_ylabel('Density')
    ax7.set_title('DISTANCE DISTRIBUTION: Kept vs All', fontsize=11, fontweight='bold')
    ax7.legend()
    
    # ═══════ Row 3: Summary ═══════
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    MAJORITY CONSENSUS ANALYSIS SUMMARY                                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                              ║
║  SAMPLE                                   SELECTION RULE                                                     ║
║  ───────────────────────                  ─────────────────────────────────────────────────────              ║
║  AD patients:        {n_ad:3d}                  Keep edge (i,j) if C[i,j] > 0.50                                  ║
║  Healthy Controls:   {n_hc:3d}                  "Edge must be present in MAJORITY of subjects"                    ║
║  TOTAL:              {n_ad+n_hc:3d}                                                                                  ║
║                                                                                                              ║
║  RESULTS                                  WHY THIS IS THE BEST TECHNIQUE                                     ║
║  ───────────────────────                  ─────────────────────────────────────────────────────              ║
║  Possible edges:     {stats['n_possible']:4d}                 1. Biological validity: >50% = population-level               ║
║  Kept edges:         {stats['n_kept']:4d}                 2. Noise removal: Excludes rare/artifactual edges             ║
║  Final sparsity:     {stats['sparsity']*100:5.1f}%              3. GP-VAR stability: Consistent eigenvalues                  ║
║                                           4. Interpretable: "Most subjects have this connection"            ║
║  QUALITY BREAKDOWN                                                                                           ║
║  ───────────────────────                  WHAT THE FINAL GRAPH REPRESENTS                                    ║
║  Unanimous (≥99%):   {stats['n_unanimous']:4d}                 ─────────────────────────────────────────────────────        ║
║  Strong (75-99%):    {stats['n_strong']:4d}                 G[i,j] = W[i,j] if C[i,j] > 0.50, else 0                     ║
║  Moderate (50-75%):  {stats['n_moderate']:4d}                                                                                ║
║                                           The edge weight is the Fisher-z averaged correlation               ║
║  Mean consensus:     {stats['mean_consensus']:.4f}               of subjects who have that connection.                        ║
║  Mean weight:        {stats['mean_weight']:.4f}                                                                              ║
║                                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    ax8.text(0.5, 0.5, summary, transform=ax8.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.suptitle('MAJORITY CONSENSUS ANALYSIS: The Recommended Technique for AD vs HC',
                fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def generate_report(C, W, G, stats, n_ad, n_hc, save_path=None):
    """Generate text report."""
    
    report = f"""
================================================================================
                    MAJORITY CONSENSUS ANALYSIS REPORT
                    Best Practice for AD vs HC with GP-VAR
================================================================================

TECHNIQUE: Majority Consensus (C > 0.50)
========================================

RULE: Keep edge (i,j) if and only if C[i,j] > 0.50

This means: "The edge must be present in MORE THAN HALF of all subjects."

WHY THIS IS THE BEST TECHNIQUE:
-------------------------------
1. BIOLOGICAL VALIDITY
   - An edge in >50% of subjects represents a population-level connection
   - Not just individual variation or noise
   
2. NOISE REMOVAL  
   - Edges in <50% of subjects are excluded
   - These may be artifacts, measurement error, or individual quirks
   
3. GP-VAR STABILITY
   - Consistent graph structure leads to stable eigenvalues
   - The Graph Laplacian will be well-conditioned
   
4. GROUP COMPARISON
   - Meaningful for comparing AD vs HC networks
   - Both groups contribute to the consensus

SAMPLE
------
  AD patients:        {n_ad}
  Healthy Controls:   {n_hc}
  TOTAL subjects:     {n_ad + n_hc}

RESULTS
-------
  Possible edges:     {stats['n_possible']}
  Kept edges:         {stats['n_kept']}
  Final sparsity:     {stats['sparsity']*100:.2f}%

EDGE QUALITY BREAKDOWN
----------------------
  Unanimous (C ≥ 99%):   {stats['n_unanimous']:4d} ({stats['n_unanimous']/stats['n_kept']*100:.1f}%)
  Strong (75% ≤ C < 99%):{stats['n_strong']:4d} ({stats['n_strong']/stats['n_kept']*100:.1f}%)
  Moderate (50% < C < 75%): {stats['n_moderate']:4d} ({stats['n_moderate']/stats['n_kept']*100:.1f}%)

QUALITY METRICS
---------------
  Mean consensus of kept edges: {stats['mean_consensus']:.4f}
  Mean weight of kept edges:    {stats['mean_weight']:.4f}

FINAL GRAPH INTERPRETATION
--------------------------
  G[i,j] = W[i,j]  if C[i,j] > 0.50
  G[i,j] = 0       otherwise

  The edge weight W is the Fisher-z averaged correlation
  computed only from subjects who have that connection.

USE FOR GP-VAR
--------------
  1. Use G as the adjacency matrix
  2. Compute Laplacian: L = D - G
  3. Eigendecomposition for graph frequencies
  4. The graph is sparse but biologically meaningful

================================================================================
                              END OF REPORT
================================================================================
"""
    
    if save_path:
        Path(save_path).write_text(report)
        print(f"✓ Report saved: {save_path}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              DATA GENERATION (SYNTHETIC)
# ═══════════════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_subjects, n_channels, n_samples, group='HC', seed=None):
    """Generate synthetic EEG-like data."""
    if seed:
        np.random.seed(seed)
    
    # Channel locations on unit sphere
    theta = np.linspace(0, 2*np.pi, int(np.ceil(np.sqrt(n_channels)*1.5)), endpoint=False)
    phi = np.linspace(0.2*np.pi, 0.8*np.pi, int(np.ceil(np.sqrt(n_channels))))
    locs = [[np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)] 
            for t in theta for p in phi][:n_channels]
    channel_locations = np.array(locs)
    
    # Spatial covariance based on distance
    D = squareform(pdist(channel_locations))
    spatial_cov = np.exp(-D**2 / (2*0.5**2))
    
    # AD has weaker connectivity
    if group == 'AD':
        spatial_cov = spatial_cov * 0.8 + 0.2 * np.eye(n_channels)
    
    L = np.linalg.cholesky(spatial_cov + 0.01*np.eye(n_channels))
    data = np.array([L @ np.random.randn(n_channels, n_samples) for _ in range(n_subjects)])
    
    return data, channel_locations


# ═══════════════════════════════════════════════════════════════════════════════════════
#                                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_majority_consensus_analysis(n_ad=35, n_hc=31, n_channels=64, n_samples=1000):
    """Run complete majority consensus analysis."""
    
    print("="*70)
    print("            MAJORITY CONSENSUS ANALYSIS")
    print("            Best Technique for AD vs HC + GP-VAR")
    print("="*70)
    
    # Generate data
    print("\n[1/4] Generating synthetic data...")
    ad_data, channel_locations = generate_synthetic_data(n_ad, n_channels, n_samples, 'AD', 42)
    hc_data, _ = generate_synthetic_data(n_hc, n_channels, n_samples, 'HC', 123)
    
    # Compute correlation matrices
    print("[2/4] Computing correlation & binary matrices...")
    ad_adj = np.array([compute_correlation_matrix(ad_data[s]) for s in range(n_ad)])
    hc_adj = np.array([compute_correlation_matrix(hc_data[s]) for s in range(n_hc)])
    
    # Binary matrices (κ = 15%)
    ad_bin = np.array([proportional_threshold(ad_adj[s], 0.15) for s in range(n_ad)])
    hc_bin = np.array([proportional_threshold(hc_adj[s], 0.15) for s in range(n_hc)])
    
    # Combined
    all_adj = np.concatenate([ad_adj, hc_adj], axis=0)
    all_bin = np.concatenate([ad_bin, hc_bin], axis=0)
    
    # Consensus and weights
    print("[3/4] Computing consensus and weights...")
    C, W = compute_consensus_and_weights(all_adj, all_bin)
    
    # Apply majority rule
    print("[4/4] Applying majority consensus rule (C > 0.50)...")
    G, mask, stats = majority_consensus_selection(C, W, threshold=0.50)
    
    # Results
    print("\n" + "="*70)
    print("                       RESULTS")
    print("="*70)
    print(f"\n  Selection rule: C > 0.50 (edge in majority of subjects)")
    print(f"\n  Possible edges: {stats['n_possible']}")
    print(f"  Kept edges:     {stats['n_kept']}")
    print(f"  Final sparsity: {stats['sparsity']*100:.2f}%")
    print(f"\n  Quality breakdown:")
    print(f"    Unanimous (≥99%):  {stats['n_unanimous']:4d}")
    print(f"    Strong (75-99%):   {stats['n_strong']:4d}")
    print(f"    Moderate (50-75%): {stats['n_moderate']:4d}")
    print(f"\n  Mean consensus: {stats['mean_consensus']:.4f}")
    print(f"  Mean weight:    {stats['mean_weight']:.4f}")
    
    # Save outputs
    print("\n" + "="*70)
    print("Saving outputs...")
    
    plot_majority_consensus_analysis(C, W, G, stats, channel_locations,
                                     n_ad, n_hc, 
                                     save_path=str(OUTPUT_DIR / "majority_consensus_analysis.png"))
    
    generate_report(C, W, G, stats, n_ad, n_hc,
                   save_path=str(OUTPUT_DIR / "majority_consensus_report.txt"))
    
    # Save matrices
    np.save(OUTPUT_DIR / "consensus_matrix_C.npy", C)
    np.save(OUTPUT_DIR / "weight_matrix_W.npy", W)
    np.save(OUTPUT_DIR / "final_graph_G.npy", G)
    
    print(f"\n{'='*70}")
    print("                    COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print(f"\nFinal graph G is ready for GP-VAR analysis!")
    
    return G, C, W, stats


if __name__ == "__main__":
    G, C, W, stats = run_majority_consensus_analysis()
