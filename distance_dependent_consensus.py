#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
        DISTANCE-DEPENDENT CONSENSUS MATRIX CONSTRUCTION
        Mathematical Implementation following Betzel et al. (2019)
═══════════════════════════════════════════════════════════════════════════════════════

MATHEMATICAL FRAMEWORK:

1. Correlation:     A^(s)_ij = |corr(x_i, x_j)|
2. Binarization:    B^(s)_ij = 𝟙[A^(s)_ij > τ^(s)]  where τ keeps top κ%
3. Consensus:       C_ij = (1/N) Σ_s B^(s)_ij
4. Weights:         W_ij = |tanh(mean(arctanh(A^(s)_ij)))|  (Fisher-z)
5. Distance:        D_ij = ||p_i - p_j||₂
6. Scoring:         S_ij = C_ij + ε·W_ij
7. Selection:       Keep edge if C_ij > 0.50 (majority consensus)
8. Final Graph:     G_ij = W_ij · 𝟙[C_ij > 0.50]

Reference: Betzel et al. (2019) Network Neuroscience
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
from typing import Dict, List, Tuple

OUTPUT_DIR = Path("./distance_dependent_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Parameters:
    """Model parameters with mathematical notation."""
    kappa: float = 0.15      # κ: proportional threshold (15%)
    epsilon: float = 0.1     # ε: weight factor in scoring
    K: int = 10              # K: number of distance bins
    threshold: float = 0.50  # Consensus threshold for majority rule


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 1: CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_adjacency_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute adjacency matrix from EEG data.
    
    Mathematical Definition:
        A_ij = |r_ij| where r_ij = corr(x_i, x_j)
        A_ii = 0 (no self-loops)
    
    Parameters
    ----------
    X : np.ndarray, shape (C, T)
        EEG data matrix, C channels × T time samples
    
    Returns
    -------
    A : np.ndarray, shape (C, C)
        Adjacency matrix with absolute correlations
    """
    # r_ij = Pearson correlation
    A = np.abs(np.corrcoef(X))
    A = np.nan_to_num(A, nan=0.0)
    np.fill_diagonal(A, 0)  # A_ii = 0
    return A


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 2: PROPORTIONAL THRESHOLDING
# ═══════════════════════════════════════════════════════════════════════════════════════

def proportional_threshold(A: np.ndarray, kappa: float) -> np.ndarray:
    """
    Binarize adjacency matrix using proportional thresholding.
    
    Mathematical Definition:
        B_ij = 𝟙[A_ij > τ]
        
        where τ = Q_{1-κ}({A_ij : i < j})
        is the (1-κ)-quantile of upper triangular elements
    
    Parameters
    ----------
    A : np.ndarray, shape (C, C)
        Continuous adjacency matrix
    kappa : float
        Proportion of edges to keep (e.g., 0.15 for 15%)
    
    Returns
    -------
    B : np.ndarray, shape (C, C)
        Binary adjacency matrix
    """
    C = A.shape[0]
    triu_idx = np.triu_indices(C, k=1)
    weights = A[triu_idx]
    
    # τ = Q_{1-κ}(weights)
    n_keep = max(1, int(kappa * len(weights)))
    tau = np.sort(weights)[-n_keep]
    
    # B_ij = 𝟙[A_ij > τ]
    B = (A > tau).astype(float)
    B = np.maximum(B, B.T)  # Ensure symmetry
    np.fill_diagonal(B, 0)
    
    return B


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 3: CONSENSUS MATRIX
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_consensus_matrix(binary_matrices: np.ndarray) -> np.ndarray:
    """
    Compute consensus matrix from binary matrices.
    
    Mathematical Definition:
        C_ij = (1/N) Σ_{s=1}^{N} B^(s)_ij
        
        C_ij ∈ [0, 1] represents fraction of subjects with edge (i,j)
    
    Parameters
    ----------
    binary_matrices : np.ndarray, shape (N, C, C)
        Binary adjacency matrices for N subjects
    
    Returns
    -------
    C : np.ndarray, shape (C, C)
        Consensus matrix
    """
    # C_ij = (1/N) Σ_s B^(s)_ij
    C = np.mean(binary_matrices, axis=0)
    return C


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 4: WEIGHT MATRIX (FISHER-Z)
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_weight_matrix(adjacency_matrices: np.ndarray, 
                          binary_matrices: np.ndarray) -> np.ndarray:
    """
    Compute weight matrix using Fisher-z transformation.
    
    Mathematical Definition:
        Let S_ij = {s : B^(s)_ij = 1} be subjects with edge (i,j)
        
        W_ij = |tanh( (1/|S_ij|) Σ_{s ∈ S_ij} arctanh(A^(s)_ij) )|
        
        Fisher-z: z = arctanh(r), r = tanh(z)
    
    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (N, C, C)
        Continuous adjacency matrices
    binary_matrices : np.ndarray, shape (N, C, C)
        Binary adjacency matrices
    
    Returns
    -------
    W : np.ndarray, shape (C, C)
        Weight matrix with Fisher-z averaged correlations
    """
    N, C, _ = adjacency_matrices.shape
    W = np.zeros((C, C))
    
    for i in range(C):
        for j in range(i + 1, C):
            # S_ij = {s : B^(s)_ij = 1}
            S_ij = binary_matrices[:, i, j] > 0
            
            if np.any(S_ij):
                # Get correlations for subjects in S_ij
                r_values = adjacency_matrices[S_ij, i, j]
                
                # Clip to avoid arctanh(±1) = ±∞
                r_clipped = np.clip(r_values, -0.999, 0.999)
                
                # Fisher-z: z = arctanh(r)
                z_values = np.arctanh(r_clipped)
                
                # Average in z-space
                z_mean = np.mean(z_values)
                
                # Back-transform: r = tanh(z), take absolute
                W[i, j] = np.abs(np.tanh(z_mean))
                W[j, i] = W[i, j]
    
    return W


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 5: DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_distance_matrix(channel_locations: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix between channels.
    
    Mathematical Definition:
        D_ij = ||p_i - p_j||₂ = √[(p_ix - p_jx)² + (p_iy - p_jy)² + (p_iz - p_jz)²]
    
    Parameters
    ----------
    channel_locations : np.ndarray, shape (C, 3)
        3D coordinates of each channel
    
    Returns
    -------
    D : np.ndarray, shape (C, C)
        Distance matrix
    """
    # D_ij = ||p_i - p_j||₂
    D = squareform(pdist(channel_locations, metric='euclidean'))
    return D


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 6: DISTANCE BINNING
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_distance_bins(D: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create K distance bins using percentiles.
    
    Mathematical Definition:
        Bin boundaries: d_0, d_1, ..., d_K
        where d_k = Q_{k/K}({D_ij : i < j})
        
        Edge (i,j) ∈ Bin b if d_{b-1} ≤ D_ij < d_b
    
    Parameters
    ----------
    D : np.ndarray, shape (C, C)
        Distance matrix
    K : int
        Number of bins
    
    Returns
    -------
    bin_edges : np.ndarray, shape (K+1,)
        Bin boundaries [d_0, d_1, ..., d_K]
    bin_assignments : np.ndarray, shape (C, C)
        Bin index for each edge (0 to K-1)
    """
    C = D.shape[0]
    triu_idx = np.triu_indices(C, k=1)
    distances = D[triu_idx]
    
    # d_k = Q_{k/K}(distances) for k = 0, 1, ..., K
    percentiles = np.linspace(0, 100, K + 1)
    bin_edges = np.percentile(distances, percentiles)
    bin_edges[-1] += 1e-10  # Ensure max distance is included
    
    # Assign each edge to a bin
    bin_assignments = np.zeros((C, C), dtype=int)
    for idx in range(len(distances)):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        d = distances[idx]
        for b in range(K):
            if bin_edges[b] <= d < bin_edges[b + 1]:
                bin_assignments[i, j] = b
                bin_assignments[j, i] = b
                break
    
    return bin_edges, bin_assignments


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 7: SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_score_matrix(C: np.ndarray, W: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute edge scores.
    
    Mathematical Definition:
        S_ij = C_ij + ε · W_ij
        
        - C_ij ∈ [0, 1]: consensus (primary)
        - ε · W_ij ∈ [0, ~0.1]: weight contribution (tie-breaker)
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix
    W : np.ndarray
        Weight matrix
    epsilon : float
        Weight factor (e.g., 0.1)
    
    Returns
    -------
    S : np.ndarray
        Score matrix
    """
    # S_ij = C_ij + ε · W_ij
    S = C + epsilon * W
    return S


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    STEP 8: MAJORITY CONSENSUS SELECTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def majority_consensus_selection(C: np.ndarray, W: np.ndarray, 
                                  threshold: float = 0.50) -> np.ndarray:
    """
    Apply majority consensus selection to build final graph.
    
    Mathematical Definition:
        G_ij = W_ij · 𝟙[C_ij > threshold]
        
        Default threshold = 0.50 (majority rule)
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix
    W : np.ndarray
        Weight matrix
    threshold : float
        Consensus threshold (default 0.50)
    
    Returns
    -------
    G : np.ndarray
        Final graph adjacency matrix
    """
    # G_ij = W_ij · 𝟙[C_ij > threshold]
    mask = C > threshold
    G = np.where(mask, W, 0)
    return G


# ═══════════════════════════════════════════════════════════════════════════════════════
#                    DISTANCE DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════

def analyze_distance_distribution(G: np.ndarray, C: np.ndarray, W: np.ndarray,
                                   S: np.ndarray, bin_assignments: np.ndarray,
                                   bin_edges: np.ndarray, K: int) -> List[Dict]:
    """
    Analyze edge distribution across distance bins.
    
    For each bin b, compute:
        - n_b^possible: number of possible edges
        - n_b^kept: number of kept edges
        - Retention_b = n_b^kept / n_b^possible
        - Mean consensus, weight, score
    """
    n_channels = G.shape[0]
    triu_idx = np.triu_indices(n_channels, k=1)
    
    stats = []
    for b in range(K):
        # Edges in this bin
        in_bin = bin_assignments[triu_idx] == b
        n_possible = np.sum(in_bin)
        
        # Kept edges in this bin
        G_triu = G[triu_idx]
        kept = (G_triu > 0) & in_bin
        n_kept = np.sum(kept)
        
        # Statistics
        retention = n_kept / n_possible if n_possible > 0 else 0
        
        C_triu = C[triu_idx]
        W_triu = W[triu_idx]
        S_triu = S[triu_idx]
        
        mean_C = np.mean(C_triu[kept]) if n_kept > 0 else 0
        mean_W = np.mean(W_triu[kept]) if n_kept > 0 else 0
        mean_S = np.mean(S_triu[kept]) if n_kept > 0 else 0
        
        stats.append({
            'bin': b + 1,
            'distance_range': (bin_edges[b], bin_edges[b + 1]),
            'n_possible': n_possible,
            'n_kept': n_kept,
            'retention': retention,
            'mean_consensus': mean_C,
            'mean_weight': mean_W,
            'mean_score': mean_S
        })
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def plot_complete_analysis(C, W, S, G, D, bin_stats, params, n_ad, n_hc, save_path=None):
    """Create comprehensive visualization of distance-dependent analysis."""
    
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    K = len(bin_stats)
    n_channels = C.shape[0]
    n_possible = n_channels * (n_channels - 1) // 2
    n_kept = np.sum(G > 0) // 2
    
    # ═══════ Row 1: Matrices ═══════
    matrices = [
        (C, 'Consensus C\nC_ij = (1/N)Σ B^(s)_ij', 'YlOrRd', 0, 1),
        (W, 'Weight W\nFisher-z averaged', 'viridis', 0, None),
        (S, f'Score S\nS_ij = C_ij + {params.epsilon}·W_ij', 'plasma', 0, None),
        (G, f'Final Graph G\nG_ij = W_ij·𝟙[C_ij>{params.threshold}]', 'hot', 0, None)
    ]
    
    for idx, (M, title, cmap, vmin, vmax) in enumerate(matrices):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # ═══════ Row 2: Distance Analysis ═══════
    
    # Distance matrix
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(D, cmap='Blues')
    ax5.set_title('Distance Matrix D\nD_ij = ||p_i - p_j||₂', fontsize=10, fontweight='bold')
    plt.colorbar(im5, ax=ax5, fraction=0.046, label='Distance')
    
    # Edges per bin
    ax6 = fig.add_subplot(gs[1, 1])
    bins = [s['bin'] for s in bin_stats]
    n_possible_per_bin = [s['n_possible'] for s in bin_stats]
    n_kept_per_bin = [s['n_kept'] for s in bin_stats]
    
    x = np.arange(K)
    width = 0.35
    ax6.bar(x - width/2, n_possible_per_bin, width, label='Possible', color='lightgray')
    ax6.bar(x + width/2, n_kept_per_bin, width, label='Kept (C>0.5)', color='green')
    ax6.set_xlabel('Distance Bin')
    ax6.set_ylabel('Number of Edges')
    ax6.set_title('Edges per Distance Bin', fontsize=10, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(bins)
    ax6.legend()
    
    # Retention rate per bin
    ax7 = fig.add_subplot(gs[1, 2])
    retention = [s['retention'] * 100 for s in bin_stats]
    colors = plt.cm.RdYlGn(np.array(retention) / max(retention))
    ax7.bar(bins, retention, color=colors, edgecolor='black')
    ax7.axhline(y=50, color='red', linestyle='--', label='50%')
    ax7.set_xlabel('Distance Bin')
    ax7.set_ylabel('Retention Rate (%)')
    ax7.set_title('Retention_b = n_kept / n_possible', fontsize=10, fontweight='bold')
    ax7.legend()
    
    # Mean metrics per bin
    ax8 = fig.add_subplot(gs[1, 3])
    mean_C = [s['mean_consensus'] for s in bin_stats]
    mean_S = [s['mean_score'] for s in bin_stats]
    ax8.plot(bins, mean_C, 'ro-', linewidth=2, markersize=8, label='Mean C')
    ax8.plot(bins, mean_S, 'bs--', linewidth=2, markersize=8, label='Mean S')
    ax8.set_xlabel('Distance Bin')
    ax8.set_ylabel('Value')
    ax8.set_title('Quality Metrics by Distance', fontsize=10, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ═══════ Row 3: Consensus Analysis ═══════
    
    # Consensus distribution
    ax9 = fig.add_subplot(gs[2, 0:2])
    triu_idx = np.triu_indices(n_channels, k=1)
    all_C = C[triu_idx]
    kept_C = all_C[all_C > params.threshold]
    
    ax9.hist(all_C, bins=50, alpha=0.5, color='gray', label='All edges', density=True)
    ax9.hist(kept_C, bins=50, alpha=0.7, color='green', label=f'Kept (C>{params.threshold})', density=True)
    ax9.axvline(x=params.threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold = {params.threshold}')
    ax9.set_xlabel('Consensus C_ij')
    ax9.set_ylabel('Density')
    ax9.set_title('Consensus Distribution with Majority Threshold', fontsize=10, fontweight='bold')
    ax9.legend()
    
    # Score vs Distance scatter
    ax10 = fig.add_subplot(gs[2, 2:4])
    D_triu = D[triu_idx]
    S_triu = S[triu_idx]
    kept_mask = G[triu_idx] > 0
    
    ax10.scatter(D_triu[~kept_mask], S_triu[~kept_mask], alpha=0.3, s=5, 
                color='gray', label='Excluded')
    ax10.scatter(D_triu[kept_mask], S_triu[kept_mask], alpha=0.5, s=10,
                color='green', label='Kept')
    ax10.set_xlabel('Distance D_ij')
    ax10.set_ylabel('Score S_ij')
    ax10.set_title('Score vs Distance (showing kept edges)', fontsize=10, fontweight='bold')
    ax10.legend()
    
    # ═══════ Row 4: Summary ═══════
    ax11 = fig.add_subplot(gs[3, :])
    ax11.axis('off')
    
    sparsity = n_kept / n_possible * 100
    mean_consensus = np.mean(C[G > 0]) if np.any(G > 0) else 0
    mean_weight = np.mean(W[G > 0]) if np.any(G > 0) else 0
    mean_score = np.mean(S[G > 0]) if np.any(G > 0) else 0
    
    summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              DISTANCE-DEPENDENT CONSENSUS MATRIX: MATHEMATICAL SUMMARY                                        ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                               ║
║  SAMPLE                                        PARAMETERS                                                                     ║
║  ────────────────────────────────              ────────────────────────────────────────────────────                           ║
║  AD subjects:        {n_ad:3d}                         κ (kappa):     {params.kappa:.2f}  (proportional threshold)                          ║
║  HC subjects:        {n_hc:3d}                         ε (epsilon):   {params.epsilon:.2f}  (score weight factor)                             ║
║  Total N:            {n_ad+n_hc:3d}                         K (bins):      {params.K:3d}  (distance bins)                                     ║
║  Channels C:         {n_channels:3d}                         threshold:     {params.threshold:.2f}  (majority consensus)                          ║
║                                                                                                                               ║
║  ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════    ║
║                                           MATHEMATICAL FRAMEWORK                                                              ║
║  ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════    ║
║                                                                                                                               ║
║  STEP 1: A^(s)_ij = |corr(x_i, x_j)|                    Individual correlation matrices                                       ║
║  STEP 2: B^(s)_ij = 𝟙[A^(s)_ij > τ^(s)]                 Binarize (top κ = {params.kappa*100:.0f}% edges)                                       ║
║  STEP 3: C_ij = (1/N) Σ_s B^(s)_ij                      Consensus matrix                                                      ║
║  STEP 4: W_ij = |tanh(mean(arctanh(A^(s)_ij)))|         Fisher-z averaged weights                                             ║
║  STEP 5: D_ij = ||p_i - p_j||₂                          Euclidean distance matrix                                             ║
║  STEP 6: S_ij = C_ij + ε·W_ij                           Edge scoring function                                                 ║
║  STEP 7: G_ij = W_ij · 𝟙[C_ij > 0.50]                   Majority consensus selection                                          ║
║                                                                                                                               ║
║  ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════    ║
║                                               RESULTS                                                                         ║
║  ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════    ║
║                                                                                                                               ║
║  Possible edges:     {n_possible:5d}                       Mean consensus (kept):   {mean_consensus:.4f}                                        ║
║  Kept edges:         {n_kept:5d}                       Mean weight (kept):      {mean_weight:.4f}                                        ║
║  Final sparsity:     {sparsity:5.2f}%                      Mean score (kept):       {mean_score:.4f}                                        ║
║                                                                                                                               ║
║  DISTANCE DISTRIBUTION (edges kept per bin):                                                                                  ║
║  Bin 1-3 (short):  {sum(s['n_kept'] for s in bin_stats[:3]):4d}    Bin 4-7 (medium): {sum(s['n_kept'] for s in bin_stats[3:7]):4d}    Bin 8-10 (long): {sum(s['n_kept'] for s in bin_stats[7:]):4d}                      ║
║                                                                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    ax11.text(0.5, 0.5, summary, transform=ax11.transAxes, fontsize=9,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.suptitle('DISTANCE-DEPENDENT CONSENSUS MATRIX ANALYSIS', 
                fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def generate_mathematical_report(C, W, S, G, D, bin_stats, params, n_ad, n_hc, save_path=None):
    """Generate detailed mathematical report."""
    
    n_channels = C.shape[0]
    n_possible = n_channels * (n_channels - 1) // 2
    n_kept = np.sum(G > 0) // 2
    sparsity = n_kept / n_possible
    
    triu_idx = np.triu_indices(n_channels, k=1)
    kept_mask = G[triu_idx] > 0
    
    mean_C = np.mean(C[triu_idx][kept_mask])
    mean_W = np.mean(W[triu_idx][kept_mask])
    mean_S = np.mean(S[triu_idx][kept_mask])
    
    bin_table = "\n".join([
        f"  Bin {s['bin']:2d}: D ∈ [{s['distance_range'][0]:.3f}, {s['distance_range'][1]:.3f}) | "
        f"Kept: {s['n_kept']:3d}/{s['n_possible']:3d} ({s['retention']*100:5.1f}%) | "
        f"C̄={s['mean_consensus']:.3f} | S̄={s['mean_score']:.3f}"
        for s in bin_stats
    ])
    
    report = f"""
================================================================================
          DISTANCE-DEPENDENT CONSENSUS MATRIX: MATHEMATICAL REPORT
================================================================================

1. SAMPLE
---------
  AD subjects (N_AD):    {n_ad}
  HC subjects (N_HC):    {n_hc}
  Total subjects (N):    {n_ad + n_hc}
  Channels (C):          {n_channels}

2. PARAMETERS
-------------
  κ (kappa):             {params.kappa}     (proportional threshold)
  ε (epsilon):           {params.epsilon}     (score weight factor)
  K (bins):              {params.K}      (distance bins)
  threshold:             {params.threshold}     (majority consensus)

================================================================================
                        MATHEMATICAL FRAMEWORK
================================================================================

STEP 1: CORRELATION MATRIX
--------------------------
  For each subject s ∈ {{1, ..., N}}:
  
    A^(s)_ij = |r_ij|  where  r_ij = corr(x_i^(s), x_j^(s))
    A^(s)_ii = 0  (no self-loops)


STEP 2: PROPORTIONAL THRESHOLDING
---------------------------------
  Binarize with threshold τ^(s) that keeps top κ = {params.kappa*100:.0f}% edges:
  
    B^(s)_ij = 𝟙[A^(s)_ij > τ^(s)]
    
  where τ^(s) = Q_{{1-κ}}({{A^(s)_ij : i < j}})


STEP 3: CONSENSUS MATRIX
------------------------
  Average binary matrices across all subjects:
  
    C_ij = (1/N) Σ_{{s=1}}^{{N}} B^(s)_ij
    
  Interpretation: C_ij = fraction of subjects with edge (i,j)


STEP 4: WEIGHT MATRIX (Fisher-z Transformation)
-----------------------------------------------
  For subjects S_ij = {{s : B^(s)_ij = 1}} who have edge (i,j):
  
    W_ij = |tanh( (1/|S_ij|) Σ_{{s ∈ S_ij}} arctanh(A^(s)_ij) )|
    
  Fisher-z ensures proper averaging of correlation coefficients.


STEP 5: DISTANCE MATRIX
-----------------------
  Euclidean distance between channel positions:
  
    D_ij = ||p_i - p_j||₂ = √[(p_ix - p_jx)² + (p_iy - p_jy)² + (p_iz - p_jz)²]


STEP 6: DISTANCE BINNING
------------------------
  Divide edges into K = {params.K} bins by distance percentiles:
  
    d_k = Q_{{k/K}}({{D_ij : i < j}})  for k = 0, 1, ..., K
    
  Edge (i,j) ∈ Bin b if d_{{b-1}} ≤ D_ij < d_b


STEP 7: SCORING FUNCTION
------------------------
  Combine consensus and weight:
  
    S_ij = C_ij + ε · W_ij
    
  where ε = {params.epsilon} is a small tie-breaker weight.


STEP 8: MAJORITY CONSENSUS SELECTION
------------------------------------
  Final graph using majority rule:
  
    G_ij = W_ij · 𝟙[C_ij > {params.threshold}]
    
  Interpretation: Keep edge only if present in >{params.threshold*100:.0f}% of subjects.


================================================================================
                              RESULTS
================================================================================

FINAL GRAPH STATISTICS
----------------------
  Possible edges:        {n_possible}
  Kept edges:            {n_kept}
  Final sparsity:        {sparsity*100:.2f}%
  
  Mean consensus (kept): {mean_C:.4f}
  Mean weight (kept):    {mean_W:.4f}
  Mean score (kept):     {mean_S:.4f}

DISTANCE DISTRIBUTION
---------------------
{bin_table}

INTERPRETATION
--------------
  • Short-range (Bins 1-3): {sum(s['n_kept'] for s in bin_stats[:3])} edges kept
    Higher retention due to volume conduction + local connectivity
    
  • Medium-range (Bins 4-7): {sum(s['n_kept'] for s in bin_stats[3:7])} edges kept
    Mix of local and distributed functional networks
    
  • Long-range (Bins 8-10): {sum(s['n_kept'] for s in bin_stats[7:])} edges kept
    True long-range functional connections (lower but present)

================================================================================
                      KEY EQUATIONS FOR THESIS
================================================================================

CONSENSUS MATRIX:
  C_ij = (1/N) Σ_s B^(s)_ij

WEIGHT MATRIX:
  W_ij = |tanh( (1/|S_ij|) Σ_{{s ∈ S_ij}} arctanh(A^(s)_ij) )|

DISTANCE:
  D_ij = ||p_i - p_j||₂

SCORING:
  S_ij = C_ij + ε · W_ij

FINAL GRAPH (MAJORITY CONSENSUS):
  G_ij = W_ij · 𝟙[C_ij > 0.50]

================================================================================
                           END OF REPORT
================================================================================
"""
    
    if save_path:
        Path(save_path).write_text(report)
        print(f"✓ Report saved: {save_path}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════════════════════

def generate_synthetic_eeg(n_subjects, n_channels, n_samples, group='HC', seed=None):
    """Generate synthetic EEG data with spatial correlation structure."""
    if seed:
        np.random.seed(seed)
    
    # Channel locations on unit sphere (simulating EEG cap)
    theta = np.linspace(0, 2*np.pi, int(np.ceil(np.sqrt(n_channels)*1.5)), endpoint=False)
    phi = np.linspace(0.2*np.pi, 0.8*np.pi, int(np.ceil(np.sqrt(n_channels))))
    locs = [[np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)] 
            for t in theta for p in phi][:n_channels]
    channel_locations = np.array(locs)
    
    # Spatial covariance (closer channels = higher correlation)
    D = squareform(pdist(channel_locations))
    spatial_cov = np.exp(-D**2 / (2*0.5**2))
    
    # AD has reduced connectivity
    if group == 'AD':
        spatial_cov = spatial_cov * 0.8 + 0.2 * np.eye(n_channels)
    
    # Generate data
    L = np.linalg.cholesky(spatial_cov + 0.01*np.eye(n_channels))
    data = np.array([L @ np.random.randn(n_channels, n_samples) for _ in range(n_subjects)])
    
    return data, channel_locations


# ═══════════════════════════════════════════════════════════════════════════════════════
#                                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_distance_dependent_analysis(n_ad=35, n_hc=31, n_channels=64, n_samples=1000):
    """Run complete distance-dependent consensus analysis."""
    
    print("="*70)
    print("     DISTANCE-DEPENDENT CONSENSUS MATRIX CONSTRUCTION")
    print("     Following Betzel et al. (2019) Mathematical Framework")
    print("="*70)
    
    params = Parameters()
    
    # Step 0: Generate data
    print(f"\n[0/8] Generating synthetic EEG data...")
    ad_data, channel_locations = generate_synthetic_eeg(n_ad, n_channels, n_samples, 'AD', 42)
    hc_data, _ = generate_synthetic_eeg(n_hc, n_channels, n_samples, 'HC', 123)
    print(f"      AD: {n_ad} subjects, HC: {n_hc} subjects, {n_channels} channels")
    
    # Step 1: Correlation matrices
    print(f"\n[1/8] Computing A^(s)_ij = |corr(x_i, x_j)|...")
    ad_adj = np.array([compute_adjacency_matrix(ad_data[s]) for s in range(n_ad)])
    hc_adj = np.array([compute_adjacency_matrix(hc_data[s]) for s in range(n_hc)])
    all_adj = np.concatenate([ad_adj, hc_adj], axis=0)
    
    # Step 2: Binarization
    print(f"[2/8] Computing B^(s)_ij = 𝟙[A^(s)_ij > τ] with κ={params.kappa}...")
    ad_bin = np.array([proportional_threshold(ad_adj[s], params.kappa) for s in range(n_ad)])
    hc_bin = np.array([proportional_threshold(hc_adj[s], params.kappa) for s in range(n_hc)])
    all_bin = np.concatenate([ad_bin, hc_bin], axis=0)
    
    # Step 3: Consensus matrix
    print(f"[3/8] Computing C_ij = (1/N) Σ_s B^(s)_ij...")
    C = compute_consensus_matrix(all_bin)
    
    # Step 4: Weight matrix
    print(f"[4/8] Computing W_ij with Fisher-z transformation...")
    W = compute_weight_matrix(all_adj, all_bin)
    
    # Step 5: Distance matrix
    print(f"[5/8] Computing D_ij = ||p_i - p_j||₂...")
    D = compute_distance_matrix(channel_locations)
    
    # Step 6: Distance binning
    print(f"[6/8] Creating K={params.K} distance bins...")
    bin_edges, bin_assignments = create_distance_bins(D, params.K)
    
    # Step 7: Scoring
    print(f"[7/8] Computing S_ij = C_ij + {params.epsilon}·W_ij...")
    S = compute_score_matrix(C, W, params.epsilon)
    
    # Step 8: Selection
    print(f"[8/8] Applying majority consensus: G_ij = W_ij · 𝟙[C_ij > {params.threshold}]...")
    G = majority_consensus_selection(C, W, params.threshold)
    
    # Analyze distance distribution
    bin_stats = analyze_distance_distribution(G, C, W, S, bin_assignments, bin_edges, params.K)
    
    # Results
    n_possible = n_channels * (n_channels - 1) // 2
    n_kept = np.sum(G > 0) // 2
    
    print("\n" + "="*70)
    print("                         RESULTS")
    print("="*70)
    print(f"\n  Final sparsity: {n_kept}/{n_possible} edges ({n_kept/n_possible*100:.2f}%)")
    print(f"\n  Distance distribution (edges kept):")
    for s in bin_stats:
        print(f"    Bin {s['bin']:2d}: {s['n_kept']:3d}/{s['n_possible']:3d} "
              f"({s['retention']*100:5.1f}%) | C̄={s['mean_consensus']:.3f}")
    
    # Save outputs
    print("\n" + "="*70)
    print("Saving outputs...")
    
    plot_complete_analysis(C, W, S, G, D, bin_stats, params, n_ad, n_hc,
                          save_path=str(OUTPUT_DIR / "distance_dependent_analysis.png"))
    
    generate_mathematical_report(C, W, S, G, D, bin_stats, params, n_ad, n_hc,
                                save_path=str(OUTPUT_DIR / "mathematical_report.txt"))
    
    # Save matrices
    np.save(OUTPUT_DIR / "consensus_C.npy", C)
    np.save(OUTPUT_DIR / "weight_W.npy", W)
    np.save(OUTPUT_DIR / "score_S.npy", S)
    np.save(OUTPUT_DIR / "final_graph_G.npy", G)
    np.save(OUTPUT_DIR / "distance_D.npy", D)
    
    print(f"\n{'='*70}")
    print("                      COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    
    return G, C, W, S, D, bin_stats


if __name__ == "__main__":
    G, C, W, S, D, bin_stats = run_distance_dependent_analysis()
