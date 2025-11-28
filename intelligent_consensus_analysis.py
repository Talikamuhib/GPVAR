#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
            INTELLIGENT CONSENSUS MATRIX ANALYSIS
            Natural Sparsity with Comprehensive Scoring Analysis
═══════════════════════════════════════════════════════════════════════════════════════

This script implements an INTELLIGENT analysis approach:

1. NATURAL SPARSITY for edge selection (keep all C > 0)
2. SCORING for edge quality assessment
3. DISTANCE-DEPENDENT analysis for spatial understanding
4. MULTI-LEVEL categorization of edges

The analysis provides:
- Edge reliability scoring
- Distance distribution analysis
- Consensus level categorization
- Network quality metrics
- Comprehensive visualizations

Author: Intelligent Consensus Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass


OUTPUT_DIR = Path("./intelligent_analysis_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class EdgeInfo:
    """Information about a single edge."""
    i: int                    # Source channel
    j: int                    # Target channel
    consensus: float          # C[i,j] - fraction of subjects
    weight: float             # W[i,j] - Fisher-z averaged correlation
    score: float              # C + ε×W - combined reliability score
    distance: float           # Euclidean distance between channels
    distance_bin: int         # Which distance bin (0-9)
    consensus_level: str      # 'unanimous', 'strong', 'moderate', 'weak', 'minimal'


@dataclass  
class AnalysisResults:
    """Complete analysis results."""
    # Matrices
    G_final: np.ndarray           # Final graph with natural sparsity
    C: np.ndarray                 # Consensus matrix
    W: np.ndarray                 # Weight matrix
    Score: np.ndarray             # Score matrix
    
    # Edge information
    edges: List[EdgeInfo]         # All kept edges with full info
    
    # Statistics
    n_possible: int
    n_kept: int
    natural_sparsity: float
    
    # By consensus level
    n_unanimous: int              # C >= 0.99
    n_strong: int                 # 0.75 <= C < 0.99
    n_moderate: int               # 0.50 <= C < 0.75
    n_weak: int                   # 0.25 <= C < 0.50
    n_minimal: int                # 0 < C < 0.25
    
    # By distance bin
    per_bin_stats: List[Dict]
    
    # Quality metrics
    mean_consensus: float
    mean_weight: float
    mean_score: float
    network_reliability: float    # Weighted average of consensus


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
    """Proportional thresholding to binary matrix."""
    n = A.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    weights = A[triu_idx]
    k = max(1, int(sparsity * len(weights)))
    threshold = np.sort(weights)[-k] if k < len(weights) else 0
    B = (A > threshold).astype(float)
    B = np.maximum(B, B.T)
    np.fill_diagonal(B, 0)
    return B


def compute_consensus(binary_matrices: np.ndarray) -> np.ndarray:
    """Compute consensus matrix C = mean of binary matrices."""
    return np.mean(binary_matrices, axis=0)


def compute_weights_fisher_z(adjacency_matrices: np.ndarray, 
                              binary_matrices: np.ndarray) -> np.ndarray:
    """Compute weight matrix using Fisher-z averaging."""
    n_subjects, n_channels, _ = adjacency_matrices.shape
    W = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            mask = binary_matrices[:, i, j] > 0
            if np.any(mask):
                r = adjacency_matrices[mask, i, j]
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                W[i, j] = np.abs(np.tanh(np.mean(z)))
                W[j, i] = W[i, j]
    return W


def get_consensus_level(c: float) -> str:
    """Categorize consensus value."""
    if c >= 0.99:
        return 'unanimous'
    elif c >= 0.75:
        return 'strong'
    elif c >= 0.50:
        return 'moderate'
    elif c >= 0.25:
        return 'weak'
    else:
        return 'minimal'


# ═══════════════════════════════════════════════════════════════════════════════════════
#                     INTELLIGENT ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def intelligent_analysis(C: np.ndarray, 
                         W: np.ndarray,
                         channel_locations: np.ndarray,
                         n_bins: int = 10,
                         epsilon: float = 0.1) -> AnalysisResults:
    """
    Perform intelligent analysis with natural sparsity.
    
    SELECTION: Keep ALL edges where C > 0 (natural sparsity)
    ANALYSIS: Score each edge, categorize, and analyze distribution
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix
    W : np.ndarray
        Weight matrix
    channel_locations : np.ndarray
        3D channel coordinates
    n_bins : int
        Number of distance bins
    epsilon : float
        Weight for W in score calculation
    
    Returns
    -------
    AnalysisResults
        Comprehensive analysis results
    """
    n_channels = C.shape[0]
    triu_idx = np.triu_indices(n_channels, k=1)
    n_possible = len(triu_idx[0])
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Compute Score Matrix
    # ═══════════════════════════════════════════════════════════════
    Score = C + epsilon * W
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Compute Distance Matrix and Bins
    # ═══════════════════════════════════════════════════════════════
    D = squareform(pdist(channel_locations))
    distances = D[triu_idx]
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10
    
    def get_bin(d):
        for b in range(n_bins):
            if bin_edges[b] <= d < bin_edges[b + 1]:
                return b
        return n_bins - 1
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Natural Sparsity Selection + Edge Analysis
    # ═══════════════════════════════════════════════════════════════
    edges = []
    
    for idx in range(n_possible):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        c = C[i, j]
        
        if c > 0:  # NATURAL SPARSITY: keep all C > 0
            edge = EdgeInfo(
                i=i, j=j,
                consensus=c,
                weight=W[i, j],
                score=Score[i, j],
                distance=distances[idx],
                distance_bin=get_bin(distances[idx]),
                consensus_level=get_consensus_level(c)
            )
            edges.append(edge)
    
    n_kept = len(edges)
    natural_sparsity = n_kept / n_possible
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Build Final Graph (using Weight as edge value)
    # ═══════════════════════════════════════════════════════════════
    G = np.zeros((n_channels, n_channels))
    for e in edges:
        G[e.i, e.j] = e.weight
        G[e.j, e.i] = e.weight
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Categorize by Consensus Level
    # ═══════════════════════════════════════════════════════════════
    n_unanimous = sum(1 for e in edges if e.consensus_level == 'unanimous')
    n_strong = sum(1 for e in edges if e.consensus_level == 'strong')
    n_moderate = sum(1 for e in edges if e.consensus_level == 'moderate')
    n_weak = sum(1 for e in edges if e.consensus_level == 'weak')
    n_minimal = sum(1 for e in edges if e.consensus_level == 'minimal')
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Per-Bin Statistics
    # ═══════════════════════════════════════════════════════════════
    per_bin_stats = []
    for b in range(n_bins):
        bin_edges_list = [e for e in edges if e.distance_bin == b]
        n_in_bin = sum(1 for d in distances if bin_edges[b] <= d < bin_edges[b+1])
        
        if len(bin_edges_list) > 0:
            stats = {
                'bin': b,
                'distance_range': (bin_edges[b], bin_edges[b+1]),
                'n_possible': n_in_bin,
                'n_kept': len(bin_edges_list),
                'kept_fraction': len(bin_edges_list) / n_in_bin if n_in_bin > 0 else 0,
                'pct_of_total': len(bin_edges_list) / n_kept * 100,
                'mean_consensus': np.mean([e.consensus for e in bin_edges_list]),
                'mean_weight': np.mean([e.weight for e in bin_edges_list]),
                'mean_score': np.mean([e.score for e in bin_edges_list]),
                'n_unanimous': sum(1 for e in bin_edges_list if e.consensus_level == 'unanimous'),
                'n_strong': sum(1 for e in bin_edges_list if e.consensus_level == 'strong'),
                'n_moderate': sum(1 for e in bin_edges_list if e.consensus_level == 'moderate'),
            }
        else:
            stats = {
                'bin': b, 'distance_range': (bin_edges[b], bin_edges[b+1]),
                'n_possible': n_in_bin, 'n_kept': 0, 'kept_fraction': 0,
                'pct_of_total': 0, 'mean_consensus': 0, 'mean_weight': 0, 'mean_score': 0,
                'n_unanimous': 0, 'n_strong': 0, 'n_moderate': 0
            }
        per_bin_stats.append(stats)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 7: Quality Metrics
    # ═══════════════════════════════════════════════════════════════
    mean_consensus = np.mean([e.consensus for e in edges])
    mean_weight = np.mean([e.weight for e in edges])
    mean_score = np.mean([e.score for e in edges])
    
    # Network reliability: weighted by edge weight
    total_weight = sum(e.weight for e in edges)
    network_reliability = sum(e.consensus * e.weight for e in edges) / total_weight if total_weight > 0 else 0
    
    return AnalysisResults(
        G_final=G, C=C, W=W, Score=Score,
        edges=edges,
        n_possible=n_possible, n_kept=n_kept, natural_sparsity=natural_sparsity,
        n_unanimous=n_unanimous, n_strong=n_strong, n_moderate=n_moderate,
        n_weak=n_weak, n_minimal=n_minimal,
        per_bin_stats=per_bin_stats,
        mean_consensus=mean_consensus, mean_weight=mean_weight, mean_score=mean_score,
        network_reliability=network_reliability
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def plot_intelligent_analysis(results: AnalysisResults, 
                               n_ad: int, n_hc: int,
                               save_path: str = None):
    """Create comprehensive intelligent analysis visualization."""
    
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    edges = results.edges
    n_bins = len(results.per_bin_stats)
    
    # ═══════ ROW 1: Matrices ═══════
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(results.C, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_title('Consensus Matrix C\n(Fraction of subjects)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(results.W, cmap='viridis', vmin=0)
    ax2.set_title('Weight Matrix W\n(Fisher-z averaged)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(results.Score, cmap='plasma', vmin=0)
    ax3.set_title('Score Matrix\n(C + 0.1×W)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(results.G_final, cmap='hot', vmin=0)
    ax4.set_title(f'Final Graph G\n(Natural Sparsity: {results.natural_sparsity*100:.1f}%)', 
                 fontsize=11, fontweight='bold', color='green')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # ═══════ ROW 2: Consensus Level Analysis ═══════
    ax5 = fig.add_subplot(gs[1, 0])
    levels = ['Unanimous\n(≥99%)', 'Strong\n(75-99%)', 'Moderate\n(50-75%)', 
              'Weak\n(25-50%)', 'Minimal\n(<25%)']
    counts = [results.n_unanimous, results.n_strong, results.n_moderate, 
              results.n_weak, results.n_minimal]
    colors = ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax5.bar(levels, counts, color=colors, edgecolor='black')
    ax5.set_ylabel('Number of Edges')
    ax5.set_title('EDGE QUALITY BY CONSENSUS LEVEL', fontsize=11, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}\n({count/results.n_kept*100:.1f}%)', ha='center', fontsize=9)
    
    # Pie chart of consensus levels
    ax6 = fig.add_subplot(gs[1, 1])
    sizes = [results.n_unanimous, results.n_strong, results.n_moderate, 
             results.n_weak, results.n_minimal]
    labels = ['Unanimous', 'Strong', 'Moderate', 'Weak', 'Minimal']
    explode = (0.05, 0, 0, 0, 0)
    ax6.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    ax6.set_title('Consensus Distribution', fontsize=11, fontweight='bold')
    
    # Score distribution
    ax7 = fig.add_subplot(gs[1, 2])
    scores = [e.score for e in edges]
    ax7.hist(scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(results.mean_score, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {results.mean_score:.3f}')
    ax7.set_xlabel('Score (C + 0.1×W)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('SCORE DISTRIBUTION', fontsize=11, fontweight='bold')
    ax7.legend()
    
    # Score vs Consensus scatter
    ax8 = fig.add_subplot(gs[1, 3])
    consensus_vals = [e.consensus for e in edges]
    weight_vals = [e.weight for e in edges]
    scatter = ax8.scatter(consensus_vals, weight_vals, c=scores, cmap='plasma', 
                         alpha=0.6, s=10)
    ax8.set_xlabel('Consensus C')
    ax8.set_ylabel('Weight W')
    ax8.set_title('C vs W (colored by Score)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax8, label='Score')
    
    # ═══════ ROW 3: Distance Analysis ═══════
    ax9 = fig.add_subplot(gs[2, 0:2])
    bin_indices = list(range(n_bins))
    n_kept_per_bin = [s['n_kept'] for s in results.per_bin_stats]
    n_possible_per_bin = [s['n_possible'] for s in results.per_bin_stats]
    
    x = np.arange(n_bins)
    width = 0.35
    ax9.bar(x - width/2, n_possible_per_bin, width, label='Possible', color='lightgray')
    ax9.bar(x + width/2, n_kept_per_bin, width, label='Kept (C>0)', color='green')
    ax9.set_xlabel('Distance Bin (1=short, 10=long)')
    ax9.set_ylabel('Number of Edges')
    ax9.set_title('EDGES PER DISTANCE BIN', fontsize=11, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([str(i+1) for i in range(n_bins)])
    ax9.legend()
    
    # Mean score per bin
    ax10 = fig.add_subplot(gs[2, 2:4])
    mean_scores = [s['mean_score'] for s in results.per_bin_stats]
    mean_consensus = [s['mean_consensus'] for s in results.per_bin_stats]
    
    ax10.plot(range(n_bins), mean_scores, 'o-', color='purple', linewidth=2, 
             markersize=8, label='Mean Score')
    ax10.plot(range(n_bins), mean_consensus, 's--', color='red', linewidth=2, 
             markersize=8, label='Mean Consensus')
    ax10.set_xlabel('Distance Bin (1=short, 10=long)')
    ax10.set_ylabel('Value')
    ax10.set_title('MEAN SCORE & CONSENSUS BY DISTANCE', fontsize=11, fontweight='bold')
    ax10.set_xticks(range(n_bins))
    ax10.set_xticklabels([str(i+1) for i in range(n_bins)])
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # ═══════ ROW 4: Quality Analysis ═══════
    # Consensus level per distance bin (stacked)
    ax11 = fig.add_subplot(gs[3, 0:2])
    unanimous_per_bin = [s['n_unanimous'] for s in results.per_bin_stats]
    strong_per_bin = [s['n_strong'] for s in results.per_bin_stats]
    moderate_per_bin = [s['n_moderate'] for s in results.per_bin_stats]
    other_per_bin = [s['n_kept'] - s['n_unanimous'] - s['n_strong'] - s['n_moderate'] 
                    for s in results.per_bin_stats]
    
    ax11.bar(range(n_bins), unanimous_per_bin, label='Unanimous', color='#2ecc71')
    ax11.bar(range(n_bins), strong_per_bin, bottom=unanimous_per_bin, 
            label='Strong', color='#27ae60')
    ax11.bar(range(n_bins), moderate_per_bin, 
            bottom=[u+s for u,s in zip(unanimous_per_bin, strong_per_bin)],
            label='Moderate', color='#f1c40f')
    ax11.bar(range(n_bins), other_per_bin,
            bottom=[u+s+m for u,s,m in zip(unanimous_per_bin, strong_per_bin, moderate_per_bin)],
            label='Weak/Minimal', color='#e74c3c')
    ax11.set_xlabel('Distance Bin')
    ax11.set_ylabel('Number of Edges')
    ax11.set_title('CONSENSUS QUALITY BY DISTANCE', fontsize=11, fontweight='bold')
    ax11.legend(loc='upper right')
    ax11.set_xticks(range(n_bins))
    
    # Summary statistics
    ax12 = fig.add_subplot(gs[3, 2:4])
    ax12.axis('off')
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                 INTELLIGENT ANALYSIS SUMMARY                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  SAMPLE: {n_ad} AD + {n_hc} HC = {n_ad + n_hc} subjects                              ║
║                                                                      ║
║  ════════════════════════════════════════════════════════════════    ║
║  NATURAL SPARSITY RESULTS                                            ║
║  ════════════════════════════════════════════════════════════════    ║
║                                                                      ║
║    Possible edges:     {results.n_possible:,}                                     ║
║    Kept edges (C>0):   {results.n_kept:,}                                     ║
║    Natural sparsity:   {results.natural_sparsity*100:.2f}%                                  ║
║                                                                      ║
║  ════════════════════════════════════════════════════════════════    ║
║  EDGE QUALITY METRICS                                                ║
║  ════════════════════════════════════════════════════════════════    ║
║                                                                      ║
║    Mean Consensus:     {results.mean_consensus:.4f}                                  ║
║    Mean Weight:        {results.mean_weight:.4f}                                  ║
║    Mean Score:         {results.mean_score:.4f}                                  ║
║    Network Reliability:{results.network_reliability:.4f}                                  ║
║                                                                      ║
║  ════════════════════════════════════════════════════════════════    ║
║  CONSENSUS LEVEL BREAKDOWN                                           ║
║  ════════════════════════════════════════════════════════════════    ║
║                                                                      ║
║    Unanimous (≥99%):   {results.n_unanimous:4d} ({results.n_unanimous/results.n_kept*100:5.1f}%)                       ║
║    Strong (75-99%):    {results.n_strong:4d} ({results.n_strong/results.n_kept*100:5.1f}%)                       ║
║    Moderate (50-75%):  {results.n_moderate:4d} ({results.n_moderate/results.n_kept*100:5.1f}%)                       ║
║    Weak (25-50%):      {results.n_weak:4d} ({results.n_weak/results.n_kept*100:5.1f}%)                       ║
║    Minimal (<25%):     {results.n_minimal:4d} ({results.n_minimal/results.n_kept*100:5.1f}%)                       ║
║                                                                      ║
║  ════════════════════════════════════════════════════════════════    ║
║  INTERPRETATION                                                      ║
║  ════════════════════════════════════════════════════════════════    ║
║                                                                      ║
║    • High reliability edges (unanimous+strong): {results.n_unanimous + results.n_strong:4d}              ║
║    • These form the CORE network backbone                            ║
║    • Moderate edges add important variability                        ║
║    • Weak/minimal edges capture individual differences               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    ax12.text(0.02, 0.98, summary, transform=ax12.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('INTELLIGENT CONSENSUS ANALYSIS: Natural Sparsity with Edge Quality Assessment',
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Analysis figure saved to: {save_path}")
    
    return fig


def plot_top_edges_analysis(results: AnalysisResults, top_n: int = 50, save_path: str = None):
    """Analyze and visualize top edges by score."""
    
    # Sort edges by score
    sorted_edges = sorted(results.edges, key=lambda e: e.score, reverse=True)
    top_edges = sorted_edges[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top edges by score
    ax1 = axes[0, 0]
    scores = [e.score for e in top_edges]
    colors = ['#2ecc71' if e.consensus_level in ['unanimous', 'strong'] else '#f1c40f' 
              for e in top_edges]
    ax1.barh(range(top_n), scores, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Score (C + 0.1×W)')
    ax1.set_ylabel(f'Edge Rank (1 = highest)')
    ax1.set_title(f'TOP {top_n} EDGES BY SCORE', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(0, top_n, 5))
    ax1.invert_yaxis()
    
    # Score components for top edges
    ax2 = axes[0, 1]
    consensus_vals = [e.consensus for e in top_edges]
    weight_vals = [e.weight for e in top_edges]
    x = range(top_n)
    ax2.bar(x, consensus_vals, label='Consensus C', color='red', alpha=0.7)
    ax2.bar(x, [w*0.1 for w in weight_vals], bottom=consensus_vals, 
           label='0.1×Weight', color='blue', alpha=0.7)
    ax2.set_xlabel('Edge Rank')
    ax2.set_ylabel('Score Components')
    ax2.set_title('SCORE BREAKDOWN (Top Edges)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # Distance distribution of top edges
    ax3 = axes[1, 0]
    distances = [e.distance for e in top_edges]
    all_distances = [e.distance for e in results.edges]
    ax3.hist(all_distances, bins=30, alpha=0.5, color='gray', label='All edges', density=True)
    ax3.hist(distances, bins=30, alpha=0.7, color='green', label=f'Top {top_n}', density=True)
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Density')
    ax3.set_title('DISTANCE: Top Edges vs All Edges', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # Consensus level of top edges
    ax4 = axes[1, 1]
    levels = ['unanimous', 'strong', 'moderate', 'weak', 'minimal']
    top_counts = [sum(1 for e in top_edges if e.consensus_level == l) for l in levels]
    all_counts = [sum(1 for e in results.edges if e.consensus_level == l) for l in levels]
    
    x = np.arange(len(levels))
    width = 0.35
    ax4.bar(x - width/2, [c/results.n_kept*100 for c in all_counts], width, 
           label='All edges', color='lightgray')
    ax4.bar(x + width/2, [c/top_n*100 for c in top_counts], width, 
           label=f'Top {top_n}', color='green')
    ax4.set_xlabel('Consensus Level')
    ax4.set_ylabel('Percentage')
    ax4.set_title('CONSENSUS LEVEL: Top vs All', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Unan.', 'Strong', 'Mod.', 'Weak', 'Min.'])
    ax4.legend()
    
    plt.suptitle(f'Analysis of Top {top_n} Edges by Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Top edges analysis saved to: {save_path}")
    
    return fig


def generate_intelligent_report(results: AnalysisResults, n_ad: int, n_hc: int,
                                 save_path: str = None) -> str:
    """Generate comprehensive analysis report."""
    
    # Sort edges for top/bottom analysis
    sorted_by_score = sorted(results.edges, key=lambda e: e.score, reverse=True)
    top_10 = sorted_by_score[:10]
    bottom_10 = sorted_by_score[-10:]
    
    top_10_str = "\n".join([
        f"    {i+1}. Channels ({e.i:2d},{e.j:2d}): C={e.consensus:.3f}, W={e.weight:.3f}, "
        f"Score={e.score:.3f}, Dist={e.distance:.3f}, Level={e.consensus_level}"
        for i, e in enumerate(top_10)
    ])
    
    bin_table = "\n".join([
        f"    Bin {s['bin']+1:2d}: {s['n_kept']:3d}/{s['n_possible']:3d} kept "
        f"({s['kept_fraction']*100:5.1f}%) | MeanC={s['mean_consensus']:.3f} | "
        f"MeanScore={s['mean_score']:.3f} | Unan={s['n_unanimous']:2d} Strong={s['n_strong']:2d}"
        for s in results.per_bin_stats
    ])
    
    report = f"""
================================================================================
                INTELLIGENT CONSENSUS ANALYSIS REPORT
                 Natural Sparsity with Edge Quality Assessment
================================================================================

OVERVIEW
--------
This analysis uses NATURAL SPARSITY (keep all edges where C > 0) combined with
intelligent edge quality assessment using scoring (Score = C + 0.1×W).

SAMPLE
------
  • AD patients:        {n_ad}
  • Healthy Controls:   {n_hc}
  • TOTAL subjects:     {n_ad + n_hc}

================================================================================
                         NATURAL SPARSITY RESULTS
================================================================================

  Possible edges:       {results.n_possible:,}
  Kept edges (C>0):     {results.n_kept:,}
  NATURAL SPARSITY:     {results.natural_sparsity*100:.2f}%

================================================================================
                         EDGE QUALITY METRICS
================================================================================

  Mean Consensus (C):   {results.mean_consensus:.4f}
  Mean Weight (W):      {results.mean_weight:.4f}
  Mean Score:           {results.mean_score:.4f}
  Network Reliability:  {results.network_reliability:.4f}

  INTERPRETATION:
    Score = C + 0.1×W combines consistency (C) with strength (W).
    Higher score = more reliable edge (both consistent AND strong).
    Network reliability is the weighted average of consensus.

================================================================================
                       CONSENSUS LEVEL BREAKDOWN
================================================================================

  Level            Count    Percentage    Description
  ─────────────────────────────────────────────────────────────
  Unanimous        {results.n_unanimous:5d}      {results.n_unanimous/results.n_kept*100:5.1f}%        C ≥ 99% (all subjects)
  Strong           {results.n_strong:5d}      {results.n_strong/results.n_kept*100:5.1f}%        75% ≤ C < 99%
  Moderate         {results.n_moderate:5d}      {results.n_moderate/results.n_kept*100:5.1f}%        50% ≤ C < 75%
  Weak             {results.n_weak:5d}      {results.n_weak/results.n_kept*100:5.1f}%        25% ≤ C < 50%
  Minimal          {results.n_minimal:5d}      {results.n_minimal/results.n_kept*100:5.1f}%        0% < C < 25%
  ─────────────────────────────────────────────────────────────
  TOTAL            {results.n_kept:5d}      100.0%

  CORE NETWORK: Unanimous + Strong = {results.n_unanimous + results.n_strong} edges ({(results.n_unanimous + results.n_strong)/results.n_kept*100:.1f}%)
  These edges form the reliable backbone of the network.

================================================================================
                       DISTANCE DISTRIBUTION
================================================================================

{bin_table}

  OBSERVATION:
    Edges are distributed across ALL distance bins.
    Short-range edges (Bin 1-3) tend to have higher consensus.
    Long-range edges (Bin 8-10) show lower but still meaningful consensus.
    This natural distribution preserves both local and global connectivity.

================================================================================
                         TOP 10 EDGES BY SCORE
================================================================================

{top_10_str}

  These are the most RELIABLE edges in the network:
    • High consensus (consistent across subjects)
    • High weight (strong correlation)

================================================================================
                       INTELLIGENT INSIGHTS
================================================================================

1. NETWORK CORE (Unanimous + Strong edges):
   • {results.n_unanimous + results.n_strong} edges form the reliable backbone
   • These should be prioritized in network analysis
   • Represent connections present in ≥75% of subjects

2. NETWORK VARIABILITY (Moderate + Weak edges):
   • {results.n_moderate + results.n_weak} edges show individual differences
   • May capture disease-related variations
   • Important for group comparison (AD vs HC)

3. RARE CONNECTIONS (Minimal edges):
   • {results.n_minimal} edges present in <25% of subjects
   • May be artifacts or rare individual patterns
   • Consider with caution

4. DISTANCE PATTERN:
   • Short-range: Higher consensus (volume conduction + true connectivity)
   • Long-range: Lower but present consensus (true long-range connections)
   • Natural distribution preserved (no artificial bias)

5. RECOMMENDATIONS FOR GP-VAR:
   • Use FULL natural sparsity graph for complete analysis
   • Or use CORE network (unanimous + strong) for robust analysis
   • Edge weights should be W (correlation-based) or Score (reliability-based)

================================================================================
                              END OF REPORT
================================================================================
"""
    
    if save_path:
        Path(save_path).write_text(report)
        print(f"✓ Report saved to: {save_path}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_subjects: int, n_channels: int, n_samples: int,
                            group: str = 'HC', seed: int = None):
    """Generate synthetic EEG data."""
    if seed:
        np.random.seed(seed)
    
    # Channel locations
    theta = np.linspace(0, 2*np.pi, int(np.ceil(np.sqrt(n_channels)*1.5)), endpoint=False)
    phi = np.linspace(0.2*np.pi, 0.8*np.pi, int(np.ceil(np.sqrt(n_channels))))
    locs = [[np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)] 
            for t in theta for p in phi][:n_channels]
    channel_locations = np.array(locs)
    
    # Spatial covariance
    D = squareform(pdist(channel_locations))
    spatial_cov = np.exp(-D**2 / (2*0.5**2))
    if group == 'AD':
        spatial_cov = spatial_cov * 0.8 + 0.2 * np.eye(n_channels)
    
    L = np.linalg.cholesky(spatial_cov + 0.01*np.eye(n_channels))
    data = np.array([L @ np.random.randn(n_channels, n_samples) for _ in range(n_subjects)])
    
    return data, channel_locations


# ═══════════════════════════════════════════════════════════════════════════════════════
#                                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_intelligent_analysis(n_ad: int = 35, n_hc: int = 31, n_channels: int = 64,
                              n_samples: int = 1000, sparsity_binarize: float = 0.15):
    """Run complete intelligent analysis."""
    
    print("="*70)
    print("         INTELLIGENT CONSENSUS ANALYSIS")
    print("         Natural Sparsity + Edge Quality Assessment")
    print("="*70)
    
    # Generate data
    print("\n[1/5] Generating data...")
    ad_data, channel_locations = generate_synthetic_data(n_ad, n_channels, n_samples, 'AD', 42)
    hc_data, _ = generate_synthetic_data(n_hc, n_channels, n_samples, 'HC', 123)
    
    # Compute matrices
    print("[2/5] Computing correlation matrices...")
    ad_adj = np.array([compute_correlation_matrix(ad_data[s]) for s in range(n_ad)])
    hc_adj = np.array([compute_correlation_matrix(hc_data[s]) for s in range(n_hc)])
    
    print(f"[3/5] Computing binary matrices (κ={sparsity_binarize})...")
    ad_bin = np.array([proportional_threshold(ad_adj[s], sparsity_binarize) for s in range(n_ad)])
    hc_bin = np.array([proportional_threshold(hc_adj[s], sparsity_binarize) for s in range(n_hc)])
    
    # Combined
    all_adj = np.concatenate([ad_adj, hc_adj], axis=0)
    all_bin = np.concatenate([ad_bin, hc_bin], axis=0)
    
    print("[4/5] Computing consensus and weights...")
    C = compute_consensus(all_bin)
    W = compute_weights_fisher_z(all_adj, all_bin)
    
    # Intelligent analysis
    print("[5/5] Running intelligent analysis...")
    results = intelligent_analysis(C, W, channel_locations)
    
    # Print summary
    print("\n" + "="*70)
    print("                    ANALYSIS RESULTS")
    print("="*70)
    print(f"\n  NATURAL SPARSITY: {results.n_kept}/{results.n_possible} edges "
          f"({results.natural_sparsity*100:.2f}%)")
    print(f"\n  EDGE QUALITY:")
    print(f"    Mean Consensus: {results.mean_consensus:.4f}")
    print(f"    Mean Weight:    {results.mean_weight:.4f}")
    print(f"    Mean Score:     {results.mean_score:.4f}")
    print(f"    Reliability:    {results.network_reliability:.4f}")
    print(f"\n  CONSENSUS LEVELS:")
    print(f"    Unanimous (≥99%):  {results.n_unanimous:4d} ({results.n_unanimous/results.n_kept*100:.1f}%)")
    print(f"    Strong (75-99%):   {results.n_strong:4d} ({results.n_strong/results.n_kept*100:.1f}%)")
    print(f"    Moderate (50-75%): {results.n_moderate:4d} ({results.n_moderate/results.n_kept*100:.1f}%)")
    print(f"    Weak (25-50%):     {results.n_weak:4d} ({results.n_weak/results.n_kept*100:.1f}%)")
    print(f"    Minimal (<25%):    {results.n_minimal:4d} ({results.n_minimal/results.n_kept*100:.1f}%)")
    print(f"\n  CORE NETWORK: {results.n_unanimous + results.n_strong} edges (unanimous + strong)")
    
    # Generate outputs
    print("\n" + "="*70)
    print("Generating outputs...")
    
    plot_intelligent_analysis(results, n_ad, n_hc, 
                              save_path=str(OUTPUT_DIR / "1_intelligent_analysis.png"))
    plot_top_edges_analysis(results, top_n=50,
                            save_path=str(OUTPUT_DIR / "2_top_edges_analysis.png"))
    generate_intelligent_report(results, n_ad, n_hc,
                               save_path=str(OUTPUT_DIR / "3_analysis_report.txt"))
    
    # Save data
    np.save(OUTPUT_DIR / "consensus_matrix.npy", results.C)
    np.save(OUTPUT_DIR / "weight_matrix.npy", results.W)
    np.save(OUTPUT_DIR / "score_matrix.npy", results.Score)
    np.save(OUTPUT_DIR / "final_graph.npy", results.G_final)
    
    print(f"\n{'='*70}")
    print("                    COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    
    return results


if __name__ == "__main__":
    results = run_intelligent_analysis()
