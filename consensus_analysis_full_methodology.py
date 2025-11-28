#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
        DISTANCE-DEPENDENT CONSENSUS MATRIX ANALYSIS (Betzel-Style)
                    Full Methodology Demonstration
═══════════════════════════════════════════════════════════════════════════════════════

This script demonstrates the COMPLETE methodology for building consensus matrices
following the Betzel-style approach, as implemented in consensus_matrix_eeg.py.

METHODOLOGY STEPS:
==================

STEP 1: Per-Subject Correlation Matrices
----------------------------------------
    For each subject s:
        A(s) = Pearson correlation matrix from EEG data
        A(s)[i,j] = correlation between channel i and channel j

STEP 2: Proportional Thresholding → Binary Matrices
---------------------------------------------------
    For each subject s:
        B(s) = threshold(A(s), sparsity=κ)
        Keep top κ fraction of edges (e.g., κ=15%)
        B(s)[i,j] = 1 if edge in top κ%, else 0

STEP 3: Consensus Matrix C
--------------------------
    C[i,j] = (1/S) × Σ_s B(s)[i,j]
    
    C[i,j] = fraction of subjects that have edge (i,j)
    Range: 0 (no subjects) to 1 (all subjects)

STEP 4: Weight Matrix W (Fisher-z averaging)
--------------------------------------------
    For edges where C[i,j] > 0:
        z_values = arctanh(A(s)[i,j]) for subjects with edge
        W[i,j] = tanh(mean(z_values))
    
    W[i,j] = representative correlation strength

STEP 5: Distance-Dependent Edge Selection (Betzel bins)
-------------------------------------------------------
    1. Compute Euclidean distance D[i,j] between channels
    2. Divide distances into n_bins (e.g., 10 bins)
    3. For each distance bin:
       - Score edges: score = C[i,j] + ε × W[i,j]
       - Select top edges per bin
    4. Ensures edges are selected across ALL distances

STEP 6: Final Group Graph G
---------------------------
    G[i,j] = W[i,j] for selected edges
    G[i,j] = 0 otherwise

REFERENCES:
-----------
    Betzel et al. (2019). "Distance-dependent consensus thresholds for 
    generating group-representative structural brain networks."
    Network Neuroscience, 3(2), 475-496.

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from scipy.spatial.distance import pdist, squareform


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("./consensus_methodology_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
#                     STEP-BY-STEP METHODOLOGY IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def step1_compute_correlation_matrix(data: np.ndarray, absolute: bool = True) -> np.ndarray:
    """
    STEP 1: Compute Pearson correlation adjacency matrix.
    
    A(s)[i,j] = |correlation(channel_i, channel_j)|
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels × n_samples)
    absolute : bool
        Use absolute correlation values
        
    Returns
    -------
    A : np.ndarray
        Correlation matrix (n_channels × n_channels)
    """
    A = np.corrcoef(data)
    A = np.nan_to_num(A, nan=0.0)
    
    if absolute:
        A = np.abs(A)
    
    np.fill_diagonal(A, 0)
    return A


def step2_proportional_threshold(A: np.ndarray, sparsity: float) -> np.ndarray:
    """
    STEP 2: Apply proportional thresholding to create binary matrix.
    
    Keep top κ fraction of edges.
    
    Parameters
    ----------
    A : np.ndarray
        Weighted correlation matrix
    sparsity : float
        Fraction of edges to keep (0 < κ < 1)
        
    Returns
    -------
    B : np.ndarray
        Binary matrix (1 = edge present, 0 = absent)
    """
    n = A.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    edge_weights = A[triu_idx]
    
    n_edges = len(edge_weights)
    k_edges = int(np.floor(sparsity * n_edges))
    k_edges = max(1, k_edges)
    
    if k_edges < n_edges:
        threshold = np.sort(edge_weights)[::-1][k_edges]
    else:
        threshold = 0
    
    B = np.zeros_like(A)
    B[A > threshold] = 1
    B = np.maximum(B, B.T)  # Ensure symmetry
    np.fill_diagonal(B, 0)
    
    return B


def step3_compute_consensus(binary_matrices: np.ndarray) -> np.ndarray:
    """
    STEP 3: Compute consensus matrix C.
    
    C[i,j] = (1/S) × Σ_s B(s)[i,j]
           = fraction of subjects with edge (i,j)
    
    Parameters
    ----------
    binary_matrices : np.ndarray
        Stack of binary matrices (n_subjects × n_channels × n_channels)
        
    Returns
    -------
    C : np.ndarray
        Consensus matrix (values 0 to 1)
    """
    C = np.mean(binary_matrices, axis=0)
    return C


def step4_compute_weights(adjacency_matrices: np.ndarray, 
                          binary_matrices: np.ndarray) -> np.ndarray:
    """
    STEP 4: Compute weight matrix W using Fisher-z averaging.
    
    For edges where at least one subject has the connection:
        z = arctanh(r)  [Fisher z-transform]
        z_mean = average z across subjects with edge
        W = |tanh(z_mean)|  [back-transform]
    
    Parameters
    ----------
    adjacency_matrices : np.ndarray
        Stack of correlation matrices (n_subjects × n_channels × n_channels)
    binary_matrices : np.ndarray
        Stack of binary matrices (n_subjects × n_channels × n_channels)
        
    Returns
    -------
    W : np.ndarray
        Weight matrix
    """
    n_subjects, n_channels, _ = adjacency_matrices.shape
    W = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            # Find subjects where edge exists
            edge_exists = binary_matrices[:, i, j] > 0
            
            if np.any(edge_exists):
                # Get correlation values for subjects with edge
                r_values = adjacency_matrices[edge_exists, i, j]
                
                # Fisher z-transform
                r_clipped = np.clip(r_values, -0.999, 0.999)
                z_values = np.arctanh(r_clipped)
                
                # Average in z-space, then back-transform
                z_mean = np.mean(z_values)
                W[i, j] = np.abs(np.tanh(z_mean))
                W[j, i] = W[i, j]
    
    return W


def step5_natural_sparsity_selection(C: np.ndarray, 
                                      W: np.ndarray,
                                      W_full: np.ndarray = None,
                                      min_consensus: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    STEP 5: NATURAL SPARSITY - Keep all edges where consensus > 0.
    
    NO artificial sparsity cutoff!
    The sparsity emerges NATURALLY from the consensus process:
    - Edges present in at least one subject are kept
    - Sparsity reflects true agreement across subjects
    
    This is critical for:
    - Graph frequency spectrum analysis
    - GP-VAR models
    - Preserving network topology
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix (fraction of subjects with each edge)
    W : np.ndarray
        Weight matrix (Fisher-z averaged, conditional on edge existence)
    W_full : np.ndarray, optional
        Dense weight matrix (Fisher-z averaged across ALL subjects)
        If provided, used for edge weights
    min_consensus : float
        Minimum consensus threshold (default 0 = any subject)
        Set to 0.5 for majority consensus, etc.
        
    Returns
    -------
    G : np.ndarray
        Final graph with NATURAL sparsity
    selection_info : dict
        Information about natural sparsity
    """
    n_channels = C.shape[0]
    triu_idx = np.triu_indices(n_channels, k=1)
    n_possible = len(triu_idx[0])
    
    # Use full weight matrix if available, otherwise use conditional weights
    weight_source = W_full if W_full is not None else W
    
    # NATURAL SELECTION: Keep ALL edges where consensus > min_consensus
    # NO artificial cutoff!
    mask = C > min_consensus
    mask = mask & (weight_source > 0)
    np.fill_diagonal(mask, False)
    
    # Build final graph
    G = np.where(mask, weight_source, 0.0)
    G = np.maximum(G, G.T)  # Ensure symmetry
    
    # Compute statistics
    n_selected = int(np.sum(np.triu(G > 0, k=1)))
    natural_sparsity = n_selected / n_possible
    
    # Analyze consensus distribution of selected edges
    selected_consensus = C[triu_idx][G[triu_idx] > 0]
    
    selection_info = {
        'method': 'NATURAL_SPARSITY',
        'artificial_cutoff': False,
        'min_consensus_threshold': min_consensus,
        'n_possible_edges': n_possible,
        'n_selected_edges': n_selected,
        'natural_sparsity_percent': natural_sparsity * 100,
        'consensus_mean': float(np.mean(selected_consensus)) if len(selected_consensus) > 0 else 0,
        'consensus_min': float(np.min(selected_consensus)) if len(selected_consensus) > 0 else 0,
        'consensus_max': float(np.max(selected_consensus)) if len(selected_consensus) > 0 else 0,
        'edges_unanimous': int(np.sum(selected_consensus >= 0.99)),
        'edges_majority': int(np.sum(selected_consensus > 0.5)),
        'edges_any': int(np.sum(selected_consensus > 0)),
    }
    
    return G, selection_info


def step5_distance_dependent_selection(C: np.ndarray, 
                                       W: np.ndarray,
                                       channel_locations: np.ndarray,
                                       target_sparsity: float = None,
                                       n_bins: int = 10,
                                       epsilon: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    STEP 5 (Alternative): Distance-dependent edge selection (Betzel-style bins).
    
    NOTE: Use step5_natural_sparsity_selection() for NATURAL sparsity!
    This function is for when you need a specific target sparsity.
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix
    W : np.ndarray
        Weight matrix
    channel_locations : np.ndarray
        3D coordinates (n_channels × 3)
    target_sparsity : float, optional
        Target fraction of edges. If None, uses natural sparsity.
    n_bins : int
        Number of distance bins
    epsilon : float
        Weight for W in scoring
        
    Returns
    -------
    G : np.ndarray
        Final graph adjacency matrix
    selection_info : dict
        Information about edge selection
    """
    # If no target sparsity, use NATURAL sparsity
    if target_sparsity is None:
        return step5_natural_sparsity_selection(C, W)
    
    n_channels = C.shape[0]
    triu_idx = np.triu_indices(n_channels, k=1)
    n_possible = len(triu_idx[0])
    
    # Compute distance matrix
    distances_flat = pdist(channel_locations, metric='euclidean')
    D = squareform(distances_flat)
    distances = D[triu_idx]
    
    # Create distance bins
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10
    
    # Target edges per bin
    k_target = int(np.round(target_sparsity * n_possible))
    edges_per_bin = k_target // n_bins
    remaining = k_target % n_bins
    
    # Initialize
    G = np.zeros((n_channels, n_channels))
    selected_edges = set()
    selection_info = {'bins': [], 'total_selected': 0, 'method': 'DISTANCE_DEPENDENT'}
    
    # Process each bin
    for bin_idx in range(n_bins):
        bin_mask = (distances >= bin_edges[bin_idx]) & (distances < bin_edges[bin_idx + 1])
        bin_edge_indices = np.where(bin_mask)[0]
        
        if len(bin_edge_indices) == 0:
            selection_info['bins'].append({'bin': bin_idx, 'selected': 0, 'available': 0})
            continue
        
        # Score edges in this bin
        scores = []
        valid_pairs = []
        
        for idx in bin_edge_indices:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            if C[i, j] > 0:  # Only consider edges that exist in at least one subject
                score = C[i, j] + epsilon * W[i, j]
                scores.append(score)
                valid_pairs.append((i, j))
        
        if len(scores) == 0:
            selection_info['bins'].append({'bin': bin_idx, 'selected': 0, 'available': len(bin_edge_indices)})
            continue
        
        # Select top edges for this bin
        k_bin = edges_per_bin + (1 if bin_idx < remaining else 0)
        k_bin = min(k_bin, len(scores))
        
        sorted_idx = np.argsort(scores)[::-1][:k_bin]
        
        for idx in sorted_idx:
            i, j = valid_pairs[idx]
            G[i, j] = W[i, j]
            G[j, i] = W[i, j]
            selected_edges.add((min(i,j), max(i,j)))
        
        selection_info['bins'].append({
            'bin': bin_idx, 
            'distance_range': (bin_edges[bin_idx], bin_edges[bin_idx + 1]),
            'selected': k_bin, 
            'available': len(bin_edge_indices)
        })
    
    selection_info['total_selected'] = len(selected_edges)
    selection_info['target'] = k_target
    selection_info['artificial_cutoff'] = True
    
    return G, selection_info


def step6_build_overall_consensus(ad_binary: np.ndarray, 
                                  hc_binary: np.ndarray,
                                  ad_adjacency: np.ndarray,
                                  hc_adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    STEP 6: Build OVERALL consensus from ALL subjects (AD + HC).
    
    This combines all individual subject matrices to create a true
    consensus across the entire study population.
    
    Parameters
    ----------
    ad_binary : np.ndarray
        AD binary matrices (n_ad × n_channels × n_channels)
    hc_binary : np.ndarray
        HC binary matrices (n_hc × n_channels × n_channels)
    ad_adjacency : np.ndarray
        AD correlation matrices
    hc_adjacency : np.ndarray
        HC correlation matrices
        
    Returns
    -------
    C_overall : np.ndarray
        Overall consensus matrix
    W_overall : np.ndarray
        Overall weight matrix
    """
    # Stack ALL binary matrices
    all_binary = np.concatenate([ad_binary, hc_binary], axis=0)
    all_adjacency = np.concatenate([ad_adjacency, hc_adjacency], axis=0)
    
    # Compute overall consensus
    C_overall = step3_compute_consensus(all_binary)
    
    # Compute overall weights
    W_overall = step4_compute_weights(all_adjacency, all_binary)
    
    return C_overall, W_overall


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def plot_methodology_flowchart(save_path: str = None):
    """Create a flowchart showing the methodology steps."""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'DISTANCE-DEPENDENT CONSENSUS METHODOLOGY\n(Betzel-Style)', 
           fontsize=16, fontweight='bold', ha='center', va='top')
    
    # Step boxes
    steps = [
        (1, 10.0, 'STEP 1: Per-Subject Correlation\nA(s) = |corr(EEG)|', '#FFE4E1'),
        (1, 8.0, 'STEP 2: Proportional Threshold\nB(s) = top κ% edges → binary', '#E1F5FE'),
        (1, 6.0, 'STEP 3: Consensus Matrix\nC = mean(B) across subjects', '#E8F5E9'),
        (1, 4.0, 'STEP 4: Weight Matrix\nW = Fisher-z average', '#FFF3E0'),
        (1, 2.0, 'STEP 5: Distance Bins\nSelect edges per distance bin', '#F3E5F5'),
        (6, 2.0, 'STEP 6: Final Graph G\nG[i,j] = W[i,j] for selected', '#FFFDE7'),
    ]
    
    for x, y, text, color in steps:
        box = FancyBboxPatch((x-0.8, y-0.7), 3.6, 1.4,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    for i in range(4):
        ax.annotate('', xy=(2.8, steps[i+1][1]+0.7), xytext=(2.8, steps[i][1]-0.7),
                   arrowprops=arrow_style)
    
    # Arrow to final
    ax.annotate('', xy=(5.2, 2.0), xytext=(4.6, 2.0), arrowprops=arrow_style)
    
    # Formulas
    formulas = [
        (6, 10.0, r'$A^{(s)}_{ij} = |\rho(x_i, x_j)|$'),
        (6, 8.0, r'$B^{(s)}_{ij} = \mathbf{1}[A^{(s)}_{ij} > \tau_\kappa]$'),
        (6, 6.0, r'$C_{ij} = \frac{1}{S}\sum_s B^{(s)}_{ij}$'),
        (6, 4.0, r'$W_{ij} = \tanh(\bar{z}_{ij})$'),
    ]
    
    for x, y, formula in formulas:
        ax.text(x+2.5, y, formula, fontsize=14, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # Key insight box
    insight_text = """KEY INSIGHT: Distance-Dependent Selection

Traditional approach: Select top edges globally → biased toward short-range

Betzel approach: 
  1. Divide edges into distance bins
  2. Select top edges FROM EACH BIN
  3. Ensures representation across all distances

This preserves both local and long-range connectivity!"""
    
    ax.text(5, 0.3, insight_text, fontsize=9, ha='center', va='bottom',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Methodology flowchart saved to: {save_path}")
    
    return fig


def plot_full_pipeline_demo(ad_adjacency: np.ndarray,
                            hc_adjacency: np.ndarray,
                            ad_binary: np.ndarray,
                            hc_binary: np.ndarray,
                            C_ad: np.ndarray,
                            C_hc: np.ndarray,
                            C_overall: np.ndarray,
                            W_overall: np.ndarray,
                            G_final: np.ndarray,
                            channel_locations: np.ndarray,
                            n_ad: int,
                            n_hc: int,
                            save_path: str = None):
    """
    Create comprehensive figure showing the full methodology pipeline.
    """
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 6, figure=fig, hspace=0.35, wspace=0.3)
    
    n_channels = C_ad.shape[0]
    triu_idx = np.triu_indices(n_channels, k=1)
    
    # ═══════ ROW 1: Individual Subject Matrices ═══════
    fig.text(0.5, 0.97, 'STEP 1-2: Individual Subject Matrices → Binary', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Sample AD subjects
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(ad_adjacency[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'AD Subject {i+1}\nCorrelation A(s)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Sample HC subjects
    for i in range(3):
        ax = fig.add_subplot(gs[0, i+3])
        ax.imshow(hc_adjacency[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'HC Subject {i+1}\nCorrelation A(s)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # ═══════ ROW 2: Binary Matrices ═══════
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(ad_binary[i], cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f'AD Subject {i+1}\nBinary B(s)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i+3])
        ax.imshow(hc_binary[i], cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f'HC Subject {i+1}\nBinary B(s)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # ═══════ ROW 3: Group Consensus ═══════
    fig.text(0.5, 0.62, 'STEP 3: Group Consensus Matrices C = mean(B)', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax_cad = fig.add_subplot(gs[2, 0:2])
    im = ax_cad.imshow(C_ad, cmap='YlOrRd', vmin=0, vmax=1)
    ax_cad.set_title(f'AD CONSENSUS (N={n_ad})\nC_AD = mean(B_AD)', fontsize=12, fontweight='bold', color='darkred')
    plt.colorbar(im, ax=ax_cad, fraction=0.046, label='Fraction of subjects')
    
    ax_chc = fig.add_subplot(gs[2, 2:4])
    im = ax_chc.imshow(C_hc, cmap='YlGnBu', vmin=0, vmax=1)
    ax_chc.set_title(f'HC CONSENSUS (N={n_hc})\nC_HC = mean(B_HC)', fontsize=12, fontweight='bold', color='darkblue')
    plt.colorbar(im, ax=ax_chc, fraction=0.046, label='Fraction of subjects')
    
    ax_coverall = fig.add_subplot(gs[2, 4:6])
    im = ax_coverall.imshow(C_overall, cmap='viridis', vmin=0, vmax=1)
    ax_coverall.set_title(f'OVERALL CONSENSUS (N={n_ad+n_hc})\nC_ALL = mean(B_ALL)', 
                         fontsize=12, fontweight='bold', color='darkgreen')
    plt.colorbar(im, ax=ax_coverall, fraction=0.046, label='Fraction of subjects')
    
    # ═══════ ROW 4: Weights and Distance Selection ═══════
    fig.text(0.5, 0.42, 'STEP 4-5: Weight Matrix & Distance-Dependent Selection', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax_w = fig.add_subplot(gs[3, 0:2])
    im = ax_w.imshow(W_overall, cmap='plasma', vmin=0)
    ax_w.set_title('Weight Matrix W\n(Fisher-z averaged)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_w, fraction=0.046, label='Weight')
    
    # Distance matrix
    ax_d = fig.add_subplot(gs[3, 2:4])
    D = squareform(pdist(channel_locations))
    im = ax_d.imshow(D, cmap='coolwarm')
    ax_d.set_title('Distance Matrix D\n(Euclidean between channels)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_d, fraction=0.046, label='Distance')
    
    # Final graph
    ax_g = fig.add_subplot(gs[3, 4:6])
    im = ax_g.imshow(G_final, cmap='hot', vmin=0)
    ax_g.set_title('FINAL GRAPH G\n(Distance-dependent selected)', fontsize=12, fontweight='bold', color='green')
    plt.colorbar(im, ax=ax_g, fraction=0.046, label='Edge weight')
    
    # ═══════ ROW 5: Verification and Statistics ═══════
    fig.text(0.5, 0.22, 'VERIFICATION: Consensus is Correct', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Scatter: AD vs HC
    ax_scatter1 = fig.add_subplot(gs[4, 0:2])
    ad_vals = C_ad[triu_idx]
    hc_vals = C_hc[triu_idx]
    ax_scatter1.scatter(hc_vals, ad_vals, alpha=0.3, s=5, c='purple')
    ax_scatter1.plot([0, 1], [0, 1], 'k--', lw=2)
    r = np.corrcoef(ad_vals, hc_vals)[0, 1]
    ax_scatter1.set_xlabel('HC Consensus')
    ax_scatter1.set_ylabel('AD Consensus')
    ax_scatter1.set_title(f'AD vs HC Consensus\nr = {r:.4f}', fontsize=11)
    ax_scatter1.set_xlim(-0.05, 1.05)
    ax_scatter1.set_ylim(-0.05, 1.05)
    ax_scatter1.grid(True, alpha=0.3)
    
    # Scatter: Groups vs Overall
    ax_scatter2 = fig.add_subplot(gs[4, 2:4])
    overall_vals = C_overall[triu_idx]
    ax_scatter2.scatter(ad_vals, overall_vals, alpha=0.3, s=5, c='red', label='AD')
    ax_scatter2.scatter(hc_vals, overall_vals, alpha=0.3, s=5, c='blue', label='HC')
    ax_scatter2.plot([0, 1], [0, 1], 'k--', lw=2)
    r_ad = np.corrcoef(ad_vals, overall_vals)[0, 1]
    r_hc = np.corrcoef(hc_vals, overall_vals)[0, 1]
    ax_scatter2.set_xlabel('Group Consensus')
    ax_scatter2.set_ylabel('Overall Consensus')
    ax_scatter2.set_title(f'Groups vs Overall\nAD: r={r_ad:.3f}, HC: r={r_hc:.3f}', fontsize=11)
    ax_scatter2.set_xlim(-0.05, 1.05)
    ax_scatter2.set_ylim(-0.05, 1.05)
    ax_scatter2.legend()
    ax_scatter2.grid(True, alpha=0.3)
    
    # Verification: Overall = weighted avg
    ax_verify = fig.add_subplot(gs[4, 4:6])
    n_total = n_ad + n_hc
    expected = (n_ad * ad_vals + n_hc * hc_vals) / n_total
    ax_verify.scatter(expected, overall_vals, alpha=0.5, s=5, c='green')
    ax_verify.plot([0, 1], [0, 1], 'r-', lw=3)
    max_err = np.max(np.abs(expected - overall_vals))
    ax_verify.set_xlabel('Expected: (n_AD×C_AD + n_HC×C_HC)/(n_AD+n_HC)')
    ax_verify.set_ylabel('Actual Overall Consensus')
    ax_verify.set_title(f'PROOF: Expected = Actual\nMax error = {max_err:.2e}', 
                       fontsize=11, fontweight='bold', color='green')
    ax_verify.set_xlim(-0.05, 1.05)
    ax_verify.set_ylim(-0.05, 1.05)
    ax_verify.grid(True, alpha=0.3)
    
    plt.suptitle('COMPLETE METHODOLOGY: Distance-Dependent Consensus (Betzel-Style)\n'
                f'AD: {n_ad} subjects, HC: {n_hc} subjects, Overall: {n_ad+n_hc} subjects',
                fontsize=16, fontweight='bold', y=1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Full pipeline figure saved to: {save_path}")
    
    return fig


def plot_distance_bin_analysis(C: np.ndarray,
                               W: np.ndarray,
                               channel_locations: np.ndarray,
                               n_bins: int = 10,
                               save_path: str = None):
    """
    Visualize the distance-dependent edge selection process.
    """
    n_channels = C.shape[0]
    triu_idx = np.triu_indices(n_channels, k=1)
    
    # Compute distances
    D = squareform(pdist(channel_locations))
    distances = D[triu_idx]
    consensus_vals = C[triu_idx]
    weight_vals = W[triu_idx]
    
    # Create bins
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Consensus vs Distance
    ax1 = axes[0, 0]
    ax1.scatter(distances, consensus_vals, alpha=0.3, s=5, c='blue')
    for i in range(n_bins):
        ax1.axvline(bin_edges[i], color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Euclidean Distance', fontsize=11)
    ax1.set_ylabel('Consensus C[i,j]', fontsize=11)
    ax1.set_title('Consensus vs Distance\n(Red lines = bin boundaries)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight vs Distance
    ax2 = axes[0, 1]
    ax2.scatter(distances, weight_vals, alpha=0.3, s=5, c='green')
    for i in range(n_bins):
        ax2.axvline(bin_edges[i], color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Euclidean Distance', fontsize=11)
    ax2.set_ylabel('Weight W[i,j]', fontsize=11)
    ax2.set_title('Weight vs Distance', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Score vs Distance
    ax3 = axes[0, 2]
    scores = consensus_vals + 0.1 * weight_vals
    ax3.scatter(distances, scores, alpha=0.3, s=5, c='purple')
    for i in range(n_bins):
        ax3.axvline(bin_edges[i], color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Euclidean Distance', fontsize=11)
    ax3.set_ylabel('Score = C + 0.1×W', fontsize=11)
    ax3.set_title('Selection Score vs Distance', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Edges per bin
    ax4 = axes[1, 0]
    edges_per_bin = []
    bin_centers = []
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        edges_per_bin.append(np.sum(mask))
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
    ax4.bar(range(n_bins), edges_per_bin, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Distance Bin', fontsize=11)
    ax4.set_ylabel('Number of Edges', fontsize=11)
    ax4.set_title('Edges Available per Distance Bin', fontsize=12)
    ax4.set_xticks(range(n_bins))
    
    # Plot 5: Mean consensus per bin
    ax5 = axes[1, 1]
    mean_consensus_per_bin = []
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        if np.sum(mask) > 0:
            mean_consensus_per_bin.append(np.mean(consensus_vals[mask]))
        else:
            mean_consensus_per_bin.append(0)
    ax5.bar(range(n_bins), mean_consensus_per_bin, color='coral', edgecolor='black')
    ax5.set_xlabel('Distance Bin', fontsize=11)
    ax5.set_ylabel('Mean Consensus', fontsize=11)
    ax5.set_title('Mean Consensus per Distance Bin', fontsize=12)
    ax5.set_xticks(range(n_bins))
    
    # Plot 6: Explanation
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    explanation = """
    DISTANCE-DEPENDENT SELECTION (Betzel-Style)
    ═══════════════════════════════════════════
    
    WHY IS THIS IMPORTANT?
    
    Problem with global selection:
    • Short-range connections have higher correlation
    • Global top-K selection is biased toward short-range
    • Long-range connections (important for integration) are lost
    
    Solution - Distance bins:
    1. Divide edges into distance bins (percentiles)
    2. Select top edges FROM EACH BIN
    3. Ensures representation across ALL distances
    
    FORMULA:
    Score(i,j) = C(i,j) + ε × W(i,j)
    
    Where:
    • C = consensus (fraction of subjects)
    • W = weight (Fisher-z averaged correlation)
    • ε = small weight (default 0.1)
    
    This preserves BOTH:
    ✓ Local connectivity (segregation)
    ✓ Long-range connectivity (integration)
    """
    
    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Distance-Dependent Edge Selection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Distance bin analysis saved to: {save_path}")
    
    return fig


def generate_methodology_report(n_ad: int, n_hc: int, n_channels: int,
                                sparsity_binarize: float, 
                                use_natural_sparsity: bool = True,
                                selection_info: dict = None,
                                save_path: str = None) -> str:
    """Generate detailed methodology report."""
    
    n_total = n_ad + n_hc
    n_possible = n_channels * (n_channels - 1) // 2
    
    # Get sparsity info
    if selection_info and use_natural_sparsity:
        final_sparsity_pct = selection_info.get('natural_sparsity_percent', 'N/A')
        n_selected = selection_info.get('n_selected_edges', 'N/A')
        edges_unanimous = selection_info.get('edges_unanimous', 'N/A')
        edges_majority = selection_info.get('edges_majority', 'N/A')
    else:
        final_sparsity_pct = 'N/A'
        n_selected = 'N/A'
        edges_unanimous = 'N/A'
        edges_majority = 'N/A'
    
    sparsity_section = f"""
STEP 5: NATURAL SPARSITY (No Artificial Cutoff)
-----------------------------------------------
  
  *** THIS IS THE KEY DIFFERENCE FROM TYPICAL APPROACHES ***
  
  NATURAL SPARSITY means:
    • Keep ALL edges where consensus C[i,j] > 0
    • NO arbitrary percentage cutoff (e.g., NOT "keep only 10%")
    • Sparsity emerges NATURALLY from the consensus process
  
  Final graph includes edge (i,j) if:
    • At least ONE subject has this edge (C[i,j] > 0)
    • The edge has valid weight (W[i,j] > 0)
  
  WHY NATURAL SPARSITY?
    ✓ Preserves true network topology
    ✓ Critical for valid graph frequency spectrum
    ✓ Required for GP-VAR analysis
    ✓ No arbitrary information loss
    ✓ Sparsity reflects actual subject agreement
  
  RESULTS:
    • Possible edges: {n_possible:,}
    • Selected edges: {n_selected}
    • Natural sparsity: {final_sparsity_pct:.2f}% (NOT artificially imposed!)
    • Edges with unanimous consensus (100%): {edges_unanimous}
    • Edges with majority consensus (>50%): {edges_majority}
""" if use_natural_sparsity else f"""
STEP 5: Distance-Dependent Edge Selection (Betzel-style)
---------------------------------------------------------
  
  1. Compute Euclidean distance D[i,j] from 3D channel coordinates
  2. Divide distances into bins (percentile-based)
  3. Select top edges from EACH bin based on score:
  
     Score[i,j] = C[i,j] + 0.1 × W[i,j]
     
  4. Artificial sparsity is IMPOSED (e.g., 10%)
  
  WARNING: This may lose important connections!
"""
    
    report = f"""
================================================================================
        CONSENSUS MATRIX METHODOLOGY REPORT
                (Betzel-Style Implementation)
================================================================================

OVERVIEW
--------
This analysis builds group-representative brain connectivity networks using
the consensus approach (Betzel et al., 2019).

*** SPARSITY MODE: {'NATURAL (recommended for GP-VAR)' if use_natural_sparsity else 'ARTIFICIAL'} ***

SAMPLE
------
  • AD patients:        {n_ad}
  • Healthy Controls:   {n_hc}
  • TOTAL subjects:     {n_total}
  • EEG channels:       {n_channels}
  • Possible edges:     {n_possible:,}

================================================================================
                           METHODOLOGY STEPS
================================================================================

STEP 1: Per-Subject Correlation Matrices
----------------------------------------
  For each subject s = 1, ..., {n_total}:
    
    A(s)[i,j] = |Pearson correlation(channel_i, channel_j)|
    
  • Absolute value used to capture both positive and negative correlations
  • Diagonal set to 0 (no self-connections)
  • OUTPUT: DENSE matrix (all edges have values)

STEP 2: Proportional Thresholding → Binary Matrices
---------------------------------------------------
  For each subject s:
    
    B(s)[i,j] = 1  if A(s)[i,j] in top {sparsity_binarize*100:.0f}% of edges
              = 0  otherwise
    
  • Sparsity κ = {sparsity_binarize} ({sparsity_binarize*100:.0f}% of edges kept per subject)
  • Ensures comparable density across subjects
  • OUTPUT: SPARSE binary matrix (~{sparsity_binarize*100:.0f}% edges per subject)

STEP 3: Consensus Matrix C
--------------------------
  C[i,j] = (1/{n_total}) × Σ_s B(s)[i,j]
  
  • C[i,j] = fraction of subjects with edge (i,j)
  • Range: 0 (no subjects) to 1 (all subjects)
  • High consensus = consistent connection across subjects
  • OUTPUT: Values 0-1 for each edge

STEP 4: Weight Matrix W (Fisher-z averaging)
--------------------------------------------
  For edges where C[i,j] > 0:
    
    z_values = arctanh(A(s)[i,j]) for subjects with edge
    W[i,j] = |tanh(mean(z_values))|
    
  • Fisher z-transform handles non-normality of correlations
  • Averaging in z-space, then back-transform
  • W represents typical correlation strength for each edge
{sparsity_section}
STEP 6: Final Graph G
---------------------
  G[i,j] = W[i,j]  for selected edges
         = 0       otherwise

================================================================================
              WHY NATURAL SPARSITY IS CRITICAL FOR GP-VAR
================================================================================

The graph frequency spectrum (eigenvalues of the Laplacian) is used by GP-VAR
models to decompose signals into graph frequency components.

PROBLEM with artificial sparsity:
---------------------------------
  • Cutting at arbitrary % (e.g., 10%) removes valid edges
  • Changes the Laplacian eigenvalues
  • Distorts the graph frequency spectrum
  • GP-VAR results become unreliable

SOLUTION with natural sparsity:
-------------------------------
  • Sparsity emerges from subject agreement
  • All edges with ANY consensus are kept
  • Graph structure reflects true connectivity
  • Valid graph frequency spectrum for GP-VAR

================================================================================
                         OVERALL CONSENSUS
================================================================================

The OVERALL consensus is computed from ALL {n_total} subjects (AD + HC):

  C_overall[i,j] = (1/{n_total}) × Σ_{{s ∈ AD ∪ HC}} B(s)[i,j]

Mathematically equivalent to:

  C_overall = ({n_ad} × C_AD + {n_hc} × C_HC) / {n_total}

This represents:
  • Brain connectivity patterns shared across the entire population
  • Template for graph-based signal processing (GP-VAR)
  • Reference for comparing individual groups

================================================================================
                              REFERENCES
================================================================================

Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2019). 
  Distance-dependent consensus thresholds for generating group-representative 
  structural brain networks. Network Neuroscience, 3(2), 475-496.

================================================================================
"""
    
    if save_path:
        Path(save_path).write_text(report)
        print(f"✓ Methodology report saved to: {save_path}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════════════
#                              SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def generate_synthetic_eeg_data(n_subjects: int, 
                                 n_channels: int, 
                                 n_samples: int,
                                 group: str = 'HC',
                                 seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic EEG data with spatial structure.
    
    Returns
    -------
    data : np.ndarray
        Shape (n_subjects, n_channels, n_samples)
    channel_locations : np.ndarray
        Shape (n_channels, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate 3D channel locations (spherical head model)
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(n_channels)*2), endpoint=False)
    phi = np.linspace(0.2*np.pi, 0.8*np.pi, int(np.sqrt(n_channels)))
    
    locs = []
    for t in theta:
        for p in phi:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            locs.append([x, y, z])
    
    channel_locations = np.array(locs[:n_channels])
    
    # Generate spatially correlated EEG data
    # Distance matrix
    D = squareform(pdist(channel_locations))
    
    # Spatial covariance (closer channels more correlated)
    sigma = 0.5  # spatial scale
    spatial_cov = np.exp(-D**2 / (2 * sigma**2))
    
    # AD has reduced connectivity
    if group == 'AD':
        spatial_cov = spatial_cov * 0.8 + 0.2 * np.eye(n_channels)
    
    # Generate data
    L = np.linalg.cholesky(spatial_cov + 0.01 * np.eye(n_channels))
    
    data = np.zeros((n_subjects, n_channels, n_samples))
    for s in range(n_subjects):
        white_noise = np.random.randn(n_channels, n_samples)
        data[s] = L @ white_noise
    
    return data, channel_locations


# ═══════════════════════════════════════════════════════════════════════════════════════
#                                    MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_full_methodology_demo(n_ad: int = 35,
                              n_hc: int = 31,
                              n_channels: int = 64,
                              n_samples: int = 1000,
                              sparsity_binarize: float = 0.15,
                              use_natural_sparsity: bool = True,
                              n_bins: int = 10):
    """
    Run the complete methodology demonstration.
    
    Parameters
    ----------
    n_ad : int
        Number of AD subjects
    n_hc : int
        Number of HC subjects
    n_channels : int
        Number of EEG channels
    n_samples : int
        Number of time samples per subject
    sparsity_binarize : float
        Sparsity for per-subject binarization (Step 2)
    use_natural_sparsity : bool
        If True (default), use NATURAL sparsity - no artificial cutoff
        If False, use distance-dependent selection with fixed sparsity
    n_bins : int
        Number of distance bins (only used if use_natural_sparsity=False)
    """
    print("="*70)
    print("   CONSENSUS MATRIX METHODOLOGY DEMONSTRATION")
    print("="*70)
    print(f"\n   SPARSITY MODE: {'NATURAL (no artificial cutoff)' if use_natural_sparsity else 'ARTIFICIAL (fixed %)'}")
    print("="*70)
    
    # Generate synthetic data
    print("\n[1/7] Generating synthetic EEG data...")
    ad_data, channel_locations = generate_synthetic_eeg_data(
        n_ad, n_channels, n_samples, group='AD', seed=42)
    hc_data, _ = generate_synthetic_eeg_data(
        n_hc, n_channels, n_samples, group='HC', seed=123)
    
    print(f"  ✓ AD data: {ad_data.shape}")
    print(f"  ✓ HC data: {hc_data.shape}")
    print(f"  ✓ Channel locations: {channel_locations.shape}")
    
    # STEP 1: Compute correlation matrices
    print("\n[2/7] STEP 1: Computing per-subject correlation matrices...")
    ad_adjacency = np.array([step1_compute_correlation_matrix(ad_data[s]) for s in range(n_ad)])
    hc_adjacency = np.array([step1_compute_correlation_matrix(hc_data[s]) for s in range(n_hc)])
    print(f"  ✓ AD adjacency matrices: {ad_adjacency.shape}")
    print(f"  ✓ HC adjacency matrices: {hc_adjacency.shape}")
    
    # STEP 2: Proportional thresholding
    print(f"\n[3/7] STEP 2: Proportional thresholding (κ={sparsity_binarize})...")
    ad_binary = np.array([step2_proportional_threshold(ad_adjacency[s], sparsity_binarize) 
                         for s in range(n_ad)])
    hc_binary = np.array([step2_proportional_threshold(hc_adjacency[s], sparsity_binarize) 
                         for s in range(n_hc)])
    print(f"  ✓ AD binary matrices: {ad_binary.shape}")
    print(f"  ✓ HC binary matrices: {hc_binary.shape}")
    
    # STEP 3: Consensus matrices
    print("\n[4/7] STEP 3: Computing consensus matrices...")
    C_ad = step3_compute_consensus(ad_binary)
    C_hc = step3_compute_consensus(hc_binary)
    C_overall, W_overall = step6_build_overall_consensus(
        ad_binary, hc_binary, ad_adjacency, hc_adjacency)
    print(f"  ✓ AD consensus mean: {np.mean(C_ad):.4f}")
    print(f"  ✓ HC consensus mean: {np.mean(C_hc):.4f}")
    print(f"  ✓ Overall consensus mean: {np.mean(C_overall):.4f}")
    
    # STEP 4: Weight matrix
    print("\n[5/7] STEP 4: Computing weight matrix (Fisher-z)...")
    print(f"  ✓ Overall weight mean: {np.mean(W_overall):.4f}")
    
    # STEP 5-6: NATURAL SPARSITY selection
    print(f"\n[6/7] STEP 5-6: Building final graph...")
    
    if use_natural_sparsity:
        print("  → Using NATURAL SPARSITY (keeping all edges where consensus > 0)")
        G_final, selection_info = step5_natural_sparsity_selection(C_overall, W_overall)
        print(f"  ✓ NATURAL sparsity: {selection_info['natural_sparsity_percent']:.2f}%")
        print(f"  ✓ Selected {selection_info['n_selected_edges']} edges (out of {selection_info['n_possible_edges']} possible)")
        print(f"  ✓ Edges with unanimous consensus (100%): {selection_info['edges_unanimous']}")
        print(f"  ✓ Edges with majority consensus (>50%): {selection_info['edges_majority']}")
    else:
        print(f"  → Using ARTIFICIAL sparsity with distance-dependent bins")
        G_final, selection_info = step5_distance_dependent_selection(
            C_overall, W_overall, channel_locations, target_sparsity=0.10, n_bins=n_bins)
        print(f"  ✓ Selected {selection_info['total_selected']} edges (target: {selection_info['target']})")
    
    # Generate outputs
    print("\n[7/7] Generating outputs...")
    
    # Flowchart
    plot_methodology_flowchart(str(OUTPUT_DIR / "1_methodology_flowchart.png"))
    
    # Full pipeline
    plot_full_pipeline_demo(
        ad_adjacency, hc_adjacency,
        ad_binary, hc_binary,
        C_ad, C_hc, C_overall, W_overall, G_final,
        channel_locations, n_ad, n_hc,
        str(OUTPUT_DIR / "2_full_pipeline_demo.png")
    )
    
    # Distance bin analysis
    plot_distance_bin_analysis(
        C_overall, W_overall, channel_locations, n_bins,
        str(OUTPUT_DIR / "3_distance_bin_analysis.png")
    )
    
    # Methodology report
    generate_methodology_report(
        n_ad, n_hc, n_channels, sparsity_binarize, 
        use_natural_sparsity=use_natural_sparsity,
        selection_info=selection_info,
        save_path=str(OUTPUT_DIR / "4_methodology_report.txt")
    )
    
    # Save matrices
    np.save(OUTPUT_DIR / "AD_consensus.npy", C_ad)
    np.save(OUTPUT_DIR / "HC_consensus.npy", C_hc)
    np.save(OUTPUT_DIR / "Overall_consensus.npy", C_overall)
    np.save(OUTPUT_DIR / "Overall_weights.npy", W_overall)
    np.save(OUTPUT_DIR / "Final_graph.npy", G_final)
    np.save(OUTPUT_DIR / "channel_locations.npy", channel_locations)
    
    print(f"\n{'='*70}")
    print("                    ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nFiles generated:")
    print("  1. 1_methodology_flowchart.png    - Visual methodology overview")
    print("  2. 2_full_pipeline_demo.png       - Complete pipeline with data")
    print("  3. 3_distance_bin_analysis.png    - Distance-dependent selection")
    print("  4. 4_methodology_report.txt       - Detailed text report")
    print("  5. AD_consensus.npy               - AD consensus matrix")
    print("  6. HC_consensus.npy               - HC consensus matrix")
    print("  7. Overall_consensus.npy          - Overall consensus matrix")
    print("  8. Overall_weights.npy            - Weight matrix")
    print("  9. Final_graph.npy                - Final graph G")
    print(" 10. channel_locations.npy          - 3D channel coordinates")
    
    return {
        'ad_adjacency': ad_adjacency,
        'hc_adjacency': hc_adjacency,
        'ad_binary': ad_binary,
        'hc_binary': hc_binary,
        'C_ad': C_ad,
        'C_hc': C_hc,
        'C_overall': C_overall,
        'W_overall': W_overall,
        'G_final': G_final,
        'channel_locations': channel_locations,
        'selection_info': selection_info
    }


if __name__ == "__main__":
    results = run_full_methodology_demo(
        n_ad=35,
        n_hc=31,
        n_channels=64,
        n_samples=1000,
        sparsity_binarize=0.15,        # Per-subject binarization (Step 2)
        use_natural_sparsity=True,      # NATURAL sparsity - NO artificial cutoff!
        n_bins=10
    )
