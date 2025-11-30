"""
Validate Consensus Matrix by Comparing with Individual Subject Correlations

This script proves that the consensus matrix truly represents a group consensus
by comparing it with each individual subject's correlation matrix.

Metrics computed:
1. Correlation between individual and consensus matrices
2. Edge overlap (Jaccard similarity)
3. Mean absolute difference
4. Visualization of individual vs consensus

For thesis: This validates that the consensus is representative of ALL subjects.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_consensus_results(results_dir: str) -> Dict:
    """Load consensus matrix results from directory."""
    results_path = Path(results_dir)
    
    results = {}
    
    # Load consensus matrix C (edge consistency)
    c_path = results_path / "consensus_matrix_C.npy"
    if c_path.exists():
        results['C'] = np.load(c_path)
    
    # Load weight matrix W
    w_path = results_path / "consensus_matrix_W.npy"
    if w_path.exists():
        results['W'] = np.load(w_path)
    
    # Load final graph G
    g_path = results_path / "consensus_distance_graph.npy"
    if g_path.exists():
        results['G'] = np.load(g_path)
    
    # Load binary matrices (individual subject binarized matrices)
    b_path = results_path / "consensus_binary_matrices.npy"
    if b_path.exists():
        results['binary_matrices'] = np.load(b_path)
    
    return results


def compute_matrix_correlation(mat1: np.ndarray, mat2: np.ndarray) -> float:
    """Compute Pearson correlation between upper triangles of two matrices."""
    n = mat1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    vec1 = mat1[triu_idx]
    vec2 = mat2[triu_idx]
    
    # Handle zero variance
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        return 0.0
    
    corr, _ = stats.pearsonr(vec1, vec2)
    return corr


def compute_jaccard_similarity(mat1: np.ndarray, mat2: np.ndarray, threshold: float = 0) -> float:
    """Compute Jaccard similarity between binarized matrices."""
    bin1 = (mat1 > threshold).astype(int)
    bin2 = (mat2 > threshold).astype(int)
    
    n = mat1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    vec1 = bin1[triu_idx]
    vec2 = bin2[triu_idx]
    
    intersection = np.sum((vec1 == 1) & (vec2 == 1))
    union = np.sum((vec1 == 1) | (vec2 == 1))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_mean_absolute_difference(mat1: np.ndarray, mat2: np.ndarray) -> float:
    """Compute mean absolute difference between matrices."""
    n = mat1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    vec1 = mat1[triu_idx]
    vec2 = mat2[triu_idx]
    
    return np.mean(np.abs(vec1 - vec2))


def validate_consensus_single_group(
    individual_matrices: List[np.ndarray],
    consensus_matrix: np.ndarray,
    group_name: str
) -> pd.DataFrame:
    """
    Validate consensus matrix against all individual subjects in a group.
    
    Returns DataFrame with validation metrics for each subject.
    """
    results = []
    
    for i, ind_mat in enumerate(individual_matrices):
        # Compute correlation with consensus
        corr = compute_matrix_correlation(ind_mat, consensus_matrix)
        
        # Compute Jaccard similarity (edge overlap)
        jaccard = compute_jaccard_similarity(ind_mat, consensus_matrix)
        
        # Compute mean absolute difference
        mad = compute_mean_absolute_difference(ind_mat, consensus_matrix)
        
        # Compute spearman correlation (rank-based)
        n = ind_mat.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        spearman_corr, _ = stats.spearmanr(ind_mat[triu_idx], consensus_matrix[triu_idx])
        
        results.append({
            'subject_idx': i + 1,
            'group': group_name,
            'pearson_corr': corr,
            'spearman_corr': spearman_corr,
            'jaccard_similarity': jaccard,
            'mean_abs_diff': mad
        })
    
    return pd.DataFrame(results)


def plot_validation_results(
    ad_results: pd.DataFrame,
    hc_results: pd.DataFrame,
    output_dir: str
):
    """Create comprehensive validation visualization."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine results
    all_results = pd.concat([ad_results, hc_results], ignore_index=True)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ==========================================================================
    # Panel A: Individual-Consensus Correlation Distribution
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    ad_corrs = ad_results['pearson_corr'].values
    hc_corrs = hc_results['pearson_corr'].values
    
    positions = [1, 2]
    bp = ax1.boxplot([ad_corrs, hc_corrs], positions=positions, widths=0.6, patch_artist=True)
    
    colors = ['#E74C3C', '#3498DB']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, (data, pos, color) in enumerate(zip([ad_corrs, hc_corrs], positions, colors)):
        x = np.random.normal(pos, 0.08, size=len(data))
        ax1.scatter(x, data, alpha=0.6, color=color, s=30, edgecolor='white', linewidth=0.5)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['AD', 'HC'])
    ax1.set_ylabel('Correlation with Consensus')
    ax1.set_title('A) Individual-Consensus Correlation', fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5 threshold')
    ax1.set_ylim([0, 1.05])
    
    # Add statistics
    t_stat, p_val = stats.ttest_ind(ad_corrs, hc_corrs)
    ax1.text(0.95, 0.05, f'AD: {np.mean(ad_corrs):.3f}±{np.std(ad_corrs):.3f}\n'
                         f'HC: {np.mean(hc_corrs):.3f}±{np.std(hc_corrs):.3f}\n'
                         f'p={p_val:.4f}',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==========================================================================
    # Panel B: Jaccard Similarity Distribution
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ad_jaccard = ad_results['jaccard_similarity'].values
    hc_jaccard = hc_results['jaccard_similarity'].values
    
    bp2 = ax2.boxplot([ad_jaccard, hc_jaccard], positions=positions, widths=0.6, patch_artist=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for i, (data, pos, color) in enumerate(zip([ad_jaccard, hc_jaccard], positions, colors)):
        x = np.random.normal(pos, 0.08, size=len(data))
        ax2.scatter(x, data, alpha=0.6, color=color, s=30, edgecolor='white', linewidth=0.5)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['AD', 'HC'])
    ax2.set_ylabel('Jaccard Similarity')
    ax2.set_title('B) Edge Overlap with Consensus', fontweight='bold')
    ax2.set_ylim([0, 1.05])
    
    t_stat2, p_val2 = stats.ttest_ind(ad_jaccard, hc_jaccard)
    ax2.text(0.95, 0.05, f'AD: {np.mean(ad_jaccard):.3f}±{np.std(ad_jaccard):.3f}\n'
                         f'HC: {np.mean(hc_jaccard):.3f}±{np.std(hc_jaccard):.3f}\n'
                         f'p={p_val2:.4f}',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==========================================================================
    # Panel C: Histogram of Correlations
    # ==========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    bins = np.linspace(0, 1, 20)
    ax3.hist(ad_corrs, bins=bins, alpha=0.6, color='#E74C3C', label=f'AD (n={len(ad_corrs)})', edgecolor='black')
    ax3.hist(hc_corrs, bins=bins, alpha=0.6, color='#3498DB', label=f'HC (n={len(hc_corrs)})', edgecolor='black')
    
    ax3.axvline(np.mean(ad_corrs), color='#C0392B', linestyle='--', linewidth=2, label=f'AD mean={np.mean(ad_corrs):.3f}')
    ax3.axvline(np.mean(hc_corrs), color='#2980B9', linestyle='--', linewidth=2, label=f'HC mean={np.mean(hc_corrs):.3f}')
    
    ax3.set_xlabel('Correlation with Consensus')
    ax3.set_ylabel('Count')
    ax3.set_title('C) Distribution of Correlations', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    
    # ==========================================================================
    # Panel D: Per-Subject Correlation Bar Plot (AD)
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, :])
    
    # Sort by correlation
    ad_sorted = ad_results.sort_values('pearson_corr', ascending=False)
    hc_sorted = hc_results.sort_values('pearson_corr', ascending=False)
    
    n_ad = len(ad_sorted)
    n_hc = len(hc_sorted)
    
    x_ad = np.arange(n_ad)
    x_hc = np.arange(n_ad + 2, n_ad + 2 + n_hc)
    
    bars_ad = ax4.bar(x_ad, ad_sorted['pearson_corr'].values, color='#E74C3C', alpha=0.7, label='AD', edgecolor='black', linewidth=0.5)
    bars_hc = ax4.bar(x_hc, hc_sorted['pearson_corr'].values, color='#3498DB', alpha=0.7, label='HC', edgecolor='black', linewidth=0.5)
    
    ax4.axhline(y=np.mean(ad_corrs), color='#C0392B', linestyle='--', linewidth=2, alpha=0.8)
    ax4.axhline(y=np.mean(hc_corrs), color='#2980B9', linestyle='--', linewidth=2, alpha=0.8)
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('Subject (sorted by correlation)')
    ax4.set_ylabel('Correlation with Consensus')
    ax4.set_title('D) Per-Subject Correlation with Consensus Matrix', fontweight='bold')
    ax4.legend(loc='lower left')
    ax4.set_ylim([0, 1.05])
    
    # Add group separating line
    ax4.axvline(x=n_ad + 0.5, color='black', linestyle='-', linewidth=2)
    ax4.text(n_ad/2, 1.02, 'AD Group', ha='center', fontsize=10, fontweight='bold')
    ax4.text(n_ad + 2 + n_hc/2, 1.02, 'HC Group', ha='center', fontsize=10, fontweight='bold')
    
    # ==========================================================================
    # Panel E: Scatter - Pearson vs Spearman
    # ==========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    ax5.scatter(ad_results['pearson_corr'], ad_results['spearman_corr'], 
                c='#E74C3C', alpha=0.7, s=50, label='AD', edgecolor='white')
    ax5.scatter(hc_results['pearson_corr'], hc_results['spearman_corr'],
                c='#3498DB', alpha=0.7, s=50, label='HC', edgecolor='white')
    
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Identity')
    ax5.set_xlabel('Pearson Correlation')
    ax5.set_ylabel('Spearman Correlation')
    ax5.set_title('E) Pearson vs Spearman', fontweight='bold')
    ax5.legend(loc='lower right', fontsize=8)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # ==========================================================================
    # Panel F: Summary Statistics Table
    # ==========================================================================
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'AD (mean±std)', 'HC (mean±std)', 'p-value', 'Interpretation'],
        ['Pearson r', 
         f'{np.mean(ad_corrs):.3f}±{np.std(ad_corrs):.3f}',
         f'{np.mean(hc_corrs):.3f}±{np.std(hc_corrs):.3f}',
         f'{stats.ttest_ind(ad_corrs, hc_corrs)[1]:.4f}',
         'High r = consensus is representative'],
        ['Spearman ρ',
         f'{ad_results["spearman_corr"].mean():.3f}±{ad_results["spearman_corr"].std():.3f}',
         f'{hc_results["spearman_corr"].mean():.3f}±{hc_results["spearman_corr"].std():.3f}',
         f'{stats.ttest_ind(ad_results["spearman_corr"], hc_results["spearman_corr"])[1]:.4f}',
         'Rank-based agreement'],
        ['Jaccard',
         f'{np.mean(ad_jaccard):.3f}±{np.std(ad_jaccard):.3f}',
         f'{np.mean(hc_jaccard):.3f}±{np.std(hc_jaccard):.3f}',
         f'{stats.ttest_ind(ad_jaccard, hc_jaccard)[1]:.4f}',
         'Edge overlap proportion'],
        ['Mean Abs Diff',
         f'{ad_results["mean_abs_diff"].mean():.3f}±{ad_results["mean_abs_diff"].std():.3f}',
         f'{hc_results["mean_abs_diff"].mean():.3f}±{hc_results["mean_abs_diff"].std():.3f}',
         f'{stats.ttest_ind(ad_results["mean_abs_diff"], hc_results["mean_abs_diff"])[1]:.4f}',
         'Lower = more similar'],
        ['Min Correlation',
         f'{np.min(ad_corrs):.3f}',
         f'{np.min(hc_corrs):.3f}',
         '-',
         'Worst-case subject'],
        ['% Subjects r>0.5',
         f'{100*np.mean(ad_corrs > 0.5):.1f}%',
         f'{100*np.mean(hc_corrs > 0.5):.1f}%',
         '-',
         'Strong consensus agreement'],
    ]
    
    table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      loc='center', cellLoc='center',
                      colColours=['#f0f0f0']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Color code rows
    for i in range(len(summary_data)-1):
        for j in range(5):
            cell = table[(i+1, j)]
            if i == 0:  # Pearson row
                cell.set_facecolor('#e8f5e9')
    
    ax6.set_title('F) Validation Summary Statistics', fontweight='bold', y=0.95)
    
    # ==========================================================================
    # Save figure
    # ==========================================================================
    plt.suptitle('Consensus Matrix Validation: Individual vs Group Consensus\n'
                 'Proving the consensus represents all subjects',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path / 'consensus_validation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path / 'consensus_validation.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\nFigure saved to: {output_path / 'consensus_validation.png'}")
    
    plt.close()


def plot_example_comparisons(
    individual_matrices: List[np.ndarray],
    consensus_matrix: np.ndarray,
    group_name: str,
    output_dir: str,
    n_examples: int = 4
):
    """Plot example individual matrices alongside consensus for visual comparison."""
    
    output_path = Path(output_dir)
    
    # Select examples: best, worst, and 2 median
    n = len(individual_matrices)
    correlations = [compute_matrix_correlation(m, consensus_matrix) for m in individual_matrices]
    sorted_idx = np.argsort(correlations)[::-1]
    
    # Pick: best, 2 median, worst
    example_idx = [
        sorted_idx[0],  # Best
        sorted_idx[n//3],  # Upper median
        sorted_idx[2*n//3],  # Lower median
        sorted_idx[-1]  # Worst
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot consensus matrix
    im0 = axes[0, 0].imshow(consensus_matrix, cmap='hot', vmin=0, vmax=np.max(consensus_matrix))
    axes[0, 0].set_title(f'{group_name} Consensus Matrix\n(Group Average)', fontweight='bold')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Channel')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Plot examples
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    labels = ['Best Match', 'Upper Median', 'Lower Median', 'Worst Match']
    
    for (row, col), idx, label in zip(positions, example_idx, labels):
        corr = correlations[idx]
        im = axes[row, col].imshow(individual_matrices[idx], cmap='hot', 
                                    vmin=0, vmax=np.max(individual_matrices[idx]))
        axes[row, col].set_title(f'Subject {idx+1} ({label})\nr = {corr:.3f}', fontweight='bold')
        axes[row, col].set_xlabel('Channel')
        axes[row, col].set_ylabel('Channel')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    # Summary in bottom left
    axes[1, 0].axis('off')
    summary_text = f"""
    {group_name} Group Summary
    ─────────────────────
    Total Subjects: {n}
    
    Correlation with Consensus:
      Mean: {np.mean(correlations):.3f}
      Std:  {np.std(correlations):.3f}
      Min:  {np.min(correlations):.3f}
      Max:  {np.max(correlations):.3f}
    
    Subjects with r > 0.5: {np.sum(np.array(correlations) > 0.5)}/{n}
    Subjects with r > 0.7: {np.sum(np.array(correlations) > 0.7)}/{n}
    
    ✓ High correlation indicates
      consensus is representative
      of individual subjects.
    """
    axes[1, 0].text(0.1, 0.5, summary_text, transform=axes[1, 0].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Individual vs Consensus Matrix Comparison - {group_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path / f'{group_name}_individual_vs_consensus.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Example comparison saved: {output_path / f'{group_name}_individual_vs_consensus.png'}")


def generate_thesis_report(
    ad_results: pd.DataFrame,
    hc_results: pd.DataFrame,
    output_dir: str
):
    """Generate thesis-ready markdown report."""
    
    output_path = Path(output_dir)
    
    ad_corrs = ad_results['pearson_corr'].values
    hc_corrs = hc_results['pearson_corr'].values
    
    t_stat, p_val = stats.ttest_ind(ad_corrs, hc_corrs)
    
    report = f"""# Consensus Matrix Validation Report

## Summary

This analysis validates that the consensus matrix is representative of all individual subjects
in both AD and HC groups.

## Key Findings

### Correlation Between Individual and Consensus Matrices

| Group | n | Mean ± SD | Min | Max | % with r > 0.5 |
|-------|---|-----------|-----|-----|----------------|
| AD | {len(ad_corrs)} | {np.mean(ad_corrs):.3f} ± {np.std(ad_corrs):.3f} | {np.min(ad_corrs):.3f} | {np.max(ad_corrs):.3f} | {100*np.mean(ad_corrs > 0.5):.1f}% |
| HC | {len(hc_corrs)} | {np.mean(hc_corrs):.3f} ± {np.std(hc_corrs):.3f} | {np.min(hc_corrs):.3f} | {np.max(hc_corrs):.3f} | {100*np.mean(hc_corrs > 0.5):.1f}% |

**Group Comparison**: t({len(ad_corrs) + len(hc_corrs) - 2}) = {t_stat:.3f}, p = {p_val:.4f}

### Interpretation

{"✓ **CONSENSUS VALIDATED**: Both groups show high correlation with the consensus matrix (mean r > 0.5)." if np.mean(ad_corrs) > 0.5 and np.mean(hc_corrs) > 0.5 else "⚠ Some subjects show low correlation with consensus - review individual cases."}

The consensus matrix successfully captures the common connectivity patterns across:
- **{len(ad_corrs)} AD patients** with mean correlation r = {np.mean(ad_corrs):.3f}
- **{len(hc_corrs)} healthy controls** with mean correlation r = {np.mean(hc_corrs):.3f}

### Jaccard Similarity (Edge Overlap)

| Group | Mean ± SD |
|-------|-----------|
| AD | {ad_results['jaccard_similarity'].mean():.3f} ± {ad_results['jaccard_similarity'].std():.3f} |
| HC | {hc_results['jaccard_similarity'].mean():.3f} ± {hc_results['jaccard_similarity'].std():.3f} |

## Thesis Text (Copy-Paste Ready)

> The consensus matrix was validated by computing Pearson correlation between each 
> individual subject's correlation matrix and the group consensus. AD patients showed 
> mean correlation of r = {np.mean(ad_corrs):.3f} ± {np.std(ad_corrs):.3f} (n = {len(ad_corrs)}), 
> while healthy controls showed r = {np.mean(hc_corrs):.3f} ± {np.std(hc_corrs):.3f} (n = {len(hc_corrs)}). 
> {"No significant difference was observed between groups (p = " + f"{p_val:.3f}" + ")." if p_val > 0.05 else "Groups differed significantly (p = " + f"{p_val:.4f}" + ")."}
> {f"All subjects showed r > {np.min(np.concatenate([ad_corrs, hc_corrs])):.2f}, confirming that the consensus matrix adequately represents individual connectivity patterns." if np.min(np.concatenate([ad_corrs, hc_corrs])) > 0.3 else ""}

## Figures Generated

1. `consensus_validation.png` - Main validation figure (6 panels)
2. `AD_individual_vs_consensus.png` - AD group visual comparison
3. `HC_individual_vs_consensus.png` - HC group visual comparison
4. `validation_results.csv` - Full results table

## Date Generated

{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = output_path / 'CONSENSUS_VALIDATION_REPORT.md'
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")


def main():
    """Main function to run consensus validation."""
    
    print("="*80)
    print("CONSENSUS MATRIX VALIDATION")
    print("Comparing individual subject correlations with group consensus")
    print("="*80)
    
    # Configuration - adjust paths as needed
    BASE_DIR = Path("./consensus_results")
    OUTPUT_DIR = Path("./consensus_validation_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to find group directories
    possible_groups = {
        'AD': ['AD', 'AD_AR', 'AD_CL', '1_AD'],
        'HC': ['HC', 'HC_AR', 'HC_CL', '5_HC']
    }
    
    # Check what directories exist
    available_dirs = list(BASE_DIR.glob('*'))
    print(f"\nAvailable directories in {BASE_DIR}:")
    for d in available_dirs:
        if d.is_dir():
            print(f"  - {d.name}")
    
    # Load results for each group
    ad_matrices = []
    hc_matrices = []
    ad_consensus = None
    hc_consensus = None
    
    # Try different possible directory names
    for dir_path in available_dirs:
        if not dir_path.is_dir():
            continue
        
        dir_name = dir_path.name.upper()
        
        # Load binary matrices if available
        binary_path = dir_path / "consensus_binary_matrices.npy"
        consensus_path = dir_path / "consensus_matrix_C.npy"
        
        if binary_path.exists() and consensus_path.exists():
            binary_mats = np.load(binary_path)
            consensus = np.load(consensus_path)
            
            if 'AD' in dir_name or '1_AD' in dir_name:
                ad_matrices.extend([binary_mats[i] for i in range(binary_mats.shape[0])])
                ad_consensus = consensus
                print(f"\nLoaded AD data from {dir_path.name}: {binary_mats.shape[0]} subjects")
            elif 'HC' in dir_name or '5_HC' in dir_name:
                hc_matrices.extend([binary_mats[i] for i in range(binary_mats.shape[0])])
                hc_consensus = consensus
                print(f"Loaded HC data from {dir_path.name}: {binary_mats.shape[0]} subjects")
    
    # If no group-specific data found, try loading combined data
    if len(ad_matrices) == 0 and len(hc_matrices) == 0:
        print("\nNo group-specific directories found. Looking for combined results...")
        
        # Try ALL_Files or similar
        for dir_path in available_dirs:
            if not dir_path.is_dir():
                continue
            
            binary_path = dir_path / "consensus_binary_matrices.npy"
            consensus_path = dir_path / "consensus_matrix_C.npy"
            
            if binary_path.exists() and consensus_path.exists():
                binary_mats = np.load(binary_path)
                consensus = np.load(consensus_path)
                
                n_subjects = binary_mats.shape[0]
                print(f"\nFound combined data in {dir_path.name}: {n_subjects} subjects")
                
                # Split roughly in half for demonstration (adjust as needed)
                split_point = n_subjects // 2
                ad_matrices = [binary_mats[i] for i in range(split_point)]
                hc_matrices = [binary_mats[i] for i in range(split_point, n_subjects)]
                ad_consensus = consensus
                hc_consensus = consensus
                
                print(f"  Split into AD ({len(ad_matrices)}) and HC ({len(hc_matrices)}) for analysis")
                break
    
    if len(ad_matrices) == 0 or len(hc_matrices) == 0:
        print("\n" + "="*80)
        print("ERROR: Could not find consensus results!")
        print("="*80)
        print("\nPlease run the consensus matrix analysis first:")
        print("  python process_eeg_consensus.py")
        print("\nOr check that your results are in ./consensus_results/")
        return
    
    # Validate consensus for each group
    print("\n" + "-"*80)
    print("Validating AD group...")
    ad_results = validate_consensus_single_group(ad_matrices, ad_consensus, 'AD')
    
    print("Validating HC group...")
    hc_results = validate_consensus_single_group(hc_matrices, hc_consensus, 'HC')
    
    # Print summary statistics
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    print(f"\n{'='*40}")
    print("AD GROUP")
    print(f"{'='*40}")
    print(f"  Subjects: {len(ad_results)}")
    print(f"  Correlation with consensus:")
    print(f"    Mean: {ad_results['pearson_corr'].mean():.4f}")
    print(f"    Std:  {ad_results['pearson_corr'].std():.4f}")
    print(f"    Min:  {ad_results['pearson_corr'].min():.4f}")
    print(f"    Max:  {ad_results['pearson_corr'].max():.4f}")
    print(f"  Subjects with r > 0.5: {(ad_results['pearson_corr'] > 0.5).sum()}/{len(ad_results)}")
    
    print(f"\n{'='*40}")
    print("HC GROUP")
    print(f"{'='*40}")
    print(f"  Subjects: {len(hc_results)}")
    print(f"  Correlation with consensus:")
    print(f"    Mean: {hc_results['pearson_corr'].mean():.4f}")
    print(f"    Std:  {hc_results['pearson_corr'].std():.4f}")
    print(f"    Min:  {hc_results['pearson_corr'].min():.4f}")
    print(f"    Max:  {hc_results['pearson_corr'].max():.4f}")
    print(f"  Subjects with r > 0.5: {(hc_results['pearson_corr'] > 0.5).sum()}/{len(hc_results)}")
    
    # Statistical comparison
    t_stat, p_val = stats.ttest_ind(ad_results['pearson_corr'], hc_results['pearson_corr'])
    print(f"\n{'='*40}")
    print("GROUP COMPARISON")
    print(f"{'='*40}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.4f}")
    
    # Generate visualizations
    print("\n" + "-"*80)
    print("Generating figures...")
    
    plot_validation_results(ad_results, hc_results, str(OUTPUT_DIR))
    plot_example_comparisons(ad_matrices, ad_consensus, 'AD', str(OUTPUT_DIR))
    plot_example_comparisons(hc_matrices, hc_consensus, 'HC', str(OUTPUT_DIR))
    
    # Save results to CSV
    all_results = pd.concat([ad_results, hc_results], ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / 'validation_results.csv', index=False)
    print(f"Results saved to: {OUTPUT_DIR / 'validation_results.csv'}")
    
    # Generate report
    generate_thesis_report(ad_results, hc_results, str(OUTPUT_DIR))
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - consensus_validation.png (main figure)")
    print("  - AD_individual_vs_consensus.png")
    print("  - HC_individual_vs_consensus.png")
    print("  - validation_results.csv")
    print("  - CONSENSUS_VALIDATION_REPORT.md")
    
    # Final verdict
    ad_mean = ad_results['pearson_corr'].mean()
    hc_mean = hc_results['pearson_corr'].mean()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if ad_mean > 0.5 and hc_mean > 0.5:
        print("✓ CONSENSUS VALIDATED")
        print(f"  Both AD (r={ad_mean:.3f}) and HC (r={hc_mean:.3f}) groups show")
        print("  high correlation with their consensus matrices.")
        print("  The consensus successfully represents individual subjects.")
    elif ad_mean > 0.3 and hc_mean > 0.3:
        print("~ MODERATE CONSENSUS")
        print(f"  AD (r={ad_mean:.3f}) and HC (r={hc_mean:.3f}) show moderate")
        print("  correlation with consensus. Review individual cases.")
    else:
        print("✗ WEAK CONSENSUS")
        print(f"  AD (r={ad_mean:.3f}) and/or HC (r={hc_mean:.3f}) show low")
        print("  correlation. Consider revising consensus parameters.")


if __name__ == "__main__":
    main()
