"""
=============================================================================
COMPARING CONNECTIVITY MATRICES - EACH SUBJECT vs SAVED CONSENSUS
=============================================================================

This script compares each individual subject against a PRE-COMPUTED consensus 
matrix loaded from a saved .npy file.

KEY FEATURES: 
- Calculates the SPARSITY/DENSITY of the consensus matrix
- Matches density by keeping only the TOP N edges for fair comparison
- Uses TWO complementary metrics:
  * PEARSON CORRELATION - weight similarity
  * JACCARD SIMILARITY - binary edge overlap

OUTPUT FIGURES:
- figure1_consensus_overview.png  - Consensus matrix visualization
- figure2_pearson_analysis.png    - Pearson correlation analysis  
- figure3_jaccard_analysis.png    - Jaccard similarity analysis
- figure4_combined_summary.png    - Combined publication-ready summary

RUN: python compare_subject_to_saved_consensus.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# FIGURE STYLE SETTINGS
# =============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'AD': '#E74C3C',       # Red
    'HC': '#3498DB',       # Blue
    'consensus': '#F39C12', # Orange
    'shared': '#27AE60',   # Green
    'mean_line': '#2C3E50', # Dark
}

print("="*70)
print("SUBJECT vs CONSENSUS COMPARISON")
print("Metrics: PEARSON CORRELATION + JACCARD SIMILARITY")
print("="*70)

# =============================================================================
# PATHS
# =============================================================================
CONSENSUS_MATRIX_PATH = "/home/muhibt/GPVAR/consensus_results/ALL_Files/consensus_distance_graph.npy"
OUTPUT_FOLDER = Path("subject_vs_consensus_results")

# File lists
AD_FILES = [
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
]

HC_FILES = [
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_subject_id(filepath):
    """Extract subject ID from filepath."""
    for part in Path(filepath).parts:
        if part.startswith('sub-'):
            return part
    return Path(filepath).stem


def load_eeg_data(filepath):
    """Load EEG data from .set file."""
    try:
        import mne
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        if raw.get_montage() is None:
            try:
                from mne.channels import make_standard_montage
                raw.set_montage(make_standard_montage("biosemi128"), on_missing='warn')
            except:
                pass
        return raw.get_data()
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def compute_correlation_matrix(data):
    """Compute absolute Pearson correlation matrix."""
    corr = np.abs(np.corrcoef(data))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    return corr


def calculate_sparsity(matrix):
    """Calculate sparsity info for a matrix."""
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    values = matrix[triu_idx]
    n_edges = np.sum(values > 0)
    total = len(values)
    return {
        'n_channels': n,
        'total_possible_edges': total,
        'n_edges': int(n_edges),
        'density': n_edges / total,
        'sparsity': 1 - n_edges / total,
        'mean_weight': np.mean(values[values > 0]) if n_edges > 0 else 0,
        'max_weight': np.max(values),
        'min_nonzero': np.min(values[values > 0]) if n_edges > 0 else 0
    }


def threshold_to_density(matrix, n_edges):
    """Keep top N edges as binary matrix."""
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    values = matrix[triu_idx]
    
    if n_edges >= len(values):
        threshold = 0
    else:
        threshold = np.sort(values)[::-1][n_edges - 1]
    
    binary = np.zeros_like(matrix)
    binary[matrix >= threshold] = 1
    np.fill_diagonal(binary, 0)
    return np.maximum(binary, binary.T)


def cohens_d(g1, g2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1-1)*g1.var() + (n2-1)*g2.var()) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0


def compare_to_consensus(subject_matrix, consensus_matrix, n_edges):
    """Compare subject to consensus with density matching."""
    n = subject_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    cons_values = consensus_matrix[triu_idx]
    cons_binary = (cons_values > 0).astype(int)
    
    subj_binary = threshold_to_density(subject_matrix, n_edges)[triu_idx].astype(int)
    subj_values = subject_matrix[triu_idx]
    
    # Pearson correlation
    r, p = stats.pearsonr(subj_values, cons_values)
    
    # Jaccard similarity
    intersection = np.sum((subj_binary == 1) & (cons_binary == 1))
    union = np.sum((subj_binary == 1) | (cons_binary == 1))
    jaccard = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = 2 * intersection / (np.sum(subj_binary) + np.sum(cons_binary))
    
    return {
        'Pearson_r': r, 'Pearson_p': p,
        'Jaccard': jaccard, 'Dice': dice,
        'N_Shared': int(intersection),
        'N_Subject': int(np.sum(subj_binary)),
        'N_Consensus': int(np.sum(cons_binary))
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_figure1(consensus_matrix, sparsity_info, best_matrix, worst_matrix, 
                   best_info, worst_info, n_edges, output_folder):
    """Figure 1: Consensus Matrix Overview"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    n = consensus_matrix.shape[0]
    
    # 1A: Consensus weighted
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(consensus_matrix[consensus_matrix > 0], 95) if np.any(consensus_matrix > 0) else 1
    im1 = ax1.imshow(consensus_matrix, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title(f'(A) Consensus Matrix\n{n_edges:,} edges, {sparsity_info["density"]*100:.1f}% density', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Channel')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('Weight')
    
    # 1B: Consensus binary
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow((consensus_matrix > 0).astype(float), cmap='Blues', vmin=0, vmax=1)
    ax2.set_title(f'(B) Consensus Binary\n{n_edges:,} edges', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Channel')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 1C: Sparsity info box
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    info_text = f"""
    ══════════════════════════════
         CONSENSUS PROPERTIES
    ══════════════════════════════
    
    Matrix Size:    {n} × {n}
    
    Total Possible: {sparsity_info['total_possible_edges']:,}
    
    Actual Edges:   {n_edges:,}
    
    Density:        {sparsity_info['density']*100:.2f}%
    
    ──────────────────────────────
    Edge Weights:
      Mean: {sparsity_info['mean_weight']:.4f}
      Max:  {sparsity_info['max_weight']:.4f}
      Min:  {sparsity_info['min_nonzero']:.4f}
    ══════════════════════════════
    """
    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='steelblue', alpha=0.9))
    
    # 1D: Best subject
    ax4 = fig.add_subplot(gs[1, 0])
    best_bin = threshold_to_density(best_matrix, n_edges)
    im4 = ax4.imshow(best_bin, cmap='Greens', vmin=0, vmax=1)
    ax4.set_title(f'(D) Best Match: {best_info["id"]}\nPearson={best_info["pearson"]:.3f}, Jaccard={best_info["jaccard"]:.3f}', 
                  fontweight='bold', fontsize=11, color='darkgreen')
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Channel')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 1E: Worst subject
    ax5 = fig.add_subplot(gs[1, 1])
    worst_bin = threshold_to_density(worst_matrix, n_edges)
    im5 = ax5.imshow(worst_bin, cmap='Reds', vmin=0, vmax=1)
    ax5.set_title(f'(E) Worst Match: {worst_info["id"]}\nPearson={worst_info["pearson"]:.3f}, Jaccard={worst_info["jaccard"]:.3f}', 
                  fontweight='bold', fontsize=11, color='darkred')
    ax5.set_xlabel('Channel')
    ax5.set_ylabel('Channel')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 1F: Edge overlap (RGB)
    ax6 = fig.add_subplot(gs[1, 2])
    cons_bin = consensus_matrix > 0
    best_b = best_bin > 0
    
    overlap = np.zeros((n, n, 3))
    shared = cons_bin & best_b
    cons_only = cons_bin & ~best_b
    subj_only = ~cons_bin & best_b
    
    overlap[:,:,1] = shared * 0.8   # Green = shared
    overlap[:,:,0] = cons_only * 0.8  # Red = consensus only
    overlap[:,:,2] = subj_only * 0.8  # Blue = subject only
    
    ax6.imshow(overlap)
    ax6.set_title('(F) Edge Overlap (Best Subject)\nGreen=Shared, Red=Consensus, Blue=Subject', 
                  fontweight='bold', fontsize=11)
    ax6.set_xlabel('Channel')
    ax6.set_ylabel('Channel')
    
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label=f'Shared ({np.sum(shared)//2:,})'),
        Patch(facecolor='red', alpha=0.8, label=f'Consensus only ({np.sum(cons_only)//2:,})'),
        Patch(facecolor='blue', alpha=0.8, label=f'Subject only ({np.sum(subj_only)//2:,})')
    ]
    ax6.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Figure 1: Consensus Matrix Overview', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_folder / 'figure1_consensus_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ figure1_consensus_overview.png")


def create_figure2(df, ad_df, hc_df, output_folder):
    """Figure 2: Pearson Correlation Analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    t_stat, p_val = stats.ttest_ind(ad_df['Pearson_r'], hc_df['Pearson_r'])
    d = cohens_d(ad_df['Pearson_r'], hc_df['Pearson_r'])
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    
    # 2A: Boxplot
    ax1 = fig.add_subplot(gs[0, 0])
    bp = ax1.boxplot([ad_df['Pearson_r'], hc_df['Pearson_r']], labels=['AD', 'HC'],
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['AD'])
    bp['boxes'][1].set_facecolor(COLORS['HC'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    # Add points
    for i, (data, color) in enumerate([(ad_df['Pearson_r'], COLORS['AD']), 
                                        (hc_df['Pearson_r'], COLORS['HC'])]):
        x = np.random.normal(i+1, 0.06, size=len(data))
        ax1.scatter(x, data, alpha=0.6, color=color, s=40, edgecolor='white', zorder=3)
    
    # Significance bar
    y_max = max(ad_df['Pearson_r'].max(), hc_df['Pearson_r'].max())
    ax1.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', lw=1.5)
    ax1.text(1.5, y_max + 0.03, f'{sig} (p={p_val:.4f})', ha='center', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax1.set_title(f'(A) Group Comparison\nCohen\'s d = {d:.3f}', fontweight='bold', fontsize=12)
    
    # 2B: Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(df['Pearson_r'].min() - 0.03, df['Pearson_r'].max() + 0.03, 20)
    ax2.hist(ad_df['Pearson_r'], bins=bins, alpha=0.6, color=COLORS['AD'], 
             label=f'AD (n={len(ad_df)})', edgecolor='white')
    ax2.hist(hc_df['Pearson_r'], bins=bins, alpha=0.6, color=COLORS['HC'], 
             label=f'HC (n={len(hc_df)})', edgecolor='white')
    ax2.axvline(ad_df['Pearson_r'].mean(), color=COLORS['AD'], ls='--', lw=2.5)
    ax2.axvline(hc_df['Pearson_r'].mean(), color=COLORS['HC'], ls='--', lw=2.5)
    ax2.set_xlabel('Pearson r', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('(B) Distribution', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left')
    
    # 2C: Violin
    ax3 = fig.add_subplot(gs[0, 2])
    parts = ax3.violinplot([ad_df['Pearson_r'], hc_df['Pearson_r']], positions=[1, 2],
                           showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['AD'], COLORS['HC']][i])
        pc.set_alpha(0.7)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['AD', 'HC'])
    ax3.set_ylabel('Pearson r', fontsize=12)
    ax3.set_title('(C) Violin Plot', fontweight='bold', fontsize=12)
    
    # 2D: All subjects ranked
    ax4 = fig.add_subplot(gs[1, :])
    df_sorted = df.sort_values('Pearson_r', ascending=False).reset_index(drop=True)
    colors = [COLORS['AD'] if g == 'AD' else COLORS['HC'] for g in df_sorted['Group']]
    ax4.bar(range(len(df_sorted)), df_sorted['Pearson_r'], color=colors, alpha=0.8, edgecolor='white')
    ax4.axhline(df['Pearson_r'].mean(), color=COLORS['mean_line'], ls='-', lw=2, 
                label=f'Mean: {df["Pearson_r"].mean():.3f}')
    ax4.axhline(ad_df['Pearson_r'].mean(), color=COLORS['AD'], ls='--', lw=2)
    ax4.axhline(hc_df['Pearson_r'].mean(), color=COLORS['HC'], ls='--', lw=2)
    ax4.set_xlabel('Subject Rank', fontsize=12)
    ax4.set_ylabel('Pearson r', fontsize=12)
    ax4.set_title('(D) All Subjects Ranked by Pearson Correlation', fontweight='bold', fontsize=12)
    ax4.set_xlim([-1, len(df_sorted)])
    
    legend_elements = [Patch(facecolor=COLORS['AD'], alpha=0.8, label='AD'),
                       Patch(facecolor=COLORS['HC'], alpha=0.8, label='HC')]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Figure 2: Pearson Correlation Analysis\n(Weight Similarity)', 
                 fontsize=15, fontweight='bold', y=0.99)
    
    # Stats annotation
    fig.text(0.5, 0.01, f't = {t_stat:.3f}, p = {p_val:.4f}, Cohen\'s d = {d:.3f}',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_folder / 'figure2_pearson_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ figure2_pearson_analysis.png")


def create_figure3(df, ad_df, hc_df, n_edges, output_folder):
    """Figure 3: Jaccard Similarity Analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    t_stat, p_val = stats.ttest_ind(ad_df['Jaccard'], hc_df['Jaccard'])
    d = cohens_d(ad_df['Jaccard'], hc_df['Jaccard'])
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    
    # 3A: Boxplot
    ax1 = fig.add_subplot(gs[0, 0])
    bp = ax1.boxplot([ad_df['Jaccard'], hc_df['Jaccard']], labels=['AD', 'HC'],
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['AD'])
    bp['boxes'][1].set_facecolor(COLORS['HC'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_df['Jaccard'], COLORS['AD']), 
                                        (hc_df['Jaccard'], COLORS['HC'])]):
        x = np.random.normal(i+1, 0.06, size=len(data))
        ax1.scatter(x, data, alpha=0.6, color=color, s=40, edgecolor='white', zorder=3)
    
    y_max = max(ad_df['Jaccard'].max(), hc_df['Jaccard'].max())
    ax1.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', lw=1.5)
    ax1.text(1.5, y_max + 0.03, f'{sig} (p={p_val:.4f})', ha='center', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Jaccard Similarity', fontsize=12)
    ax1.set_title(f'(A) Group Comparison\nCohen\'s d = {d:.3f}', fontweight='bold', fontsize=12)
    
    # 3B: Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(df['Jaccard'].min() - 0.02, df['Jaccard'].max() + 0.02, 20)
    ax2.hist(ad_df['Jaccard'], bins=bins, alpha=0.6, color=COLORS['AD'], 
             label=f'AD (n={len(ad_df)})', edgecolor='white')
    ax2.hist(hc_df['Jaccard'], bins=bins, alpha=0.6, color=COLORS['HC'], 
             label=f'HC (n={len(hc_df)})', edgecolor='white')
    ax2.axvline(ad_df['Jaccard'].mean(), color=COLORS['AD'], ls='--', lw=2.5)
    ax2.axvline(hc_df['Jaccard'].mean(), color=COLORS['HC'], ls='--', lw=2.5)
    ax2.set_xlabel('Jaccard', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('(B) Distribution', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left')
    
    # 3C: Shared edges histogram
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(df['N_Shared'].min() - 20, df['N_Shared'].max() + 20, 20)
    ax3.hist(ad_df['N_Shared'], bins=bins, alpha=0.6, color=COLORS['AD'], edgecolor='white')
    ax3.hist(hc_df['N_Shared'], bins=bins, alpha=0.6, color=COLORS['HC'], edgecolor='white')
    ax3.axvline(n_edges, color='black', ls='-', lw=3, label=f'Consensus: {n_edges:,}')
    ax3.set_xlabel('Shared Edges', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('(C) Shared Edges Distribution', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left')
    
    # 3D: All subjects ranked
    ax4 = fig.add_subplot(gs[1, :])
    df_sorted = df.sort_values('Jaccard', ascending=False).reset_index(drop=True)
    colors = [COLORS['AD'] if g == 'AD' else COLORS['HC'] for g in df_sorted['Group']]
    ax4.bar(range(len(df_sorted)), df_sorted['Jaccard'], color=colors, alpha=0.8, edgecolor='white')
    ax4.axhline(df['Jaccard'].mean(), color=COLORS['mean_line'], ls='-', lw=2,
                label=f'Mean: {df["Jaccard"].mean():.3f}')
    ax4.axhline(ad_df['Jaccard'].mean(), color=COLORS['AD'], ls='--', lw=2)
    ax4.axhline(hc_df['Jaccard'].mean(), color=COLORS['HC'], ls='--', lw=2)
    ax4.set_xlabel('Subject Rank', fontsize=12)
    ax4.set_ylabel('Jaccard Similarity', fontsize=12)
    ax4.set_title('(D) All Subjects Ranked by Jaccard Similarity', fontweight='bold', fontsize=12)
    ax4.set_xlim([-1, len(df_sorted)])
    
    legend_elements = [Patch(facecolor=COLORS['AD'], alpha=0.8, label='AD'),
                       Patch(facecolor=COLORS['HC'], alpha=0.8, label='HC')]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Figure 3: Jaccard Similarity Analysis\n(Binary Edge Overlap: J = |A∩B| / |A∪B|)', 
                 fontsize=15, fontweight='bold', y=0.99)
    
    fig.text(0.5, 0.01, f't = {t_stat:.3f}, p = {p_val:.4f}, Cohen\'s d = {d:.3f}',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_folder / 'figure3_jaccard_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ figure3_jaccard_analysis.png")


def create_figure4(df, ad_df, hc_df, sparsity_info, n_edges, output_folder):
    """Figure 4: Combined Summary (Publication Ready)"""
    
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Stats
    t_p, p_p = stats.ttest_ind(ad_df['Pearson_r'], hc_df['Pearson_r'])
    t_j, p_j = stats.ttest_ind(ad_df['Jaccard'], hc_df['Jaccard'])
    d_p = cohens_d(ad_df['Pearson_r'], hc_df['Pearson_r'])
    d_j = cohens_d(ad_df['Jaccard'], hc_df['Jaccard'])
    sig_p = '***' if p_p < 0.001 else '**' if p_p < 0.01 else '*' if p_p < 0.05 else 'ns'
    sig_j = '***' if p_j < 0.001 else '**' if p_j < 0.01 else '*' if p_j < 0.05 else 'ns'
    
    n_ad, n_hc = len(ad_df), len(hc_df)
    
    # 4A: Pearson boxplot
    ax1 = fig.add_subplot(gs[0, 0])
    bp1 = ax1.boxplot([ad_df['Pearson_r'], hc_df['Pearson_r']], labels=['AD', 'HC'],
                      patch_artist=True, widths=0.6)
    bp1['boxes'][0].set_facecolor(COLORS['AD'])
    bp1['boxes'][1].set_facecolor(COLORS['HC'])
    for box in bp1['boxes']:
        box.set_alpha(0.7)
    for i, (data, color) in enumerate([(ad_df['Pearson_r'], COLORS['AD']), 
                                        (hc_df['Pearson_r'], COLORS['HC'])]):
        x = np.random.normal(i+1, 0.05, size=len(data))
        ax1.scatter(x, data, alpha=0.5, color=color, s=35, edgecolor='white', zorder=3)
    y_max = max(ad_df['Pearson_r'].max(), hc_df['Pearson_r'].max())
    ax1.plot([1, 2], [y_max + 0.015, y_max + 0.015], 'k-', lw=1.5)
    ax1.text(1.5, y_max + 0.02, sig_p, ha='center', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pearson r', fontsize=12)
    ax1.set_title(f'(A) Pearson Correlation\np = {p_p:.4f}', fontweight='bold')
    
    # 4B: Jaccard boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    bp2 = ax2.boxplot([ad_df['Jaccard'], hc_df['Jaccard']], labels=['AD', 'HC'],
                      patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor(COLORS['AD'])
    bp2['boxes'][1].set_facecolor(COLORS['HC'])
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    for i, (data, color) in enumerate([(ad_df['Jaccard'], COLORS['AD']), 
                                        (hc_df['Jaccard'], COLORS['HC'])]):
        x = np.random.normal(i+1, 0.05, size=len(data))
        ax2.scatter(x, data, alpha=0.5, color=color, s=35, edgecolor='white', zorder=3)
    y_max = max(ad_df['Jaccard'].max(), hc_df['Jaccard'].max())
    ax2.plot([1, 2], [y_max + 0.015, y_max + 0.015], 'k-', lw=1.5)
    ax2.text(1.5, y_max + 0.02, sig_j, ha='center', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Jaccard', fontsize=12)
    ax2.set_title(f'(B) Jaccard Similarity\np = {p_j:.4f}', fontweight='bold')
    
    # 4C: Scatter Pearson vs Jaccard
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(ad_df['Pearson_r'], ad_df['Jaccard'], c=COLORS['AD'], 
                label=f'AD (n={n_ad})', alpha=0.7, s=50, edgecolor='white')
    ax3.scatter(hc_df['Pearson_r'], hc_df['Jaccard'], c=COLORS['HC'], 
                label=f'HC (n={n_hc})', alpha=0.7, s=50, edgecolor='white')
    r_pj, _ = stats.pearsonr(df['Pearson_r'], df['Jaccard'])
    ax3.set_xlabel('Pearson r', fontsize=12)
    ax3.set_ylabel('Jaccard', fontsize=12)
    ax3.set_title(f'(C) Metrics Correlation\nr = {r_pj:.3f}', fontweight='bold')
    ax3.legend(loc='lower right')
    
    # 4D: Violin comparison
    ax4 = fig.add_subplot(gs[1, 0])
    positions = [1, 2, 4, 5]
    data_v = [ad_df['Pearson_r'], hc_df['Pearson_r'], ad_df['Jaccard'], hc_df['Jaccard']]
    parts = ax4.violinplot(data_v, positions=positions, showmeans=True, showmedians=True, widths=0.7)
    colors_v = [COLORS['AD'], COLORS['HC'], COLORS['AD'], COLORS['HC']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.7)
    ax4.set_xticks([1.5, 4.5])
    ax4.set_xticklabels(['Pearson r', 'Jaccard'])
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('(D) Distribution Comparison', fontweight='bold')
    
    # 4E: Bar chart means
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(2)
    width = 0.35
    ad_means = [ad_df['Pearson_r'].mean(), ad_df['Jaccard'].mean()]
    hc_means = [hc_df['Pearson_r'].mean(), hc_df['Jaccard'].mean()]
    ad_stds = [ad_df['Pearson_r'].std(), ad_df['Jaccard'].std()]
    hc_stds = [hc_df['Pearson_r'].std(), hc_df['Jaccard'].std()]
    ax5.bar(x - width/2, ad_means, width, yerr=ad_stds, label='AD', 
            color=COLORS['AD'], alpha=0.8, capsize=5)
    ax5.bar(x + width/2, hc_means, width, yerr=hc_stds, label='HC', 
            color=COLORS['HC'], alpha=0.8, capsize=5)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Pearson r', 'Jaccard'])
    ax5.set_ylabel('Mean ± SD', fontsize=12)
    ax5.set_title('(E) Group Means', fontweight='bold')
    ax5.legend()
    
    # Significance stars
    for i, (pv, ym) in enumerate([(p_p, max(ad_means[0]+ad_stds[0], hc_means[0]+hc_stds[0])),
                                   (p_j, max(ad_means[1]+ad_stds[1], hc_means[1]+hc_stds[1]))]):
        s = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
        ax5.text(i, ym + 0.02, s, ha='center', fontsize=12, fontweight='bold')
    
    # 4F: Shared edges
    ax6 = fig.add_subplot(gs[1, 2])
    bins = np.linspace(df['N_Shared'].min() - 10, df['N_Shared'].max() + 10, 15)
    ax6.hist(ad_df['N_Shared'], bins=bins, alpha=0.6, color=COLORS['AD'], 
             label=f'AD (μ={ad_df["N_Shared"].mean():.0f})', edgecolor='white')
    ax6.hist(hc_df['N_Shared'], bins=bins, alpha=0.6, color=COLORS['HC'], 
             label=f'HC (μ={hc_df["N_Shared"].mean():.0f})', edgecolor='white')
    ax6.axvline(n_edges, color='black', ls='-', lw=2.5, label=f'Consensus: {n_edges:,}')
    ax6.set_xlabel('Shared Edges', fontsize=12)
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_title('(F) Shared Edges', fontweight='bold')
    ax6.legend(fontsize=9)
    
    # 4G: Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    SUBJECT vs CONSENSUS: SUMMARY STATISTICS                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                          ║
║   CONSENSUS: {n_edges:,} edges ({sparsity_info['density']*100:.1f}% density)  |  SUBJECTS: {len(df)} total ({n_ad} AD + {n_hc} HC)                                     ║
║   Method: Distance-dependent consensus with natural sparsity  |  Density Matching: Top {n_edges:,} edges per subject                ║
║                                                                                                                          ║
╠═══════════════════════════════════════════════╦══════════════════════════════════════════════════════════════════════════╣
║   PEARSON CORRELATION (Weight Similarity)     ║   JACCARD SIMILARITY (Binary Edge Overlap)                              ║
╠═══════════════════════════════════════════════╬══════════════════════════════════════════════════════════════════════════╣
║   AD:  {ad_df['Pearson_r'].mean():.3f} ± {ad_df['Pearson_r'].std():.3f}                              ║   AD:  {ad_df['Jaccard'].mean():.3f} ± {ad_df['Jaccard'].std():.3f}    (Shared: {ad_df['N_Shared'].mean():.0f} edges)                       ║
║   HC:  {hc_df['Pearson_r'].mean():.3f} ± {hc_df['Pearson_r'].std():.3f}                              ║   HC:  {hc_df['Jaccard'].mean():.3f} ± {hc_df['Jaccard'].std():.3f}    (Shared: {hc_df['N_Shared'].mean():.0f} edges)                       ║
║   p = {p_p:.4f} {sig_p}  |  Cohen's d = {d_p:.3f}           ║   p = {p_j:.4f} {sig_j}  |  Cohen's d = {d_j:.3f}                                           ║
╠═══════════════════════════════════════════════╩══════════════════════════════════════════════════════════════════════════╣
║   INTERPRETATION:                                                                                                        ║
║   • Pearson r: Measures how similar CONNECTION STRENGTHS are to the group consensus                                     ║
║   • Jaccard:   Measures if the SAME CONNECTIONS are present (binary edge overlap after density matching)               ║
║   • Both metrics are complementary: Pearson focuses on weights, Jaccard on topology                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    ax7.text(0.5, 0.5, summary, transform=ax7.transAxes, fontsize=9.5,
             ha='center', va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.95))
    
    plt.suptitle('Figure 4: Combined Summary - Pearson Correlation + Jaccard Similarity', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(output_folder / 'figure4_combined_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ figure4_combined_summary.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create output folder
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output folder: {output_folder.absolute()}")
    
    # Load consensus
    print("\n" + "="*50)
    print("LOADING CONSENSUS MATRIX")
    print("="*50)
    
    if Path(CONSENSUS_MATRIX_PATH).exists():
        consensus = np.load(CONSENSUS_MATRIX_PATH)
        use_synthetic = False
        print(f"✓ Loaded: {CONSENSUS_MATRIX_PATH}")
    else:
        print(f"✗ Not found, using synthetic data")
        use_synthetic = True
        np.random.seed(42)
        n = 128
        consensus = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = abs(i - j)
                prob = 0.6 * np.exp(-d/10) if d < 20 else 0.05
                if np.random.rand() < prob:
                    w = np.random.rand() * 0.5 + 0.1
                    consensus[i, j] = consensus[j, i] = w
    
    n_channels = consensus.shape[0]
    sparsity_info = calculate_sparsity(consensus)
    n_edges = sparsity_info['n_edges']
    
    print(f"  Shape: {consensus.shape}")
    print(f"  Edges: {n_edges:,} / {sparsity_info['total_possible_edges']:,}")
    print(f"  Density: {sparsity_info['density']*100:.2f}%")
    
    # Save consensus and sparsity info
    np.save(output_folder / 'consensus_matrix.npy', consensus)
    with open(output_folder / 'sparsity_info.txt', 'w') as f:
        for k, v in sparsity_info.items():
            f.write(f"{k}: {v}\n")
    
    # Load subjects
    print("\n" + "="*50)
    print("LOADING SUBJECTS")
    print("="*50)
    
    all_matrices, subject_ids, group_labels = [], [], []
    
    real_files = any(Path(f).exists() for f in AD_FILES + HC_FILES)
    
    if real_files and not use_synthetic:
        for f in AD_FILES:
            if Path(f).exists():
                data = load_eeg_data(f)
                if data is not None:
                    m = compute_correlation_matrix(data)
                    if m.shape[0] == n_channels:
                        all_matrices.append(m)
                        subject_ids.append(extract_subject_id(f))
                        group_labels.append('AD')
        
        for f in HC_FILES:
            if Path(f).exists():
                data = load_eeg_data(f)
                if data is not None:
                    m = compute_correlation_matrix(data)
                    if m.shape[0] == n_channels:
                        all_matrices.append(m)
                        subject_ids.append(extract_subject_id(f))
                        group_labels.append('HC')
    
    if len(all_matrices) == 0:
        print("  Using synthetic subjects...")
        np.random.seed(42)
        base = np.random.rand(n_channels, n_channels) * 0.3
        base = (base + base.T) / 2
        np.fill_diagonal(base, 0)
        
        for i in range(35):
            m = np.clip(base + np.random.randn(n_channels, n_channels) * 0.08, 0, 1)
            m = (m + m.T) / 2
            np.fill_diagonal(m, 0)
            all_matrices.append(m)
            subject_ids.append(f'sub-300{i+1:02d}')
            group_labels.append('AD')
        
        for i in range(31):
            m = np.clip(base + np.random.randn(n_channels, n_channels) * 0.08, 0, 1)
            m = (m + m.T) / 2
            np.fill_diagonal(m, 0)
            all_matrices.append(m)
            subject_ids.append(f'sub-100{i+1:02d}')
            group_labels.append('HC')
    
    n_ad = sum(1 for g in group_labels if g == 'AD')
    n_hc = sum(1 for g in group_labels if g == 'HC')
    print(f"  Total: {len(all_matrices)} ({n_ad} AD + {n_hc} HC)")
    
    # Compare subjects
    print("\n" + "="*50)
    print("COMPUTING SIMILARITY METRICS")
    print("="*50)
    
    results = []
    for i, (m, sid, g) in enumerate(zip(all_matrices, subject_ids, group_labels)):
        comp = compare_to_consensus(m, consensus, n_edges)
        results.append({'Subject_ID': sid, 'Group': g, **comp})
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(all_matrices)}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('Pearson_r', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    
    # Save CSV
    df.to_csv(output_folder / 'subject_vs_saved_consensus.csv', index=False)
    print(f"  ✓ Saved CSV")
    
    # Save NPY files
    np.save(output_folder / 'all_subject_matrices.npy', np.stack(all_matrices))
    np.save(output_folder / 'subject_ids.npy', np.array(subject_ids))
    np.save(output_folder / 'group_labels.npy', np.array(group_labels))
    
    ad_df = df[df['Group'] == 'AD']
    hc_df = df[df['Group'] == 'HC']
    
    # Find best/worst
    best_idx = df['Pearson_r'].idxmax()
    worst_idx = df['Pearson_r'].idxmin()
    best_sid = df.loc[best_idx, 'Subject_ID']
    worst_sid = df.loc[worst_idx, 'Subject_ID']
    best_matrix = all_matrices[subject_ids.index(best_sid)]
    worst_matrix = all_matrices[subject_ids.index(worst_sid)]
    best_info = {'id': best_sid, 'pearson': df.loc[best_idx, 'Pearson_r'], 'jaccard': df.loc[best_idx, 'Jaccard']}
    worst_info = {'id': worst_sid, 'pearson': df.loc[worst_idx, 'Pearson_r'], 'jaccard': df.loc[worst_idx, 'Jaccard']}
    
    # Create figures
    print("\n" + "="*50)
    print("CREATING FIGURES")
    print("="*50)
    
    create_figure1(consensus, sparsity_info, best_matrix, worst_matrix, best_info, worst_info, n_edges, output_folder)
    create_figure2(df, ad_df, hc_df, output_folder)
    create_figure3(df, ad_df, hc_df, n_edges, output_folder)
    create_figure4(df, ad_df, hc_df, sparsity_info, n_edges, output_folder)
    
    # Print summary
    t_p, p_p = stats.ttest_ind(ad_df['Pearson_r'], hc_df['Pearson_r'])
    t_j, p_j = stats.ttest_ind(ad_df['Jaccard'], hc_df['Jaccard'])
    sig_p = '***' if p_p < 0.001 else '**' if p_p < 0.01 else '*' if p_p < 0.05 else 'ns'
    sig_j = '***' if p_j < 0.001 else '**' if p_j < 0.01 else '*' if p_j < 0.05 else 'ns'
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"""
OUTPUT: {output_folder.absolute()}

FIGURES:
  ✓ figure1_consensus_overview.png
  ✓ figure2_pearson_analysis.png
  ✓ figure3_jaccard_analysis.png
  ✓ figure4_combined_summary.png

DATA FILES:
  ✓ subject_vs_saved_consensus.csv
  ✓ consensus_matrix.npy
  ✓ all_subject_matrices.npy

════════════════════════════════════════════════════════════
                       KEY RESULTS
════════════════════════════════════════════════════════════

PEARSON CORRELATION (weight similarity):
  AD (n={n_ad}): {ad_df['Pearson_r'].mean():.3f} ± {ad_df['Pearson_r'].std():.3f}
  HC (n={n_hc}): {hc_df['Pearson_r'].mean():.3f} ± {hc_df['Pearson_r'].std():.3f}
  p = {p_p:.4f} {sig_p}

JACCARD SIMILARITY (edge overlap):
  AD (n={n_ad}): {ad_df['Jaccard'].mean():.3f} ± {ad_df['Jaccard'].std():.3f}
  HC (n={n_hc}): {hc_df['Jaccard'].mean():.3f} ± {hc_df['Jaccard'].std():.3f}
  p = {p_j:.4f} {sig_j}

════════════════════════════════════════════════════════════
""")
    
    return {'df': df, 'consensus': consensus, 'sparsity_info': sparsity_info}


if __name__ == "__main__":
    results = main()
