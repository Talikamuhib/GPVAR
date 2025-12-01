"""
=============================================================================
COMPARING CONNECTIVITY MATRICES - EACH SUBJECT vs SAVED CONSENSUS
=============================================================================

This script compares each individual subject against a PRE-COMPUTED consensus 
matrix loaded from a saved .npy file.

KEY FEATURE: 
- First calculates the SPARSITY/DENSITY of the consensus matrix
- For each subject, matches the density by keeping only the TOP N edges
- This ensures fair comparison with the same number of edges

SIMILARITY METRICS:
- PEARSON CORRELATION: Measures weight similarity (how similar are connection strengths)
- JACCARD SIMILARITY: Measures binary edge overlap (do they have the same connections)

Consensus Matrix Path:
/home/muhibt/GPVAR/consensus_results/ALL_Files/consensus_distance_graph.npy

Outputs (saved to output folder):
- consensus_matrix.npy - Copy of the consensus matrix used
- subject_vs_saved_consensus.csv - Pearson r and Jaccard similarity for each subject
- subject_vs_saved_consensus.png - Visualization plots
- sparsity_info.txt - Sparsity/density information

RUN: python compare_subject_to_saved_consensus.py

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
import logging
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("COMPARING EACH SUBJECT vs SAVED CONSENSUS MATRIX")
print("Using: PEARSON CORRELATION + JACCARD SIMILARITY")
print("="*70)

# =============================================================================
# PATHS
# =============================================================================

# Path to the saved consensus matrix
CONSENSUS_MATRIX_PATH = "/home/muhibt/GPVAR/consensus_results/ALL_Files/consensus_distance_graph.npy"

# Output folder for all results
OUTPUT_FOLDER = Path("subject_vs_consensus_results")

# AD Group Files (Alzheimer's Disease)
AD_FILES = [
    # AD_AR subgroup
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
    # AD_CL subgroup
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

# HC Group Files (Healthy Controls)
HC_FILES = [
    # HC_AR subgroup
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
    # HC_CL subgroup
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

def create_output_folder(folder_path):
    """Create output folder if it doesn't exist."""
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def extract_subject_id(filepath):
    """Extract subject ID from filepath."""
    path = Path(filepath)
    for part in path.parts:
        if part.startswith('sub-'):
            return part
    return path.stem


def load_eeg_data(filepath):
    """Load EEG data from .set file (EEGLAB format)."""
    try:
        import mne
        from mne.channels import make_standard_montage
        
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        
        if raw.get_montage() is None:
            try:
                biosemi_montage = make_standard_montage("biosemi128")
                raw.set_montage(biosemi_montage, on_missing='warn')
            except Exception:
                pass
        
        data = raw.get_data()
        return data
        
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def compute_correlation_matrix(data, absolute=True):
    """Compute Pearson correlation matrix from EEG data."""
    corr_matrix = np.corrcoef(data)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    if absolute:
        corr_matrix = np.abs(corr_matrix)
        
    np.fill_diagonal(corr_matrix, 0)
    return corr_matrix


def calculate_sparsity(matrix, threshold=0):
    """
    Calculate sparsity/density of a matrix.
    
    Parameters
    ----------
    matrix : ndarray
        Symmetric connectivity matrix
    threshold : float
        Values above this threshold are considered "edges"
    
    Returns
    -------
    dict with:
        - n_channels: Number of channels/nodes
        - total_possible_edges: Total possible edges (n*(n-1)/2)
        - n_edges: Number of non-zero edges
        - density: Proportion of edges present (0-1)
        - sparsity: 1 - density
    """
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    values = matrix[triu_idx]
    
    total_possible = len(values)
    n_edges = np.sum(values > threshold)
    density = n_edges / total_possible
    
    return {
        'n_channels': n,
        'total_possible_edges': total_possible,
        'n_edges': int(n_edges),
        'density': density,
        'sparsity': 1 - density,
        'mean_weight': np.mean(values[values > threshold]) if n_edges > 0 else 0,
        'max_weight': np.max(values) if len(values) > 0 else 0,
        'min_nonzero_weight': np.min(values[values > threshold]) if n_edges > 0 else 0
    }


def threshold_matrix_to_density(matrix, n_edges_to_keep):
    """
    Threshold a matrix to keep only the top N edges by weight.
    
    Parameters
    ----------
    matrix : ndarray
        Symmetric connectivity matrix
    n_edges_to_keep : int
        Number of top edges to keep
    
    Returns
    -------
    thresholded_matrix : ndarray
        Binary matrix with same density as target
    """
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    values = matrix[triu_idx]
    
    # Get threshold for top N edges
    if n_edges_to_keep >= len(values):
        threshold = 0
    elif n_edges_to_keep <= 0:
        threshold = np.max(values) + 1
    else:
        # Sort descending and get the Nth value
        sorted_values = np.sort(values)[::-1]
        threshold = sorted_values[n_edges_to_keep - 1]
    
    # Create binary matrix
    binary_matrix = np.zeros_like(matrix)
    binary_matrix[matrix >= threshold] = 1
    np.fill_diagonal(binary_matrix, 0)
    
    # Make symmetric
    binary_matrix = np.maximum(binary_matrix, binary_matrix.T)
    
    return binary_matrix


def compute_pearson_correlation(subject_values, consensus_values):
    """
    Compute Pearson correlation coefficient between two vectors.
    
    PEARSON CORRELATION measures linear relationship between weights:
    - r = 1.0: Perfect positive correlation (weights match exactly)
    - r = 0.5: Moderate correlation
    - r = 0.0: No linear relationship
    
    Formula:
        r = Σ(x - x̄)(y - ȳ) / [√Σ(x - x̄)² × √Σ(y - ȳ)²]
    
    Parameters
    ----------
    subject_values : ndarray
        Subject's edge weights (flattened upper triangle)
    consensus_values : ndarray
        Consensus edge weights (flattened upper triangle)
    
    Returns
    -------
    r : float
        Pearson correlation coefficient
    p : float
        Two-tailed p-value
    """
    r, p = stats.pearsonr(subject_values, consensus_values)
    return r, p


def compute_jaccard_similarity(subject_binary, consensus_binary):
    """
    Compute Jaccard similarity between two binary edge sets.
    
    JACCARD SIMILARITY measures overlap between binary edge sets:
    - J = 1.0: Perfect overlap (identical edges)
    - J = 0.5: 50% overlap
    - J = 0.0: No shared edges
    
    Formula:
        J = |A ∩ B| / |A ∪ B|
        J = (shared edges) / (total unique edges)
    
    Parameters
    ----------
    subject_binary : ndarray
        Subject's binary edges (1 = edge, 0 = no edge)
    consensus_binary : ndarray
        Consensus binary edges
    
    Returns
    -------
    jaccard : float
        Jaccard similarity coefficient [0, 1]
    intersection : int
        Number of shared edges
    union : int
        Total unique edges
    """
    intersection = np.sum((subject_binary == 1) & (consensus_binary == 1))
    union = np.sum((subject_binary == 1) | (consensus_binary == 1))
    jaccard = intersection / union if union > 0 else 0
    return jaccard, int(intersection), int(union)


def compare_to_consensus_density_matched(subject_matrix, consensus_matrix, consensus_n_edges):
    """
    Compare a subject's connectivity matrix to the consensus,
    matching the density by keeping top N edges.
    
    Uses TWO key metrics:
    1. PEARSON CORRELATION - measures weight similarity
    2. JACCARD SIMILARITY - measures binary edge overlap
    
    Parameters
    ----------
    subject_matrix : ndarray
        Subject's full correlation matrix
    consensus_matrix : ndarray
        Consensus matrix (already sparse/distance-dependent)
    consensus_n_edges : int
        Number of edges in the consensus matrix
    
    Returns
    -------
    dict with similarity metrics
    """
    n = subject_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    # Get consensus binary edges
    consensus_values = consensus_matrix[triu_idx]
    consensus_binary = (consensus_values > 0).astype(int)
    
    # Threshold subject matrix to match consensus density
    subject_binary_matrix = threshold_matrix_to_density(subject_matrix, consensus_n_edges)
    subject_binary = subject_binary_matrix[triu_idx].astype(int)
    
    # Get weighted values for correlation
    subject_values = subject_matrix[triu_idx]
    
    results = {}
    
    # =========================================================================
    # METRIC 1: PEARSON CORRELATION (Weight Similarity)
    # =========================================================================
    # Measures: How similar are the CONNECTION STRENGTHS?
    # Range: [-1, 1], higher = more similar weights
    # =========================================================================
    
    # Pearson on FULL weighted matrices (all 8,128 edges)
    r_full, p_full = compute_pearson_correlation(subject_values, consensus_values)
    results['Pearson_r'] = r_full
    results['Pearson_p'] = p_full
    
    # Pearson on EDGES ONLY (where consensus > 0)
    consensus_edge_mask = consensus_values > 0
    if np.sum(consensus_edge_mask) > 2:
        r_edges, p_edges = compute_pearson_correlation(
            subject_values[consensus_edge_mask], 
            consensus_values[consensus_edge_mask]
        )
        results['Pearson_r_edges'] = r_edges
        results['Pearson_p_edges'] = p_edges
    else:
        results['Pearson_r_edges'] = np.nan
        results['Pearson_p_edges'] = np.nan
    
    # =========================================================================
    # METRIC 2: JACCARD SIMILARITY (Binary Edge Overlap)
    # =========================================================================
    # Measures: Do they have the SAME CONNECTIONS?
    # Range: [0, 1], higher = more overlap
    # Note: Subject is density-matched to consensus for fair comparison
    # =========================================================================
    
    jaccard, intersection, union = compute_jaccard_similarity(subject_binary, consensus_binary)
    results['Jaccard'] = jaccard
    results['N_Shared_Edges'] = intersection
    results['N_Union_Edges'] = union
    results['N_Subject_Edges'] = int(np.sum(subject_binary))
    results['N_Consensus_Edges'] = int(np.sum(consensus_binary))
    
    # =========================================================================
    # ADDITIONAL METRICS
    # =========================================================================
    
    # Dice coefficient (related to Jaccard)
    dice = 2 * intersection / (np.sum(subject_binary) + np.sum(consensus_binary))
    results['Dice'] = dice if (np.sum(subject_binary) + np.sum(consensus_binary)) > 0 else 0
    
    # Spearman correlation (rank-based)
    rho, _ = stats.spearmanr(subject_values, consensus_values)
    results['Spearman_rho'] = rho
    
    # Cosine similarity
    norm_subj = np.linalg.norm(subject_values)
    norm_cons = np.linalg.norm(consensus_values)
    if norm_subj > 0 and norm_cons > 0:
        results['Cosine_Similarity'] = np.dot(subject_values, consensus_values) / (norm_subj * norm_cons)
    else:
        results['Cosine_Similarity'] = 0
    
    # Mean absolute difference
    results['Mean_Abs_Diff'] = np.mean(np.abs(subject_values - consensus_values))
    
    # RMSD
    results['RMSD'] = np.sqrt(np.mean((subject_values - consensus_values)**2))
    
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Main function to compute each subject's similarity to saved consensus."""
    
    # =========================================================================
    # STEP 0: CREATE OUTPUT FOLDER
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 0: CREATING OUTPUT FOLDER")
    print("="*70)
    
    output_folder = create_output_folder(OUTPUT_FOLDER)
    print(f"✓ Output folder created: {output_folder.absolute()}")
    
    # =========================================================================
    # STEP 1: LOAD THE SAVED CONSENSUS MATRIX
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING SAVED CONSENSUS MATRIX")
    print("="*70)
    
    consensus_path = Path(CONSENSUS_MATRIX_PATH)
    
    if consensus_path.exists():
        print(f"✓ Loading consensus from: {CONSENSUS_MATRIX_PATH}")
        consensus_matrix = np.load(CONSENSUS_MATRIX_PATH)
        print(f"  • Matrix shape: {consensus_matrix.shape}")
        print(f"  • Matrix dtype: {consensus_matrix.dtype}")
        use_synthetic = False
    else:
        print(f"✗ Consensus matrix not found at: {CONSENSUS_MATRIX_PATH}")
        print("  Using synthetic consensus for demonstration...")
        use_synthetic = True
        
        # Create synthetic sparse consensus (simulating distance-dependence)
        np.random.seed(42)
        n_channels = 128
        
        # Create a sparse distance-dependent pattern
        base = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                distance = abs(i - j)
                if distance < 20:
                    prob = 0.6 * np.exp(-distance / 10)
                else:
                    prob = 0.05
                if np.random.rand() < prob:
                    weight = np.random.rand() * 0.5 + 0.1
                    base[i, j] = weight
                    base[j, i] = weight
        
        consensus_matrix = base
    
    n_channels = consensus_matrix.shape[0]
    
    # =========================================================================
    # STEP 2: CALCULATE SPARSITY OF CONSENSUS MATRIX
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: CALCULATING SPARSITY OF CONSENSUS MATRIX")
    print("="*70)
    
    sparsity_info = calculate_sparsity(consensus_matrix, threshold=0)
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    CONSENSUS MATRIX SPARSITY ANALYSIS                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Matrix Shape:           {n_channels} x {n_channels}                                         │
│  Total Possible Edges:   {sparsity_info['total_possible_edges']:,}                                        │
│  Non-Zero Edges:         {sparsity_info['n_edges']:,}                                          │
│                                                                            │
│  DENSITY:                {sparsity_info['density']:.4f}  ({sparsity_info['density']*100:.2f}%)                              │
│  SPARSITY:               {sparsity_info['sparsity']:.4f}  ({sparsity_info['sparsity']*100:.2f}%)                              │
│                                                                            │
│  Edge Weights (non-zero):                                                  │
│    Mean:  {sparsity_info['mean_weight']:.4f}                                                     │
│    Max:   {sparsity_info['max_weight']:.4f}                                                     │
│    Min:   {sparsity_info['min_nonzero_weight']:.4f}                                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")
    
    consensus_n_edges = sparsity_info['n_edges']
    consensus_density = sparsity_info['density']
    
    # Save sparsity info
    sparsity_file = output_folder / "sparsity_info.txt"
    with open(sparsity_file, 'w') as f:
        f.write("CONSENSUS MATRIX SPARSITY ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Source: {CONSENSUS_MATRIX_PATH}\n")
        f.write(f"Matrix Shape: {n_channels} x {n_channels}\n\n")
        f.write(f"Total Possible Edges: {sparsity_info['total_possible_edges']:,}\n")
        f.write(f"Non-Zero Edges: {sparsity_info['n_edges']:,}\n\n")
        f.write(f"DENSITY: {sparsity_info['density']:.6f} ({sparsity_info['density']*100:.4f}%)\n")
        f.write(f"SPARSITY: {sparsity_info['sparsity']:.6f} ({sparsity_info['sparsity']*100:.4f}%)\n\n")
        f.write(f"Edge Weights (non-zero):\n")
        f.write(f"  Mean: {sparsity_info['mean_weight']:.6f}\n")
        f.write(f"  Max:  {sparsity_info['max_weight']:.6f}\n")
        f.write(f"  Min:  {sparsity_info['min_nonzero_weight']:.6f}\n")
    print(f"✓ Sparsity info saved to: {sparsity_file}")
    
    # Save consensus matrix to output folder
    consensus_save_path = output_folder / "consensus_matrix.npy"
    np.save(consensus_save_path, consensus_matrix)
    print(f"✓ Consensus matrix saved to: {consensus_save_path}")
    
    # =========================================================================
    # STEP 3: LOAD SUBJECT DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: LOADING SUBJECT DATA")
    print("="*70)
    
    all_matrices = []
    subject_ids = []
    group_labels = []
    
    # Check if real EEG files exist
    real_files_exist = False
    for filepath in AD_FILES + HC_FILES:
        if Path(filepath).exists():
            real_files_exist = True
            break
    
    if real_files_exist and not use_synthetic:
        logger.info("Loading real EEG files...")
        
        # Load AD subjects
        print(f"\nLoading AD subjects...")
        for filepath in AD_FILES:
            if Path(filepath).exists():
                data = load_eeg_data(filepath)
                if data is not None:
                    corr_matrix = compute_correlation_matrix(data)
                    if corr_matrix.shape[0] == n_channels:
                        all_matrices.append(corr_matrix)
                        subject_ids.append(extract_subject_id(filepath))
                        group_labels.append('AD')
                    else:
                        logger.warning(f"Dimension mismatch for {filepath}")
        
        print(f"  Loaded {sum(1 for g in group_labels if g == 'AD')} AD subjects")
        
        # Load HC subjects
        print(f"\nLoading HC subjects...")
        for filepath in HC_FILES:
            if Path(filepath).exists():
                data = load_eeg_data(filepath)
                if data is not None:
                    corr_matrix = compute_correlation_matrix(data)
                    if corr_matrix.shape[0] == n_channels:
                        all_matrices.append(corr_matrix)
                        subject_ids.append(extract_subject_id(filepath))
                        group_labels.append('HC')
                    else:
                        logger.warning(f"Dimension mismatch for {filepath}")
        
        print(f"  Loaded {sum(1 for g in group_labels if g == 'HC')} HC subjects")
    
    # If no real data, create synthetic
    if len(all_matrices) == 0:
        print("\nUsing synthetic subject data for demonstration...")
        
        np.random.seed(42)
        
        # AD-specific pattern
        ad_mod = np.zeros((n_channels, n_channels))
        ad_mod[:50, :50] = 0.15
        ad_mod[80:, 80:] = -0.05
        ad_mod = (ad_mod + ad_mod.T) / 2
        
        # HC-specific pattern
        hc_mod = np.zeros((n_channels, n_channels))
        hc_mod[40:80, 40:80] = 0.1
        hc_mod = (hc_mod + hc_mod.T) / 2
        
        # Base pattern similar to consensus
        base_pattern = np.random.rand(n_channels, n_channels) * 0.3
        base_pattern = (base_pattern + base_pattern.T) / 2
        np.fill_diagonal(base_pattern, 0)
        
        # Generate AD subjects
        n_ad = 35
        for i in range(n_ad):
            noise = np.random.randn(n_channels, n_channels) * 0.08
            noise = (noise + noise.T) / 2
            subj = base_pattern + ad_mod + noise
            np.fill_diagonal(subj, 0)
            subj = np.clip(subj, 0, 1)
            all_matrices.append(subj)
            subject_ids.append(f'sub-300{i+1:02d}')
            group_labels.append('AD')
        
        # Generate HC subjects
        n_hc = 31
        for i in range(n_hc):
            noise = np.random.randn(n_channels, n_channels) * 0.08
            noise = (noise + noise.T) / 2
            subj = base_pattern + hc_mod + noise
            np.fill_diagonal(subj, 0)
            subj = np.clip(subj, 0, 1)
            all_matrices.append(subj)
            subject_ids.append(f'sub-100{i+1:02d}')
            group_labels.append('HC')
    
    n_subjects = len(all_matrices)
    n_ad = sum(1 for g in group_labels if g == 'AD')
    n_hc = sum(1 for g in group_labels if g == 'HC')
    
    print(f"\n✓ Data loaded:")
    print(f"  • Total subjects: {n_subjects}")
    print(f"  • AD subjects: {n_ad}")
    print(f"  • HC subjects: {n_hc}")
    print(f"  • Channels: {n_channels}")
    
    # =========================================================================
    # STEP 4: COMPARE EACH SUBJECT TO CONSENSUS (DENSITY MATCHED)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPARING EACH SUBJECT TO CONSENSUS")
    print("        Using: PEARSON CORRELATION + JACCARD SIMILARITY")
    print("="*70)
    print(f"\n  ► Matching each subject to consensus density: {consensus_n_edges} edges ({consensus_density*100:.2f}%)")
    
    results_list = []
    
    for i, (matrix, subj_id, group) in enumerate(zip(all_matrices, subject_ids, group_labels)):
        comparison = compare_to_consensus_density_matched(matrix, consensus_matrix, consensus_n_edges)
        
        results_list.append({
            'Subject_ID': subj_id,
            'Group': group,
            'Pearson_r': comparison['Pearson_r'],
            'Pearson_p': comparison['Pearson_p'],
            'Jaccard': comparison['Jaccard'],
            'N_Shared_Edges': comparison['N_Shared_Edges'],
            'N_Subject_Edges': comparison['N_Subject_Edges'],
            'N_Consensus_Edges': comparison['N_Consensus_Edges'],
            'Dice': comparison['Dice'],
            'Spearman_rho': comparison['Spearman_rho'],
            'Cosine_Similarity': comparison['Cosine_Similarity'],
            'Mean_Abs_Diff': comparison['Mean_Abs_Diff'],
            'RMSD': comparison['RMSD']
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_subjects} subjects...")
    
    # Create DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Add combined score (average of normalized Pearson and Jaccard)
    df_results['Combined_Score'] = (df_results['Pearson_r'] + df_results['Jaccard']) / 2
    
    # Sort by Combined Score descending
    df_results = df_results.sort_values('Combined_Score', ascending=False).reset_index(drop=True)
    df_results['Rank'] = range(1, len(df_results) + 1)
    
    # =========================================================================
    # STEP 5: SAVE RESULTS TO CSV
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: SAVING RESULTS TO CSV")
    print("="*70)
    
    csv_filename = output_folder / 'subject_vs_saved_consensus.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"✓ Results saved to: {csv_filename}")
    
    # Print summary table
    print("\n" + "-"*110)
    print(f"{'Rank':<6} {'Subject_ID':<15} {'Group':<6} {'Pearson_r':<12} {'Jaccard':<10} {'Shared':<8} {'Combined':<10}")
    print("-"*110)
    
    for _, row in df_results.head(10).iterrows():
        print(f"{row['Rank']:<6} {row['Subject_ID']:<15} {row['Group']:<6} "
              f"{row['Pearson_r']:<12.4f} {row['Jaccard']:<10.4f} "
              f"{row['N_Shared_Edges']:<8} {row['Combined_Score']:<10.4f}")
    
    print("...")
    
    for _, row in df_results.tail(5).iterrows():
        print(f"{row['Rank']:<6} {row['Subject_ID']:<15} {row['Group']:<6} "
              f"{row['Pearson_r']:<12.4f} {row['Jaccard']:<10.4f} "
              f"{row['N_Shared_Edges']:<8} {row['Combined_Score']:<10.4f}")
    
    print("-"*110)
    
    # =========================================================================
    # STEP 6: SUMMARY STATISTICS BY GROUP
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: SUMMARY STATISTICS BY GROUP")
    print("="*70)
    
    ad_results = df_results[df_results['Group'] == 'AD']
    hc_results = df_results[df_results['Group'] == 'HC']
    
    # Statistical tests for BOTH metrics
    t_stat_p, p_val_p = stats.ttest_ind(ad_results['Pearson_r'], hc_results['Pearson_r'])
    t_stat_j, p_val_j = stats.ttest_ind(ad_results['Jaccard'], hc_results['Jaccard'])
    t_stat_c, p_val_c = stats.ttest_ind(ad_results['Combined_Score'], hc_results['Combined_Score'])
    
    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
    
    d_pearson = cohens_d(ad_results['Pearson_r'], hc_results['Pearson_r'])
    d_jaccard = cohens_d(ad_results['Jaccard'], hc_results['Jaccard'])
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│      SUMMARY: SUBJECT vs CONSENSUS (DENSITY-MATCHED COMPARISON)           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  CONSENSUS MATRIX:                                                         │
│    Edges: {consensus_n_edges:,} / {sparsity_info['total_possible_edges']:,} ({consensus_density*100:.2f}% density)                       │
│                                                                            │
│  DENSITY MATCHING:                                                         │
│    Each subject's correlation matrix is thresholded to keep               │
│    the TOP {consensus_n_edges:,} edges (matching consensus density)                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ╔════════════════════════════════════════════════════════════════════╗   │
│  ║  PEARSON CORRELATION (weight similarity)                           ║   │
│  ╚════════════════════════════════════════════════════════════════════╝   │
│    AD (n={n_ad}):   {ad_results['Pearson_r'].mean():.3f} ± {ad_results['Pearson_r'].std():.3f}  [{ad_results['Pearson_r'].min():.3f}, {ad_results['Pearson_r'].max():.3f}]   │
│    HC (n={n_hc}):   {hc_results['Pearson_r'].mean():.3f} ± {hc_results['Pearson_r'].std():.3f}  [{hc_results['Pearson_r'].min():.3f}, {hc_results['Pearson_r'].max():.3f}]   │
│    t-test: t = {t_stat_p:>7.3f}, p = {p_val_p:.4f}  {'***' if p_val_p < 0.001 else '**' if p_val_p < 0.01 else '*' if p_val_p < 0.05 else 'ns':>4}                       │
│    Cohen's d = {d_pearson:.3f}                                                       │
│                                                                            │
│  ╔════════════════════════════════════════════════════════════════════╗   │
│  ║  JACCARD SIMILARITY (binary edge overlap)                          ║   │
│  ╚════════════════════════════════════════════════════════════════════╝   │
│    AD (n={n_ad}):   {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}  [{ad_results['Jaccard'].min():.3f}, {ad_results['Jaccard'].max():.3f}]   │
│    HC (n={n_hc}):   {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}  [{hc_results['Jaccard'].min():.3f}, {hc_results['Jaccard'].max():.3f}]   │
│    t-test: t = {t_stat_j:>7.3f}, p = {p_val_j:.4f}  {'***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns':>4}                       │
│    Cohen's d = {d_jaccard:.3f}                                                       │
│                                                                            │
│  SHARED EDGES (out of {consensus_n_edges:,})                                              │
│    AD mean: {ad_results['N_Shared_Edges'].mean():.0f} edges ({ad_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}%)                                    │
│    HC mean: {hc_results['N_Shared_Edges'].mean():.0f} edges ({hc_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}%)                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")
    
    # =========================================================================
    # STEP 7: CREATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Colors
    ad_color = '#E74C3C'
    hc_color = '#3498DB'
    
    fig = plt.figure(figsize=(20, 16))
    
    # -------------------------------------------------------------------------
    # Row 1: Consensus Matrix Visualizations
    # -------------------------------------------------------------------------
    
    # Plot 1: Consensus Matrix
    ax1 = fig.add_subplot(4, 4, 1)
    vmax = np.percentile(consensus_matrix[consensus_matrix > 0], 95) if np.any(consensus_matrix > 0) else 1
    im1 = ax1.imshow(consensus_matrix, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title(f'Consensus Matrix\n({consensus_n_edges:,} edges, {consensus_density*100:.1f}% density)', 
                  fontweight='bold', fontsize=10)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Channel')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot 2: Consensus Binary (edges only)
    ax2 = fig.add_subplot(4, 4, 2)
    consensus_binary = (consensus_matrix > 0).astype(float)
    im2 = ax2.imshow(consensus_binary, cmap='Greys', vmin=0, vmax=1)
    ax2.set_title(f'Consensus Binary Edges\n({consensus_n_edges:,} edges)', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Channel')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot 3: Example subject (best match)
    ax3 = fig.add_subplot(4, 4, 3)
    best_idx = df_results['Combined_Score'].idxmax()
    best_subj_id = df_results.loc[best_idx, 'Subject_ID']
    best_matrix_idx = subject_ids.index(best_subj_id)
    best_subj_binary = threshold_matrix_to_density(all_matrices[best_matrix_idx], consensus_n_edges)
    im3 = ax3.imshow(best_subj_binary, cmap='Greys', vmin=0, vmax=1)
    ax3.set_title(f'Best Subject: {best_subj_id}\n(Pearson={df_results.loc[best_idx, "Pearson_r"]:.3f}, Jaccard={df_results.loc[best_idx, "Jaccard"]:.3f})', 
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Channel')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Plot 4: Worst subject
    ax4 = fig.add_subplot(4, 4, 4)
    worst_idx = df_results['Combined_Score'].idxmin()
    worst_subj_id = df_results.loc[worst_idx, 'Subject_ID']
    worst_matrix_idx = subject_ids.index(worst_subj_id)
    worst_subj_binary = threshold_matrix_to_density(all_matrices[worst_matrix_idx], consensus_n_edges)
    im4 = ax4.imshow(worst_subj_binary, cmap='Greys', vmin=0, vmax=1)
    ax4.set_title(f'Worst Subject: {worst_subj_id}\n(Pearson={df_results.loc[worst_idx, "Pearson_r"]:.3f}, Jaccard={df_results.loc[worst_idx, "Jaccard"]:.3f})', 
                  fontweight='bold', fontsize=10)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Channel')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # -------------------------------------------------------------------------
    # Row 2: PEARSON CORRELATION Visualizations
    # -------------------------------------------------------------------------
    
    # Plot 5: Pearson Boxplot
    ax5 = fig.add_subplot(4, 4, 5)
    bp = ax5.boxplot([ad_results['Pearson_r'], hc_results['Pearson_r']], 
                     labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Pearson_r'], ad_color), 
                                        (hc_results['Pearson_r'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax5.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white')
    
    sig_str_p = '***' if p_val_p < 0.001 else '**' if p_val_p < 0.01 else '*' if p_val_p < 0.05 else 'ns'
    ax5.set_ylabel('Pearson r', fontsize=11)
    ax5.set_title(f'PEARSON CORRELATION by Group\n(p = {p_val_p:.4f} {sig_str_p}, d = {d_pearson:.2f})', 
                  fontweight='bold', fontsize=10, color='darkgreen')
    ax5.axhline(y=df_results['Pearson_r'].mean(), color='gray', linestyle='--', linewidth=1.5)
    
    # Plot 6: Pearson Histogram
    ax6 = fig.add_subplot(4, 4, 6)
    bins = np.linspace(df_results['Pearson_r'].min() - 0.02, 
                       df_results['Pearson_r'].max() + 0.02, 20)
    ax6.hist(ad_results['Pearson_r'], bins=bins, alpha=0.6, color=ad_color, 
             label=f'AD (μ={ad_results["Pearson_r"].mean():.3f})', edgecolor='white')
    ax6.hist(hc_results['Pearson_r'], bins=bins, alpha=0.6, color=hc_color, 
             label=f'HC (μ={hc_results["Pearson_r"].mean():.3f})', edgecolor='white')
    ax6.set_xlabel('Pearson r', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('PEARSON Distribution', fontweight='bold', fontsize=10, color='darkgreen')
    ax6.legend(loc='upper left', fontsize=9)
    
    # -------------------------------------------------------------------------
    # Row 2 continued: JACCARD Visualizations
    # -------------------------------------------------------------------------
    
    # Plot 7: Jaccard Boxplot
    ax7 = fig.add_subplot(4, 4, 7)
    bp2 = ax7.boxplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                      labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor(ad_color)
    bp2['boxes'][1].set_facecolor(hc_color)
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Jaccard'], ad_color), 
                                        (hc_results['Jaccard'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax7.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white')
    
    sig_str_j = '***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns'
    ax7.set_ylabel('Jaccard Similarity', fontsize=11)
    ax7.set_title(f'JACCARD SIMILARITY by Group\n(p = {p_val_j:.4f} {sig_str_j}, d = {d_jaccard:.2f})', 
                  fontweight='bold', fontsize=10, color='darkblue')
    ax7.axhline(y=df_results['Jaccard'].mean(), color='gray', linestyle='--', linewidth=1.5)
    
    # Plot 8: Jaccard Histogram
    ax8 = fig.add_subplot(4, 4, 8)
    bins = np.linspace(df_results['Jaccard'].min() - 0.02, 
                       df_results['Jaccard'].max() + 0.02, 20)
    ax8.hist(ad_results['Jaccard'], bins=bins, alpha=0.6, color=ad_color, 
             label=f'AD (μ={ad_results["Jaccard"].mean():.3f})', edgecolor='white')
    ax8.hist(hc_results['Jaccard'], bins=bins, alpha=0.6, color=hc_color, 
             label=f'HC (μ={hc_results["Jaccard"].mean():.3f})', edgecolor='white')
    ax8.set_xlabel('Jaccard', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.set_title('JACCARD Distribution', fontweight='bold', fontsize=10, color='darkblue')
    ax8.legend(loc='upper left', fontsize=9)
    
    # -------------------------------------------------------------------------
    # Row 3: Scatter Plots and Rankings
    # -------------------------------------------------------------------------
    
    # Plot 9: Pearson vs Jaccard
    ax9 = fig.add_subplot(4, 4, 9)
    ax9.scatter(ad_results['Pearson_r'], ad_results['Jaccard'], 
                c=ad_color, label=f'AD (n={n_ad})', alpha=0.7, s=60, edgecolor='white')
    ax9.scatter(hc_results['Pearson_r'], hc_results['Jaccard'], 
                c=hc_color, label=f'HC (n={n_hc})', alpha=0.7, s=60, edgecolor='white')
    ax9.set_xlabel('Pearson r (weight similarity)', fontsize=11)
    ax9.set_ylabel('Jaccard (edge overlap)', fontsize=11)
    ax9.set_title('PEARSON vs JACCARD\n(Two complementary metrics)', fontweight='bold', fontsize=10)
    ax9.legend(loc='lower right', fontsize=9)
    
    # Add correlation line
    r_pj, _ = stats.pearsonr(df_results['Pearson_r'], df_results['Jaccard'])
    ax9.annotate(f'r = {r_pj:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
    
    # Plot 10: Shared Edges Histogram
    ax10 = fig.add_subplot(4, 4, 10)
    bins = np.linspace(df_results['N_Shared_Edges'].min() - 10, 
                       df_results['N_Shared_Edges'].max() + 10, 20)
    ax10.hist(ad_results['N_Shared_Edges'], bins=bins, alpha=0.6, color=ad_color, 
              label=f'AD (μ={ad_results["N_Shared_Edges"].mean():.0f})', edgecolor='white')
    ax10.hist(hc_results['N_Shared_Edges'], bins=bins, alpha=0.6, color=hc_color, 
              label=f'HC (μ={hc_results["N_Shared_Edges"].mean():.0f})', edgecolor='white')
    ax10.axvline(consensus_n_edges, color='black', linestyle='--', linewidth=2, 
                 label=f'Consensus: {consensus_n_edges}')
    ax10.set_xlabel('Shared Edges', fontsize=11)
    ax10.set_ylabel('Count', fontsize=11)
    ax10.set_title('Number of Shared Edges', fontweight='bold', fontsize=10)
    ax10.legend(loc='upper left', fontsize=9)
    
    # Plot 11-12: Bar chart - All subjects ranked by Combined Score
    ax11 = fig.add_subplot(4, 4, (11, 12))
    colors_bar = [ad_color if g == 'AD' else hc_color for g in df_results['Group']]
    
    # Plot both Pearson and Jaccard as stacked-style
    width = 0.4
    x_pos = np.arange(n_subjects)
    ax11.bar(x_pos - width/2, df_results['Pearson_r'], width, color='darkgreen', alpha=0.7, label='Pearson r')
    ax11.bar(x_pos + width/2, df_results['Jaccard'], width, color='darkblue', alpha=0.7, label='Jaccard')
    
    ax11.axhline(y=df_results['Pearson_r'].mean(), color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7)
    ax11.axhline(y=df_results['Jaccard'].mean(), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax11.set_xlabel('Subject Rank (by Combined Score)', fontsize=11)
    ax11.set_ylabel('Similarity', fontsize=11)
    ax11.set_title('All Subjects: PEARSON (green) vs JACCARD (blue)', fontweight='bold', fontsize=10)
    ax11.set_xlim([-1, n_subjects])
    ax11.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Row 4: Summary and Additional Plots
    # -------------------------------------------------------------------------
    
    # Plot 13: Violin plots for both metrics
    ax13 = fig.add_subplot(4, 4, 13)
    positions = [1, 2, 4, 5]
    data_violin = [ad_results['Pearson_r'], hc_results['Pearson_r'], 
                   ad_results['Jaccard'], hc_results['Jaccard']]
    parts = ax13.violinplot(data_violin, positions=positions, showmeans=True, showmedians=True)
    
    colors_violin = [ad_color, hc_color, ad_color, hc_color]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.6)
    
    ax13.set_xticks([1.5, 4.5])
    ax13.set_xticklabels(['Pearson r', 'Jaccard'])
    ax13.set_ylabel('Similarity', fontsize=11)
    ax13.set_title('Distribution Comparison\n(Red=AD, Blue=HC)', fontweight='bold', fontsize=10)
    
    # Plot 14: Combined Score Boxplot
    ax14 = fig.add_subplot(4, 4, 14)
    bp3 = ax14.boxplot([ad_results['Combined_Score'], hc_results['Combined_Score']], 
                       labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp3['boxes'][0].set_facecolor(ad_color)
    bp3['boxes'][1].set_facecolor(hc_color)
    for box in bp3['boxes']:
        box.set_alpha(0.7)
    
    sig_str_c = '***' if p_val_c < 0.001 else '**' if p_val_c < 0.01 else '*' if p_val_c < 0.05 else 'ns'
    ax14.set_ylabel('Combined Score', fontsize=11)
    ax14.set_title(f'COMBINED SCORE\n(Pearson + Jaccard) / 2\n(p = {p_val_c:.4f} {sig_str_c})', 
                   fontweight='bold', fontsize=10)
    
    # Plot 15-16: Summary text
    ax15 = fig.add_subplot(4, 4, (15, 16))
    ax15.axis('off')
    
    best_subj = df_results.iloc[0]
    worst_subj = df_results.iloc[-1]
    
    summary_text = f"""
    ══════════════════════════════════════════════════════════════════════════════════════════
                         SIMILARITY METRICS: PEARSON CORRELATION + JACCARD
    ══════════════════════════════════════════════════════════════════════════════════════════
    
    CONSENSUS: {consensus_n_edges:,} edges ({consensus_density*100:.2f}% density) | SUBJECTS: {n_subjects} ({n_ad} AD + {n_hc} HC)
    
    ┌──────────────────────────────────────────┬──────────────────────────────────────────┐
    │  PEARSON CORRELATION                     │  JACCARD SIMILARITY                      │
    │  (Weight similarity)                     │  (Binary edge overlap)                   │
    ├──────────────────────────────────────────┼──────────────────────────────────────────┤
    │  AD: {ad_results['Pearson_r'].mean():.3f} ± {ad_results['Pearson_r'].std():.3f}                          │  AD: {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}                          │
    │  HC: {hc_results['Pearson_r'].mean():.3f} ± {hc_results['Pearson_r'].std():.3f}                          │  HC: {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}                          │
    │  p = {p_val_p:.4f} {sig_str_p}  (d = {d_pearson:.2f})               │  p = {p_val_j:.4f} {sig_str_j}  (d = {d_jaccard:.2f})               │
    └──────────────────────────────────────────┴──────────────────────────────────────────┘
    
    INTERPRETATION:
    • Pearson r measures: Do connection STRENGTHS match the consensus?
    • Jaccard measures: Do they have the SAME CONNECTIONS (binary overlap)?
    • Combined Score = (Pearson + Jaccard) / 2
    
    BEST:  {best_subj['Subject_ID']} ({best_subj['Group']}) - Pearson={best_subj['Pearson_r']:.3f}, Jaccard={best_subj['Jaccard']:.3f}
    WORST: {worst_subj['Subject_ID']} ({worst_subj['Group']}) - Pearson={worst_subj['Pearson_r']:.3f}, Jaccard={worst_subj['Jaccard']:.3f}
    """
    
    ax15.text(0.02, 0.98, summary_text, transform=ax15.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Subject vs Consensus: PEARSON CORRELATION + JACCARD SIMILARITY',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_filename = output_folder / 'subject_vs_saved_consensus.png'
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {fig_filename}")
    
    plt.close()
    
    # =========================================================================
    # STEP 8: SAVE NPY FILES
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 8: SAVING NPY FILES")
    print("="*70)
    
    # Save all subject matrices
    all_matrices_array = np.stack(all_matrices, axis=0)
    np.save(output_folder / 'all_subject_matrices.npy', all_matrices_array)
    print(f"✓ Saved: {output_folder}/all_subject_matrices.npy (shape: {all_matrices_array.shape})")
    
    np.save(output_folder / 'subject_ids.npy', np.array(subject_ids))
    print(f"✓ Saved: {output_folder}/subject_ids.npy")
    
    np.save(output_folder / 'group_labels.npy', np.array(group_labels))
    print(f"✓ Saved: {output_folder}/group_labels.npy")
    
    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"""
OUTPUT FOLDER: {output_folder.absolute()}

OUTPUT FILES:
  ✓ consensus_matrix.npy           - Consensus matrix used
  ✓ sparsity_info.txt              - Sparsity/density information  
  ✓ subject_vs_saved_consensus.csv - All subject similarity metrics
  ✓ subject_vs_saved_consensus.png - Visualization figure
  ✓ all_subject_matrices.npy       - All individual subject matrices
  ✓ subject_ids.npy                - Subject ID array
  ✓ group_labels.npy               - Group labels (AD/HC)

CONSENSUS SPARSITY:
  Edges: {consensus_n_edges:,} / {sparsity_info['total_possible_edges']:,}
  Density: {consensus_density*100:.2f}%

══════════════════════════════════════════════════════════════════════════════
                              KEY RESULTS
══════════════════════════════════════════════════════════════════════════════

  PEARSON CORRELATION (weight similarity):
    AD: {ad_results['Pearson_r'].mean():.3f} ± {ad_results['Pearson_r'].std():.3f}
    HC: {hc_results['Pearson_r'].mean():.3f} ± {hc_results['Pearson_r'].std():.3f}
    p = {p_val_p:.4f} {sig_str_p}  |  Cohen's d = {d_pearson:.3f}

  JACCARD SIMILARITY (binary edge overlap):
    AD: {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}
    HC: {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}
    p = {p_val_j:.4f} {sig_str_j}  |  Cohen's d = {d_jaccard:.3f}

══════════════════════════════════════════════════════════════════════════════
""")
    
    return {
        'df_results': df_results,
        'consensus_matrix': consensus_matrix,
        'sparsity_info': sparsity_info,
        'all_matrices': all_matrices,
        'subject_ids': subject_ids,
        'group_labels': group_labels,
        'output_folder': output_folder
    }


if __name__ == "__main__":
    results = main()
