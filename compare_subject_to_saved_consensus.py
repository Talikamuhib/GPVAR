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
print("(With Density Matching)")
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


def compare_to_consensus_density_matched(subject_matrix, consensus_matrix, consensus_n_edges):
    """
    Compare a subject's connectivity matrix to the consensus,
    matching the density by keeping top N edges.
    
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
    # METRIC 1: Pearson correlation on FULL weighted matrices
    # =========================================================================
    r_full, p_full = stats.pearsonr(subject_values, consensus_values)
    results['pearson_r_full'] = r_full
    results['pearson_p_full'] = p_full
    
    # =========================================================================
    # METRIC 2: Pearson correlation on EDGES ONLY (where consensus > 0)
    # =========================================================================
    consensus_edge_mask = consensus_values > 0
    if np.sum(consensus_edge_mask) > 2:
        r_edges, p_edges = stats.pearsonr(
            subject_values[consensus_edge_mask], 
            consensus_values[consensus_edge_mask]
        )
        results['pearson_r_edges'] = r_edges
        results['pearson_p_edges'] = p_edges
    else:
        results['pearson_r_edges'] = np.nan
        results['pearson_p_edges'] = np.nan
    
    # =========================================================================
    # METRIC 3: Jaccard similarity (binary edge overlap)
    # =========================================================================
    intersection = np.sum((subject_binary == 1) & (consensus_binary == 1))
    union = np.sum((subject_binary == 1) | (consensus_binary == 1))
    results['jaccard'] = intersection / union if union > 0 else 0
    results['n_shared_edges'] = int(intersection)
    results['n_subject_edges'] = int(np.sum(subject_binary))
    results['n_consensus_edges'] = int(np.sum(consensus_binary))
    
    # =========================================================================
    # METRIC 4: Dice coefficient (another overlap measure)
    # =========================================================================
    dice = 2 * intersection / (np.sum(subject_binary) + np.sum(consensus_binary))
    results['dice'] = dice if (np.sum(subject_binary) + np.sum(consensus_binary)) > 0 else 0
    
    # =========================================================================
    # METRIC 5: Spearman correlation
    # =========================================================================
    rho, _ = stats.spearmanr(subject_values, consensus_values)
    results['spearman_rho'] = rho
    
    # =========================================================================
    # METRIC 6: Cosine similarity
    # =========================================================================
    norm_subj = np.linalg.norm(subject_values)
    norm_cons = np.linalg.norm(consensus_values)
    if norm_subj > 0 and norm_cons > 0:
        results['cosine_similarity'] = np.dot(subject_values, consensus_values) / (norm_subj * norm_cons)
    else:
        results['cosine_similarity'] = 0
    
    # =========================================================================
    # METRIC 7: Mean absolute difference
    # =========================================================================
    results['mean_abs_diff'] = np.mean(np.abs(subject_values - consensus_values))
    
    # =========================================================================
    # METRIC 8: RMSD
    # =========================================================================
    results['rmsd'] = np.sqrt(np.mean((subject_values - consensus_values)**2))
    
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
        # Add some edges with distance-like decay
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                distance = abs(i - j)
                # Higher probability of connection for nearby channels
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
    print("STEP 4: COMPARING EACH SUBJECT TO CONSENSUS (DENSITY MATCHED)")
    print("="*70)
    print(f"\n  ► Matching each subject to consensus density: {consensus_n_edges} edges ({consensus_density*100:.2f}%)")
    
    results_list = []
    
    for i, (matrix, subj_id, group) in enumerate(zip(all_matrices, subject_ids, group_labels)):
        comparison = compare_to_consensus_density_matched(matrix, consensus_matrix, consensus_n_edges)
        
        results_list.append({
            'Subject_ID': subj_id,
            'Group': group,
            'Pearson_r_full': comparison['pearson_r_full'],
            'Pearson_r_edges': comparison['pearson_r_edges'],
            'Jaccard': comparison['jaccard'],
            'Dice': comparison['dice'],
            'N_Shared_Edges': comparison['n_shared_edges'],
            'N_Subject_Edges': comparison['n_subject_edges'],
            'N_Consensus_Edges': comparison['n_consensus_edges'],
            'Spearman_rho': comparison['spearman_rho'],
            'Cosine_Similarity': comparison['cosine_similarity'],
            'Mean_Abs_Diff': comparison['mean_abs_diff'],
            'RMSD': comparison['rmsd']
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_subjects} subjects...")
    
    # Create DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Sort by Jaccard descending (since we're comparing binary edge overlap)
    df_results = df_results.sort_values('Jaccard', ascending=False).reset_index(drop=True)
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
    print("\n" + "-"*100)
    print(f"{'Rank':<6} {'Subject_ID':<15} {'Group':<6} {'Jaccard':<10} {'Shared':<8} {'Pearson_full':<14} {'Dice':<10}")
    print("-"*100)
    
    for _, row in df_results.head(10).iterrows():
        print(f"{row['Rank']:<6} {row['Subject_ID']:<15} {row['Group']:<6} "
              f"{row['Jaccard']:<10.4f} {row['N_Shared_Edges']:<8} "
              f"{row['Pearson_r_full']:<14.4f} {row['Dice']:<10.4f}")
    
    print("...")
    
    for _, row in df_results.tail(5).iterrows():
        print(f"{row['Rank']:<6} {row['Subject_ID']:<15} {row['Group']:<6} "
              f"{row['Jaccard']:<10.4f} {row['N_Shared_Edges']:<8} "
              f"{row['Pearson_r_full']:<14.4f} {row['Dice']:<10.4f}")
    
    print("-"*100)
    
    # =========================================================================
    # STEP 6: SUMMARY STATISTICS BY GROUP
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: SUMMARY STATISTICS BY GROUP")
    print("="*70)
    
    ad_results = df_results[df_results['Group'] == 'AD']
    hc_results = df_results[df_results['Group'] == 'HC']
    
    # Statistical tests
    t_stat_j, p_val_j = stats.ttest_ind(ad_results['Jaccard'], hc_results['Jaccard'])
    t_stat_r, p_val_r = stats.ttest_ind(ad_results['Pearson_r_full'], hc_results['Pearson_r_full'])
    t_stat_d, p_val_d = stats.ttest_ind(ad_results['Dice'], hc_results['Dice'])
    
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
│  JACCARD SIMILARITY (binary edge overlap)                                  │
│  ──────────────────────────────────────────────────────────────────────    │
│    ALL (n={n_subjects}):  {df_results['Jaccard'].mean():.3f} ± {df_results['Jaccard'].std():.3f}  [{df_results['Jaccard'].min():.3f}, {df_results['Jaccard'].max():.3f}]   │
│    AD (n={n_ad}):   {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}  [{ad_results['Jaccard'].min():.3f}, {ad_results['Jaccard'].max():.3f}]   │
│    HC (n={n_hc}):   {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}  [{hc_results['Jaccard'].min():.3f}, {hc_results['Jaccard'].max():.3f}]   │
│    t-test: t = {t_stat_j:>7.3f}, p = {p_val_j:.4f}  {'***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns':>4}                       │
│                                                                            │
│  SHARED EDGES (out of {consensus_n_edges:,})                                              │
│  ──────────────────────────────────────────────────────────────────────    │
│    AD mean: {ad_results['N_Shared_Edges'].mean():.0f} edges ({ad_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}%)                                    │
│    HC mean: {hc_results['N_Shared_Edges'].mean():.0f} edges ({hc_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}%)                                    │
│                                                                            │
│  PEARSON r (full weighted matrix)                                          │
│  ──────────────────────────────────────────────────────────────────────    │
│    AD:  {ad_results['Pearson_r_full'].mean():.3f} ± {ad_results['Pearson_r_full'].std():.3f}                                              │
│    HC:  {hc_results['Pearson_r_full'].mean():.3f} ± {hc_results['Pearson_r_full'].std():.3f}                                              │
│    t-test: t = {t_stat_r:>7.3f}, p = {p_val_r:.4f}  {'***' if p_val_r < 0.001 else '**' if p_val_r < 0.01 else '*' if p_val_r < 0.05 else 'ns':>4}                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")
    
    # =========================================================================
    # STEP 7: CREATE VISUALIZATIONS (THESIS QUALITY)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: CREATING THESIS-QUALITY VISUALIZATIONS")
    print("="*70)
    
    # Set thesis-quality style
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Colors
    ad_color = '#E74C3C'
    hc_color = '#3498DB'
    consensus_color = '#2ECC71'
    overlap_color = '#9B59B6'
    
    # =========================================================================
    # FIGURE 1: Main Results Overview
    # =========================================================================
    print("  Creating Figure 1: Main Results Overview...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # Plot 1: Consensus Matrix
    ax1 = fig.add_subplot(3, 4, 1)
    vmax = np.percentile(consensus_matrix[consensus_matrix > 0], 95) if np.any(consensus_matrix > 0) else 1
    im1 = ax1.imshow(consensus_matrix, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title(f'Consensus Matrix\n({consensus_n_edges:,} edges, {consensus_density*100:.1f}% density)', 
                  fontweight='bold', fontsize=10)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Channel')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot 2: Consensus Binary (edges only)
    ax2 = fig.add_subplot(3, 4, 2)
    consensus_binary = (consensus_matrix > 0).astype(float)
    im2 = ax2.imshow(consensus_binary, cmap='Greys', vmin=0, vmax=1)
    ax2.set_title(f'Consensus Binary Edges\n({consensus_n_edges:,} edges)', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Channel')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot 3: Example subject (best match)
    ax3 = fig.add_subplot(3, 4, 3)
    best_idx = df_results['Jaccard'].idxmax()
    best_subj_id = df_results.loc[best_idx, 'Subject_ID']
    best_matrix_idx = subject_ids.index(best_subj_id)
    best_subj_binary = threshold_matrix_to_density(all_matrices[best_matrix_idx], consensus_n_edges)
    im3 = ax3.imshow(best_subj_binary, cmap='Greys', vmin=0, vmax=1)
    ax3.set_title(f'Best Subject: {best_subj_id}\n(Jaccard={df_results.loc[best_idx, "Jaccard"]:.3f})', 
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Channel')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Plot 4: Jaccard Boxplot
    ax4 = fig.add_subplot(3, 4, 4)
    bp = ax4.boxplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                     labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Jaccard'], ad_color), 
                                        (hc_results['Jaccard'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax4.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white')
    
    sig_str_j = '***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns'
    ax4.set_ylabel('Jaccard Similarity', fontsize=11)
    ax4.set_title(f'Jaccard by Group\n(p = {p_val_j:.4f} {sig_str_j})', fontweight='bold', fontsize=10)
    ax4.axhline(y=df_results['Jaccard'].mean(), color='gray', linestyle='--', linewidth=1.5)
    
    # Plot 5: Shared Edges Histogram
    ax5 = fig.add_subplot(3, 4, 5)
    bins = np.linspace(df_results['N_Shared_Edges'].min() - 10, 
                       df_results['N_Shared_Edges'].max() + 10, 20)
    ax5.hist(ad_results['N_Shared_Edges'], bins=bins, alpha=0.6, color=ad_color, 
             label=f'AD (μ={ad_results["N_Shared_Edges"].mean():.0f})', edgecolor='white')
    ax5.hist(hc_results['N_Shared_Edges'], bins=bins, alpha=0.6, color=hc_color, 
             label=f'HC (μ={hc_results["N_Shared_Edges"].mean():.0f})', edgecolor='white')
    ax5.axvline(consensus_n_edges, color='black', linestyle='--', linewidth=2, label=f'Consensus: {consensus_n_edges}')
    ax5.set_xlabel('Shared Edges', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Number of Shared Edges', fontweight='bold', fontsize=10)
    ax5.legend(loc='upper left', fontsize=9)
    
    # Plot 6: Jaccard vs Pearson
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.scatter(ad_results['Pearson_r_full'], ad_results['Jaccard'], 
                c=ad_color, label=f'AD (n={n_ad})', alpha=0.7, s=50, edgecolor='white')
    ax6.scatter(hc_results['Pearson_r_full'], hc_results['Jaccard'], 
                c=hc_color, label=f'HC (n={n_hc})', alpha=0.7, s=50, edgecolor='white')
    ax6.set_xlabel('Pearson r (full)', fontsize=11)
    ax6.set_ylabel('Jaccard', fontsize=11)
    ax6.set_title('Jaccard vs Pearson r', fontweight='bold', fontsize=10)
    ax6.legend(loc='lower right', fontsize=9)
    
    # Plot 7: Bar chart - All subjects ranked by Jaccard
    ax7 = fig.add_subplot(3, 4, (7, 8))
    colors_bar = [ad_color if g == 'AD' else hc_color for g in df_results['Group']]
    ax7.bar(range(n_subjects), df_results['Jaccard'], color=colors_bar, alpha=0.8)
    ax7.axhline(y=df_results['Jaccard'].mean(), color='black', linestyle='--', linewidth=1.5,
                label=f'Mean: {df_results["Jaccard"].mean():.3f}')
    ax7.set_xlabel('Subject Rank (by Jaccard)', fontsize=11)
    ax7.set_ylabel('Jaccard Similarity', fontsize=11)
    ax7.set_title('All Subjects Ranked by Edge Overlap with Consensus', fontweight='bold', fontsize=10)
    ax7.set_xlim([-1, n_subjects])
    ax7.legend(loc='upper right')
    
    # Plot 8: Dice vs Jaccard
    ax8 = fig.add_subplot(3, 4, 9)
    ax8.scatter(df_results['Jaccard'], df_results['Dice'], 
                c=[ad_color if g == 'AD' else hc_color for g in df_results['Group']],
                alpha=0.7, s=50, edgecolor='white')
    ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal
    ax8.set_xlabel('Jaccard', fontsize=11)
    ax8.set_ylabel('Dice', fontsize=11)
    ax8.set_title('Dice vs Jaccard\n(Both measure overlap)', fontweight='bold', fontsize=10)
    
    # Plot 9: Violin plot - Jaccard
    ax9 = fig.add_subplot(3, 4, 10)
    parts = ax9.violinplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                           positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([ad_color, hc_color][i])
        pc.set_alpha(0.6)
    ax9.set_xticks([1, 2])
    ax9.set_xticklabels(['AD', 'HC'])
    ax9.set_ylabel('Jaccard', fontsize=11)
    ax9.set_title('Jaccard Distribution', fontweight='bold', fontsize=10)
    
    # Plot 10-12: Summary text
    ax10 = fig.add_subplot(3, 4, (11, 12))
    ax10.axis('off')
    
    best_subj = df_results.iloc[0]
    worst_subj = df_results.iloc[-1]
    
    summary_text = f"""
    ══════════════════════════════════════════════════════════════════════════════
                    DENSITY-MATCHED COMPARISON: RESULTS SUMMARY
    ══════════════════════════════════════════════════════════════════════════════
    
    CONSENSUS MATRIX
    ────────────────────────────────────────────────────────────────────────────────
    Shape:             {n_channels} x {n_channels}
    Non-zero edges:    {consensus_n_edges:,} / {sparsity_info['total_possible_edges']:,}
    Density:           {consensus_density*100:.2f}%
    
    DENSITY MATCHING
    ────────────────────────────────────────────────────────────────────────────────
    For each subject, we threshold their correlation matrix to keep
    exactly {consensus_n_edges:,} edges (top edges by weight).
    This ensures fair binary comparison with the consensus.
    
    SUBJECTS: {n_subjects} total ({n_ad} AD + {n_hc} HC)
    
    JACCARD SIMILARITY (edge overlap)
    ────────────────────────────────────────────────────────────────────────────────
    AD:  {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}  |  Shared edges: {ad_results['N_Shared_Edges'].mean():.0f} ({ad_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}%)
    HC:  {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}  |  Shared edges: {hc_results['N_Shared_Edges'].mean():.0f} ({hc_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}%)
    t-test: t = {t_stat_j:.3f}, p = {p_val_j:.4f}  {sig_str_j}
    
    EXTREME SUBJECTS
    ────────────────────────────────────────────────────────────────────────────────
    Best match:   {best_subj['Subject_ID']} ({best_subj['Group']}) - Jaccard = {best_subj['Jaccard']:.4f}
    Worst match:  {worst_subj['Subject_ID']} ({worst_subj['Group']}) - Jaccard = {worst_subj['Jaccard']:.4f}
    """
    
    ax10.text(0.02, 0.98, summary_text, transform=ax10.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Subject vs Consensus: Density-Matched Comparison',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_filename = output_folder / 'subject_vs_saved_consensus.png'
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 1 saved: {fig_filename}")
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Matrix Comparison (Consensus vs Best/Worst Subjects)
    # =========================================================================
    print("  Creating Figure 2: Matrix Comparisons...")
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    
    # Get best and worst subjects
    best_idx = df_results['Jaccard'].idxmax()
    worst_idx = df_results['Jaccard'].idxmin()
    best_subj_id = df_results.loc[best_idx, 'Subject_ID']
    worst_subj_id = df_results.loc[worst_idx, 'Subject_ID']
    best_matrix_idx = subject_ids.index(best_subj_id)
    worst_matrix_idx = subject_ids.index(worst_subj_id)
    
    # Consensus matrix
    ax = axes2[0, 0]
    vmax = np.percentile(consensus_matrix[consensus_matrix > 0], 95) if np.any(consensus_matrix > 0) else 1
    im = ax.imshow(consensus_matrix, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title(f'(A) Consensus Matrix\n({consensus_n_edges} edges)', fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Weight')
    
    # Consensus binary
    ax = axes2[0, 1]
    consensus_binary = (consensus_matrix > 0).astype(float)
    im = ax.imshow(consensus_binary, cmap='Greys', vmin=0, vmax=1)
    ax.set_title('(B) Consensus Binary', fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Edge')
    
    # Best subject weighted
    ax = axes2[0, 2]
    best_matrix = all_matrices[best_matrix_idx]
    im = ax.imshow(best_matrix, cmap='hot', vmin=0, vmax=np.percentile(best_matrix, 95))
    ax.set_title(f'(C) Best Subject: {best_subj_id}\n(Full correlation)', fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax, fraction=0.046, label='|r|')
    
    # Best subject thresholded
    ax = axes2[0, 3]
    best_binary = threshold_matrix_to_density(best_matrix, consensus_n_edges)
    im = ax.imshow(best_binary, cmap='Greys', vmin=0, vmax=1)
    ax.set_title(f'(D) Best Subject Thresholded\n(Top {consensus_n_edges} edges)', fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Edge')
    
    # Edge overlap visualization (best subject)
    ax = axes2[1, 0]
    overlap_matrix = np.zeros((n_channels, n_channels, 3))
    # Red = consensus only, Blue = subject only, Green = both
    consensus_only = (consensus_binary == 1) & (best_binary == 0)
    subject_only = (consensus_binary == 0) & (best_binary == 1)
    both = (consensus_binary == 1) & (best_binary == 1)
    overlap_matrix[:, :, 0] = consensus_only.astype(float)  # Red
    overlap_matrix[:, :, 2] = subject_only.astype(float)   # Blue
    overlap_matrix[:, :, 1] = both.astype(float)           # Green
    ax.imshow(overlap_matrix)
    ax.set_title(f'(E) Edge Overlap (Best Subject)\nGreen=Both, Red=Consensus, Blue=Subject', fontweight='bold', fontsize=10)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    
    # Worst subject weighted
    ax = axes2[1, 1]
    worst_matrix = all_matrices[worst_matrix_idx]
    im = ax.imshow(worst_matrix, cmap='hot', vmin=0, vmax=np.percentile(worst_matrix, 95))
    ax.set_title(f'(F) Worst Subject: {worst_subj_id}\n(Full correlation)', fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax, fraction=0.046, label='|r|')
    
    # Worst subject thresholded
    ax = axes2[1, 2]
    worst_binary = threshold_matrix_to_density(worst_matrix, consensus_n_edges)
    im = ax.imshow(worst_binary, cmap='Greys', vmin=0, vmax=1)
    ax.set_title(f'(G) Worst Subject Thresholded\n(Top {consensus_n_edges} edges)', fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Edge')
    
    # Edge overlap visualization (worst subject)
    ax = axes2[1, 3]
    overlap_matrix_worst = np.zeros((n_channels, n_channels, 3))
    consensus_only_w = (consensus_binary == 1) & (worst_binary == 0)
    subject_only_w = (consensus_binary == 0) & (worst_binary == 1)
    both_w = (consensus_binary == 1) & (worst_binary == 1)
    overlap_matrix_worst[:, :, 0] = consensus_only_w.astype(float)
    overlap_matrix_worst[:, :, 2] = subject_only_w.astype(float)
    overlap_matrix_worst[:, :, 1] = both_w.astype(float)
    ax.imshow(overlap_matrix_worst)
    ax.set_title(f'(H) Edge Overlap (Worst Subject)\nGreen=Both, Red=Consensus, Blue=Subject', fontweight='bold', fontsize=10)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    
    plt.suptitle('Matrix Comparison: Consensus vs Individual Subjects', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig2_filename = output_folder / 'figure2_matrix_comparison.png'
    plt.savefig(fig2_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 2 saved: {fig2_filename}")
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Statistical Analysis (Thesis Quality)
    # =========================================================================
    print("  Creating Figure 3: Statistical Analysis...")
    
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 10))
    
    # Plot A: Jaccard Distribution with Statistics
    ax = axes3[0, 0]
    # Histogram
    bins = np.linspace(df_results['Jaccard'].min() - 0.005, 
                       df_results['Jaccard'].max() + 0.005, 25)
    ax.hist(ad_results['Jaccard'], bins=bins, alpha=0.7, color=ad_color, 
            label=f'AD (n={n_ad})', edgecolor='white', linewidth=0.5)
    ax.hist(hc_results['Jaccard'], bins=bins, alpha=0.7, color=hc_color, 
            label=f'HC (n={n_hc})', edgecolor='white', linewidth=0.5)
    ax.axvline(ad_results['Jaccard'].mean(), color=ad_color, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(hc_results['Jaccard'].mean(), color=hc_color, linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Jaccard Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(A) Jaccard Similarity Distribution', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add statistics annotation
    stats_text = f"AD: {ad_results['Jaccard'].mean():.3f}±{ad_results['Jaccard'].std():.3f}\n"
    stats_text += f"HC: {hc_results['Jaccard'].mean():.3f}±{hc_results['Jaccard'].std():.3f}\n"
    stats_text += f"p = {p_val_j:.2e}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot B: Boxplot with Individual Points
    ax = axes3[0, 1]
    bp = ax.boxplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                    labels=['AD', 'HC'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_alpha(0.6)
    bp['medians'][0].set_color('black')
    bp['medians'][1].set_color('black')
    
    # Add jittered points
    for i, (data, color) in enumerate([(ad_results['Jaccard'], ad_color), 
                                        (hc_results['Jaccard'], hc_color)]):
        x = np.random.normal(i+1, 0.08, size=len(data))
        ax.scatter(x, data, alpha=0.6, color=color, s=40, edgecolor='white', linewidth=0.5, zorder=3)
    
    # Add significance bar
    y_max = max(ad_results['Jaccard'].max(), hc_results['Jaccard'].max())
    sig_str = '***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns'
    ax.plot([1, 1, 2, 2], [y_max + 0.005, y_max + 0.008, y_max + 0.008, y_max + 0.005], 'k-', linewidth=1)
    ax.text(1.5, y_max + 0.01, sig_str, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('(B) Group Comparison', fontweight='bold')
    ax.set_ylim([df_results['Jaccard'].min() - 0.01, y_max + 0.02])
    
    # Plot C: Shared Edges Count
    ax = axes3[0, 2]
    bins_edges = np.linspace(df_results['N_Shared_Edges'].min() - 5, 
                             df_results['N_Shared_Edges'].max() + 5, 20)
    ax.hist(ad_results['N_Shared_Edges'], bins=bins_edges, alpha=0.7, color=ad_color, 
            label=f'AD', edgecolor='white', linewidth=0.5)
    ax.hist(hc_results['N_Shared_Edges'], bins=bins_edges, alpha=0.7, color=hc_color, 
            label=f'HC', edgecolor='white', linewidth=0.5)
    ax.axvline(consensus_n_edges * 0.5, color='gray', linestyle=':', linewidth=2, 
               label=f'50% of consensus ({consensus_n_edges//2})')
    ax.set_xlabel('Number of Shared Edges')
    ax.set_ylabel('Frequency')
    ax.set_title('(C) Shared Edge Count Distribution', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # Plot D: Violin Plot
    ax = axes3[1, 0]
    parts = ax.violinplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                          positions=[1, 2], showmeans=True, showmedians=True, widths=0.8)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([ad_color, hc_color][i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('darkgray')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['AD', 'HC'])
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('(D) Distribution Shape (Violin Plot)', fontweight='bold')
    
    # Plot E: Correlation between Jaccard and Shared Edges
    ax = axes3[1, 1]
    ax.scatter(ad_results['N_Shared_Edges'], ad_results['Jaccard'], 
               c=ad_color, label='AD', alpha=0.7, s=50, edgecolor='white')
    ax.scatter(hc_results['N_Shared_Edges'], hc_results['Jaccard'], 
               c=hc_color, label='HC', alpha=0.7, s=50, edgecolor='white')
    
    # Add regression line
    all_shared = df_results['N_Shared_Edges'].values
    all_jaccard = df_results['Jaccard'].values
    z = np.polyfit(all_shared, all_jaccard, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(all_shared.min(), all_shared.max(), 100)
    ax.plot(x_line, p_line(x_line), 'k--', alpha=0.5, linewidth=1.5)
    
    corr_sj, p_sj = stats.pearsonr(all_shared, all_jaccard)
    ax.set_xlabel('Number of Shared Edges')
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title(f'(E) Shared Edges vs Jaccard\n(r = {corr_sj:.3f})', fontweight='bold')
    ax.legend(loc='lower right')
    
    # Plot F: Effect Size Visualization
    ax = axes3[1, 2]
    
    # Calculate effect sizes
    def cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        var1, var2 = g1.var(), g2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (g1.mean() - g2.mean()) / pooled_std
    
    metrics = ['Jaccard', 'Dice', 'N_Shared_Edges', 'Pearson_r_full']
    effect_sizes = []
    metric_labels = ['Jaccard', 'Dice', 'Shared Edges', 'Pearson r']
    
    for metric in metrics:
        d = cohens_d(ad_results[metric], hc_results[metric])
        effect_sizes.append(d)
    
    colors_es = [ad_color if d > 0 else hc_color for d in effect_sizes]
    bars = ax.barh(range(len(metrics)), effect_sizes, color=colors_es, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title("(F) Effect Sizes (AD vs HC)\n(|d|>0.2=small, |d|>0.8=large)", fontweight='bold')
    
    # Add value labels
    for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
        ax.text(d + 0.05 if d > 0 else d - 0.05, i, f'{d:.2f}', 
                va='center', ha='left' if d > 0 else 'right', fontsize=10)
    
    plt.suptitle('Statistical Analysis of Subject-Consensus Similarity', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig3_filename = output_folder / 'figure3_statistical_analysis.png'
    plt.savefig(fig3_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 3 saved: {fig3_filename}")
    plt.close()
    
    # =========================================================================
    # FIGURE 4: Subject Ranking & Individual Analysis
    # =========================================================================
    print("  Creating Figure 4: Subject Ranking...")
    
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot A: All subjects ranked by Jaccard (horizontal bars)
    ax = axes4[0, 0]
    sorted_df = df_results.sort_values('Jaccard', ascending=True)
    colors_rank = [ad_color if g == 'AD' else hc_color for g in sorted_df['Group']]
    y_pos = np.arange(len(sorted_df))
    bars = ax.barh(y_pos, sorted_df['Jaccard'], color=colors_rank, alpha=0.8, height=0.8)
    ax.axvline(df_results['Jaccard'].mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'Mean: {df_results["Jaccard"].mean():.3f}')
    ax.set_xlabel('Jaccard Similarity')
    ax.set_ylabel('Subject (ranked)')
    ax.set_title('(A) All Subjects Ranked by Jaccard', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_yticks([])  # Hide individual labels for clarity
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ad_color, alpha=0.8, label=f'AD (n={n_ad})'),
                       Patch(facecolor=hc_color, alpha=0.8, label=f'HC (n={n_hc})')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Plot B: Top 15 and Bottom 15 subjects
    ax = axes4[0, 1]
    top_15 = df_results.head(15)
    bottom_15 = df_results.tail(15).iloc[::-1]
    
    # Combine with gap
    combined_labels = list(top_15['Subject_ID']) + ['...'] + list(bottom_15['Subject_ID'])
    combined_values = list(top_15['Jaccard']) + [np.nan] + list(bottom_15['Jaccard'])
    combined_colors = [ad_color if g == 'AD' else hc_color for g in top_15['Group']]
    combined_colors += ['white']
    combined_colors += [ad_color if g == 'AD' else hc_color for g in bottom_15['Group']]
    
    y_pos = np.arange(len(combined_labels))
    bars = ax.barh(y_pos, combined_values, color=combined_colors, alpha=0.8, height=0.7,
                   edgecolor=['black' if c != 'white' else 'white' for c in combined_colors])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Jaccard Similarity')
    ax.set_title('(B) Top 15 and Bottom 15 Subjects', fontweight='bold')
    ax.axvline(df_results['Jaccard'].mean(), color='black', linestyle='--', linewidth=1.5)
    
    # Plot C: Jaccard over ranks
    ax = axes4[1, 0]
    ranks = np.arange(1, n_subjects + 1)
    colors_by_rank = [ad_color if g == 'AD' else hc_color for g in df_results['Group']]
    ax.scatter(ranks, df_results['Jaccard'], c=colors_by_rank, s=60, alpha=0.7, edgecolor='white')
    
    # Fit and plot trend line
    z = np.polyfit(ranks, df_results['Jaccard'], 2)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(1, n_subjects, 100)
    ax.plot(x_fit, p_fit(x_fit), 'k-', alpha=0.5, linewidth=2, label='Trend')
    
    ax.axhline(df_results['Jaccard'].mean(), color='gray', linestyle='--', linewidth=1.5,
               label=f'Mean: {df_results["Jaccard"].mean():.3f}')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('(C) Jaccard vs Rank', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Plot D: Group proportion at each rank quartile
    ax = axes4[1, 1]
    
    # Divide into quartiles
    q1 = df_results.head(n_subjects // 4)
    q2 = df_results.iloc[n_subjects // 4: n_subjects // 2]
    q3 = df_results.iloc[n_subjects // 2: 3 * n_subjects // 4]
    q4 = df_results.tail(n_subjects - 3 * n_subjects // 4)
    
    quartiles = [q1, q2, q3, q4]
    quartile_labels = ['Q1\n(Top 25%)', 'Q2', 'Q3', 'Q4\n(Bottom 25%)']
    
    ad_counts = [sum(q['Group'] == 'AD') for q in quartiles]
    hc_counts = [sum(q['Group'] == 'HC') for q in quartiles]
    
    x = np.arange(4)
    width = 0.35
    bars1 = ax.bar(x - width/2, ad_counts, width, label='AD', color=ad_color, alpha=0.8)
    bars2 = ax.bar(x + width/2, hc_counts, width, label='HC', color=hc_color, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(quartile_labels)
    ax.set_ylabel('Number of Subjects')
    ax.set_title('(D) Group Distribution by Rank Quartile', fontweight='bold')
    ax.legend()
    
    # Add count labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Subject Ranking Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig4_filename = output_folder / 'figure4_subject_ranking.png'
    plt.savefig(fig4_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 4 saved: {fig4_filename}")
    plt.close()
    
    # =========================================================================
    # FIGURE 5: Edge-Level Analysis
    # =========================================================================
    print("  Creating Figure 5: Edge-Level Analysis...")
    
    fig5, axes5 = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get edge-level data
    triu_idx = np.triu_indices(n_channels, k=1)
    consensus_edges = consensus_matrix[triu_idx]
    consensus_binary_vec = (consensus_edges > 0).astype(int)
    
    # Calculate edge frequency across subjects
    edge_frequency = np.zeros(len(triu_idx[0]))
    for matrix in all_matrices:
        subj_binary = threshold_matrix_to_density(matrix, consensus_n_edges)
        edge_frequency += subj_binary[triu_idx]
    edge_frequency = edge_frequency / n_subjects * 100  # Convert to percentage
    
    # Plot A: Edge frequency histogram
    ax = axes5[0, 0]
    ax.hist(edge_frequency, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_xlabel('Edge Frequency (%)')
    ax.set_ylabel('Number of Edges')
    ax.set_title('(A) How Often Each Edge Appears\n(Across All Subjects)', fontweight='bold')
    ax.legend()
    
    # Plot B: Edge frequency for consensus edges only
    ax = axes5[0, 1]
    consensus_edge_freq = edge_frequency[consensus_binary_vec == 1]
    ax.hist(consensus_edge_freq, bins=20, color=consensus_color, alpha=0.7, edgecolor='white')
    ax.axvline(consensus_edge_freq.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Mean: {consensus_edge_freq.mean():.1f}%')
    ax.set_xlabel('Edge Frequency (%)')
    ax.set_ylabel('Number of Consensus Edges')
    ax.set_title('(B) Consensus Edge Recovery Rate', fontweight='bold')
    ax.legend()
    
    # Plot C: Scatter - Consensus weight vs Edge frequency
    ax = axes5[0, 2]
    consensus_nonzero_mask = consensus_edges > 0
    ax.scatter(consensus_edges[consensus_nonzero_mask], 
               edge_frequency[consensus_nonzero_mask],
               alpha=0.5, s=20, c='steelblue')
    
    # Fit regression
    if np.sum(consensus_nonzero_mask) > 2:
        corr_wf, p_wf = stats.pearsonr(consensus_edges[consensus_nonzero_mask], 
                                        edge_frequency[consensus_nonzero_mask])
        ax.set_title(f'(C) Consensus Weight vs Recovery\n(r = {corr_wf:.3f})', fontweight='bold')
    else:
        ax.set_title('(C) Consensus Weight vs Recovery', fontweight='bold')
    ax.set_xlabel('Consensus Edge Weight')
    ax.set_ylabel('Edge Frequency (%)')
    
    # Plot D: Edge overlap between AD and HC
    ax = axes5[1, 0]
    
    # Calculate group-specific edge frequencies
    ad_edge_freq = np.zeros(len(triu_idx[0]))
    hc_edge_freq = np.zeros(len(triu_idx[0]))
    
    for i, (matrix, group) in enumerate(zip(all_matrices, group_labels)):
        subj_binary = threshold_matrix_to_density(matrix, consensus_n_edges)
        if group == 'AD':
            ad_edge_freq += subj_binary[triu_idx]
        else:
            hc_edge_freq += subj_binary[triu_idx]
    
    ad_edge_freq = ad_edge_freq / n_ad * 100
    hc_edge_freq = hc_edge_freq / n_hc * 100
    
    ax.scatter(ad_edge_freq, hc_edge_freq, alpha=0.3, s=10, c='gray')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('AD Edge Frequency (%)')
    ax.set_ylabel('HC Edge Frequency (%)')
    corr_adHC, _ = stats.pearsonr(ad_edge_freq, hc_edge_freq)
    ax.set_title(f'(D) AD vs HC Edge Consistency\n(r = {corr_adHC:.3f})', fontweight='bold')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    # Plot E: Consensus edge recovery by group
    ax = axes5[1, 1]
    
    # For each subject, count how many consensus edges they recover
    ad_recovery = []
    hc_recovery = []
    for i, (matrix, group) in enumerate(zip(all_matrices, group_labels)):
        subj_binary = threshold_matrix_to_density(matrix, consensus_n_edges)
        subj_edges = subj_binary[triu_idx]
        recovery = np.sum((subj_edges == 1) & (consensus_binary_vec == 1)) / consensus_n_edges * 100
        if group == 'AD':
            ad_recovery.append(recovery)
        else:
            hc_recovery.append(recovery)
    
    bp = ax.boxplot([ad_recovery, hc_recovery], labels=['AD', 'HC'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_alpha(0.6)
    
    for i, (data, color) in enumerate([(ad_recovery, ad_color), (hc_recovery, hc_color)]):
        x = np.random.normal(i+1, 0.06, size=len(data))
        ax.scatter(x, data, alpha=0.6, color=color, s=30, edgecolor='white', zorder=3)
    
    ax.set_ylabel('Consensus Edge Recovery (%)')
    ax.set_title('(E) Consensus Edge Recovery by Group', fontweight='bold')
    
    # Plot F: Summary statistics table
    ax = axes5[1, 2]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['─' * 20, '─' * 15],
        ['Total edges (possible)', f'{sparsity_info["total_possible_edges"]:,}'],
        ['Consensus edges', f'{consensus_n_edges:,}'],
        ['Consensus density', f'{consensus_density*100:.2f}%'],
        ['', ''],
        ['Avg. edges in subject', f'{consensus_n_edges:,}'],
        ['(after thresholding)', ''],
        ['', ''],
        ['Consensus edge recovery:', ''],
        [f'  AD mean', f'{np.mean(ad_recovery):.1f}%'],
        [f'  HC mean', f'{np.mean(hc_recovery):.1f}%'],
        ['', ''],
        ['Highly consistent edges', f'{np.sum(edge_frequency > 80):,}'],
        ['(present in >80% subjects)', f'({np.sum(edge_frequency > 80)/len(edge_frequency)*100:.1f}%)'],
    ]
    
    table_text = '\n'.join([f'{row[0]:<25} {row[1]:>15}' for row in table_data])
    ax.text(0.1, 0.95, table_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('(F) Edge Statistics Summary', fontweight='bold')
    
    plt.suptitle('Edge-Level Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig5_filename = output_folder / 'figure5_edge_analysis.png'
    plt.savefig(fig5_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 5 saved: {fig5_filename}")
    plt.close()
    
    # =========================================================================
    # FIGURE 6: Publication-Ready Summary Figure
    # =========================================================================
    print("  Creating Figure 6: Publication Summary...")
    
    fig6 = plt.figure(figsize=(12, 10))
    
    # Use gridspec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig6, hspace=0.35, wspace=0.35)
    
    # A: Consensus matrix
    ax1 = fig6.add_subplot(gs[0, 0])
    vmax = np.percentile(consensus_matrix[consensus_matrix > 0], 95) if np.any(consensus_matrix > 0) else 1
    im1 = ax1.imshow(consensus_matrix, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_title(f'A. Consensus\n({consensus_density*100:.1f}% density)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Channel', fontsize=10)
    ax1.set_ylabel('Channel', fontsize=10)
    cbar = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    
    # B: Best subject overlap
    ax2 = fig6.add_subplot(gs[0, 1])
    ax2.imshow(overlap_matrix)
    best_jaccard = df_results.loc[best_idx, 'Jaccard']
    ax2.set_title(f'B. Best Match\n(J={best_jaccard:.3f})', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Channel', fontsize=10)
    ax2.set_ylabel('Channel', fontsize=10)
    
    # C: Worst subject overlap
    ax3 = fig6.add_subplot(gs[0, 2])
    ax3.imshow(overlap_matrix_worst)
    worst_jaccard = df_results.loc[worst_idx, 'Jaccard']
    ax3.set_title(f'C. Worst Match\n(J={worst_jaccard:.3f})', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Channel', fontsize=10)
    ax3.set_ylabel('Channel', fontsize=10)
    
    # D: Main result - Boxplot
    ax4 = fig6.add_subplot(gs[1, 0])
    bp = ax4.boxplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                     labels=['AD', 'HC'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Jaccard'], ad_color), 
                                        (hc_results['Jaccard'], hc_color)]):
        x = np.random.normal(i+1, 0.06, size=len(data))
        ax4.scatter(x, data, alpha=0.6, color=color, s=35, edgecolor='white', zorder=3)
    
    y_max = max(ad_results['Jaccard'].max(), hc_results['Jaccard'].max())
    sig_str = '***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns'
    ax4.plot([1, 1, 2, 2], [y_max + 0.004, y_max + 0.006, y_max + 0.006, y_max + 0.004], 'k-', linewidth=1)
    ax4.text(1.5, y_max + 0.007, sig_str, ha='center', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('Jaccard Similarity', fontsize=10)
    ax4.set_title('D. Group Comparison', fontweight='bold', fontsize=11)
    
    # E: Distribution
    ax5 = fig6.add_subplot(gs[1, 1])
    bins = np.linspace(df_results['Jaccard'].min() - 0.005, 
                       df_results['Jaccard'].max() + 0.005, 20)
    ax5.hist(ad_results['Jaccard'], bins=bins, alpha=0.7, color=ad_color, 
             label=f'AD (n={n_ad})', edgecolor='white')
    ax5.hist(hc_results['Jaccard'], bins=bins, alpha=0.7, color=hc_color, 
             label=f'HC (n={n_hc})', edgecolor='white')
    ax5.set_xlabel('Jaccard Similarity', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('E. Distribution', fontweight='bold', fontsize=11)
    ax5.legend(loc='upper right', fontsize=9)
    
    # F: Subject ranking
    ax6 = fig6.add_subplot(gs[1, 2])
    colors_bar = [ad_color if g == 'AD' else hc_color for g in df_results['Group']]
    ax6.bar(range(n_subjects), df_results['Jaccard'], color=colors_bar, alpha=0.8, width=1.0)
    ax6.axhline(df_results['Jaccard'].mean(), color='black', linestyle='--', linewidth=1.5)
    ax6.set_xlabel('Subject Rank', fontsize=10)
    ax6.set_ylabel('Jaccard', fontsize=10)
    ax6.set_title('F. All Subjects Ranked', fontweight='bold', fontsize=11)
    ax6.set_xlim([-1, n_subjects])
    
    # G-I: Summary panel
    ax7 = fig6.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    ══════════════════════════════════════════════════════════════════════════════════════════════════════════
                                           DENSITY-MATCHED COMPARISON: KEY RESULTS
    ══════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    CONSENSUS MATRIX                                         METHODOLOGY
    ────────────────────────────────────                     ────────────────────────────────────
    • Channels: {n_channels}                                          • For each subject, the correlation matrix
    • Non-zero edges: {consensus_n_edges:,} / {sparsity_info["total_possible_edges"]:,}                    is thresholded to keep the TOP {consensus_n_edges} edges
    • Density: {consensus_density*100:.2f}%                                       • This matches the consensus density exactly
                                                             • Jaccard similarity measures binary overlap
    
    JACCARD SIMILARITY (Edge Overlap)                        STATISTICAL COMPARISON
    ────────────────────────────────────                     ────────────────────────────────────
    • AD (n={n_ad}):  {ad_results['Jaccard'].mean():.4f} ± {ad_results['Jaccard'].std():.4f}                      • t-statistic: {t_stat_j:.3f}
    • HC (n={n_hc}):  {hc_results['Jaccard'].mean():.4f} ± {hc_results['Jaccard'].std():.4f}                      • p-value: {p_val_j:.2e}  {sig_str}
    • Difference: {(ad_results['Jaccard'].mean() - hc_results['Jaccard'].mean())*100:.2f}% higher in AD                  • Cohen's d: {cohens_d(ad_results['Jaccard'], hc_results['Jaccard']):.3f}
    
    SHARED EDGES                                             INTERPRETATION
    ────────────────────────────────────                     ────────────────────────────────────
    • AD mean: {ad_results['N_Shared_Edges'].mean():.0f} edges ({ad_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}% of consensus)              Matrix overlap: GREEN = Both | RED = Consensus only | BLUE = Subject only
    • HC mean: {hc_results['N_Shared_Edges'].mean():.0f} edges ({hc_results['N_Shared_Edges'].mean()/consensus_n_edges*100:.1f}% of consensus)
    ══════════════════════════════════════════════════════════════════════════════════════════════════════════
    """
    
    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    
    plt.suptitle('Subject vs Consensus: Density-Matched Comparison Results', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    fig6_filename = output_folder / 'figure6_publication_summary.png'
    plt.savefig(fig6_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 6 saved: {fig6_filename}")
    plt.close()
    
    print(f"\n  ✓ All 6 figures saved to: {output_folder}/")
    
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
  ✓ consensus_matrix.npy        - Consensus matrix used
  ✓ sparsity_info.txt           - Sparsity/density information  
  ✓ subject_vs_saved_consensus.csv - All subject similarity metrics
  ✓ subject_vs_saved_consensus.png - Visualization figure
  ✓ all_subject_matrices.npy    - All individual subject matrices
  ✓ subject_ids.npy             - Subject ID array
  ✓ group_labels.npy            - Group labels (AD/HC)

CONSENSUS SPARSITY:
  Edges: {consensus_n_edges:,} / {sparsity_info['total_possible_edges']:,}
  Density: {consensus_density*100:.2f}%

KEY METRICS:
  Jaccard (AD): {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}
  Jaccard (HC): {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}
  Difference: p = {p_val_j:.4f} {sig_str_j}
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
