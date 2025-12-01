"""
=============================================================================
COMPARING CONNECTIVITY MATRICES - EACH SUBJECT vs OVERALL CONSENSUS
=============================================================================

This script compares each individual subject against the OVERALL consensus 
matrix (combining all subjects from both AD and HC groups).

Outputs:
- CSV file with Pearson r and Jaccard similarity for each subject
- Informative visualization plots

RUN: python compare_subject_overall_consensus.py

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("COMPARING EACH SUBJECT vs OVERALL CONSENSUS MATRIX")
print("="*70)

# =============================================================================
# REAL EEG FILE PATHS
# =============================================================================

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


def fisher_z_transform(r):
    """Apply Fisher z-transformation: z = arctanh(r)"""
    r_clipped = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r_clipped)


def fisher_z_inverse(z):
    """Apply inverse Fisher z-transformation: r = tanh(z)"""
    return np.tanh(z)


def compute_consensus_matrix(corr_matrices):
    """Compute consensus matrix using Fisher-z averaging."""
    if len(corr_matrices) == 0:
        raise ValueError("No matrices provided")
    
    stack = np.stack(corr_matrices, axis=0)
    z_stack = fisher_z_transform(stack)
    z_mean = np.mean(z_stack, axis=0)
    consensus = np.abs(fisher_z_inverse(z_mean))
    np.fill_diagonal(consensus, 0)
    
    return consensus


def compare_to_consensus(subject_matrix, consensus_matrix, sparsity=0.15):
    """Compare a subject's connectivity matrix to the consensus."""
    n = subject_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    vec_subj = subject_matrix[triu_idx]
    vec_cons = consensus_matrix[triu_idx]
    
    results = {}
    
    # Pearson correlation
    r, p = stats.pearsonr(vec_subj, vec_cons)
    results['pearson_r'] = r
    results['pearson_p'] = p
    
    # Spearman correlation
    rho, _ = stats.spearmanr(vec_subj, vec_cons)
    results['spearman_rho'] = rho
    
    # Jaccard similarity (top edges)
    n_edges = int(sparsity * len(vec_subj))
    thresh_subj = np.sort(vec_subj)[::-1][n_edges]
    thresh_cons = np.sort(vec_cons)[::-1][n_edges]
    bin_subj = (vec_subj >= thresh_subj).astype(int)
    bin_cons = (vec_cons >= thresh_cons).astype(int)
    intersection = np.sum((bin_subj == 1) & (bin_cons == 1))
    union = np.sum((bin_subj == 1) | (bin_cons == 1))
    results['jaccard'] = intersection / union if union > 0 else 0
    
    # Cosine similarity
    cos_sim = np.dot(vec_subj, vec_cons) / (np.linalg.norm(vec_subj) * np.linalg.norm(vec_cons))
    results['cosine_similarity'] = cos_sim
    
    # Mean absolute difference
    results['mean_abs_diff'] = np.mean(np.abs(vec_subj - vec_cons))
    
    # RMSD
    results['rmsd'] = np.sqrt(np.mean((vec_subj - vec_cons)**2))
    
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Main function to compute each subject's similarity to overall consensus."""
    
    # =========================================================================
    # STEP 1: LOAD OR CREATE DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    all_matrices = []
    subject_ids = []
    group_labels = []
    
    # Try to load real EEG data
    use_synthetic = True
    
    for filepath in AD_FILES:
        if Path(filepath).exists():
            use_synthetic = False
            break
    
    if not use_synthetic:
        logger.info("Loading real EEG files...")
        
        for filepath in AD_FILES:
            if Path(filepath).exists():
                data = load_eeg_data(filepath)
                if data is not None:
                    corr_matrix = compute_correlation_matrix(data)
                    all_matrices.append(corr_matrix)
                    subject_ids.append(extract_subject_id(filepath))
                    group_labels.append('AD')
        
        for filepath in HC_FILES:
            if Path(filepath).exists():
                data = load_eeg_data(filepath)
                if data is not None:
                    corr_matrix = compute_correlation_matrix(data)
                    all_matrices.append(corr_matrix)
                    subject_ids.append(extract_subject_id(filepath))
                    group_labels.append('HC')
    
    if use_synthetic or len(all_matrices) == 0:
        print("\nUsing synthetic data for demonstration...")
        print("(Real EEG files not found at specified paths)")
        
        np.random.seed(42)
        n_channels = 128
        
        # Create baseline patterns
        base_pattern = np.random.rand(n_channels, n_channels) * 0.3
        base_pattern = (base_pattern + base_pattern.T) / 2
        np.fill_diagonal(base_pattern, 0)
        
        # AD-specific pattern
        ad_mod = np.zeros((n_channels, n_channels))
        ad_mod[:50, :50] = 0.15
        ad_mod[80:, 80:] = -0.05
        ad_mod = (ad_mod + ad_mod.T) / 2
        
        # HC-specific pattern
        hc_mod = np.zeros((n_channels, n_channels))
        hc_mod[40:80, 40:80] = 0.1
        hc_mod = (hc_mod + hc_mod.T) / 2
        
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
    n_channels = all_matrices[0].shape[0]
    n_ad = sum(1 for g in group_labels if g == 'AD')
    n_hc = sum(1 for g in group_labels if g == 'HC')
    
    print(f"\n✓ Data loaded:")
    print(f"  • Total subjects: {n_subjects}")
    print(f"  • AD subjects: {n_ad}")
    print(f"  • HC subjects: {n_hc}")
    print(f"  • Channels: {n_channels}")
    
    # =========================================================================
    # STEP 2: COMPUTE OVERALL CONSENSUS MATRIX
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: COMPUTING OVERALL CONSENSUS MATRIX")
    print("="*70)
    
    overall_consensus = compute_consensus_matrix(all_matrices)
    print(f"✓ Overall consensus computed from ALL {n_subjects} subjects")
    print(f"  • Matrix shape: {overall_consensus.shape}")
    print(f"  • Mean connectivity: {np.mean(overall_consensus[np.triu_indices(n_channels, k=1)]):.4f}")
    
    # =========================================================================
    # STEP 3: COMPARE EACH SUBJECT TO OVERALL CONSENSUS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: COMPARING EACH SUBJECT TO OVERALL CONSENSUS")
    print("="*70)
    
    results_list = []
    
    for i, (matrix, subj_id, group) in enumerate(zip(all_matrices, subject_ids, group_labels)):
        comparison = compare_to_consensus(matrix, overall_consensus)
        
        results_list.append({
            'Subject_ID': subj_id,
            'Group': group,
            'Pearson_r': comparison['pearson_r'],
            'Pearson_p': comparison['pearson_p'],
            'Spearman_rho': comparison['spearman_rho'],
            'Jaccard': comparison['jaccard'],
            'Cosine_Similarity': comparison['cosine_similarity'],
            'Mean_Abs_Diff': comparison['mean_abs_diff'],
            'RMSD': comparison['rmsd']
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_subjects} subjects...")
    
    # Create DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Sort by Pearson_r descending
    df_results = df_results.sort_values('Pearson_r', ascending=False).reset_index(drop=True)
    df_results['Rank'] = range(1, len(df_results) + 1)
    
    # =========================================================================
    # STEP 4: SAVE RESULTS TO CSV
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: SAVING RESULTS TO CSV")
    print("="*70)
    
    csv_filename = 'subject_consensus_similarity.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"✓ Results saved to: {csv_filename}")
    
    # Print summary table
    print("\n" + "-"*90)
    print(f"{'Rank':<6} {'Subject_ID':<15} {'Group':<6} {'Pearson_r':<12} {'Jaccard':<12} {'Cosine':<12}")
    print("-"*90)
    
    for _, row in df_results.head(10).iterrows():
        print(f"{row['Rank']:<6} {row['Subject_ID']:<15} {row['Group']:<6} "
              f"{row['Pearson_r']:<12.4f} {row['Jaccard']:<12.4f} {row['Cosine_Similarity']:<12.4f}")
    
    print("...")
    
    for _, row in df_results.tail(5).iterrows():
        print(f"{row['Rank']:<6} {row['Subject_ID']:<15} {row['Group']:<6} "
              f"{row['Pearson_r']:<12.4f} {row['Jaccard']:<12.4f} {row['Cosine_Similarity']:<12.4f}")
    
    print("-"*90)
    
    # =========================================================================
    # STEP 5: SUMMARY STATISTICS BY GROUP
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: SUMMARY STATISTICS BY GROUP")
    print("="*70)
    
    ad_results = df_results[df_results['Group'] == 'AD']
    hc_results = df_results[df_results['Group'] == 'HC']
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│              SUMMARY: SUBJECT SIMILARITY TO OVERALL CONSENSUS              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ALL SUBJECTS (n={n_subjects})                                               │
│  ──────────────────────────────────────────────────────────────────────    │
│    Pearson r:  {df_results['Pearson_r'].mean():.3f} ± {df_results['Pearson_r'].std():.3f}  (range: [{df_results['Pearson_r'].min():.3f}, {df_results['Pearson_r'].max():.3f}])  │
│    Jaccard:    {df_results['Jaccard'].mean():.3f} ± {df_results['Jaccard'].std():.3f}  (range: [{df_results['Jaccard'].min():.3f}, {df_results['Jaccard'].max():.3f}])  │
│                                                                            │
│  AD GROUP (n={n_ad})                                                        │
│  ──────────────────────────────────────────────────────────────────────    │
│    Pearson r:  {ad_results['Pearson_r'].mean():.3f} ± {ad_results['Pearson_r'].std():.3f}  (range: [{ad_results['Pearson_r'].min():.3f}, {ad_results['Pearson_r'].max():.3f}])  │
│    Jaccard:    {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}  (range: [{ad_results['Jaccard'].min():.3f}, {ad_results['Jaccard'].max():.3f}])  │
│                                                                            │
│  HC GROUP (n={n_hc})                                                        │
│  ──────────────────────────────────────────────────────────────────────    │
│    Pearson r:  {hc_results['Pearson_r'].mean():.3f} ± {hc_results['Pearson_r'].std():.3f}  (range: [{hc_results['Pearson_r'].min():.3f}, {hc_results['Pearson_r'].max():.3f}])  │
│    Jaccard:    {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}  (range: [{hc_results['Jaccard'].min():.3f}, {hc_results['Jaccard'].max():.3f}])  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Statistical test between groups
    t_stat_r, p_val_r = stats.ttest_ind(ad_results['Pearson_r'], hc_results['Pearson_r'])
    t_stat_j, p_val_j = stats.ttest_ind(ad_results['Jaccard'], hc_results['Jaccard'])
    
    print(f"  Statistical Comparison (AD vs HC):")
    print(f"    Pearson r: t = {t_stat_r:.3f}, p = {p_val_r:.4f}")
    print(f"    Jaccard:   t = {t_stat_j:.3f}, p = {p_val_j:.4f}")
    
    # =========================================================================
    # STEP 6: CREATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Colors
    ad_color = '#E74C3C'
    hc_color = '#3498DB'
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Overall Consensus Matrix
    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.imshow(overall_consensus, cmap='hot', vmin=0, 
                     vmax=np.percentile(overall_consensus, 95))
    ax1.set_title('Overall Consensus Matrix\n(All Subjects Combined)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Channel')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot 2: Boxplot - Pearson r by Group
    ax2 = fig.add_subplot(3, 3, 2)
    bp = ax2.boxplot([ad_results['Pearson_r'], hc_results['Pearson_r']], 
                     labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Pearson_r'], ad_color), 
                                        (hc_results['Pearson_r'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white')
    
    ax2.set_ylabel('Pearson r', fontsize=11)
    ax2.set_title(f'Pearson r by Group\n(t-test p = {p_val_r:.4f})', fontweight='bold', fontsize=11)
    ax2.axhline(y=df_results['Pearson_r'].mean(), color='gray', linestyle='--', linewidth=1.5)
    
    # Plot 3: Boxplot - Jaccard by Group
    ax3 = fig.add_subplot(3, 3, 3)
    bp2 = ax3.boxplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                      labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor(ad_color)
    bp2['boxes'][1].set_facecolor(hc_color)
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Jaccard'], ad_color), 
                                        (hc_results['Jaccard'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax3.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white')
    
    ax3.set_ylabel('Jaccard Similarity', fontsize=11)
    ax3.set_title(f'Jaccard by Group\n(t-test p = {p_val_j:.4f})', fontweight='bold', fontsize=11)
    ax3.axhline(y=df_results['Jaccard'].mean(), color='gray', linestyle='--', linewidth=1.5)
    
    # Plot 4: Scatter - Pearson r vs Jaccard
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.scatter(ad_results['Pearson_r'], ad_results['Jaccard'], 
                c=ad_color, label=f'AD (n={n_ad})', alpha=0.7, s=50, edgecolor='white')
    ax4.scatter(hc_results['Pearson_r'], hc_results['Jaccard'], 
                c=hc_color, label=f'HC (n={n_hc})', alpha=0.7, s=50, edgecolor='white')
    ax4.set_xlabel('Pearson r', fontsize=11)
    ax4.set_ylabel('Jaccard Similarity', fontsize=11)
    ax4.set_title('Pearson r vs Jaccard\n(Each point = 1 subject)', fontweight='bold', fontsize=11)
    ax4.legend(loc='lower right', fontsize=10)
    
    # Plot 5: Histogram - Pearson r
    ax5 = fig.add_subplot(3, 3, 5)
    bins = np.linspace(df_results['Pearson_r'].min() - 0.02, 
                       df_results['Pearson_r'].max() + 0.02, 20)
    ax5.hist(ad_results['Pearson_r'], bins=bins, alpha=0.6, color=ad_color, 
             label=f'AD (μ={ad_results["Pearson_r"].mean():.3f})', edgecolor='white')
    ax5.hist(hc_results['Pearson_r'], bins=bins, alpha=0.6, color=hc_color, 
             label=f'HC (μ={hc_results["Pearson_r"].mean():.3f})', edgecolor='white')
    ax5.axvline(ad_results['Pearson_r'].mean(), color=ad_color, linestyle='--', linewidth=2)
    ax5.axvline(hc_results['Pearson_r'].mean(), color=hc_color, linestyle='--', linewidth=2)
    ax5.set_xlabel('Pearson r', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Distribution of Pearson r', fontweight='bold', fontsize=11)
    ax5.legend(loc='upper left', fontsize=9)
    
    # Plot 6: Bar Chart - All Subjects Ranked
    ax6 = fig.add_subplot(3, 3, 6)
    colors_bar = [ad_color if g == 'AD' else hc_color for g in df_results['Group']]
    ax6.bar(range(n_subjects), df_results['Pearson_r'], color=colors_bar, alpha=0.8)
    ax6.axhline(y=df_results['Pearson_r'].mean(), color='black', linestyle='--', linewidth=1.5)
    ax6.set_xlabel('Subject Rank', fontsize=11)
    ax6.set_ylabel('Pearson r', fontsize=11)
    ax6.set_title('All Subjects Ranked by Similarity', fontweight='bold', fontsize=11)
    ax6.set_xlim([-1, n_subjects])
    
    # Plot 7-9: Summary text
    ax7 = fig.add_subplot(3, 3, (7, 9))
    ax7.axis('off')
    
    best_subj = df_results.iloc[0]
    worst_subj = df_results.iloc[-1]
    
    summary_text = f"""
    ══════════════════════════════════════════════════════════════════════════
                    EACH SUBJECT vs OVERALL CONSENSUS: RESULTS
    ══════════════════════════════════════════════════════════════════════════
    
    DATA SUMMARY
    ────────────────────────────────────────────────────────────────────────────
    Total Subjects:    {n_subjects} ({n_ad} AD + {n_hc} HC)
    Channels:          {n_channels}
    Edges analyzed:    {n_channels * (n_channels - 1) // 2}
    
    PEARSON r (Subject vs Overall Consensus)
    ────────────────────────────────────────────────────────────────────────────
    All subjects:      {df_results['Pearson_r'].mean():.3f} ± {df_results['Pearson_r'].std():.3f}
    AD group:          {ad_results['Pearson_r'].mean():.3f} ± {ad_results['Pearson_r'].std():.3f}
    HC group:          {hc_results['Pearson_r'].mean():.3f} ± {hc_results['Pearson_r'].std():.3f}
    t-test (AD vs HC): t = {t_stat_r:.3f}, p = {p_val_r:.4f}  {'***' if p_val_r < 0.001 else '**' if p_val_r < 0.01 else '*' if p_val_r < 0.05 else 'ns'}
    
    JACCARD SIMILARITY (15% edge threshold)
    ────────────────────────────────────────────────────────────────────────────
    All subjects:      {df_results['Jaccard'].mean():.3f} ± {df_results['Jaccard'].std():.3f}
    AD group:          {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}
    HC group:          {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}
    t-test (AD vs HC): t = {t_stat_j:.3f}, p = {p_val_j:.4f}  {'***' if p_val_j < 0.001 else '**' if p_val_j < 0.01 else '*' if p_val_j < 0.05 else 'ns'}
    
    EXTREME SUBJECTS
    ────────────────────────────────────────────────────────────────────────────
    Most similar:      {best_subj['Subject_ID']} ({best_subj['Group']}) - r = {best_subj['Pearson_r']:.4f}
    Least similar:     {worst_subj['Subject_ID']} ({worst_subj['Group']}) - r = {worst_subj['Pearson_r']:.4f}
    
    INTERPRETATION
    ────────────────────────────────────────────────────────────────────────────
    • The overall consensus represents the "average" connectivity pattern
    • Higher Pearson r = subject is more similar to the group average
    • {'AD and HC show SIGNIFICANTLY DIFFERENT similarity to consensus' if p_val_r < 0.05 else 'No significant difference between AD and HC'}
    • This can identify outliers or subjects with atypical connectivity
    """
    
    ax7.text(0.02, 0.98, summary_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Each Subject vs Overall Consensus: Connectivity Similarity Analysis',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_filename = 'subject_consensus_similarity.png'
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {fig_filename}")
    
    plt.show()
    
    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"""
OUTPUT FILES:
  ✓ {csv_filename} - All subject similarity metrics
  ✓ {fig_filename} - Visualization figure

CSV COLUMNS:
  • Subject_ID: Subject identifier
  • Group: AD or HC
  • Pearson_r: Correlation with overall consensus
  • Jaccard: Edge overlap similarity (15% sparsity)
  • Rank: Rank by Pearson r (1 = highest similarity)
""")
    
    return {
        'df_results': df_results,
        'overall_consensus': overall_consensus,
        'all_matrices': all_matrices,
        'subject_ids': subject_ids,
        'group_labels': group_labels
    }


if __name__ == "__main__":
    results = main()
