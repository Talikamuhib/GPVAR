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
    # Find subject folder (sub-XXXXX pattern)
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
    """
    Compute consensus matrix from individual correlation matrices.
    Uses Fisher-z averaging for robust mean estimation.
    """
    if len(corr_matrices) == 0:
        raise ValueError("No matrices provided")
    
    stack = np.stack(corr_matrices, axis=0)
    z_stack = fisher_z_transform(stack)
    z_mean = np.mean(z_stack, axis=0)
    consensus = np.abs(fisher_z_inverse(z_mean))
    np.fill_diagonal(consensus, 0)
    
    return consensus


def compare_to_consensus(subject_matrix, consensus_matrix, sparsity=0.15):
    """
    Compare a subject's connectivity matrix to the consensus.
    
    Parameters
    ----------
    subject_matrix : ndarray
        Individual subject's connectivity matrix
    consensus_matrix : ndarray
        Group consensus connectivity matrix
    sparsity : float
        Fraction of edges to keep for Jaccard calculation (default 0.15 = 15%)
    
    Returns
    -------
    dict with keys:
        - pearson_r: Pearson correlation coefficient
        - pearson_p: p-value for Pearson correlation
        - spearman_rho: Spearman correlation
        - jaccard: Jaccard similarity (at given sparsity)
        - cosine_similarity: Cosine similarity
        - mean_abs_diff: Mean absolute difference
        - rmsd: Root mean square difference
    """
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
    n_edges = min(n_edges, len(vec_subj) - 1)  # Ensure valid index
    n_edges = max(n_edges, 1)  # At least 1 edge
    thresh_subj = np.sort(vec_subj)[::-1][n_edges - 1]
    thresh_cons = np.sort(vec_cons)[::-1][n_edges - 1]
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
        # Load real EEG data
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
        
        # AD-specific pattern (stronger frontal)
        ad_mod = np.zeros((n_channels, n_channels))
        ad_mod[:50, :50] = 0.15  # Frontal enhancement
        ad_mod[80:, 80:] = -0.05  # Posterior reduction
        ad_mod = (ad_mod + ad_mod.T) / 2
        
        # HC-specific pattern (stronger parietal)
        hc_mod = np.zeros((n_channels, n_channels))
        hc_mod[40:80, 40:80] = 0.1  # Central/parietal enhancement
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
    
    # Set up figure with beautiful styling
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 14))
    
    # Custom colors
    ad_color = '#E74C3C'  # Red for AD
    hc_color = '#3498DB'  # Blue for HC
    consensus_color = '#9B59B6'  # Purple for consensus
    
    # -------------------------------------------------------------------------
    # Plot 1: Overall Consensus Matrix
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(3, 4, 1)
    im1 = ax1.imshow(overall_consensus, cmap='hot', vmin=0, 
                     vmax=np.percentile(overall_consensus, 95))
    ax1.set_title('Overall Consensus Matrix\n(All Subjects)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Channel')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('Correlation')
    
    # -------------------------------------------------------------------------
    # Plot 2: Boxplot - Pearson r by Group
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(3, 4, 2)
    bp = ax2.boxplot([ad_results['Pearson_r'], hc_results['Pearson_r']], 
                     labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(ad_color)
    bp['boxes'][1].set_facecolor(hc_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    # Add individual points
    for i, (data, color) in enumerate([(ad_results['Pearson_r'], ad_color), 
                                        (hc_results['Pearson_r'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white', linewidth=0.5)
    
    ax2.set_ylabel('Pearson r', fontsize=11)
    ax2.set_title(f'Pearson r by Group\n(p = {p_val_r:.4f})', fontweight='bold', fontsize=11)
    ax2.axhline(y=df_results['Pearson_r'].mean(), color='gray', linestyle='--', 
                linewidth=1.5, label=f'Overall mean: {df_results["Pearson_r"].mean():.3f}')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim([min(df_results['Pearson_r']) - 0.05, max(df_results['Pearson_r']) + 0.05])
    
    # -------------------------------------------------------------------------
    # Plot 3: Boxplot - Jaccard by Group
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(3, 4, 3)
    bp2 = ax3.boxplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                      labels=['AD', 'HC'], patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor(ad_color)
    bp2['boxes'][1].set_facecolor(hc_color)
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    
    for i, (data, color) in enumerate([(ad_results['Jaccard'], ad_color), 
                                        (hc_results['Jaccard'], hc_color)]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax3.scatter(x, data, alpha=0.5, color=color, s=30, edgecolor='white', linewidth=0.5)
    
    ax3.set_ylabel('Jaccard Similarity', fontsize=11)
    ax3.set_title(f'Jaccard by Group\n(p = {p_val_j:.4f})', fontweight='bold', fontsize=11)
    ax3.axhline(y=df_results['Jaccard'].mean(), color='gray', linestyle='--', 
                linewidth=1.5, label=f'Overall mean: {df_results["Jaccard"].mean():.3f}')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_ylim([min(df_results['Jaccard']) - 0.05, max(df_results['Jaccard']) + 0.05])
    
    # -------------------------------------------------------------------------
    # Plot 4: Scatter - Pearson r vs Jaccard
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.scatter(ad_results['Pearson_r'], ad_results['Jaccard'], 
                c=ad_color, label=f'AD (n={n_ad})', alpha=0.7, s=50, edgecolor='white')
    ax4.scatter(hc_results['Pearson_r'], hc_results['Jaccard'], 
                c=hc_color, label=f'HC (n={n_hc})', alpha=0.7, s=50, edgecolor='white')
    ax4.set_xlabel('Pearson r', fontsize=11)
    ax4.set_ylabel('Jaccard Similarity', fontsize=11)
    ax4.set_title('Pearson r vs Jaccard\n(Each point = 1 subject)', fontweight='bold', fontsize=11)
    ax4.legend(loc='lower right', fontsize=10)
    
    # Add regression line
    z = np.polyfit(df_results['Pearson_r'], df_results['Jaccard'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_results['Pearson_r'].min(), df_results['Pearson_r'].max(), 100)
    ax4.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)
    
    # Correlation annotation
    corr_rj, _ = stats.pearsonr(df_results['Pearson_r'], df_results['Jaccard'])
    ax4.annotate(f'r = {corr_rj:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Plot 5: Histogram - Pearson r Distribution
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(3, 4, 5)
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
    
    # -------------------------------------------------------------------------
    # Plot 6: Histogram - Jaccard Distribution
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(3, 4, 6)
    bins_j = np.linspace(df_results['Jaccard'].min() - 0.02, 
                         df_results['Jaccard'].max() + 0.02, 20)
    ax6.hist(ad_results['Jaccard'], bins=bins_j, alpha=0.6, color=ad_color, 
             label=f'AD (μ={ad_results["Jaccard"].mean():.3f})', edgecolor='white')
    ax6.hist(hc_results['Jaccard'], bins=bins_j, alpha=0.6, color=hc_color, 
             label=f'HC (μ={hc_results["Jaccard"].mean():.3f})', edgecolor='white')
    ax6.axvline(ad_results['Jaccard'].mean(), color=ad_color, linestyle='--', linewidth=2)
    ax6.axvline(hc_results['Jaccard'].mean(), color=hc_color, linestyle='--', linewidth=2)
    ax6.set_xlabel('Jaccard Similarity', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Distribution of Jaccard', fontweight='bold', fontsize=11)
    ax6.legend(loc='upper left', fontsize=9)
    
    # -------------------------------------------------------------------------
    # Plot 7: Bar Chart - All Subjects Ranked by Pearson r
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(3, 4, (7, 8))
    colors_bar = [ad_color if g == 'AD' else hc_color for g in df_results['Group']]
    bars = ax7.bar(range(n_subjects), df_results['Pearson_r'], color=colors_bar, 
                   alpha=0.8, edgecolor='none')
    ax7.axhline(y=df_results['Pearson_r'].mean(), color='black', linestyle='--', 
                linewidth=1.5, label=f'Mean: {df_results["Pearson_r"].mean():.3f}')
    ax7.set_xlabel('Subject Rank (by Pearson r)', fontsize=11)
    ax7.set_ylabel('Pearson r', fontsize=11)
    ax7.set_title('All Subjects Ranked by Similarity to Consensus', fontweight='bold', fontsize=11)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ad_color, alpha=0.8, label=f'AD (n={n_ad})'),
                       Patch(facecolor=hc_color, alpha=0.8, label=f'HC (n={n_hc})')]
    ax7.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax7.set_xlim([-1, n_subjects])
    
    # -------------------------------------------------------------------------
    # Plot 8: Violin Plot - Detailed Distribution
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(3, 4, 9)
    parts = ax8.violinplot([ad_results['Pearson_r'], hc_results['Pearson_r']], 
                           positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([ad_color, hc_color][i])
        pc.set_alpha(0.6)
    ax8.set_xticks([1, 2])
    ax8.set_xticklabels(['AD', 'HC'])
    ax8.set_ylabel('Pearson r', fontsize=11)
    ax8.set_title('Violin Plot: Pearson r', fontweight='bold', fontsize=11)
    
    # -------------------------------------------------------------------------
    # Plot 9: Violin Plot - Jaccard
    # -------------------------------------------------------------------------
    ax9 = fig.add_subplot(3, 4, 10)
    parts2 = ax9.violinplot([ad_results['Jaccard'], hc_results['Jaccard']], 
                            positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor([ad_color, hc_color][i])
        pc.set_alpha(0.6)
    ax9.set_xticks([1, 2])
    ax9.set_xticklabels(['AD', 'HC'])
    ax9.set_ylabel('Jaccard', fontsize=11)
    ax9.set_title('Violin Plot: Jaccard', fontweight='bold', fontsize=11)
    
    # -------------------------------------------------------------------------
    # Plot 10: Individual Subject Comparison (Top/Bottom Examples)
    # -------------------------------------------------------------------------
    ax10 = fig.add_subplot(3, 4, 11)
    
    # Find best and worst subjects
    best_idx = df_results['Pearson_r'].idxmax()
    worst_idx = df_results['Pearson_r'].idxmin()
    best_subj_id = df_results.loc[best_idx, 'Subject_ID']
    worst_subj_id = df_results.loc[worst_idx, 'Subject_ID']
    
    # Get their matrices
    best_matrix_idx = subject_ids.index(best_subj_id)
    worst_matrix_idx = subject_ids.index(worst_subj_id)
    
    best_matrix = all_matrices[best_matrix_idx]
    worst_matrix = all_matrices[worst_matrix_idx]
    
    # Plot difference from consensus
    best_diff = best_matrix - overall_consensus
    vmax_diff = np.percentile(np.abs(best_diff), 95)
    im10 = ax10.imshow(best_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    ax10.set_title(f'Best Subject - Consensus\n({best_subj_id}, r={df_results.loc[best_idx, "Pearson_r"]:.3f})', 
                   fontweight='bold', fontsize=10)
    plt.colorbar(im10, ax=ax10, fraction=0.046)
    
    # -------------------------------------------------------------------------
    # Plot 11: Summary Statistics Text
    # -------------------------------------------------------------------------
    ax11 = fig.add_subplot(3, 4, 12)
    ax11.axis('off')
    
    summary_text = f"""
    ═══════════════════════════════════════════
    SUBJECT SIMILARITY TO OVERALL CONSENSUS
    ═══════════════════════════════════════════
    
    DATA SUMMARY
    ────────────────────────────────────────────
    Total Subjects:    {n_subjects}
    AD Subjects:       {n_ad}
    HC Subjects:       {n_hc}
    Channels:          {n_channels}
    
    PEARSON r (Subject vs Consensus)
    ────────────────────────────────────────────
    Overall:   {df_results['Pearson_r'].mean():.3f} ± {df_results['Pearson_r'].std():.3f}
    AD:        {ad_results['Pearson_r'].mean():.3f} ± {ad_results['Pearson_r'].std():.3f}
    HC:        {hc_results['Pearson_r'].mean():.3f} ± {hc_results['Pearson_r'].std():.3f}
    t-test:    p = {p_val_r:.4f}
    
    JACCARD SIMILARITY (15% sparsity)
    ────────────────────────────────────────────
    Overall:   {df_results['Jaccard'].mean():.3f} ± {df_results['Jaccard'].std():.3f}
    AD:        {ad_results['Jaccard'].mean():.3f} ± {ad_results['Jaccard'].std():.3f}
    HC:        {hc_results['Jaccard'].mean():.3f} ± {hc_results['Jaccard'].std():.3f}
    t-test:    p = {p_val_j:.4f}
    
    BEST/WORST SUBJECTS
    ────────────────────────────────────────────
    Highest r: {best_subj_id} (r={df_results['Pearson_r'].max():.3f})
    Lowest r:  {worst_subj_id} (r={df_results['Pearson_r'].min():.3f})
    """
    
    ax11.text(0.02, 0.98, summary_text, transform=ax11.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Each Subject vs Overall Consensus: Connectivity Similarity Analysis',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig_filename = 'subject_consensus_similarity.png'
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {fig_filename}")
    
    plt.show()
    
    # =========================================================================
    # ADDITIONAL FIGURE: Individual Subject Ranking
    # =========================================================================
    print("\nCreating detailed ranking figure...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Horizontal bar chart with subject IDs (top 20 + bottom 20)
    ax = axes2[0, 0]
    
    # Top 20
    top_20 = df_results.head(20)
    y_pos_top = np.arange(20)
    colors_top = [ad_color if g == 'AD' else hc_color for g in top_20['Group']]
    bars_top = ax.barh(y_pos_top, top_20['Pearson_r'], color=colors_top, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos_top)
    ax.set_yticklabels(top_20['Subject_ID'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_title('TOP 20 Subjects (Highest Similarity)', fontweight='bold', fontsize=11)
    ax.axvline(df_results['Pearson_r'].mean(), color='black', linestyle='--', linewidth=1.5)
    
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars_top, top_20['Pearson_r'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=8)
    
    # Plot 2: Bottom 20
    ax = axes2[0, 1]
    bottom_20 = df_results.tail(20).iloc[::-1]  # Reverse for display
    y_pos_bot = np.arange(20)
    colors_bot = [ad_color if g == 'AD' else hc_color for g in bottom_20['Group']]
    bars_bot = ax.barh(y_pos_bot, bottom_20['Pearson_r'], color=colors_bot, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos_bot)
    ax.set_yticklabels(bottom_20['Subject_ID'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_title('BOTTOM 20 Subjects (Lowest Similarity)', fontweight='bold', fontsize=11)
    ax.axvline(df_results['Pearson_r'].mean(), color='black', linestyle='--', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars_bot, bottom_20['Pearson_r'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=8)
    
    # Plot 3: Scatter with labels for outliers
    ax = axes2[1, 0]
    ax.scatter(df_results['Pearson_r'], df_results['Jaccard'], 
               c=[ad_color if g == 'AD' else hc_color for g in df_results['Group']],
               alpha=0.6, s=60, edgecolor='white', linewidth=0.5)
    
    # Label outliers (top 5 and bottom 5)
    for _, row in df_results.head(5).iterrows():
        ax.annotate(row['Subject_ID'], (row['Pearson_r'], row['Jaccard']),
                    fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    for _, row in df_results.tail(5).iterrows():
        ax.annotate(row['Subject_ID'], (row['Pearson_r'], row['Jaccard']),
                    fontsize=7, alpha=0.8, xytext=(5, -10), textcoords='offset points')
    
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_ylabel('Jaccard Similarity', fontsize=11)
    ax.set_title('All Subjects: Pearson r vs Jaccard\n(Outliers labeled)', fontweight='bold', fontsize=11)
    
    legend_elements = [Patch(facecolor=ad_color, alpha=0.8, label=f'AD (n={n_ad})'),
                       Patch(facecolor=hc_color, alpha=0.8, label=f'HC (n={n_hc})')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Plot 4: Group comparison with confidence intervals
    ax = axes2[1, 1]
    
    # Calculate 95% CI
    def ci_95(data):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
        return mean, ci
    
    ad_mean_r, ad_ci_r = ci_95(ad_results['Pearson_r'])
    hc_mean_r, hc_ci_r = ci_95(hc_results['Pearson_r'])
    ad_mean_j, ad_ci_j = ci_95(ad_results['Jaccard'])
    hc_mean_j, hc_ci_j = ci_95(hc_results['Jaccard'])
    
    x = np.array([0.5, 1.5, 3.5, 4.5])
    means = [ad_mean_r, hc_mean_r, ad_mean_j, hc_mean_j]
    cis = [ad_ci_r, hc_ci_r, ad_ci_j, hc_ci_j]
    colors_ci = [ad_color, hc_color, ad_color, hc_color]
    
    bars = ax.bar(x, means, yerr=cis, color=colors_ci, alpha=0.8, 
                  capsize=5, width=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xticks([1, 4])
    ax.set_xticklabels(['Pearson r', 'Jaccard'], fontsize=11)
    ax.set_ylabel('Similarity Score', fontsize=11)
    ax.set_title('Group Comparison (Mean ± 95% CI)', fontweight='bold', fontsize=11)
    
    # Add significance stars
    if p_val_r < 0.001:
        sig_r = '***'
    elif p_val_r < 0.01:
        sig_r = '**'
    elif p_val_r < 0.05:
        sig_r = '*'
    else:
        sig_r = 'ns'
    
    if p_val_j < 0.001:
        sig_j = '***'
    elif p_val_j < 0.01:
        sig_j = '**'
    elif p_val_j < 0.05:
        sig_j = '*'
    else:
        sig_j = 'ns'
    
    ax.annotate(sig_r, xy=(1, max(ad_mean_r + ad_ci_r, hc_mean_r + hc_ci_r) + 0.03),
                ha='center', fontsize=14, fontweight='bold')
    ax.annotate(sig_j, xy=(4, max(ad_mean_j + ad_ci_j, hc_mean_j + hc_ci_j) + 0.03),
                ha='center', fontsize=14, fontweight='bold')
    
    legend_elements = [Patch(facecolor=ad_color, alpha=0.8, label='AD'),
                       Patch(facecolor=hc_color, alpha=0.8, label='HC')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle('Subject Ranking: Similarity to Overall Consensus',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig2_filename = 'subject_ranking_detailed.png'
    plt.savefig(fig2_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {fig2_filename}")
    
    plt.show()
    
    # =========================================================================
    # STEP 7: SAVE DATA IN .NPY FORMAT
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVING DATA IN .NPY FORMAT")
    print("="*70)
    
    # Save overall consensus matrix
    np.save('overall_consensus_matrix.npy', overall_consensus)
    print(f"✓ Saved: overall_consensus_matrix.npy (shape: {overall_consensus.shape})")
    
    # Save all individual subject matrices as a 3D array
    all_matrices_array = np.stack(all_matrices, axis=0)
    np.save('all_subject_matrices.npy', all_matrices_array)
    print(f"✓ Saved: all_subject_matrices.npy (shape: {all_matrices_array.shape})")
    
    # Save subject IDs and group labels
    np.save('subject_ids.npy', np.array(subject_ids))
    print(f"✓ Saved: subject_ids.npy ({len(subject_ids)} subjects)")
    
    np.save('group_labels.npy', np.array(group_labels))
    print(f"✓ Saved: group_labels.npy ({len(group_labels)} labels)")
    
    # Save AD and HC matrices separately
    ad_matrices = [all_matrices[i] for i in range(len(all_matrices)) if group_labels[i] == 'AD']
    hc_matrices = [all_matrices[i] for i in range(len(all_matrices)) if group_labels[i] == 'HC']
    
    ad_matrices_array = np.stack(ad_matrices, axis=0)
    hc_matrices_array = np.stack(hc_matrices, axis=0)
    
    np.save('ad_subject_matrices.npy', ad_matrices_array)
    print(f"✓ Saved: ad_subject_matrices.npy (shape: {ad_matrices_array.shape})")
    
    np.save('hc_subject_matrices.npy', hc_matrices_array)
    print(f"✓ Saved: hc_subject_matrices.npy (shape: {hc_matrices_array.shape})")
    
    # Save group-specific consensus matrices
    ad_consensus = compute_consensus_matrix(ad_matrices)
    hc_consensus = compute_consensus_matrix(hc_matrices)
    
    np.save('ad_consensus_matrix.npy', ad_consensus)
    print(f"✓ Saved: ad_consensus_matrix.npy (shape: {ad_consensus.shape})")
    
    np.save('hc_consensus_matrix.npy', hc_consensus)
    print(f"✓ Saved: hc_consensus_matrix.npy (shape: {hc_consensus.shape})")
    
    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"""
OUTPUT FILES:
  CSV:
  ✓ {csv_filename} - All subject similarity metrics
  
  FIGURES (PNG):
  ✓ {fig_filename} - Main visualization figure
  ✓ {fig2_filename} - Detailed ranking figure
  
  NUMPY DATA (.npy):
  ✓ overall_consensus_matrix.npy - Consensus matrix from all subjects
  ✓ all_subject_matrices.npy - All individual subject matrices (3D array)
  ✓ subject_ids.npy - Subject ID array
  ✓ group_labels.npy - Group label array (AD/HC)
  ✓ ad_subject_matrices.npy - AD group individual matrices
  ✓ hc_subject_matrices.npy - HC group individual matrices
  ✓ ad_consensus_matrix.npy - AD group consensus matrix
  ✓ hc_consensus_matrix.npy - HC group consensus matrix

CSV COLUMNS:
  • Subject_ID: Subject identifier
  • Group: AD or HC
  • Pearson_r: Correlation with overall consensus
  • Pearson_p: p-value for correlation
  • Spearman_rho: Rank correlation with consensus
  • Jaccard: Edge overlap similarity (15% sparsity)
  • Cosine_Similarity: Cosine similarity of edge vectors
  • Mean_Abs_Diff: Mean absolute edge difference
  • RMSD: Root mean square difference
  • Rank: Rank by Pearson r (1 = highest similarity)

HOW TO LOAD .NPY FILES:
  >>> import numpy as np
  >>> consensus = np.load('overall_consensus_matrix.npy')
  >>> all_matrices = np.load('all_subject_matrices.npy')
  >>> subject_ids = np.load('subject_ids.npy', allow_pickle=True)
  >>> group_labels = np.load('group_labels.npy', allow_pickle=True)
""")
    
    return {
        'df_results': df_results,
        'overall_consensus': overall_consensus,
        'all_matrices': all_matrices,
        'subject_ids': subject_ids,
        'group_labels': group_labels,
        'ad_consensus': ad_consensus,
        'hc_consensus': hc_consensus
    }


if __name__ == "__main__":
    results = main()
