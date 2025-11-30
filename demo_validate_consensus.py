"""
=============================================================================
DEMONSTRATION: How to Validate a Consensus Graph
=============================================================================

This script demonstrates how to prove that a consensus matrix is valid by
comparing it with individual subject correlation matrices.

RUN THIS: python demo_validate_consensus.py

What this proves:
1. The consensus matrix represents ALL subjects (not just a few)
2. Each individual subject's connectivity is captured in the consensus
3. The consensus is a true "average" pattern across the group

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HOW TO VALIDATE A CONSENSUS GRAPH FOR YOUR THESIS")
print("="*70)

# =============================================================================
# STEP 1: Create Example Data (In your case, you'll load real data)
# =============================================================================
print("\n" + "="*70)
print("STEP 1: Understanding the Data Structure")
print("="*70)

np.random.seed(42)
n_channels = 64  # Number of EEG channels
n_subjects_ad = 35  # AD patients
n_subjects_hc = 31  # Healthy controls

print(f"""
Your data consists of:
- N channels (electrodes): {n_channels}
- Individual correlation matrices: {n_channels} x {n_channels} per subject
- AD group: {n_subjects_ad} subjects
- HC group: {n_subjects_hc} subjects
- Consensus matrix: {n_channels} x {n_channels} (one per group)
""")

# Create synthetic individual correlation matrices
# In reality, these come from Pearson correlation of EEG signals

def create_synthetic_subject(base_pattern, noise_level=0.3):
    """Create a synthetic subject matrix similar to base pattern."""
    noise = np.random.randn(n_channels, n_channels) * noise_level
    mat = base_pattern + noise
    mat = (mat + mat.T) / 2  # Symmetrize
    np.fill_diagonal(mat, 0)  # No self-connections
    mat = np.clip(mat, 0, 1)  # Keep in valid range
    return mat

# Create a "true" underlying pattern (what consensus should capture)
true_pattern = np.random.rand(n_channels, n_channels) * 0.5 + 0.3
true_pattern = (true_pattern + true_pattern.T) / 2
np.fill_diagonal(true_pattern, 0)

# Generate individual subjects with variations around the true pattern
ad_subjects = [create_synthetic_subject(true_pattern, noise_level=0.25) for _ in range(n_subjects_ad)]
hc_subjects = [create_synthetic_subject(true_pattern, noise_level=0.20) for _ in range(n_subjects_hc)]

# Create consensus matrices (average of individual matrices)
ad_consensus = np.mean(ad_subjects, axis=0)
hc_consensus = np.mean(hc_subjects, axis=0)

print("✓ Data structure created (synthetic example)")

# =============================================================================
# STEP 2: The Key Validation Metric - Correlation with Consensus
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Computing Individual-Consensus Correlation")
print("="*70)

print("""
THE KEY QUESTION: Does the consensus matrix represent each individual?

METHOD: Compute Pearson correlation between each subject's matrix and 
        the consensus matrix (using upper triangle values only).

INTERPRETATION:
  r > 0.7  → Strong agreement (individual well-represented)
  r > 0.5  → Moderate agreement (acceptable)
  r < 0.5  → Weak agreement (individual may be an outlier)
""")

def compute_correlation_with_consensus(individual_matrix, consensus_matrix):
    """Compute correlation between individual and consensus (upper triangle)."""
    n = individual_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    ind_values = individual_matrix[triu_idx]
    cons_values = consensus_matrix[triu_idx]
    
    correlation, p_value = stats.pearsonr(ind_values, cons_values)
    return correlation, p_value

# Compute correlations for all subjects
ad_correlations = []
for i, subj in enumerate(ad_subjects):
    corr, _ = compute_correlation_with_consensus(subj, ad_consensus)
    ad_correlations.append(corr)

hc_correlations = []
for i, subj in enumerate(hc_subjects):
    corr, _ = compute_correlation_with_consensus(subj, hc_consensus)
    hc_correlations.append(corr)

ad_correlations = np.array(ad_correlations)
hc_correlations = np.array(hc_correlations)

print("\n--- RESULTS ---")
print(f"\nAD Group (n={n_subjects_ad}):")
print(f"  Mean correlation with consensus: {np.mean(ad_correlations):.4f}")
print(f"  Std deviation:                   {np.std(ad_correlations):.4f}")
print(f"  Min correlation:                 {np.min(ad_correlations):.4f}")
print(f"  Max correlation:                 {np.max(ad_correlations):.4f}")
print(f"  Subjects with r > 0.5:           {np.sum(ad_correlations > 0.5)}/{n_subjects_ad} ({100*np.mean(ad_correlations > 0.5):.1f}%)")

print(f"\nHC Group (n={n_subjects_hc}):")
print(f"  Mean correlation with consensus: {np.mean(hc_correlations):.4f}")
print(f"  Std deviation:                   {np.std(hc_correlations):.4f}")
print(f"  Min correlation:                 {np.min(hc_correlations):.4f}")
print(f"  Max correlation:                 {np.max(hc_correlations):.4f}")
print(f"  Subjects with r > 0.5:           {np.sum(hc_correlations > 0.5)}/{n_subjects_hc} ({100*np.mean(hc_correlations > 0.5):.1f}%)")

# =============================================================================
# STEP 3: Statistical Test - Do Groups Differ?
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Group Comparison")
print("="*70)

t_stat, p_value = stats.ttest_ind(ad_correlations, hc_correlations)
cohens_d = (np.mean(ad_correlations) - np.mean(hc_correlations)) / np.sqrt(
    (np.std(ad_correlations)**2 + np.std(hc_correlations)**2) / 2
)

print(f"""
Question: Do AD and HC subjects agree equally well with their consensus?

Statistical Test: Independent samples t-test
  t-statistic: {t_stat:.4f}
  p-value:     {p_value:.4f}
  Cohen's d:   {cohens_d:.4f}

Interpretation:
  {"Groups show similar consensus agreement (p > 0.05)" if p_value > 0.05 else "Groups differ in consensus agreement (p < 0.05)"}
""")

# =============================================================================
# STEP 4: Visualization
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Creating Validation Figures")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel A: Boxplot comparison
ax1 = axes[0, 0]
bp = ax1.boxplot([ad_correlations, hc_correlations], 
                  labels=['AD', 'HC'],
                  patch_artist=True)
bp['boxes'][0].set_facecolor('#E74C3C')
bp['boxes'][1].set_facecolor('#3498DB')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_alpha(0.7)

# Add scatter points
for i, (data, color) in enumerate(zip([ad_correlations, hc_correlations], ['#E74C3C', '#3498DB'])):
    x = np.random.normal(i+1, 0.04, size=len(data))
    ax1.scatter(x, data, alpha=0.5, color=color, s=30)

ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='r=0.5 threshold')
ax1.set_ylabel('Correlation with Consensus', fontsize=12)
ax1.set_title('A) Individual-Consensus Correlation\nby Group', fontweight='bold')
ax1.set_ylim([0, 1.05])

# Panel B: Histogram
ax2 = axes[0, 1]
bins = np.linspace(0, 1, 20)
ax2.hist(ad_correlations, bins=bins, alpha=0.6, color='#E74C3C', label='AD', edgecolor='black')
ax2.hist(hc_correlations, bins=bins, alpha=0.6, color='#3498DB', label='HC', edgecolor='black')
ax2.axvline(np.mean(ad_correlations), color='#C0392B', linestyle='--', linewidth=2)
ax2.axvline(np.mean(hc_correlations), color='#2980B9', linestyle='--', linewidth=2)
ax2.set_xlabel('Correlation with Consensus')
ax2.set_ylabel('Count')
ax2.set_title('B) Distribution of Correlations', fontweight='bold')
ax2.legend()

# Panel C: Per-subject bar chart (sorted)
ax3 = axes[0, 2]
ad_sorted = np.sort(ad_correlations)[::-1]
hc_sorted = np.sort(hc_correlations)[::-1]

x_ad = np.arange(len(ad_sorted))
x_hc = np.arange(len(ad_sorted) + 2, len(ad_sorted) + 2 + len(hc_sorted))

ax3.bar(x_ad, ad_sorted, color='#E74C3C', alpha=0.7, label='AD')
ax3.bar(x_hc, hc_sorted, color='#3498DB', alpha=0.7, label='HC')
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
ax3.axhline(y=np.mean(ad_correlations), color='#C0392B', linestyle='--', linewidth=2)
ax3.axhline(y=np.mean(hc_correlations), color='#2980B9', linestyle='--', linewidth=2)
ax3.set_xlabel('Subject (sorted)')
ax3.set_ylabel('Correlation with Consensus')
ax3.set_title('C) Every Subject vs Consensus', fontweight='bold')
ax3.set_ylim([0, 1.05])
ax3.legend(loc='lower left')

# Panel D: Example individual vs consensus (AD)
ax4 = axes[1, 0]
best_ad_idx = np.argmax(ad_correlations)
im4 = ax4.imshow(ad_subjects[best_ad_idx], cmap='hot', vmin=0, vmax=1)
ax4.set_title(f'D) Best AD Subject (r={ad_correlations[best_ad_idx]:.3f})', fontweight='bold')
ax4.set_xlabel('Channel')
ax4.set_ylabel('Channel')
plt.colorbar(im4, ax=ax4, fraction=0.046)

# Panel E: AD Consensus
ax5 = axes[1, 1]
im5 = ax5.imshow(ad_consensus, cmap='hot', vmin=0, vmax=1)
ax5.set_title('E) AD Consensus Matrix', fontweight='bold')
ax5.set_xlabel('Channel')
ax5.set_ylabel('Channel')
plt.colorbar(im5, ax=ax5, fraction=0.046)

# Panel F: Summary statistics
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
╔══════════════════════════════════════════════════════╗
║          CONSENSUS VALIDATION SUMMARY                ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  AD Group (n={n_subjects_ad:2d})                                    ║
║    Mean r = {np.mean(ad_correlations):.3f} ± {np.std(ad_correlations):.3f}                         ║
║    Range: [{np.min(ad_correlations):.3f}, {np.max(ad_correlations):.3f}]                         ║
║    {np.sum(ad_correlations > 0.5):2d}/{n_subjects_ad} subjects with r > 0.5                    ║
║                                                      ║
║  HC Group (n={n_subjects_hc:2d})                                    ║
║    Mean r = {np.mean(hc_correlations):.3f} ± {np.std(hc_correlations):.3f}                         ║
║    Range: [{np.min(hc_correlations):.3f}, {np.max(hc_correlations):.3f}]                         ║
║    {np.sum(hc_correlations > 0.5):2d}/{n_subjects_hc} subjects with r > 0.5                    ║
║                                                      ║
║  Group comparison: p = {p_value:.4f}                        ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║  ✓ CONSENSUS IS VALID if:                            ║
║    • Mean r > 0.5 for both groups                    ║
║    • Most subjects show r > 0.5                      ║
║    • No systematic outliers                          ║
╚══════════════════════════════════════════════════════╝
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.suptitle('Consensus Matrix Validation: Is it Representative of All Subjects?',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('consensus_validation_demo.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: consensus_validation_demo.png")
plt.show()

# =============================================================================
# STEP 5: Thesis-Ready Interpretation
# =============================================================================
print("\n" + "="*70)
print("STEP 5: How to Report This in Your Thesis")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    THESIS TEXT (COPY-PASTE READY)                    ║
╚══════════════════════════════════════════════════════════════════════╝

METHODS SECTION:
----------------
"To validate that the consensus matrix adequately represents individual 
subjects, we computed Pearson correlation coefficients between each 
subject's correlation matrix and the group consensus matrix. Correlations 
were computed using the upper triangle of each matrix (excluding the 
diagonal), yielding {n_channels*(n_channels-1)//2} edge comparisons per subject."


RESULTS SECTION:
----------------
"Consensus matrix validation confirmed that individual subjects were 
well-represented by their group consensus. In the AD group (n={n_subjects_ad}), 
subjects showed mean correlation of r = {np.mean(ad_correlations):.3f} ± {np.std(ad_correlations):.3f} 
with the consensus matrix, with {np.sum(ad_correlations > 0.5)}/{n_subjects_ad} ({100*np.mean(ad_correlations > 0.5):.0f}%) 
showing r > 0.5. Similarly, HC subjects (n={n_subjects_hc}) showed mean correlation 
of r = {np.mean(hc_correlations):.3f} ± {np.std(hc_correlations):.3f}, with {np.sum(hc_correlations > 0.5)}/{n_subjects_hc} ({100*np.mean(hc_correlations > 0.5):.0f}%) 
exceeding r > 0.5. {"No significant difference in consensus agreement was observed between groups (p = " + f"{p_value:.3f}" + ")." if p_value > 0.05 else "Groups differed significantly in consensus agreement (p = " + f"{p_value:.4f}" + ")."}
These results confirm that the consensus matrices capture the common 
connectivity patterns across individual subjects in each group."


FIGURE CAPTION:
---------------
"Figure X. Consensus matrix validation. (A) Boxplots showing correlation 
between individual subject matrices and group consensus for AD (red) and 
HC (blue). (B) Distribution of correlation values. (C) Per-subject 
correlation sorted by magnitude. (D) Example individual AD subject matrix 
with highest consensus agreement. (E) AD group consensus matrix. 
(F) Summary statistics confirming consensus validity."

╔══════════════════════════════════════════════════════════════════════╗
║                         KEY INTERPRETATION                           ║
╚══════════════════════════════════════════════════════════════════════╝

Your consensus is VALID if:

  ✓ Mean correlation > 0.5 for both groups
    → AD: {np.mean(ad_correlations):.3f} {"✓" if np.mean(ad_correlations) > 0.5 else "✗"}
    → HC: {np.mean(hc_correlations):.3f} {"✓" if np.mean(hc_correlations) > 0.5 else "✗"}
  
  ✓ Majority of subjects have r > 0.5
    → AD: {100*np.mean(ad_correlations > 0.5):.0f}% {"✓" if np.mean(ad_correlations > 0.5) > 0.5 else "✗"}
    → HC: {100*np.mean(hc_correlations > 0.5):.0f}% {"✓" if np.mean(hc_correlations > 0.5) > 0.5 else "✗"}
  
  ✓ Minimum correlation is reasonable (> 0.3)
    → AD min: {np.min(ad_correlations):.3f} {"✓" if np.min(ad_correlations) > 0.3 else "✗"}
    → HC min: {np.min(hc_correlations):.3f} {"✓" if np.min(hc_correlations) > 0.3 else "✗"}

OVERALL: {"✓ CONSENSUS IS VALID - represents all subjects well" if np.mean(ad_correlations) > 0.5 and np.mean(hc_correlations) > 0.5 else "⚠ Review individual subjects with low correlation"}
""")

# =============================================================================
# STEP 6: Additional Metrics (Optional)
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Additional Validation Metrics")
print("="*70)

print("""
Besides Pearson correlation, you can also report:

1. SPEARMAN CORRELATION (rank-based)
   - Less sensitive to outliers
   - Should be similar to Pearson if relationship is monotonic
""")

ad_spearman = [stats.spearmanr(s[np.triu_indices(n_channels, k=1)], 
                                ad_consensus[np.triu_indices(n_channels, k=1)])[0] 
               for s in ad_subjects]
hc_spearman = [stats.spearmanr(s[np.triu_indices(n_channels, k=1)], 
                                hc_consensus[np.triu_indices(n_channels, k=1)])[0] 
               for s in hc_subjects]

print(f"   AD Spearman: {np.mean(ad_spearman):.3f} ± {np.std(ad_spearman):.3f}")
print(f"   HC Spearman: {np.mean(hc_spearman):.3f} ± {np.std(hc_spearman):.3f}")

print("""
2. JACCARD SIMILARITY (edge overlap)
   - What proportion of edges overlap?
   - Binarize matrices first (keep top 15% edges)
""")

def jaccard_similarity(mat1, mat2, sparsity=0.15):
    n = mat1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    n_edges = int(sparsity * len(triu_idx[0]))
    
    # Binarize: keep top edges
    thresh1 = np.sort(mat1[triu_idx])[::-1][n_edges]
    thresh2 = np.sort(mat2[triu_idx])[::-1][n_edges]
    
    bin1 = (mat1[triu_idx] > thresh1).astype(int)
    bin2 = (mat2[triu_idx] > thresh2).astype(int)
    
    intersection = np.sum((bin1 == 1) & (bin2 == 1))
    union = np.sum((bin1 == 1) | (bin2 == 1))
    
    return intersection / union if union > 0 else 0

ad_jaccard = [jaccard_similarity(s, ad_consensus) for s in ad_subjects]
hc_jaccard = [jaccard_similarity(s, hc_consensus) for s in hc_subjects]

print(f"   AD Jaccard: {np.mean(ad_jaccard):.3f} ± {np.std(ad_jaccard):.3f}")
print(f"   HC Jaccard: {np.mean(hc_jaccard):.3f} ± {np.std(hc_jaccard):.3f}")

print("""
3. MEAN ABSOLUTE DIFFERENCE
   - How different are individual edges?
   - Lower is better
""")

ad_mad = [np.mean(np.abs(s[np.triu_indices(n_channels, k=1)] - 
                          ad_consensus[np.triu_indices(n_channels, k=1)])) 
          for s in ad_subjects]
hc_mad = [np.mean(np.abs(s[np.triu_indices(n_channels, k=1)] - 
                          hc_consensus[np.triu_indices(n_channels, k=1)])) 
          for s in hc_subjects]

print(f"   AD Mean Abs Diff: {np.mean(ad_mad):.3f} ± {np.std(ad_mad):.3f}")
print(f"   HC Mean Abs Diff: {np.mean(hc_mad):.3f} ± {np.std(hc_mad):.3f}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: HOW TO VALIDATE YOUR CONSENSUS")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     VALIDATION CHECKLIST                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. □ Load your consensus matrix (C or G from consensus_results/)   │
│                                                                     │
│  2. □ Load individual subject matrices (binary_matrices.npy)        │
│                                                                     │
│  3. □ For each subject, compute:                                    │
│       - Pearson correlation with consensus                          │
│       - (Optional) Spearman, Jaccard, Mean Abs Diff                 │
│                                                                     │
│  4. □ Check validation criteria:                                    │
│       - Mean r > 0.5 for both groups                                │
│       - Majority of subjects have r > 0.5                           │
│       - No extreme outliers (min r > 0.3)                           │
│                                                                     │
│  5. □ Create validation figure (boxplot + histogram)                │
│                                                                     │
│  6. □ Report in thesis:                                             │
│       - Mean ± std correlation for each group                       │
│       - Group comparison p-value                                    │
│       - % subjects exceeding threshold                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

TO RUN ON YOUR REAL DATA:
  1. First run: python process_eeg_consensus.py
  2. Then run:  python validate_consensus_matrix.py

This will validate your actual consensus matrices!
""")

print("\n" + "="*70)
print("DEMO COMPLETE - Check 'consensus_validation_demo.png'")
print("="*70)
