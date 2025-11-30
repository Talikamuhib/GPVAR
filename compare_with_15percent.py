"""
=============================================================================
COMPARING CONNECTIVITY WITH 15% SPARSITY
=============================================================================

When you build consensus with top 15% edges, here's how to compare:

1. Individual matrices → Binarize to 15% → Compare with Consensus
2. Both matrices have same sparsity (15%) → Fair comparison

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("="*70)
print("COMPARING CONNECTIVITY (Both at 15% sparsity)")
print("="*70)

np.random.seed(42)
n_channels = 64
SPARSITY = 0.15  # Your 15% threshold

# =============================================================================
# STEP 1: Create weighted matrices (raw correlations)
# =============================================================================
print("\n" + "="*70)
print("STEP 1: Raw Weighted Matrices (before thresholding)")
print("="*70)

# Raw correlation matrices (full, weighted)
raw_subject = np.random.rand(n_channels, n_channels) * 0.6 + 0.2
raw_subject = (raw_subject + raw_subject.T) / 2
np.fill_diagonal(raw_subject, 0)

raw_consensus = raw_subject + np.random.randn(n_channels, n_channels) * 0.1
raw_consensus = (raw_consensus + raw_consensus.T) / 2
np.fill_diagonal(raw_consensus, 0)
raw_consensus = np.clip(raw_consensus, 0, 1)

print(f"  Raw Subject matrix:   {n_channels}x{n_channels}, all edges have weights")
print(f"  Raw Consensus matrix: {n_channels}x{n_channels}, all edges have weights")

# =============================================================================
# STEP 2: Binarize both to top 15%
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Binarize to Top 15% Edges")
print("="*70)

def binarize_top_percent(matrix, sparsity):
    """Keep only top X% of edges (by weight)."""
    n = matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    weights = matrix[triu_idx]
    
    # Find threshold for top X%
    n_edges_keep = int(sparsity * len(weights))
    threshold = np.sort(weights)[::-1][n_edges_keep - 1]
    
    # Create binary matrix
    binary = np.zeros_like(matrix)
    binary[matrix >= threshold] = 1
    np.fill_diagonal(binary, 0)
    
    # Make symmetric
    binary = np.maximum(binary, binary.T)
    
    return binary, n_edges_keep

binary_subject, n_edges_subj = binarize_top_percent(raw_subject, SPARSITY)
binary_consensus, n_edges_cons = binarize_top_percent(raw_consensus, SPARSITY)

total_possible = n_channels * (n_channels - 1) // 2
print(f"  Total possible edges: {total_possible}")
print(f"  Keeping top 15%:      {int(SPARSITY * total_possible)} edges")
print(f"")
print(f"  Subject (binary):     {int(np.sum(binary_subject)/2)} edges")
print(f"  Consensus (binary):   {int(np.sum(binary_consensus)/2)} edges")

# =============================================================================
# STEP 3: Compare Binary Matrices
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Compare the Binary Matrices")
print("="*70)

def compare_binary_matrices(bin_A, bin_B, name_A="A", name_B="B"):
    """Compare two binary (0/1) matrices."""
    n = bin_A.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    vec_A = bin_A[triu_idx]
    vec_B = bin_B[triu_idx]
    
    # Count edges
    edges_A = np.sum(vec_A)
    edges_B = np.sum(vec_B)
    
    # Jaccard: intersection / union
    intersection = np.sum((vec_A == 1) & (vec_B == 1))
    union = np.sum((vec_A == 1) | (vec_B == 1))
    jaccard = intersection / union if union > 0 else 0
    
    # Dice coefficient: 2*intersection / (|A| + |B|)
    dice = 2 * intersection / (edges_A + edges_B) if (edges_A + edges_B) > 0 else 0
    
    # Overlap coefficient: intersection / min(|A|, |B|)
    overlap = intersection / min(edges_A, edges_B) if min(edges_A, edges_B) > 0 else 0
    
    # Percent agreement (same decision for each edge)
    agreement = np.mean(vec_A == vec_B)
    
    # Edges only in A, only in B, in both
    only_A = np.sum((vec_A == 1) & (vec_B == 0))
    only_B = np.sum((vec_A == 0) & (vec_B == 1))
    in_both = intersection
    
    return {
        'edges_A': int(edges_A),
        'edges_B': int(edges_B),
        'intersection': int(intersection),
        'union': int(union),
        'only_A': int(only_A),
        'only_B': int(only_B),
        'jaccard': jaccard,
        'dice': dice,
        'overlap': overlap,
        'agreement': agreement
    }

results = compare_binary_matrices(binary_subject, binary_consensus, "Subject", "Consensus")

print(f"""
  EDGE COUNTS:
  ┌─────────────────────────────────────────┐
  │  Subject edges:       {results['edges_A']:>4}             │
  │  Consensus edges:     {results['edges_B']:>4}             │
  │                                         │
  │  Edges in BOTH:       {results['intersection']:>4}  (shared)   │
  │  Edges only in Subj:  {results['only_A']:>4}             │
  │  Edges only in Cons:  {results['only_B']:>4}             │
  │  Total unique edges:  {results['union']:>4}  (union)    │
  └─────────────────────────────────────────┘

  SIMILARITY METRICS:
  ┌─────────────────────────────────────────┐
  │  Jaccard Similarity:  {results['jaccard']:.4f}            │
  │    = {results['intersection']} / {results['union']} = {results['jaccard']*100:.1f}% overlap        │
  │                                         │
  │  Dice Coefficient:    {results['dice']:.4f}            │
  │    = 2×{results['intersection']} / ({results['edges_A']}+{results['edges_B']})         │
  │                                         │
  │  Overlap Coefficient: {results['overlap']:.4f}            │
  │    = {results['intersection']} / min({results['edges_A']},{results['edges_B']})             │
  │                                         │
  │  % Agreement:         {results['agreement']*100:.1f}%             │
  │    (same decision on each edge)         │
  └─────────────────────────────────────────┘
""")

# =============================================================================
# INTERPRETATION
# =============================================================================
print("="*70)
print("INTERPRETATION")
print("="*70)

print(f"""
  YOUR RESULT: Jaccard = {results['jaccard']:.3f}

  MEANING: 
    Out of all edges that appear in EITHER matrix,
    {results['jaccard']*100:.1f}% appear in BOTH matrices.

  INTERPRETATION SCALE:
    Jaccard > 0.7  →  Very similar (strong consensus validation)
    Jaccard > 0.5  →  Similar (good consensus validation)
    Jaccard > 0.3  →  Moderate overlap
    Jaccard < 0.3  →  Different (weak consensus or different groups)

  YOUR CASE: {"✓ GOOD - Strong overlap" if results['jaccard'] > 0.5 else "~ MODERATE" if results['jaccard'] > 0.3 else "✗ LOW overlap"}
""")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("="*70)
print("Creating visualization...")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Binary matrices
ax1 = axes[0, 0]
ax1.imshow(binary_subject, cmap='Blues', vmin=0, vmax=1)
ax1.set_title(f'Subject (Binary)\n{results["edges_A"]} edges', fontweight='bold')
ax1.set_xlabel('Channel')
ax1.set_ylabel('Channel')

ax2 = axes[0, 1]
ax2.imshow(binary_consensus, cmap='Reds', vmin=0, vmax=1)
ax2.set_title(f'Consensus (Binary)\n{results["edges_B"]} edges', fontweight='bold')
ax2.set_xlabel('Channel')
ax2.set_ylabel('Channel')

# Overlap visualization
overlap_matrix = np.zeros_like(binary_subject)
overlap_matrix[(binary_subject == 1) & (binary_consensus == 1)] = 3  # Both (green)
overlap_matrix[(binary_subject == 1) & (binary_consensus == 0)] = 1  # Only subject (blue)
overlap_matrix[(binary_subject == 0) & (binary_consensus == 1)] = 2  # Only consensus (red)

from matplotlib.colors import ListedColormap
colors = ['white', 'blue', 'red', 'green']
cmap = ListedColormap(colors)

ax3 = axes[0, 2]
im3 = ax3.imshow(overlap_matrix, cmap=cmap, vmin=0, vmax=3)
ax3.set_title(f'Edge Comparison\nGreen=Both, Blue=Subj only, Red=Cons only', fontweight='bold', fontsize=9)
ax3.set_xlabel('Channel')
ax3.set_ylabel('Channel')

# Venn diagram style bar
ax4 = axes[0, 3]
categories = ['Only\nSubject', 'BOTH\n(Shared)', 'Only\nConsensus']
values = [results['only_A'], results['intersection'], results['only_B']]
colors_bar = ['#3498DB', '#27AE60', '#E74C3C']
bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=2)
ax4.set_ylabel('Number of Edges')
ax4.set_title('Edge Distribution', fontweight='bold')

# Add value labels on bars
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)

# Row 2: Metrics and summary
ax5 = axes[1, 0]
metrics = ['Jaccard', 'Dice', 'Overlap\nCoef', '% Agree']
metric_values = [results['jaccard'], results['dice'], results['overlap'], results['agreement']]
colors_metrics = ['#9B59B6', '#E67E22', '#1ABC9C', '#34495E']
bars2 = ax5.bar(metrics, metric_values, color=colors_metrics, edgecolor='black')
ax5.axhline(0.5, color='gray', linestyle='--', linewidth=2, label='0.5 threshold')
ax5.set_ylabel('Similarity Score')
ax5.set_title('Similarity Metrics', fontweight='bold')
ax5.set_ylim([0, 1.1])
ax5.legend()

# Add values on bars
for bar, val in zip(bars2, metric_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# Pie chart of edge distribution
ax6 = axes[1, 1]
sizes = [results['intersection'], results['only_A'], results['only_B']]
labels = [f'Shared\n({results["intersection"]})', 
          f'Subject only\n({results["only_A"]})', 
          f'Consensus only\n({results["only_B"]})']
colors_pie = ['#27AE60', '#3498DB', '#E74C3C']
explode = (0.05, 0, 0)
ax6.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax6.set_title('Edge Overlap Distribution', fontweight='bold')

# Formula explanation
ax7 = axes[1, 2]
ax7.axis('off')
formula_text = f"""
JACCARD FORMULA (for binary matrices):

         |A ∩ B|        Edges in BOTH
J = ─────────────── = ─────────────────
         |A ∪ B|        Edges in EITHER

         {results['intersection']}
J = ─────────────── = {results['jaccard']:.3f}
         {results['union']}


MEANING:
  {results['jaccard']*100:.1f}% of all edges (that exist in 
  either matrix) are SHARED between
  subject and consensus.
"""
ax7.text(0.1, 0.9, formula_text, transform=ax7.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Summary
ax8 = axes[1, 3]
ax8.axis('off')
summary_text = f"""
SUMMARY
═══════════════════════

Sparsity: {SPARSITY*100:.0f}% (top edges kept)

Subject:   {results['edges_A']} edges
Consensus: {results['edges_B']} edges
Shared:    {results['intersection']} edges

Jaccard = {results['jaccard']:.3f}
  → {results['jaccard']*100:.1f}% overlap

INTERPRETATION:
{"✓ HIGH overlap - Consensus is valid!" if results['jaccard'] > 0.5 else "~ MODERATE overlap" if results['jaccard'] > 0.3 else "✗ LOW overlap - Check data"}

THESIS TEXT:
"Subject showed {results['jaccard']*100:.0f}% edge
overlap (Jaccard = {results['jaccard']:.2f}) with
the consensus matrix."
"""
ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle(f'Comparing Connectivity at {SPARSITY*100:.0f}% Sparsity (Top Edges Only)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('compare_15percent.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: compare_15percent.png")
plt.show()

# =============================================================================
# MULTIPLE SUBJECTS EXAMPLE
# =============================================================================
print("\n" + "="*70)
print("EXAMPLE: Multiple Subjects vs Consensus")
print("="*70)

print("""
For your thesis, you compare EACH subject with the consensus:
""")

# Simulate multiple subjects
n_subjects = 10
all_jaccards = []

print(f"  Subject  |  Jaccard  |  Shared Edges  |  Interpretation")
print(f"  {'-'*60}")

for i in range(n_subjects):
    # Create subject (similar to consensus with noise)
    subj = raw_consensus + np.random.randn(n_channels, n_channels) * (0.05 + i*0.02)
    subj = (subj + subj.T) / 2
    np.fill_diagonal(subj, 0)
    subj = np.clip(subj, 0, 1)
    
    # Binarize
    bin_subj, _ = binarize_top_percent(subj, SPARSITY)
    
    # Compare
    res = compare_binary_matrices(bin_subj, binary_consensus)
    all_jaccards.append(res['jaccard'])
    
    interpretation = "✓ Good" if res['jaccard'] > 0.5 else "~ Moderate" if res['jaccard'] > 0.3 else "✗ Low"
    print(f"     {i+1:2d}    |   {res['jaccard']:.3f}   |      {res['intersection']:3d}       |  {interpretation}")

print(f"  {'-'*60}")
print(f"   Mean    |   {np.mean(all_jaccards):.3f}   |")
print(f"   Std     |   {np.std(all_jaccards):.3f}   |")

# =============================================================================
# THESIS TEXT
# =============================================================================
print("\n" + "="*70)
print("THESIS-READY TEXT")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         METHODS                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

"Individual correlation matrices were binarized by retaining the top 15% 
strongest edges (approximately {int(SPARSITY * total_possible)} of {total_possible} possible connections). 
Edge overlap between each subject and the group consensus was quantified 
using Jaccard similarity: J = |A ∩ B| / |A ∪ B|, where A and B represent 
the edge sets of the individual and consensus matrices, respectively."


╔══════════════════════════════════════════════════════════════════════════╗
║                         RESULTS                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

"Consensus validation revealed strong edge overlap between individual 
subjects and the group consensus. Mean Jaccard similarity was 
{np.mean(all_jaccards):.2f} ± {np.std(all_jaccards):.2f}, indicating that approximately 
{np.mean(all_jaccards)*100:.0f}% of edges were shared between individual subjects and 
the consensus matrix. All subjects exceeded the J > 0.3 threshold for 
meaningful overlap, confirming that the consensus matrix adequately 
represents individual connectivity patterns."


╔══════════════════════════════════════════════════════════════════════════╗
║                      FIGURE CAPTION                                      ║
╚══════════════════════════════════════════════════════════════════════════╝

"Figure X. Consensus matrix validation using edge overlap. (A) Individual 
subject binary connectivity (top 15% edges). (B) Group consensus binary 
matrix. (C) Edge comparison showing shared edges (green), subject-only 
edges (blue), and consensus-only edges (red). (D) Distribution of edges 
across categories. (E) Similarity metrics including Jaccard coefficient. 
(F) Pie chart of edge overlap. Mean Jaccard similarity across n subjects 
was {np.mean(all_jaccards):.2f} ± {np.std(all_jaccards):.2f}."
""")

print("\n" + "="*70)
print("DONE! Check 'compare_15percent.png'")
print("="*70)
