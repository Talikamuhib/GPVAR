"""
=============================================================================
HOW TO COMPARE TWO GRAPHS: Are They the Same or Different?
=============================================================================

This script shows multiple methods to compare:
  1. AD consensus vs HC consensus (group comparison)
  2. Individual subject vs consensus (validation)
  3. Any two connectivity matrices

RUN: python compare_two_graphs.py

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cosine, correlation
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HOW TO COMPARE TWO GRAPHS")
print("="*70)

# =============================================================================
# Create Example Graphs
# =============================================================================
np.random.seed(42)
n_channels = 64

# Create two example graphs (like AD consensus vs HC consensus)
# Graph 1: AD-like pattern (more connectivity in certain regions)
graph_AD = np.random.rand(n_channels, n_channels) * 0.5
graph_AD[:20, :20] += 0.3  # Stronger frontal connectivity
graph_AD = (graph_AD + graph_AD.T) / 2
np.fill_diagonal(graph_AD, 0)

# Graph 2: HC-like pattern (different connectivity pattern)
graph_HC = np.random.rand(n_channels, n_channels) * 0.5
graph_HC[30:50, 30:50] += 0.3  # Stronger parietal connectivity
graph_HC = (graph_HC + graph_HC.T) / 2
np.fill_diagonal(graph_HC, 0)

# Graph 3: Very similar to Graph 1 (for comparison)
graph_similar = graph_AD + np.random.randn(n_channels, n_channels) * 0.05
graph_similar = (graph_similar + graph_similar.T) / 2
np.fill_diagonal(graph_similar, 0)
graph_similar = np.clip(graph_similar, 0, 1)

print("\nExample graphs created:")
print(f"  - Graph AD: {n_channels}x{n_channels} (AD-like pattern)")
print(f"  - Graph HC: {n_channels}x{n_channels} (HC-like pattern)")
print(f"  - Graph Similar: {n_channels}x{n_channels} (similar to AD)")

# =============================================================================
# METHOD 1: PEARSON CORRELATION
# =============================================================================
print("\n" + "="*70)
print("METHOD 1: PEARSON CORRELATION")
print("="*70)

print("""
WHAT IT MEASURES: Linear relationship between edge weights
RANGE: -1 to +1
INTERPRETATION:
  r ≈ 1.0  → Graphs are nearly identical
  r > 0.7  → Graphs are similar
  r ≈ 0.5  → Moderate similarity
  r < 0.3  → Graphs are different
  r ≈ 0    → No relationship
""")

def pearson_correlation(G1, G2):
    """Compute Pearson correlation between two graphs (upper triangle)."""
    n = G1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vec1 = G1[triu_idx]
    vec2 = G2[triu_idx]
    r, p = stats.pearsonr(vec1, vec2)
    return r, p

r_AD_HC, p_AD_HC = pearson_correlation(graph_AD, graph_HC)
r_AD_sim, p_AD_sim = pearson_correlation(graph_AD, graph_similar)

print(f"Results:")
print(f"  AD vs HC:      r = {r_AD_HC:.4f}, p = {p_AD_HC:.2e}  → {'DIFFERENT' if r_AD_HC < 0.7 else 'SIMILAR'}")
print(f"  AD vs Similar: r = {r_AD_sim:.4f}, p = {p_AD_sim:.2e}  → {'DIFFERENT' if r_AD_sim < 0.7 else 'SIMILAR'}")

# =============================================================================
# METHOD 2: COSINE SIMILARITY
# =============================================================================
print("\n" + "="*70)
print("METHOD 2: COSINE SIMILARITY")
print("="*70)

print("""
WHAT IT MEASURES: Angle between edge weight vectors
RANGE: 0 to 1 (for positive weights)
INTERPRETATION:
  cos ≈ 1.0  → Graphs point in same direction (similar pattern)
  cos > 0.8  → High similarity
  cos < 0.5  → Different patterns
""")

def cosine_similarity(G1, G2):
    """Compute cosine similarity between two graphs."""
    n = G1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vec1 = G1[triu_idx]
    vec2 = G2[triu_idx]
    # cosine distance = 1 - cosine similarity
    cos_sim = 1 - cosine(vec1, vec2)
    return cos_sim

cos_AD_HC = cosine_similarity(graph_AD, graph_HC)
cos_AD_sim = cosine_similarity(graph_AD, graph_similar)

print(f"Results:")
print(f"  AD vs HC:      cosine = {cos_AD_HC:.4f}  → {'DIFFERENT' if cos_AD_HC < 0.8 else 'SIMILAR'}")
print(f"  AD vs Similar: cosine = {cos_AD_sim:.4f}  → {'DIFFERENT' if cos_AD_sim < 0.8 else 'SIMILAR'}")

# =============================================================================
# METHOD 3: JACCARD SIMILARITY (Edge Overlap)
# =============================================================================
print("\n" + "="*70)
print("METHOD 3: JACCARD SIMILARITY (Edge Overlap)")
print("="*70)

print("""
WHAT IT MEASURES: Proportion of shared edges (after binarization)
RANGE: 0 to 1
INTERPRETATION:
  J ≈ 1.0  → Same edges present in both graphs
  J > 0.5  → Majority of edges overlap
  J < 0.3  → Few edges overlap (different topology)
""")

def jaccard_similarity(G1, G2, sparsity=0.15):
    """Compute Jaccard similarity after binarizing graphs."""
    n = G1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    n_edges = int(sparsity * len(triu_idx[0]))
    
    # Binarize: keep top edges
    vec1 = G1[triu_idx]
    vec2 = G2[triu_idx]
    
    thresh1 = np.sort(vec1)[::-1][min(n_edges, len(vec1)-1)]
    thresh2 = np.sort(vec2)[::-1][min(n_edges, len(vec2)-1)]
    
    bin1 = (vec1 >= thresh1).astype(int)
    bin2 = (vec2 >= thresh2).astype(int)
    
    intersection = np.sum((bin1 == 1) & (bin2 == 1))
    union = np.sum((bin1 == 1) | (bin2 == 1))
    
    return intersection / union if union > 0 else 0

jac_AD_HC = jaccard_similarity(graph_AD, graph_HC)
jac_AD_sim = jaccard_similarity(graph_AD, graph_similar)

print(f"Results (keeping top 15% edges):")
print(f"  AD vs HC:      Jaccard = {jac_AD_HC:.4f}  → {'DIFFERENT' if jac_AD_HC < 0.5 else 'SIMILAR'}")
print(f"  AD vs Similar: Jaccard = {jac_AD_sim:.4f}  → {'DIFFERENT' if jac_AD_sim < 0.5 else 'SIMILAR'}")

# =============================================================================
# METHOD 4: FROBENIUS NORM (Matrix Distance)
# =============================================================================
print("\n" + "="*70)
print("METHOD 4: FROBENIUS NORM (Matrix Distance)")
print("="*70)

print("""
WHAT IT MEASURES: Euclidean distance between matrices
RANGE: 0 to ∞
INTERPRETATION:
  ||A-B||_F ≈ 0  → Graphs are identical
  Small value    → Graphs are similar
  Large value    → Graphs are different
  
  Normalized: divide by ||A||_F to get relative difference
""")

def frobenius_distance(G1, G2, normalized=True):
    """Compute Frobenius norm of difference."""
    diff = G1 - G2
    frob_dist = np.linalg.norm(diff, 'fro')
    
    if normalized:
        norm_G1 = np.linalg.norm(G1, 'fro')
        frob_dist = frob_dist / norm_G1 if norm_G1 > 0 else frob_dist
    
    return frob_dist

frob_AD_HC = frobenius_distance(graph_AD, graph_HC)
frob_AD_sim = frobenius_distance(graph_AD, graph_similar)

print(f"Results (normalized):")
print(f"  AD vs HC:      ||diff||_F = {frob_AD_HC:.4f}  → {'DIFFERENT' if frob_AD_HC > 0.3 else 'SIMILAR'}")
print(f"  AD vs Similar: ||diff||_F = {frob_AD_sim:.4f}  → {'DIFFERENT' if frob_AD_sim > 0.3 else 'SIMILAR'}")

# =============================================================================
# METHOD 5: EDGE-WISE STATISTICAL TEST
# =============================================================================
print("\n" + "="*70)
print("METHOD 5: EDGE-WISE STATISTICAL TEST")
print("="*70)

print("""
WHAT IT MEASURES: How many edges are significantly different?
METHOD: Paired t-test or Wilcoxon test on each edge across subjects
INTERPRETATION:
  % significant edges > 20%  → Graphs are systematically different
  % significant edges < 5%   → Graphs are essentially the same
""")

def edge_wise_comparison(G1, G2):
    """Compare edge weights directly."""
    n = G1.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    vec1 = G1[triu_idx]
    vec2 = G2[triu_idx]
    
    # Compute difference statistics
    diff = vec1 - vec2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # One-sample t-test: is mean difference different from 0?
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    
    # Count edges with large difference (> 1 std of original)
    threshold = np.std(vec1)
    n_different = np.sum(np.abs(diff) > threshold)
    pct_different = 100 * n_different / len(diff)
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_stat': t_stat,
        'p_value': p_val,
        'n_different': n_different,
        'pct_different': pct_different
    }

stats_AD_HC = edge_wise_comparison(graph_AD, graph_HC)
stats_AD_sim = edge_wise_comparison(graph_AD, graph_similar)

print(f"\nAD vs HC:")
print(f"  Mean edge difference: {stats_AD_HC['mean_diff']:.4f} ± {stats_AD_HC['std_diff']:.4f}")
print(f"  t-test: t = {stats_AD_HC['t_stat']:.2f}, p = {stats_AD_HC['p_value']:.2e}")
print(f"  Edges with large difference: {stats_AD_HC['n_different']} ({stats_AD_HC['pct_different']:.1f}%)")

print(f"\nAD vs Similar:")
print(f"  Mean edge difference: {stats_AD_sim['mean_diff']:.4f} ± {stats_AD_sim['std_diff']:.4f}")
print(f"  t-test: t = {stats_AD_sim['t_stat']:.2f}, p = {stats_AD_sim['p_value']:.2e}")
print(f"  Edges with large difference: {stats_AD_sim['n_different']} ({stats_AD_sim['pct_different']:.1f}%)")

# =============================================================================
# METHOD 6: GRAPH LAPLACIAN SPECTRUM COMPARISON
# =============================================================================
print("\n" + "="*70)
print("METHOD 6: LAPLACIAN SPECTRUM COMPARISON")
print("="*70)

print("""
WHAT IT MEASURES: Similarity of graph structure via eigenvalues
WHY USEFUL: Eigenvalues capture global topology (connectivity patterns)
INTERPRETATION:
  Similar spectra → Similar graph structure
  Different spectra → Different topology
""")

def laplacian_spectrum(G):
    """Compute Laplacian eigenvalues."""
    degrees = np.sum(G, axis=1)
    L = np.diag(degrees) - G
    eigenvalues = np.linalg.eigvalsh(L)
    return np.sort(eigenvalues)

def spectrum_similarity(G1, G2):
    """Compare Laplacian spectra."""
    eig1 = laplacian_spectrum(G1)
    eig2 = laplacian_spectrum(G2)
    
    # Correlation between spectra
    r, p = stats.pearsonr(eig1, eig2)
    
    # Euclidean distance between spectra
    dist = np.linalg.norm(eig1 - eig2)
    
    return r, dist

r_spec_AD_HC, dist_spec_AD_HC = spectrum_similarity(graph_AD, graph_HC)
r_spec_AD_sim, dist_spec_AD_sim = spectrum_similarity(graph_AD, graph_similar)

print(f"Results:")
print(f"  AD vs HC:      spectrum r = {r_spec_AD_HC:.4f}, distance = {dist_spec_AD_HC:.2f}")
print(f"  AD vs Similar: spectrum r = {r_spec_AD_sim:.4f}, distance = {dist_spec_AD_sim:.2f}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: ALL COMPARISON METRICS")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRAPH COMPARISON SUMMARY                             │
├─────────────────────┬──────────────────┬──────────────────┬─────────────┤
│ METRIC              │ AD vs HC         │ AD vs Similar    │ THRESHOLD   │
├─────────────────────┼──────────────────┼──────────────────┼─────────────┤""")
print(f"│ Pearson r           │ {r_AD_HC:>16.4f} │ {r_AD_sim:>16.4f} │ > 0.7 same  │")
print(f"│ Cosine similarity   │ {cos_AD_HC:>16.4f} │ {cos_AD_sim:>16.4f} │ > 0.8 same  │")
print(f"│ Jaccard similarity  │ {jac_AD_HC:>16.4f} │ {jac_AD_sim:>16.4f} │ > 0.5 same  │")
print(f"│ Frobenius (norm)    │ {frob_AD_HC:>16.4f} │ {frob_AD_sim:>16.4f} │ < 0.3 same  │")
print(f"│ % edges different   │ {stats_AD_HC['pct_different']:>15.1f}% │ {stats_AD_sim['pct_different']:>15.1f}% │ < 10% same  │")
print(f"│ Spectrum r          │ {r_spec_AD_HC:>16.4f} │ {r_spec_AD_sim:>16.4f} │ > 0.9 same  │")
print("""└─────────────────────┴──────────────────┴──────────────────┴─────────────┘
""")

# Determine if graphs are same or different
print("CONCLUSION:")
if r_AD_HC > 0.7 and jac_AD_HC > 0.5:
    print("  AD vs HC: SIMILAR graphs")
else:
    print("  AD vs HC: DIFFERENT graphs ← This is expected for AD vs HC!")

if r_AD_sim > 0.7 and jac_AD_sim > 0.5:
    print("  AD vs Similar: SIMILAR graphs ← Validation passed!")
else:
    print("  AD vs Similar: DIFFERENT graphs")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("Creating visualization...")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Graph matrices
im1 = axes[0, 0].imshow(graph_AD, cmap='hot', vmin=0, vmax=1)
axes[0, 0].set_title('Graph A (AD)', fontweight='bold')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

im2 = axes[0, 1].imshow(graph_HC, cmap='hot', vmin=0, vmax=1)
axes[0, 1].set_title('Graph B (HC)', fontweight='bold')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

im3 = axes[0, 2].imshow(graph_AD - graph_HC, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0, 2].set_title('Difference (A - B)', fontweight='bold')
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

# Scatter plot of edge weights
triu_idx = np.triu_indices(n_channels, k=1)
axes[0, 3].scatter(graph_AD[triu_idx], graph_HC[triu_idx], alpha=0.3, s=5)
axes[0, 3].plot([0, 1], [0, 1], 'r--', label='Identity')
axes[0, 3].set_xlabel('Graph A edges')
axes[0, 3].set_ylabel('Graph B edges')
axes[0, 3].set_title(f'Edge Comparison\nr = {r_AD_HC:.3f}', fontweight='bold')
axes[0, 3].legend()

# Row 2: Spectra and statistics
eig_AD = laplacian_spectrum(graph_AD)
eig_HC = laplacian_spectrum(graph_HC)

axes[1, 0].plot(eig_AD, 'r-', label='Graph A', linewidth=2)
axes[1, 0].plot(eig_HC, 'b-', label='Graph B', linewidth=2)
axes[1, 0].set_xlabel('Eigenvalue index')
axes[1, 0].set_ylabel('λ')
axes[1, 0].set_title('Laplacian Spectra', fontweight='bold')
axes[1, 0].legend()

# Histogram of differences
diff = graph_AD[triu_idx] - graph_HC[triu_idx]
axes[1, 1].hist(diff, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(np.mean(diff), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(diff):.3f}')
axes[1, 1].set_xlabel('Edge difference (A - B)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Distribution of Differences', fontweight='bold')
axes[1, 1].legend()

# Metrics bar chart
metrics = ['Pearson r', 'Cosine', 'Jaccard', 'Spectrum r']
values_AD_HC = [r_AD_HC, cos_AD_HC, jac_AD_HC, r_spec_AD_HC]
values_AD_sim = [r_AD_sim, cos_AD_sim, jac_AD_sim, r_spec_AD_sim]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[1, 2].bar(x - width/2, values_AD_HC, width, label='AD vs HC', color='#E74C3C')
bars2 = axes[1, 2].bar(x + width/2, values_AD_sim, width, label='AD vs Similar', color='#27AE60')
axes[1, 2].axhline(0.5, color='gray', linestyle='--', alpha=0.7)
axes[1, 2].set_ylabel('Similarity')
axes[1, 2].set_title('Comparison Metrics', fontweight='bold')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(metrics, rotation=45, ha='right')
axes[1, 2].legend()
axes[1, 2].set_ylim([0, 1.1])

# Summary text
axes[1, 3].axis('off')
summary = f"""
GRAPH COMPARISON RESULTS
════════════════════════

AD vs HC (Different groups):
  • Pearson r = {r_AD_HC:.3f}
  • Jaccard = {jac_AD_HC:.3f}
  • {stats_AD_HC['pct_different']:.1f}% edges different
  → GRAPHS ARE DIFFERENT ✓

AD vs Similar (Same pattern):
  • Pearson r = {r_AD_sim:.3f}
  • Jaccard = {jac_AD_sim:.3f}
  • {stats_AD_sim['pct_different']:.1f}% edges different
  → GRAPHS ARE SIMILAR ✓

════════════════════════
DECISION RULES:
  r > 0.7 AND Jaccard > 0.5
    → Graphs are SAME
  r < 0.5 OR Jaccard < 0.3
    → Graphs are DIFFERENT
"""
axes[1, 3].text(0.1, 0.95, summary, transform=axes[1, 3].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('How to Compare Two Graphs: Are They Same or Different?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graph_comparison_demo.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: graph_comparison_demo.png")
plt.show()

# =============================================================================
# THESIS TEXT
# =============================================================================
print("\n" + "="*70)
print("THESIS-READY TEXT")
print("="*70)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                         SAMPLE PARAGRAPHS                                ║
╚══════════════════════════════════════════════════════════════════════════╝

METHODS - How to compare graphs:
────────────────────────────────
"Graph similarity was assessed using multiple complementary metrics. 
Pearson correlation was computed between the upper triangular elements 
of the two adjacency matrices, measuring linear relationship between 
edge weights. Jaccard similarity was calculated after binarizing each 
graph (retaining top 15% of edges), quantifying edge overlap. The 
normalized Frobenius norm ||A-B||_F/||A||_F measured overall matrix 
distance. Additionally, Laplacian eigenvalue spectra were compared to 
assess structural similarity at the global topological level."


RESULTS - Reporting comparison:
───────────────────────────────
"Comparison of AD and HC consensus matrices revealed significant 
differences in connectivity patterns. Pearson correlation between 
the two consensus matrices was r = 0.XX (indicating [moderate/low] 
similarity). Jaccard similarity of J = 0.XX showed that only XX% of 
edges overlapped between groups. Edge-wise analysis identified XX% of 
connections as significantly different (|Δw| > 1σ). Laplacian spectra 
correlation was r = 0.XX, confirming [similar/different] global 
topology. These results demonstrate that AD and HC groups exhibit 
[distinct/similar] connectivity architectures."


INTERPRETATION GUIDE:
─────────────────────
• r > 0.7, Jaccard > 0.5  →  "Graphs show high similarity"
• r = 0.5-0.7             →  "Graphs show moderate similarity"
• r < 0.5, Jaccard < 0.3  →  "Graphs are substantially different"


FIGURE CAPTION:
───────────────
"Figure X. Comparison of AD and HC consensus matrices. (A) AD consensus 
adjacency matrix. (B) HC consensus adjacency matrix. (C) Difference 
matrix (AD - HC); red indicates AD > HC, blue indicates HC > AD. 
(D) Scatter plot of corresponding edge weights showing correlation 
r = 0.XX. (E) Laplacian eigenvalue spectra comparison. (F) Distribution 
of edge-wise differences. (G) Summary of similarity metrics."
""")

print("\n" + "="*70)
print("QUICK REFERENCE: WHEN ARE GRAPHS THE SAME?")
print("="*70)

print("""
┌────────────────────────────────────────────────────────────────────────┐
│                    DECISION CRITERIA                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  GRAPHS ARE THE SAME IF:                                               │
│    ✓ Pearson r > 0.7                                                   │
│    ✓ Jaccard similarity > 0.5                                          │
│    ✓ < 10% of edges significantly different                            │
│    ✓ Frobenius distance < 0.3 (normalized)                             │
│                                                                        │
│  GRAPHS ARE DIFFERENT IF:                                              │
│    ✗ Pearson r < 0.5                                                   │
│    ✗ Jaccard similarity < 0.3                                          │
│    ✗ > 20% of edges significantly different                            │
│    ✗ Frobenius distance > 0.5 (normalized)                             │
│                                                                        │
│  FOR YOUR THESIS:                                                      │
│    • Individual vs Consensus: expect r > 0.5 (validation)              │
│    • AD vs HC consensus: may be different (biological finding)         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
""")

print("\nDone! Check 'graph_comparison_demo.png' for visualization.")
