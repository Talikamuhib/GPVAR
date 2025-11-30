"""
=============================================================================
WHAT IS JACCARD SIMILARITY? - Simple Explanation
=============================================================================

Jaccard Similarity measures: "How much do two sets OVERLAP?"

FORMULA:
                    |A ∩ B|        Elements in BOTH
    Jaccard = ───────────────  =  ─────────────────────
                    |A ∪ B|        Elements in EITHER

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("WHAT IS JACCARD SIMILARITY?")
print("="*70)

# =============================================================================
# SIMPLE EXAMPLE 1: Two Sets of Friends
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    SIMPLE EXAMPLE: Friends                               ║
╚══════════════════════════════════════════════════════════════════════════╝

Imagine you and your friend each have a list of favorite movies:

  YOUR movies:    {Batman, Spiderman, Avengers, Joker, Thor}
  FRIEND movies:  {Batman, Avengers, Superman, Hulk, Thor}
  
  Movies in BOTH lists (intersection):  {Batman, Avengers, Thor} = 3 movies
  Movies in EITHER list (union):        {Batman, Spiderman, Avengers, Joker, 
                                         Thor, Superman, Hulk} = 7 movies

                  3
  Jaccard = ─────────── = 0.43 (43% overlap)
                  7

INTERPRETATION:
  • Jaccard = 1.0  → You have IDENTICAL taste (100% same movies)
  • Jaccard = 0.5  → You share HALF of your combined movies
  • Jaccard = 0.0  → You have NO movies in common
  • Jaccard = 0.43 → You share about 43% of combined movies
""")

# =============================================================================
# SIMPLE EXAMPLE 2: With Numbers
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    EXAMPLE WITH NUMBERS                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# Define two sets
set_A = {1, 2, 3, 4, 5}
set_B = {3, 4, 5, 6, 7}

intersection = set_A & set_B  # Elements in BOTH
union = set_A | set_B         # Elements in EITHER

jaccard = len(intersection) / len(union)

print(f"  Set A: {set_A}")
print(f"  Set B: {set_B}")
print(f"")
print(f"  Intersection (A ∩ B): {intersection}  →  {len(intersection)} elements")
print(f"  Union (A ∪ B):        {union}  →  {len(union)} elements")
print(f"")
print(f"  Jaccard = {len(intersection)} / {len(union)} = {jaccard:.2f}")

# =============================================================================
# EXAMPLE 3: Brain Connectivity (Your Use Case!)
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                FOR YOUR THESIS: Brain Connectivity                       ║
╚══════════════════════════════════════════════════════════════════════════╝

In brain connectivity, Jaccard measures: 
"How many EDGES (connections) are shared between two graphs?"

Example with 5 brain regions (simplified):

  Graph A (Subject 1):          Graph B (Subject 2):
  
       1───2                         1───2
       │ × │                         │   │
       3───4───5                     3───4───5
       
  Edges in A: {1-2, 1-3, 2-4, 3-4, 4-5}  (5 edges)
  Edges in B: {1-2, 1-4, 2-4, 3-4, 4-5}  (5 edges)
  
  Common edges:  {1-2, 2-4, 3-4, 4-5}    (4 edges)
  All edges:     {1-2, 1-3, 1-4, 2-4, 3-4, 4-5}  (6 edges)
  
                  4
  Jaccard = ─────────── = 0.67  (67% edge overlap)
                  6
""")

# Calculate this example
edges_A = {(1,2), (1,3), (2,4), (3,4), (4,5)}
edges_B = {(1,2), (1,4), (2,4), (3,4), (4,5)}

common = edges_A & edges_B
all_edges = edges_A | edges_B
jaccard_brain = len(common) / len(all_edges)

print(f"  Edges in A: {len(edges_A)}")
print(f"  Edges in B: {len(edges_B)}")
print(f"  Common edges: {len(common)}")
print(f"  Total unique edges: {len(all_edges)}")
print(f"  Jaccard = {jaccard_brain:.2f}")

# =============================================================================
# EXAMPLE 4: Real Brain Matrix Example
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                REAL EXAMPLE: EEG Connectivity Matrix                     ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

np.random.seed(42)

# Create two example 8x8 connectivity matrices
n = 8

# Matrix A (e.g., Subject 1 or AD consensus)
matrix_A = np.random.rand(n, n)
matrix_A = (matrix_A + matrix_A.T) / 2  # Symmetric
np.fill_diagonal(matrix_A, 0)

# Matrix B (e.g., Subject 2 or HC consensus)
matrix_B = np.random.rand(n, n)
matrix_B = (matrix_B + matrix_B.T) / 2
np.fill_diagonal(matrix_B, 0)

print("Step 1: We have two weighted connectivity matrices")
print(f"        Matrix A: {n}x{n} (continuous weights 0-1)")
print(f"        Matrix B: {n}x{n} (continuous weights 0-1)")

# Binarize - keep top 30% of edges
sparsity = 0.30
n_edges_keep = int(sparsity * n * (n-1) / 2)

print(f"\nStep 2: Binarize (keep top {int(sparsity*100)}% strongest edges)")

# Get upper triangle indices
triu_idx = np.triu_indices(n, k=1)
weights_A = matrix_A[triu_idx]
weights_B = matrix_B[triu_idx]

# Find threshold
thresh_A = np.sort(weights_A)[::-1][n_edges_keep]
thresh_B = np.sort(weights_B)[::-1][n_edges_keep]

# Binarize
binary_A = (weights_A >= thresh_A).astype(int)
binary_B = (weights_B >= thresh_B).astype(int)

print(f"        Matrix A: {np.sum(binary_A)} edges kept")
print(f"        Matrix B: {np.sum(binary_B)} edges kept")

# Calculate Jaccard
intersection = np.sum((binary_A == 1) & (binary_B == 1))
union = np.sum((binary_A == 1) | (binary_B == 1))
jaccard = intersection / union if union > 0 else 0

print(f"\nStep 3: Calculate Jaccard")
print(f"        Edges in BOTH (intersection): {intersection}")
print(f"        Edges in EITHER (union): {union}")
print(f"        Jaccard = {intersection}/{union} = {jaccard:.3f}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\nCreating visualization...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: The concept
# Panel 1: Venn diagram concept
ax1 = axes[0, 0]
from matplotlib.patches import Circle
circle1 = Circle((0.35, 0.5), 0.3, color='red', alpha=0.5, label='Set A')
circle2 = Circle((0.65, 0.5), 0.3, color='blue', alpha=0.5, label='Set B')
ax1.add_patch(circle1)
ax1.add_patch(circle2)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.text(0.25, 0.5, 'Only\nA', ha='center', va='center', fontsize=10)
ax1.text(0.5, 0.5, 'A∩B', ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(0.75, 0.5, 'Only\nB', ha='center', va='center', fontsize=10)
ax1.set_title('Jaccard = (A∩B) / (A∪B)\nOverlap / Total', fontweight='bold', fontsize=11)
ax1.axis('off')

# Panel 2: Formula
ax2 = axes[0, 1]
ax2.axis('off')
formula_text = """
JACCARD FORMULA:

         |A ∩ B|
J = ─────────────
         |A ∪ B|

    Elements in BOTH
= ───────────────────
    Elements in EITHER


RANGE: 0 to 1

• J = 1.0 → Identical
• J = 0.5 → Half overlap  
• J = 0.0 → No overlap
"""
ax2.text(0.1, 0.9, formula_text, transform=ax2.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 3: Examples scale
ax3 = axes[0, 2]
examples = [0, 0.25, 0.5, 0.75, 1.0]
colors = ['#d62728', '#ff7f0e', '#ffff00', '#90EE90', '#2ca02c']
labels = ['No overlap\n(Different)', 'Low\noverlap', 'Half\noverlap', 'High\noverlap', 'Perfect\n(Same)']

bars = ax3.barh(range(5), examples, color=colors, edgecolor='black', height=0.6)
ax3.set_yticks(range(5))
ax3.set_yticklabels(labels)
ax3.set_xlabel('Jaccard Similarity')
ax3.set_title('Interpretation Scale', fontweight='bold')
ax3.set_xlim(0, 1.1)
for i, (bar, val) in enumerate(zip(bars, examples)):
    ax3.text(val + 0.02, i, f'{val:.2f}', va='center', fontweight='bold')

# Panel 4: Brain example
ax4 = axes[0, 3]
ax4.axis('off')
brain_text = """
FOR BRAIN CONNECTIVITY:

Graph A (e.g., AD patient):
  Has edges: {1-2, 1-3, 2-4, 3-4}
  
Graph B (e.g., HC control):
  Has edges: {1-2, 2-3, 2-4, 3-5}

Common edges: {1-2, 2-4}  = 2
Total edges:  {1-2,1-3,2-3,2-4,3-4,3-5} = 6

Jaccard = 2/6 = 0.33

→ "33% of connections are
   shared between graphs"
"""
ax4.text(0.05, 0.95, brain_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
ax4.set_title('Brain Connectivity Example', fontweight='bold')

# Row 2: Matrix example
# Panel 5: Matrix A
binary_A_mat = np.zeros((n, n))
binary_A_mat[triu_idx] = binary_A
binary_A_mat = binary_A_mat + binary_A_mat.T

ax5 = axes[1, 0]
ax5.imshow(binary_A_mat, cmap='Reds', vmin=0, vmax=1)
ax5.set_title(f'Graph A (Binary)\n{int(np.sum(binary_A))} edges', fontweight='bold')
ax5.set_xlabel('Channel')
ax5.set_ylabel('Channel')

# Panel 6: Matrix B
binary_B_mat = np.zeros((n, n))
binary_B_mat[triu_idx] = binary_B
binary_B_mat = binary_B_mat + binary_B_mat.T

ax6 = axes[1, 1]
ax6.imshow(binary_B_mat, cmap='Blues', vmin=0, vmax=1)
ax6.set_title(f'Graph B (Binary)\n{int(np.sum(binary_B))} edges', fontweight='bold')
ax6.set_xlabel('Channel')
ax6.set_ylabel('Channel')

# Panel 7: Overlap (intersection)
intersection_mat = binary_A_mat * binary_B_mat

ax7 = axes[1, 2]
ax7.imshow(intersection_mat, cmap='Greens', vmin=0, vmax=1)
ax7.set_title(f'A ∩ B (Shared Edges)\n{intersection} edges', fontweight='bold')
ax7.set_xlabel('Channel')
ax7.set_ylabel('Channel')

# Panel 8: Result
ax8 = axes[1, 3]
ax8.axis('off')

result_text = f"""
╔═══════════════════════════════╗
║     JACCARD CALCULATION       ║
╠═══════════════════════════════╣
║                               ║
║  Edges in A:        {int(np.sum(binary_A)):3d}      ║
║  Edges in B:        {int(np.sum(binary_B)):3d}      ║
║                               ║
║  Intersection (∩):  {intersection:3d}      ║
║  Union (∪):         {union:3d}      ║
║                               ║
║  ─────────────────────────    ║
║                               ║
║  Jaccard = {intersection}/{union} = {jaccard:.3f}   ║
║                               ║
╠═══════════════════════════════╣
║  INTERPRETATION:              ║
║  {jaccard*100:.0f}% of edges are shared     ║
║  between the two graphs       ║
╚═══════════════════════════════╝
"""
ax8.text(0.05, 0.95, result_text, transform=ax8.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('Understanding Jaccard Similarity for Graph Comparison',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('jaccard_explained.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: jaccard_explained.png")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: WHAT IS JACCARD?")
print("="*70)

print("""
┌────────────────────────────────────────────────────────────────────────┐
│                         JACCARD SIMILARITY                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  SIMPLE DEFINITION:                                                    │
│  "What fraction of items appear in BOTH sets?"                         │
│                                                                        │
│  FORMULA:                                                              │
│                 Items in BOTH                                          │
│    Jaccard = ─────────────────────                                     │
│                Items in EITHER                                         │
│                                                                        │
│  FOR BRAIN GRAPHS:                                                     │
│                 Shared edges                                           │
│    Jaccard = ─────────────────────                                     │
│                Total unique edges                                      │
│                                                                        │
│  RANGE:                                                                │
│    0 = No overlap (completely different graphs)                        │
│    1 = Perfect overlap (identical graphs)                              │
│                                                                        │
│  THRESHOLD FOR YOUR THESIS:                                            │
│    J > 0.5 → "Graphs are similar" (majority of edges shared)           │
│    J < 0.3 → "Graphs are different" (few edges shared)                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

WHY USE JACCARD FOR BRAIN CONNECTIVITY?

  1. It's INTUITIVE: "What % of connections are shared?"
  
  2. It handles BINARY edges: After thresholding your correlation
     matrices, edges are either present (1) or absent (0)
  
  3. It's SYMMETRIC: Jaccard(A,B) = Jaccard(B,A)
  
  4. It's commonly used in network neuroscience literature

EXAMPLE FOR YOUR THESIS:

  "Individual subject connectivity matrices showed Jaccard similarity 
   of J = 0.65 ± 0.12 with the group consensus, indicating that 65% 
   of edges were shared on average, confirming the consensus matrix 
   adequately represents individual subjects."
""")
