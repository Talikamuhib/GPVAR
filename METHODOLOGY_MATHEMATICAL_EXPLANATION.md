# Mathematical Methodology: Consensus Matrix Construction

## Overview

This document presents the mathematical framework for constructing a group-level consensus connectivity matrix from individual EEG recordings, suitable for Graph Signal Processing (GP-VAR) analysis.

---

## 1. Individual Correlation Matrices

### 1.1 Pearson Correlation

For each subject $s \in \{1, 2, ..., N\}$, we have EEG data:

$$X^{(s)} \in \mathbb{R}^{C \times T}$$

where:
- $C$ = number of channels (e.g., 64)
- $T$ = number of time samples
- $N$ = total number of subjects ($N_{AD}$ + $N_{HC}$)

The Pearson correlation between channels $i$ and $j$ for subject $s$:

$$r_{ij}^{(s)} = \frac{\sum_{t=1}^{T}(x_i^{(s)}(t) - \bar{x}_i^{(s)})(x_j^{(s)}(t) - \bar{x}_j^{(s)})}{\sqrt{\sum_{t=1}^{T}(x_i^{(s)}(t) - \bar{x}_i^{(s)})^2} \cdot \sqrt{\sum_{t=1}^{T}(x_j^{(s)}(t) - \bar{x}_j^{(s)})^2}}$$

### 1.2 Adjacency Matrix

We use absolute correlation to form the adjacency matrix:

$$A_{ij}^{(s)} = |r_{ij}^{(s)}|$$

with $A_{ii}^{(s)} = 0$ (no self-loops).

---

## 2. Proportional Thresholding (Binarization)

### 2.1 Definition

Each subject's continuous adjacency matrix is binarized using proportional thresholding with parameter $\kappa$ (typically $\kappa = 0.15$ or 15%):

$$B_{ij}^{(s)} = \begin{cases} 1 & \text{if } A_{ij}^{(s)} > \tau^{(s)} \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 Threshold Calculation

The threshold $\tau^{(s)}$ is chosen such that exactly $\kappa$ fraction of possible edges are retained:

$$\tau^{(s)} = \text{percentile}\left(\{A_{ij}^{(s)} : i < j\}, \, (1-\kappa) \times 100\right)$$

This ensures each subject has the same edge density $\kappa$, allowing fair comparison across subjects.

### 2.3 Number of Edges

For $C$ channels, the number of possible edges is:

$$E_{possible} = \frac{C(C-1)}{2}$$

After thresholding, each subject has approximately:

$$E_{subject} \approx \kappa \times E_{possible}$$

---

## 3. Consensus Matrix

### 3.1 Definition

The consensus matrix $\mathbf{C}$ represents the fraction of subjects who have a connection between channels $i$ and $j$:

$$C_{ij} = \frac{1}{N} \sum_{s=1}^{N} B_{ij}^{(s)}$$

### 3.2 Interpretation

- $C_{ij} = 1.0$: Edge $(i,j)$ exists in ALL subjects (unanimous)
- $C_{ij} = 0.75$: Edge $(i,j)$ exists in 75% of subjects
- $C_{ij} = 0.50$: Edge $(i,j)$ exists in half of subjects
- $C_{ij} = 0.0$: Edge $(i,j)$ exists in NO subjects

### 3.3 Properties

- $C_{ij} \in [0, 1]$
- $C_{ij} = C_{ji}$ (symmetric)
- $C_{ii} = 0$ (no self-loops)

---

## 4. Weight Matrix (Fisher-z Transformation)

### 4.1 Problem with Simple Averaging

Correlation coefficients are bounded $r \in [-1, 1]$, making simple averaging biased. Fisher-z transformation addresses this.

### 4.2 Fisher-z Transformation

For a correlation coefficient $r$:

$$z = \text{arctanh}(r) = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right)$$

The inverse transformation:

$$r = \tanh(z) = \frac{e^{2z} - 1}{e^{2z} + 1}$$

### 4.3 Weight Matrix Computation

For each edge $(i,j)$, we compute the weight using only subjects who have that edge:

$$\mathcal{S}_{ij} = \{s : B_{ij}^{(s)} = 1\}$$

The weight is:

$$W_{ij} = \left| \tanh\left( \frac{1}{|\mathcal{S}_{ij}|} \sum_{s \in \mathcal{S}_{ij}} \text{arctanh}(A_{ij}^{(s)}) \right) \right|$$

### 4.4 Interpretation

$W_{ij}$ represents the average correlation strength for edge $(i,j)$, computed only from subjects who have that connection, using proper statistical averaging via Fisher-z transformation.

---

## 5. Majority Consensus Selection

### 5.1 Selection Rule

We apply the **majority consensus rule** to select edges for the final graph:

$$\text{Keep edge } (i,j) \iff C_{ij} > 0.50$$

### 5.2 Mathematical Justification

This rule has strong statistical interpretation:

- If $C_{ij} > 0.50$, then **more than half** of subjects have edge $(i,j)$
- This represents a **population-level** connection, not individual variation
- Edges with $C_{ij} \leq 0.50$ may be noise, artifacts, or individual quirks

### 5.3 Why 0.50?

| Threshold | Meaning | Trade-off |
|-----------|---------|-----------|
| $C > 0$ | Any subject has edge | Too permissive, includes noise |
| $C > 0.50$ | **Majority has edge** | **Optimal balance** |
| $C \geq 0.75$ | Strong majority | May be too restrictive |
| $C = 1.0$ | All subjects | Very restrictive |

The majority rule ($C > 0.50$) provides:
- Biological validity (population-level connections)
- Noise removal (excludes rare/artifactual edges)
- Stable graph for eigenvalue analysis

---

## 6. Final Graph Construction

### 6.1 Definition

The final graph adjacency matrix $\mathbf{G}$ is:

$$G_{ij} = \begin{cases} W_{ij} & \text{if } C_{ij} > 0.50 \\ 0 & \text{otherwise} \end{cases}$$

Or equivalently:

$$G_{ij} = W_{ij} \cdot \mathbb{1}[C_{ij} > 0.50]$$

where $\mathbb{1}[\cdot]$ is the indicator function.

### 6.2 Properties of Final Graph

- **Symmetric**: $G_{ij} = G_{ji}$
- **Non-negative**: $G_{ij} \geq 0$
- **Sparse**: Only edges with majority consensus are retained
- **Weighted**: Edge weights represent correlation strength

### 6.3 Sparsity

The final sparsity is:

$$\text{Sparsity} = \frac{|\{(i,j) : G_{ij} > 0, i < j\}|}{E_{possible}} = \frac{\sum_{i<j} \mathbb{1}[C_{ij} > 0.50]}{C(C-1)/2}$$

This sparsity emerges naturally from the data (not artificially imposed).

---

## 7. Graph Laplacian for GP-VAR

### 7.1 Degree Matrix

$$D_{ii} = \sum_{j=1}^{C} G_{ij}$$

### 7.2 Laplacian Matrix

$$\mathbf{L} = \mathbf{D} - \mathbf{G}$$

### 7.3 Normalized Laplacian (Optional)

$$\mathbf{L}_{norm} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{G} \mathbf{D}^{-1/2}$$

### 7.4 Eigendecomposition

$$\mathbf{L} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$$

where:
- $\mathbf{U}$ = eigenvector matrix (graph Fourier basis)
- $\mathbf{\Lambda}$ = diagonal matrix of eigenvalues (graph frequencies)

---

## 8. Complete Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MATHEMATICAL PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: EEG data X^(s) ∈ ℝ^(C×T) for s = 1, ..., N subjects

STEP 1: Correlation
        A^(s)_ij = |corr(x_i^(s), x_j^(s))|

STEP 2: Binarization (κ = 0.15)
        B^(s)_ij = 𝟙[A^(s)_ij > τ^(s)]
        where τ^(s) keeps top κ fraction of edges

STEP 3: Consensus
        C_ij = (1/N) Σ_s B^(s)_ij
        
STEP 4: Weights (Fisher-z)
        W_ij = |tanh(mean of arctanh(A^(s)_ij) for s where B^(s)_ij = 1)|

STEP 5: Majority Selection
        Keep edge (i,j) ⟺ C_ij > 0.50

STEP 6: Final Graph
        G_ij = W_ij · 𝟙[C_ij > 0.50]

OUTPUT: Sparse weighted graph G for GP-VAR analysis
```

---

## 9. Mathematical Notation Summary

| Symbol | Definition |
|--------|------------|
| $N$ | Total number of subjects |
| $C$ | Number of EEG channels |
| $T$ | Number of time samples |
| $X^{(s)}$ | EEG data matrix for subject $s$ |
| $A^{(s)}$ | Correlation-based adjacency matrix for subject $s$ |
| $B^{(s)}$ | Binary adjacency matrix for subject $s$ |
| $\kappa$ | Proportional threshold parameter (e.g., 0.15) |
| $\tau^{(s)}$ | Threshold value for subject $s$ |
| $\mathbf{C}$ | Consensus matrix |
| $\mathbf{W}$ | Weight matrix |
| $\mathbf{G}$ | Final graph adjacency matrix |
| $\mathbf{L}$ | Graph Laplacian |

---

## 10. References

1. Betzel, R. F., et al. (2019). "Distance-dependent consensus thresholds for generating group-representative structural brain networks." *Network Neuroscience*.

2. Rubinov, M., & Sporns, O. (2010). "Complex network measures of brain connectivity: Uses and interpretations." *NeuroImage*.

3. Fisher, R. A. (1921). "On the probable error of a coefficient of correlation deduced from a small sample." *Metron*.

---

## 11. Key Equations for Methodology Section

For your thesis methodology, the key equations are:

**Consensus Matrix:**
$$C_{ij} = \frac{1}{N} \sum_{s=1}^{N} B_{ij}^{(s)}$$

**Weight Matrix:**
$$W_{ij} = \left| \tanh\left( \frac{1}{|\mathcal{S}_{ij}|} \sum_{s \in \mathcal{S}_{ij}} \text{arctanh}(A_{ij}^{(s)}) \right) \right|$$

**Final Graph (Majority Consensus):**
$$G_{ij} = W_{ij} \cdot \mathbb{1}[C_{ij} > 0.50]$$

**Interpretation:**
> "An edge is included in the final graph if and only if more than half of the subjects exhibit that connection, ensuring the graph represents population-level connectivity patterns."
