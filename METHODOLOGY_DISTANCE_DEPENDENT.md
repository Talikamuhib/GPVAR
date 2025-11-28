# Mathematical Methodology: Distance-Dependent Consensus Matrix

## Overview

This document presents the complete mathematical framework for constructing a **distance-dependent group-level consensus connectivity matrix** from individual EEG recordings, following the Betzel et al. methodology adapted for EEG analysis.

---

## 1. Individual Correlation Matrices

### 1.1 Input Data

For each subject $s \in \{1, 2, ..., N\}$, we have EEG data:

$$\mathbf{X}^{(s)} \in \mathbb{R}^{C \times T}$$

where:
- $C$ = number of channels (electrodes)
- $T$ = number of time samples
- $N$ = total number of subjects ($N = N_{AD} + N_{HC}$)

### 1.2 Pearson Correlation Coefficient

The Pearson correlation between channels $i$ and $j$ for subject $s$:

$$r_{ij}^{(s)} = \frac{\text{cov}(x_i^{(s)}, x_j^{(s)})}{\sigma_{x_i^{(s)}} \cdot \sigma_{x_j^{(s)}}} = \frac{\sum_{t=1}^{T}(x_i^{(s)}(t) - \bar{x}_i^{(s)})(x_j^{(s)}(t) - \bar{x}_j^{(s)})}{\sqrt{\sum_{t=1}^{T}(x_i^{(s)}(t) - \bar{x}_i^{(s)})^2} \cdot \sqrt{\sum_{t=1}^{T}(x_j^{(s)}(t) - \bar{x}_j^{(s)})^2}}$$

### 1.3 Adjacency Matrix

We use absolute correlation to form the adjacency matrix:

$$A_{ij}^{(s)} = |r_{ij}^{(s)}|, \quad A_{ii}^{(s)} = 0$$

---

## 2. Proportional Thresholding (Binarization)

### 2.1 Definition

Each subject's continuous adjacency matrix is binarized using proportional thresholding with parameter $\kappa$:

$$B_{ij}^{(s)} = \begin{cases} 1 & \text{if } A_{ij}^{(s)} > \tau^{(s)} \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 Threshold Determination

The threshold $\tau^{(s)}$ is the $(1-\kappa)$-quantile of the upper triangular edge weights:

$$\tau^{(s)} = Q_{1-\kappa}\left(\{A_{ij}^{(s)} : i < j\}\right)$$

This ensures exactly $\kappa$ fraction of edges are retained per subject.

### 2.3 Typical Value

We use $\kappa = 0.15$ (15%), meaning each subject retains their top 15% strongest connections.

---

## 3. Consensus Matrix

### 3.1 Definition

The consensus matrix $\mathbf{C}$ captures the fraction of subjects exhibiting each connection:

$$\boxed{C_{ij} = \frac{1}{N} \sum_{s=1}^{N} B_{ij}^{(s)}}$$

### 3.2 Properties

- $C_{ij} \in [0, 1]$ for all $i, j$
- $C_{ij} = C_{ji}$ (symmetric)
- $C_{ii} = 0$ (no self-loops)
- $C_{ij} = 1$ means edge $(i,j)$ exists in ALL subjects
- $C_{ij} = 0$ means edge $(i,j)$ exists in NO subjects

---

## 4. Weight Matrix (Fisher-z Averaging)

### 4.1 Fisher-z Transformation

To properly average correlation coefficients, we apply Fisher's z-transformation:

$$z = \text{arctanh}(r) = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right)$$

Inverse transformation:

$$r = \tanh(z)$$

### 4.2 Subject Set for Each Edge

Define the set of subjects who have edge $(i,j)$:

$$\mathcal{S}_{ij} = \{s \in \{1,...,N\} : B_{ij}^{(s)} = 1\}$$

### 4.3 Weight Computation

$$\boxed{W_{ij} = \left| \tanh\left( \frac{1}{|\mathcal{S}_{ij}|} \sum_{s \in \mathcal{S}_{ij}} \text{arctanh}\left(\text{clip}(A_{ij}^{(s)}, -0.999, 0.999)\right) \right) \right|}$$

where clipping prevents numerical instability at $|r| = 1$.

---

## 5. Distance Matrix

### 5.1 Channel Locations

Let $\mathbf{p}_i \in \mathbb{R}^3$ be the 3D coordinates of channel $i$.

### 5.2 Euclidean Distance

The distance between channels $i$ and $j$:

$$\boxed{D_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2 = \sqrt{(p_{i,x} - p_{j,x})^2 + (p_{i,y} - p_{j,y})^2 + (p_{i,z} - p_{j,z})^2}}$$

### 5.3 Distance Matrix Properties

- $D_{ij} \geq 0$
- $D_{ij} = D_{ji}$ (symmetric)
- $D_{ii} = 0$

---

## 6. Distance Binning

### 6.1 Purpose

Short-range connections are typically stronger due to volume conduction and anatomical proximity. Distance binning ensures the final graph includes connections across ALL spatial scales.

### 6.2 Bin Definition

We divide edges into $K$ bins (typically $K = 10$) based on distance percentiles:

$$\text{Bin boundaries: } d_0, d_1, d_2, ..., d_K$$

where $d_k$ is the $\frac{k}{K} \times 100$-th percentile of all pairwise distances:

$$d_k = Q_{k/K}\left(\{D_{ij} : i < j\}\right)$$

### 6.3 Bin Assignment

Edge $(i,j)$ belongs to bin $b$ if:

$$d_{b-1} \leq D_{ij} < d_b$$

Define the set of edges in bin $b$:

$$\mathcal{E}_b = \{(i,j) : i < j, \, d_{b-1} \leq D_{ij} < d_b\}$$

### 6.4 Number of Edges per Bin

By construction using percentiles:

$$|\mathcal{E}_b| \approx \frac{E_{possible}}{K} = \frac{C(C-1)}{2K}$$

Each bin contains approximately equal number of possible edges.

---

## 7. Scoring Function

### 7.1 Edge Score Definition

Each edge is scored based on both consensus AND weight:

$$\boxed{S_{ij} = C_{ij} + \varepsilon \cdot W_{ij}}$$

where $\varepsilon$ is a small positive constant (typically $\varepsilon = 0.1$).

### 7.2 Interpretation

- **Primary criterion**: Consensus $C_{ij}$ (how many subjects have this edge)
- **Tie-breaker**: Weight $W_{ij}$ (how strong is the connection)
- Higher score = more reliable edge

### 7.3 Why This Scoring?

- $C_{ij}$ ranges from 0 to 1 (dominant term)
- $\varepsilon \cdot W_{ij}$ ranges from 0 to ~0.1 (tie-breaker)
- Two edges with same consensus: prefer the stronger one

---

## 8. Distance-Dependent Selection

### 8.1 The Key Idea

Instead of selecting top edges globally (which would bias toward short-range connections), we select edges **within each distance bin** to ensure representation across all spatial scales.

### 8.2 Selection Methods

#### Method A: Fixed Percentage per Bin

Select top $\rho$ fraction of edges from each bin based on score:

For each bin $b \in \{1, ..., K\}$:
$$\mathcal{E}_b^{selected} = \text{top}_{\rho \cdot |\mathcal{E}_b|}\left(\{(i,j) \in \mathcal{E}_b\}, \text{ by } S_{ij}\right)$$

Final edge set:
$$\mathcal{E}^{final} = \bigcup_{b=1}^{K} \mathcal{E}_b^{selected}$$

#### Method B: Majority Consensus per Bin (RECOMMENDED)

Select edges with majority consensus from each bin:

For each bin $b \in \{1, ..., K\}$:
$$\mathcal{E}_b^{selected} = \{(i,j) \in \mathcal{E}_b : C_{ij} > 0.50\}$$

This method:
- Uses biological criterion (majority = population-level)
- Applies it uniformly across distance bins
- Allows natural variation in edges per bin

---

## 9. Final Graph Construction

### 9.1 Edge Set

Using majority consensus with distance-dependent analysis:

$$\boxed{\mathcal{E}^{final} = \{(i,j) : i < j, \, C_{ij} > 0.50\}}$$

### 9.2 Final Graph Matrix

$$\boxed{G_{ij} = \begin{cases} W_{ij} & \text{if } (i,j) \in \mathcal{E}^{final} \text{ or } (j,i) \in \mathcal{E}^{final} \\ 0 & \text{otherwise} \end{cases}}$$

### 9.3 Properties

- $G_{ij} = G_{ji}$ (symmetric, undirected)
- $G_{ij} \geq 0$ (non-negative weights)
- $G_{ii} = 0$ (no self-loops)

---

## 10. Distance Distribution Analysis

### 10.1 Edges Kept per Bin

For each bin $b$:

$$n_b^{kept} = |\{(i,j) \in \mathcal{E}_b : G_{ij} > 0\}|$$

$$n_b^{possible} = |\mathcal{E}_b|$$

### 10.2 Retention Rate per Bin

$$\boxed{\text{Retention}_b = \frac{n_b^{kept}}{n_b^{possible}}}$$

### 10.3 Expected Pattern

| Bin | Distance | Expected Retention | Reason |
|-----|----------|-------------------|--------|
| 1-3 | Short | Higher | Volume conduction + true connectivity |
| 4-7 | Medium | Moderate | Mix of local and distributed networks |
| 8-10 | Long | Lower but present | Long-range functional connections |

---

## 11. Quality Metrics

### 11.1 Overall Sparsity

$$\text{Sparsity} = \frac{|\mathcal{E}^{final}|}{E_{possible}} = \frac{\sum_{i<j} \mathbb{1}[G_{ij} > 0]}{C(C-1)/2}$$

### 11.2 Mean Consensus of Kept Edges

$$\bar{C}_{kept} = \frac{1}{|\mathcal{E}^{final}|} \sum_{(i,j) \in \mathcal{E}^{final}} C_{ij}$$

### 11.3 Mean Weight of Kept Edges

$$\bar{W}_{kept} = \frac{1}{|\mathcal{E}^{final}|} \sum_{(i,j) \in \mathcal{E}^{final}} W_{ij}$$

### 11.4 Mean Score of Kept Edges

$$\bar{S}_{kept} = \frac{1}{|\mathcal{E}^{final}|} \sum_{(i,j) \in \mathcal{E}^{final}} S_{ij}$$

---

## 12. Complete Algorithm

```
Algorithm: Distance-Dependent Consensus Matrix Construction

INPUT:
  - EEG data X^(s) ∈ ℝ^(C×T) for s = 1, ..., N
  - Channel locations p_i ∈ ℝ³ for i = 1, ..., C
  - Parameters: κ = 0.15, ε = 0.1, K = 10

OUTPUT:
  - Final graph G ∈ ℝ^(C×C)

STEPS:

1. FOR each subject s = 1, ..., N:
   a. Compute correlation: A^(s)_ij = |corr(x_i^(s), x_j^(s))|
   b. Compute threshold: τ^(s) = Q_{1-κ}({A^(s)_ij : i < j})
   c. Binarize: B^(s)_ij = 𝟙[A^(s)_ij > τ^(s)]

2. Compute consensus matrix:
   C_ij = (1/N) Σ_s B^(s)_ij

3. Compute weight matrix:
   FOR each edge (i,j) with i < j:
     S_ij = {s : B^(s)_ij = 1}
     IF |S_ij| > 0:
       W_ij = |tanh(mean_{s ∈ S_ij} arctanh(A^(s)_ij))|

4. Compute distance matrix:
   D_ij = ||p_i - p_j||_2

5. Create distance bins:
   d_k = Q_{k/K}({D_ij : i < j}) for k = 0, 1, ..., K

6. Compute score matrix:
   S_ij = C_ij + ε · W_ij

7. Apply majority consensus selection:
   FOR each edge (i,j) with i < j:
     IF C_ij > 0.50:
       G_ij = W_ij
       G_ji = W_ij
     ELSE:
       G_ij = 0
       G_ji = 0

8. Analyze distance distribution:
   FOR each bin b = 1, ..., K:
     Count edges kept in bin b
     Compute mean consensus, weight, score in bin b

RETURN G
```

---

## 13. Mathematical Summary for Thesis

### Key Equations

**1. Consensus Matrix:**
$$C_{ij} = \frac{1}{N} \sum_{s=1}^{N} B_{ij}^{(s)}$$

**2. Weight Matrix (Fisher-z):**
$$W_{ij} = \left| \tanh\left( \frac{1}{|\mathcal{S}_{ij}|} \sum_{s \in \mathcal{S}_{ij}} \text{arctanh}(A_{ij}^{(s)}) \right) \right|$$

**3. Distance Matrix:**
$$D_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2$$

**4. Scoring Function:**
$$S_{ij} = C_{ij} + \varepsilon \cdot W_{ij}$$

**5. Distance Bins:**
$$\mathcal{E}_b = \{(i,j) : d_{b-1} \leq D_{ij} < d_b\}$$

**6. Final Graph (Majority Consensus):**
$$G_{ij} = W_{ij} \cdot \mathbb{1}[C_{ij} > 0.50]$$

---

## 14. Methodology Paragraph for Thesis

> **Consensus Matrix Construction**
>
> Group-level connectivity was established using a distance-dependent consensus approach adapted from Betzel et al. (2019). For each subject, EEG data were used to compute pairwise Pearson correlations between all channel pairs. These correlation matrices were binarized using proportional thresholding (κ = 0.15), retaining the top 15% of connections per subject to ensure uniform edge density across subjects.
>
> The consensus matrix C was computed as the element-wise mean of the binary matrices across all subjects, where C_ij represents the fraction of subjects exhibiting a connection between channels i and j. Edge weights W were computed using Fisher-z transformation to properly average correlation coefficients across subjects who exhibited each connection.
>
> To ensure the final graph captured connectivity across all spatial scales, edges were divided into K = 10 distance bins based on Euclidean distances between electrode positions. The majority consensus rule (C_ij > 0.50) was applied, retaining only edges present in more than half of the subjects. This selection criterion ensures that the final graph represents population-level connectivity patterns while excluding potentially spurious individual-level connections.
>
> The resulting sparse weighted graph G, with edge weights given by the Fisher-z averaged correlations W, was used for subsequent graph signal processing analysis.

---

## 15. References

1. Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2019). Distance-dependent consensus thresholds for generating group-representative structural brain networks. *Network Neuroscience*, 3(2), 475-496.

2. Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059-1069.

3. Fisher, R. A. (1921). On the probable error of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3-32.

4. van den Heuvel, M. P., & Sporns, O. (2011). Rich-club organization of the human connectome. *Journal of Neuroscience*, 31(44), 15775-15786.
