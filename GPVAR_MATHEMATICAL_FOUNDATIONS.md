# GP-VAR Mathematical Foundations & Implementation Verification

## 1. The Graph Signal Processing Framework

### 1.1 Graph Signals

A **graph signal** is a function defined on the nodes of a graph. For EEG:
- Graph `G = (V, E)` where `V` = electrodes, `E` = connections
- Signal `x ∈ ℝ^N` where `N` = number of electrodes
- `x[i]` = voltage at electrode `i`

### 1.2 Graph Laplacian

The **combinatorial Laplacian** is defined as:

```
L = D - A
```

Where:
- `A` = adjacency matrix (A_ij = connection strength between i and j)
- `D` = degree matrix (D_ii = Σ_j A_ij, diagonal)

**Properties:**
- Symmetric: `L = L^T`
- Positive semi-definite: all eigenvalues ≥ 0
- Smallest eigenvalue = 0 (constant eigenvector)

### 1.3 Graph Fourier Transform

The **eigendecomposition** of L:

```
L = U Λ U^T
```

Where:
- `U = [u_1, u_2, ..., u_N]` = eigenvectors (graph Fourier basis)
- `Λ = diag(λ_1, λ_2, ..., λ_N)` = eigenvalues (graph frequencies)
- `0 = λ_1 ≤ λ_2 ≤ ... ≤ λ_N`

**Graph Fourier Transform:**
```
ŝ = U^T x    (analysis)
x = U ŝ      (synthesis)
```

**Interpretation of λ:**
- Low λ (small eigenvalues): Smooth/global modes
- High λ (large eigenvalues): Rough/localized modes

---

## 2. GP-VAR Model Formulation

### 2.1 Standard VAR Model

A Vector Autoregressive model of order P:

```
x_t = Σ_{p=1}^P A_p x_{t-p} + e_t
```

Where:
- `x_t ∈ ℝ^N` = signal at time t
- `A_p ∈ ℝ^{N×N}` = coefficient matrices
- `e_t` = white noise innovation

**Problem:** `N²P` parameters (for N=64, P=10: 40,960 parameters!)

### 2.2 Graph-Structured VAR (GP-VAR)

**Key insight:** Constrain `A_p` to be **graph polynomial filters**:

```
A_p = Σ_{k=0}^K h_{p,k} L^k
```

This is a **polynomial in L** with scalar coefficients `h_{p,k}`.

**Full GP-VAR equation:**

```
x_t = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} L^k x_{t-p} + e_t
```

**Parameters:** Only `P(K+1)` scalar coefficients!
- For P=10, K=3: Only 40 parameters (vs 40,960)

### 2.3 Why L^k Makes Sense

`L^k` represents **k-hop neighborhood aggregation**:

- `L^0 = I`: Self (identity)
- `L^1 = L`: Direct neighbors (1-hop)
- `L^2 = L·L`: 2-hop neighbors
- `L^k`: k-hop neighborhood

**Physical interpretation:**
- EEG signals propagate through brain networks
- `h_{p,k}` weights the contribution from k-hop neighbors at lag p

---

## 3. Graph Fourier Domain Analysis

### 3.1 Decoupling in Fourier Domain

Applying Graph Fourier Transform `U^T` to both sides:

```
U^T x_t = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} U^T L^k x_{t-p} + U^T e_t
```

Since `L = U Λ U^T`, we have `L^k = U Λ^k U^T`, thus:

```
U^T L^k = U^T U Λ^k U^T = Λ^k U^T
```

Let `s_t = U^T x_t` (graph Fourier coefficients):

```
s_t = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} Λ^k s_{t-p} + ε_t
```

### 3.2 Per-Mode Decoupling

For each graph frequency mode `i`:

```
s_t^(i) = Σ_{p=1}^P H_p(λ_i) s_{t-p}^(i) + ε_t^(i)
```

Where:

```
H_p(λ) = Σ_{k=0}^K h_{p,k} λ^k
```

**This is remarkable:** The N-dimensional system decouples into N independent scalar AR processes, each with its own frequency response!

---

## 4. Transfer Function Derivation

### 4.1 Z-Transform Analysis

Taking the z-transform of the per-mode equation:

```
S(z, λ_i) = Σ_{p=1}^P H_p(λ_i) z^{-p} S(z, λ_i) + E(z, λ_i)
```

Rearranging:

```
S(z, λ_i) [1 - Σ_{p=1}^P H_p(λ_i) z^{-p}] = E(z, λ_i)
```

### 4.2 Transfer Function

The **transfer function** is:

```
G(z, λ) = S(z, λ) / E(z, λ) = 1 / [1 - Σ_{p=1}^P H_p(λ) z^{-p}]
```

### 4.3 Frequency Response

Evaluating on the unit circle `z = e^{jω}`:

```
G(ω, λ) = 1 / [1 - Σ_{p=1}^P H_p(λ) e^{-jωp}]
```

Where:
- `ω ∈ [0, π]` = **temporal frequency** (normalized, radians)
- `λ` = **graph frequency** (eigenvalue of L)

**The magnitude `|G(ω, λ)|` shows how much the system amplifies signals at temporal frequency ω and graph frequency λ.**

---

## 5. Implementation Verification

### 5.1 Design Matrix Construction

**Mathematical formulation:**

For each time t > P, we predict x_t from:

```
x_t[i] = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} [L^k x_{t-p}][i] + e_t[i]
```

This is a **linear regression** problem. For node i at time t:

```
y = x_t[i]
features = [[L^0 x_{t-1}][i], [L^1 x_{t-1}][i], ..., [L^K x_{t-1}][i],
            [L^0 x_{t-2}][i], ..., [L^K x_{t-P}][i]]
```

**Implementation check:**

```python
def _build_design_matrix(self, X: np.ndarray):
    n, T = X.shape
    T_valid = T - self.P
    n_obs = n * T_valid          # Total observations
    n_feat = self.P * (self.K + 1)  # Total features
    
    R = np.zeros((n_obs, n_feat))
    Y = np.zeros(n_obs)
    
    for t in range(self.P, T):
        t_idx = t - self.P
        filter_vals = []
        for p in range(1, self.P + 1):       # p = 1, 2, ..., P
            xlag = X[:, t - p]                # x_{t-p}
            for k in range(self.K + 1):       # k = 0, 1, ..., K
                filter_vals.append(self.L_powers[k] @ xlag)  # L^k x_{t-p}
        filter_vals = np.asarray(filter_vals)  # Shape: [P(K+1), N]
        
        for i in range(n):
            row_idx = t_idx * n + i
            R[row_idx, :] = filter_vals[:, i]  # [L^k x_{t-p}][i]
            Y[row_idx] = X[i, t]               # x_t[i]
    
    return R, Y
```

**✓ CORRECT:** The design matrix correctly stacks:
- Rows: All (node, time) pairs
- Columns: All `[L^k x_{t-p}]` values ordered by (p, k)

### 5.2 Coefficient Ordering

The coefficient vector `h` is ordered as:

```
h = [h_{1,0}, h_{1,1}, ..., h_{1,K},   // lag 1
     h_{2,0}, h_{2,1}, ..., h_{2,K},   // lag 2
     ...
     h_{P,0}, h_{P,1}, ..., h_{P,K}]   // lag P
```

Index formula: `h[p*(K+1) + k]` = `h_{p+1, k}`

**✓ CORRECT:** Matches the loop order in design matrix construction.

### 5.3 H_p(λ) Computation

**Mathematical formula:**

```
H_p(λ) = Σ_{k=0}^K h_{p,k} λ^k
```

**Implementation:**

```python
H_p = np.zeros((P, len(lambdas)), dtype=np.complex128)
for p in range(P):
    for i, lam in enumerate(lambdas):
        val = 0.0
        for k in range(K + 1):
            val += self.h[p*(K+1) + k] * (lam ** k)  # h_{p+1,k} * λ^k
        H_p[p, i] = val
```

**✓ CORRECT:** 
- Loop over p = 0, 1, ..., P-1 (corresponds to lags 1, 2, ..., P)
- Index `p*(K+1) + k` correctly retrieves `h_{p+1, k}`

### 5.4 Transfer Function G(ω, λ)

**Mathematical formula:**

```
G(ω, λ) = 1 / [1 - Σ_{p=1}^P H_p(λ) e^{-jωp}]
```

**Implementation:**

```python
G = np.zeros((len(omegas), len(lambdas)), dtype=np.complex128)
for w_i, w in enumerate(omegas):
    z_terms = np.exp(-1j * w * np.arange(1, P+1))  # e^{-jω}, e^{-j2ω}, ..., e^{-jPω}
    denom = 1.0 - (z_terms[:, None] * H_p).sum(axis=0)  # 1 - Σ_p H_p(λ) e^{-jωp}
    
    # Stability guard
    denom = np.where(np.abs(denom) < 1e-3, denom + 1e-3, denom)
    
    G[w_i, :] = 1.0 / denom
```

**✓ CORRECT:**
- `z_terms` = `[e^{-jω}, e^{-j2ω}, ..., e^{-jPω}]` shape (P,)
- `H_p` shape (P, N_λ)
- `z_terms[:, None] * H_p` broadcasts to (P, N_λ)
- `.sum(axis=0)` gives `Σ_p H_p(λ) e^{-jωp}` for each λ

### 5.5 Spectral Radius (Stability)

**Theory:** The system is stable iff all poles of G(z, λ) are inside the unit circle.

For a VAR(P) system, we form the **companion matrix**:

```
C = [A_1  A_2  ... A_{P-1}  A_P]
    [I    0   ... 0        0  ]
    [0    I   ... 0        0  ]
    [          ...            ]
    [0    0   ... I        0  ]
```

Where `A_p = Σ_k h_{p,k} L^k`

**Spectral radius** = max|eigenvalue(C)|

System is stable iff ρ(C) < 1.

**Implementation:**

```python
def spectral_radius(self):
    A_mats = []
    idx = 0
    for p in range(self.P):
        A_p = np.zeros((self.n, self.n))
        for k in range(self.K + 1):
            A_p += self.h[idx] * self.L_powers[k]  # A_p = Σ_k h_{p+1,k} L^k
            idx += 1
        A_mats.append(A_p)
    
    # Companion matrix
    C = np.zeros((self.n * self.P, self.n * self.P))
    C[:self.n, :self.n*self.P] = np.hstack(A_mats)  # First row block
    if self.P > 1:
        C[self.n:, :-self.n] = np.eye(self.n * (self.P - 1))  # Identity blocks
    
    vals = np.linalg.eigvals(C)
    return float(np.max(np.abs(vals)))
```

**✓ CORRECT:** Companion matrix correctly constructed.

---

## 6. Model Selection via BIC

### 6.1 Bayesian Information Criterion

```
BIC = n·log(MSE) + k·log(n)
```

Where:
- `n` = number of observations
- `k` = number of parameters = P(K+1)
- `MSE` = mean squared error

**BIC penalizes model complexity** (the k·log(n) term), preferring simpler models unless the fit improvement justifies more parameters.

**Implementation:**

```python
n_obs = Y_true.size
n_params = self.P * (self.K + 1)
BIC = n_obs * np.log(MSE + 1e-10) + n_params * np.log(n_obs)
```

**✓ CORRECT**

---

## 7. Physical Interpretation

### 7.1 What Do the Coefficients Mean?

| Coefficient | Physical Meaning |
|-------------|------------------|
| `h_{1,0}` | Self-influence at lag 1 (AR coefficient) |
| `h_{1,1}` | 1-hop neighbor influence at lag 1 |
| `h_{1,2}` | 2-hop neighbor influence at lag 1 |
| `h_{p,k}` | k-hop neighbor influence at lag p |

### 7.2 Transfer Function Interpretation

| Region | ω | λ | Meaning |
|--------|---|---|---------|
| Bottom-left | Low | Low | Global slow oscillations |
| Top-left | High | Low | Global fast oscillations |
| Bottom-right | Low | High | Localized slow oscillations |
| Top-right | High | High | Localized fast oscillations |

**High |G(ω, λ)|** means the system **amplifies** that frequency combination.

### 7.3 EEG Bands in Transfer Function

Convert normalized ω to Hz: `f = ω × fs / (2π)`

| Band | Frequency (Hz) | ω (at 100 Hz fs) |
|------|----------------|------------------|
| Delta | 0.5-4 | 0.01π - 0.08π |
| Theta | 4-8 | 0.08π - 0.16π |
| Alpha | 8-13 | 0.16π - 0.26π |
| Beta | 13-30 | 0.26π - 0.6π |
| Gamma | 30-50 | 0.6π - π |

---

## 8. Potential Issues & Corrections

### 8.1 Numerical Stability

**Issue:** When `|denom| → 0`, the transfer function blows up.

**Solution (implemented):**
```python
denom = np.where(np.abs(denom) < 1e-3, denom + 1e-3, denom)
```

This prevents division by zero near poles.

### 8.2 Laplacian Normalization

**Issue:** Large eigenvalues can cause numerical issues in `λ^k`.

**Recommendation:** Normalize L so max eigenvalue ≈ 2:
```python
L_normalized = L / (λ_max / 2)
```

### 8.3 Ridge Regularization

**Issue:** With limited data, `R^T R` may be ill-conditioned.

**Solution (implemented):**
```python
theta = solve(R^T R + λI, R^T Y)
```

Ridge parameter `λ = 5e-3` adds stability.

---

## 9. Verification Summary

| Component | Formula | Implementation | Status |
|-----------|---------|----------------|--------|
| Design matrix | `[L^k x_{t-p}]_i` | ✓ Correct loop order | ✅ |
| Coefficient indexing | `h_{p,k}` at `p*(K+1)+k` | ✓ Matches design matrix | ✅ |
| H_p(λ) | `Σ_k h_{p,k} λ^k` | ✓ Correct polynomial | ✅ |
| G(ω, λ) | `1/(1-Σ H_p e^{-jωp})` | ✓ Correct formula | ✅ |
| Spectral radius | Companion matrix eigenvalues | ✓ Correct construction | ✅ |
| BIC | `n·log(MSE) + k·log(n)` | ✓ Correct formula | ✅ |
| Ridge regression | `(R^T R + λI)^{-1} R^T Y` | ✓ Correct solve | ✅ |

**CONCLUSION: The implementation is mathematically correct.**

---

## 10. References

1. **Graph Signal Processing:** Shuman, D. I., et al. "The emerging field of signal processing on graphs." IEEE Signal Processing Magazine (2013).

2. **GP-VAR Model:** Mei, J., & Moura, J. M. "Signal processing on graphs: Causal modeling of unstructured data." IEEE TSP (2017).

3. **VAR Transfer Functions:** Lütkepohl, H. "New Introduction to Multiple Time Series Analysis." Springer (2005).

4. **Spectral Analysis of Graphs:** Chung, F. R. K. "Spectral Graph Theory." AMS (1997).
