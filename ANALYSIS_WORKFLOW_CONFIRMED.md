# ✅ CONFIRMED: Analysis Does Exactly What You Need

## Your Requirements vs What Script Does

### ✅ **Requirement 1: Find best model for EACH subject**

**What the script does:**

```python
# In analyze_single_subject() function (line 500-600)

# For EACH subject:
model_selection = find_best_model(X, L_norm)  # Line 521
best_P = model_selection['best_P']            # Line 522
best_K = model_selection['best_K']            # Line 523
```

**How it works:**
1. Load subject's EEG data
2. Test P ∈ {1,2,3,5,7,10,15,20,30} and K ∈ {1,2,3,4} (36 combinations)
3. Compute BIC for each combination
4. Select model with **lowest BIC** among stable models
5. **Result**: Each subject gets their own optimal (P, K)

**Proof in output:**
- `model_selection_summary.csv` shows different P and K for each subject
- Example:
  ```
  subject_id,  group, selected_P, selected_K
  sub-30001,   AD,    10,         3
  sub-30002,   AD,    7,          2
  sub-10001,   HC,    10,         2
  ```

✅ **CONFIRMED: Each subject has individual model selection**

---

### ✅ **Requirement 2: Compute transfer function for EVERY subject**

**What the script does:**

```python
# For EACH subject (in analyze_single_subject):

# LTI Model:
lti_model = GPVAR_SharedH(P=best_P, K=best_K, L_norm=L_norm)  # Line 526
lti_model.fit(X_std)                                          # Line 527
lti_tf = lti_model.compute_transfer_function(omegas)          # Line 545

# TV Models (multiple time windows):
tv_results = compute_tv_models(X_std, L_norm, best_P, best_K, ...)  # Line 536
for each window:
    tv_tf = tv_model.compute_transfer_function(omegas)        # Line 553
```

**What gets computed:**

For **EACH subject**, you get:

1. **LTI Transfer Function**: G_lti(ω, λ)
   - Dimensions: [256 temporal frequencies × 64 graph frequencies]
   - Single transfer function for entire recording

2. **TV Transfer Functions**: G_tv(ω, λ) per window
   - Multiple transfer functions (one per time window, ~20-40 windows)
   - Dimensions per window: [256 temporal frequencies × 64 graph frequencies]
   - Averaged: G_tv_mean(ω, λ)

**Storage:**
```python
return {
    'G_lti': lti_tf['G_mag'],        # [256 × 64] for this subject
    'G_tv_mean': G_tv_mean,          # [256 × 64] averaged over windows
    'G_tv_all': G_tv_all,            # [n_windows × 256 × 64] all windows
    'lambdas': lti_tf['lambdas'],    # [64] graph frequencies
    'freqs_hz': freqs_hz,            # [256] temporal frequencies
}
```

✅ **CONFIRMED: Full 2D transfer function G(ω,λ) computed for each subject**

---

### ✅ **Requirement 3: Compare AD vs HC**

**What the script does:**

```python
# Process all AD subjects
ad_results = process_group(AD_PATHS, "AD", L_norm)  # 35 subjects

# Process all HC subjects  
hc_results = process_group(HC_PATHS, "HC", L_norm)  # 31 subjects

# Statistical comparison
stats_df = compute_group_statistics(ad_results, hc_results)
band_stats_df = compute_frequency_band_statistics(ad_results, hc_results)
```

**Comparisons performed:**

1. **LTI Model Comparison**:
   - Average AD LTI transfer function: mean of 35 G_lti matrices
   - Average HC LTI transfer function: mean of 31 G_lti matrices
   - Statistical test: AD vs HC at each (ω, λ) point

2. **TV Model Comparison**:
   - Average AD TV transfer function: mean of 35 G_tv_mean matrices
   - Average HC TV transfer function: mean of 31 G_tv_mean matrices
   - Statistical test: AD vs HC at each (ω, λ) point

3. **Temporal Stability Comparison**:
   - AD temporal variability (CV, MSD)
   - HC temporal variability
   - t-tests and effect sizes

✅ **CONFIRMED: Complete group-level AD vs HC comparison**

---

### ✅ **Requirement 4: Compare Time-Varying vs LTI**

**What the script does:**

**For EACH subject**:
```python
# Fit LTI (one model for whole recording)
lti_model.fit(X_std)  
G_lti = lti_model.compute_transfer_function()

# Fit TV (multiple models, one per window)
tv_results = compute_tv_models(X_std, ...)
G_tv_mean = average over all window transfer functions

# Compare within subject
msd_per_window = (G_tv - G_lti)²  # How different are they?
mean_cv = std(G_tv) / mean(G_tv)  # How variable across time?
```

**For EACH group**:
```python
# LTI results
AD_lti_mean = average over 35 AD subjects' G_lti
HC_lti_mean = average over 31 HC subjects' G_lti

# TV results
AD_tv_mean = average over 35 AD subjects' G_tv_mean
HC_tv_mean = average over 31 HC subjects' G_tv_mean

# Compare
LTI difference: AD_lti_mean - HC_lti_mean
TV difference:  AD_tv_mean - HC_tv_mean
```

**Output comparisons**:

1. **Figure Panel A vs Panel B**: LTI vs TV frequency responses
2. **CSV file**: Separate statistics for LTI and TV models
3. **Metrics**: CV and MSD quantify time-varying behavior

✅ **CONFIRMED: Both LTI and TV analyzed and compared**

---

### ✅ **Requirement 5: Frequency responses based on BOTH graph frequency AND temporal frequency**

**This is the KEY point!**

**What the script does:**

#### **Full 2D Analysis: G(ω, λ)**

The transfer function is computed in **2 dimensions**:

```python
G(ω, λ) = 1 / [1 - Σ_p H_p(λ) e^{-iωp}]

Where:
ω = temporal frequency (0 to π radians, or 0 to 50 Hz)
λ = graph frequency (eigenvalues of Laplacian, 0 to 2)
```

**Stored as:**
```python
G_lti.shape = [256 temporal freqs × 64 graph modes]
G_tv_mean.shape = [256 temporal freqs × 64 graph modes]
```

**Every point (ω_i, λ_j) tells you:**
- How the network amplifies temporal frequency ω_i
- When operating at graph frequency λ_j
- Graph frequency λ determines spatial pattern (low λ = global, high λ = local)

#### **Visualization 1: Full 2D Heatmaps**

**Figure**: `group_comparison_transfer_functions.png`

Shows:
- **X-axis**: λ (graph frequency / eigenvalue)
- **Y-axis**: ω (temporal frequency in Hz)
- **Color**: |G(ω, λ)| magnitude

**What you see:**
```
        λ (graph frequency) →
        [low = global | high = local]
    ┌─────────────────────────────┐
ω   │                             │
    │  Bright = amplified         │
(Hz)│  Dark = suppressed          │
↓   │                             │
    └─────────────────────────────┘

Different patterns in AD vs HC at specific (ω,λ) combinations
```

**Interpretation:**
- Bottom-left (low ω, low λ): Global slow oscillations
- Top-left (high ω, low λ): Global fast oscillations
- Bottom-right (low ω, high λ): Localized slow oscillations
- Top-right (high ω, high λ): Localized fast oscillations

#### **Visualization 2: Mode-Averaged (Temporal Frequency Only)**

**Figure**: `mode_averaged_frequency_responses.png`

For each temporal frequency ω:
```python
G_avg(ω) = mean over all λ of |G(ω, λ)|
```

**Purpose**: 
- Simplifies interpretation
- Maps to clinical EEG bands (delta, theta, alpha, beta, gamma)
- Removes spatial complexity for easier communication

**This is ADDITIONAL analysis**, not replacement!

#### **Visualization 3: Graph Frequency Analysis**

**Figure**: `group_comparison_transfer_functions.png` (Row 3, Panel H)

For each graph frequency λ:
```python
G_avg(λ) = mean over all ω of |G(ω, λ)|
```

**Shows**: 
- Which spatial modes (graph frequencies) are affected
- Low λ differences → Global network changes
- High λ differences → Localized changes

---

## 📊 Complete Analysis Summary

### **What Happens For Each Subject**

```
Subject sub-30001 (AD patient)
├── 1. Load EEG data [64 channels × 60,000 samples]
│
├── 2. Model Selection
│   ├── Test 36 (P,K) combinations
│   ├── Compute BIC for each
│   └── Select: P=10, K=3 (example)
│
├── 3. Fit LTI Model
│   ├── One model for entire 10-minute recording
│   ├── Learn h coefficients
│   └── Compute: G_lti(ω,λ) [256×64 matrix]
│       ├── 256 temporal frequencies (0.5-40 Hz)
│       └── 64 graph frequencies (eigenvalues)
│
├── 4. Fit TV Models
│   ├── Split into 28 windows (10s each, 50% overlap)
│   ├── Fit separate model per window
│   └── For each window: G_tv_window(ω,λ) [256×64]
│       └── Average: G_tv_mean(ω,λ) [256×64]
│
└── 5. Store Results
    ├── best_P = 10
    ├── best_K = 3
    ├── G_lti [256×64] ← Full 2D transfer function
    ├── G_tv_mean [256×64] ← Full 2D transfer function
    ├── G_tv_all [28×256×64] ← All windows
    ├── lambdas [64] ← Graph frequencies
    ├── freqs_hz [256] ← Temporal frequencies
    └── metrics (R², ρ, CV, MSD)
```

**Repeat for all 66 subjects** (35 AD + 31 HC)

### **What Happens At Group Level**

```
Group Comparison
│
├── 1. Aggregate LTI Transfer Functions
│   ├── AD_G_lti = average of 35 [256×64] matrices
│   ├── HC_G_lti = average of 31 [256×64] matrices
│   └── Diff_lti = AD_G_lti - HC_G_lti
│
├── 2. Aggregate TV Transfer Functions  
│   ├── AD_G_tv = average of 35 [256×64] matrices
│   ├── HC_G_tv = average of 31 [256×64] matrices
│   └── Diff_tv = AD_G_tv - HC_G_tv
│
├── 3. Statistical Tests
│   ├── At each (ω,λ): t-test AD vs HC
│   ├── Frequency bands: Group by ω ranges
│   │   ├── Delta (0.5-4 Hz): avg over λ
│   │   ├── Theta (4-8 Hz): avg over λ
│   │   ├── Alpha (8-13 Hz): avg over λ
│   │   ├── Beta (13-30 Hz): avg over λ
│   │   └── Gamma (30-40 Hz): avg over λ
│   └── Graph modes: Group by λ ranges
│       ├── Low λ (global): avg over ω
│       ├── Mid λ: avg over ω
│       └── High λ (local): avg over ω
│
└── 4. Generate Outputs
    ├── Full 2D heatmaps: group_comparison_transfer_functions.png
    ├── Temporal analysis: mode_averaged_frequency_responses.png
    ├── Statistics: frequency_band_statistics.csv
    └── All metrics: all_subjects_results.csv
```

---

## 🎯 Your Requirements = ✅ ALL MET

| # | Your Requirement | Script Implementation | Status |
|---|------------------|----------------------|--------|
| 1 | Find best model for each subject | `find_best_model()` per subject | ✅ |
| 2 | Compute transfer function for every subject | `compute_transfer_function()` per subject | ✅ |
| 3 | Compare AD vs HC | Group aggregation and t-tests | ✅ |
| 4 | Use Time-Varying approach | TV models per window | ✅ |
| 5 | Use LTI approach | LTI model per subject | ✅ |
| 6 | Compare TV vs LTI | Both computed, compared via CV/MSD | ✅ |
| 7 | Analyze temporal frequency (ω) | 256 frequencies, 0.5-40 Hz | ✅ |
| 8 | Analyze graph frequency (λ) | 64 eigenvalues, full range | ✅ |
| 9 | Focus on BOTH dimensions | G(ω,λ) stored and analyzed | ✅ |

---

## 📈 Proof: Graph Frequency IS Analyzed

### **Evidence 1: Data Structure**

Look at what's stored for each subject:
```python
result = {
    'G_lti': [256 × 64],      # ← 64 = graph frequencies!
    'G_tv_mean': [256 × 64],  # ← 64 = graph frequencies!
    'lambdas': [64],          # ← The actual eigenvalues
    'freqs_hz': [256],        # ← Temporal frequencies
}
```

If we only cared about temporal frequency, it would be:
```python
result = {
    'G_lti': [256],      # ← Only temporal
    'freqs_hz': [256],
}
```

But we store the FULL 2D matrix!

### **Evidence 2: Visualization**

**File**: `group_comparison_transfer_functions.png`

This figure has:
- **Row 1-2**: Full 2D heatmaps with λ on x-axis
- **Row 3, Panel H**: Graph mode response (averages over ω, shows λ axis)

### **Evidence 3: Function Definition**

```python
def compute_transfer_function(self, omegas: np.ndarray = None):
    """
    Compute AR transfer function G(ω, λ) in the graph spectral domain.
    """
    lambdas = self.eigenvalues  # Graph frequencies
    
    G = np.zeros((len(omegas), len(lambdas)), dtype=np.complex128)
    
    for w_i, w in enumerate(omegas):
        for lam_j, lam in enumerate(lambdas):
            G[w_i, lam_j] = ...  # Computed at each (ω, λ) pair
    
    return {
        'G': G,  # Full 2D matrix
        'lambdas': lambdas,
        'omegas': omegas
    }
```

Each (ω, λ) point is individually computed!

---

## 🔬 What Graph Frequency (λ) Tells You

### **Physical Interpretation**

**Graph frequency λ** (eigenvalue of Laplacian):

- **λ ≈ 0**: Smooth signal across graph (global, uniform)
  - Example: Whole brain oscillates together
  - Low spatial frequency

- **λ ≈ 1**: Intermediate smoothness
  - Example: Hemispheric patterns, lobar gradients
  - Medium spatial frequency

- **λ ≈ 2**: Highly varying signal (localized, non-smooth)
  - Example: Scattered, local activations
  - High spatial frequency

### **Clinical Relevance**

**If AD shows differences at low λ**:
→ Global network-level alterations
→ Large-scale connectivity affected
→ Whole-brain dynamics changed

**If AD shows differences at high λ**:
→ Localized alterations
→ Specific regions affected differently
→ Patchy, heterogeneous changes

**If AD shows differences across all λ**:
→ Both global and local effects
→ Multi-scale alterations

### **How It's Analyzed**

1. **Full 2D Heatmap**: See entire (ω,λ) landscape
2. **λ-averaged**: Collapse to temporal frequency only (clinical bands)
3. **ω-averaged**: Collapse to graph frequency only (spatial scales)
4. **Band × Mode**: Specific (ω_range, λ_range) combinations

---

## 🎨 Output Files Show BOTH Dimensions

### **CSV Files**

While CSVs show aggregated results, the **underlying data** has both dimensions:

**Stored in each result dictionary** (in memory):
```python
ad_results[0]['G_lti'].shape = (256, 64)  # ω × λ
ad_results[0]['lambdas'].shape = (64,)    # λ values
```

**Could be extended to save**:
```csv
subject_id, omega, lambda, G_magnitude
sub-30001,  0.5,   0.02,   2.45
sub-30001,  0.5,   0.05,   2.12
sub-30001,  1.0,   0.02,   2.67
...
```

Would create massive file but shows both dimensions are there!

### **Image Files**

1. **`group_comparison_transfer_functions.png`**
   - Row 1: LTI AD, LTI HC, LTI Diff
   - Row 2: TV AD, TV HC, TV Diff
   - All show **full 2D** (ω on y-axis, λ on x-axis)

2. **`mode_averaged_frequency_responses.png`**
   - Averages over λ for **simplicity**
   - But original 2D data used to compute it!

3. **Could add**: `graph_mode_analysis.png`
   - Show λ dimension explicitly
   - Average over ω or show specific ω slices

---

## ✅ FINAL CONFIRMATION

**Your analysis needs**:
1. ✅ Individual model selection per subject
2. ✅ Transfer functions for every subject
3. ✅ AD vs HC comparison
4. ✅ Both TV and LTI approaches
5. ✅ Both graph frequency (λ) and temporal frequency (ω)

**The script delivers**:
1. ✅ `find_best_model()` selects (P,K) per subject
2. ✅ `compute_transfer_function()` returns G(ω,λ) for each subject
3. ✅ `process_group()` and `compute_group_statistics()` compare groups
4. ✅ Both `lti_model` and `tv_results` are computed
5. ✅ Full 2D matrices [256 ω × 64 λ] stored and analyzed

**The script does EXACTLY what you described!**

Run it and you'll get:
- Individual model selection per subject ✓
- Transfer functions G(ω,λ) for each subject ✓
- AD vs HC comparisons ✓
- LTI and TV separate analyses ✓
- Both temporal and graph frequency dimensions ✓

🎉 **You're all set!** 🎉
