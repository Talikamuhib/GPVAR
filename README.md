# LTI vs Time-Varying GP-VAR Analysis Pipeline

## 📋 Overview

This pipeline implements a comprehensive analysis to determine whether EEG dynamics are **Linear Time-Invariant (LTI)** or **Time-Varying (TV)** using **Graph Polynomial Vector Autoregression (GP-VAR)** models with Graph Signal Processing (GSP) techniques.

The analysis addresses a fundamental question in neuroscience: **Do brain dynamics remain constant over time, or do they evolve?**

## 🧠 Mathematical Framework

### GP-VAR Model

The Graph Polynomial VAR model combines:
- **Graph Signal Processing**: Respects brain network topology
- **Autoregressive Modeling**: Captures temporal dynamics
- **Polynomial Graph Filters**: Allows flexible spectral responses

The model equation:
```
x_t = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} L^k x_{t-p} + e_t
```

Where:
- `x_t ∈ ℝ^N`: EEG signal vector at time t (N electrodes)
- `P`: Temporal AR order (number of time lags)
- `K`: Graph polynomial order (Laplacian powers)
- `h_{p,k}`: Scalar coefficients (shared across nodes)
- `L`: Graph Laplacian (consensus brain network)
- `L^k`: k-hop graph diffusion operator

### Graph Fourier Domain

Using eigendecomposition `L = UΛU^T`, the model decouples per graph frequency:
```
s_t^(i) = Σ_{p=1}^P H_p(λ_i) s_{t-p}^(i) + ε_t^(i)
```

Where:
- `s_t = U^T x_t`: Graph Fourier coefficients
- `λ_i`: Graph eigenvalues (graph frequencies)
- `H_p(λ_i) = Σ_k h_{p,k} λ_i^k`: Frequency response

### Transfer Function

The system's frequency response in joint graph-temporal domain:
```
G(ω, λ) = 1 / (1 - Σ_{p=1}^P H_p(λ) e^{-jωp})
```

- `ω`: Temporal frequency
- `λ`: Graph frequency
- `|G(ω, λ)|`: Magnitude response

## 🔬 Analysis Pipeline

### 1. **Data Preprocessing**
- Load EEG data (.set files)
- Resample to 100 Hz
- Band-pass filter (0.5-40 Hz)
- **Global z-scoring** (critical for unbiased TV analysis)

### 2. **Model Selection**
- Grid search over P ∈ {1,2,3,5,7,10,15,20} and K ∈ {1,2,3,4}
- Use BIC on validation set
- Ensure model stability (spectral radius < 1)

### 3. **LTI Model Fitting**
- Fit single GP-VAR model on entire recording
- Compute transfer function G_LTI(ω, λ)
- Evaluate R², BIC, spectral radius

### 4. **Time-Varying Analysis**
- Split data into overlapping windows (10s, 50% overlap)
- Fit separate GP-VAR model per window
- **No per-window normalization** (avoids artificial variation)

### 5. **Statistical Testing**

#### Mean Square Deviation (MSD)
```
MSD = (1/W) Σ_w ||G_w(ω,λ) - G_LTI(ω,λ)||²
```

#### Surrogate Null Distribution
- Circular shift each channel independently
- Preserves spectrum but destroys time structure
- Compute MSD on 200 surrogates
- p-value = P(MSD_surrogate ≥ MSD_observed)

#### Confidence Interval Test
- 95% CI from window-to-window variation
- Fraction of G_LTI outside CI

### 6. **Decision Criteria**
System is **Time-Varying** if:
- p-value < 0.05 (surrogate test)
- OR >5% of G_LTI outside 95% CI

## 🚀 Usage

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
python lti_tv_gpvar_analysis.py
```

### Configuration

Edit the configuration section in the script:

```python
# Paths
CONSENSUS_LAPLACIAN_PATH = "path/to/laplacian.npy"
SUBJECT_FILES = {
    'AD': ['path/to/ad_subject1.set', ...],
    'HC': ['path/to/hc_subject1.set', ...]
}

# Analysis parameters
WINDOW_LENGTH_SEC = 10.0    # Window duration
WINDOW_OVERLAP = 0.5         # 50% overlap
N_SURROGATES = 200           # Null distribution size
ALPHA = 0.05                 # Significance level
```

### Output Structure
```
lti_vs_tv_analysis_corrected/
├── subject_summary.csv           # All subjects' metrics
├── group_comparison.png          # AD vs HC visualization
├── group_comparison_stats.json   # Statistical test results
├── AD/
│   ├── subject1_lti_vs_tv.png
│   ├── subject2_lti_vs_tv.png
│   └── ...
└── HC/
    ├── subject1_lti_vs_tv.png
    ├── subject2_lti_vs_tv.png
    └── ...
```

## 📊 Metrics Explained

### Coefficient Variation (CV)
```
CV = σ(h) / |μ(h)|
```
Measures relative variation of coefficients across windows.

### Global MSD
Average squared difference between LTI and TV transfer functions.

### Outside CI Fraction
Proportion of LTI transfer function outside TV's 95% confidence bounds.

### Spectral Radius (ρ)
Largest eigenvalue magnitude of system matrix. ρ < 1 ensures stability.

## 🎯 Key Innovations

1. **Global Standardization**: Prevents artificial time-variation from window-wise normalization
2. **Proper Null Distribution**: Circular shift surrogates preserve spectrum while destroying time structure
3. **Stability Guards**: Prevent numerical issues in unstable regimes
4. **Direct Coefficient Analysis**: Track h_{p,k} variation across windows
5. **Joint Domain Analysis**: Simultaneous graph-temporal frequency analysis

## 📈 Visualizations

Each subject gets a comprehensive 4×4 subplot figure showing:

### Row 1: Transfer Functions
- LTI |G(ω,λ)|
- TV Mean |G(ω,λ)|
- Variance across windows
- Difference (TV - LTI)

### Row 2: Temporal Analysis
- MSD per window over time
- Per-mode response comparison

### Row 3: Statistical Tests
- Regions where LTI exits TV confidence interval
- Surrogate null distribution

### Row 4: Coefficients
- Coefficient trajectories across windows
- Coefficient variation (CV) per parameter

## 🔍 Group Comparison (AD vs HC)

The pipeline includes group-level statistical analysis:

### Metrics Compared
- Global MSD
- Global Variance
- Coefficient CV
- Outside CI Fraction

### Statistical Tests
- Mann-Whitney U test for continuous metrics
- Chi-square test for TV classification proportions
- Effect size (r) computation

## ⚠️ Critical Considerations

### Why Global Z-scoring Matters
Per-window normalization forces each window to have mean=0, std=1, creating artificial differences even in truly LTI systems. Global normalization preserves genuine amplitude variations.

### Why Circular Shift Surrogates
- Preserve autocorrelation structure
- Maintain spectral properties
- Destroy only cross-channel temporal alignment
- Better than random shuffling or phase randomization for this test

### Stability Requirements
Models with spectral radius ≥ 1 are unstable and produce unreliable transfer functions. The code automatically filters these out.

## 🔮 Interpretation

### Time-Invariant Result
- Brain dynamics follow consistent rules over recording
- Single model captures behavior adequately
- Simpler interpretation and prediction

### Time-Varying Result
- Brain dynamics evolve during recording
- May indicate:
  - State transitions (alertness, drowsiness)
  - Disease progression effects
  - Cognitive state changes
  - Non-stationarity requiring adaptive models

## 📚 Mathematical Background

This analysis bridges several domains:

1. **Graph Signal Processing (GSP)**
   - Signals on graphs (brain networks)
   - Graph Fourier transform
   - Graph filters as matrix polynomials

2. **System Identification**
   - VAR models for multivariate time series
   - Transfer function analysis
   - Stability analysis

3. **Statistical Hypothesis Testing**
   - Surrogate data methods
   - Multiple comparison correction
   - Non-parametric tests

## 🤝 Contributing

To adapt this pipeline for your data:

1. Prepare consensus Laplacian matrix (brain network)
2. Organize EEG files by group
3. Adjust preprocessing parameters if needed
4. Run analysis and interpret results

## 📖 References

Key concepts used:
- Graph Signal Processing fundamentals
- GP-VAR modeling for brain signals
- Surrogate testing for time series
- EEG preprocessing standards

## 💡 Tips for Best Results

1. **Data Quality**: Ensure EEG is properly preprocessed (artifacts removed)
2. **Recording Length**: Minimum 2 minutes for reliable window analysis
3. **Parameter Selection**: Use cross-validation for P and K
4. **Group Sizes**: At least 15-20 subjects per group for group comparisons
5. **Computational Resources**: ~2-5 minutes per subject depending on recording length

## 🐛 Troubleshooting

### Common Issues

1. **"Model unstable"**: Increase ridge penalty or reduce P
2. **"Too few windows"**: Increase recording length or reduce window size
3. **"No surrogates valid"**: Check data quality, may have too much noise
4. **Memory errors**: Reduce number of surrogates or downsample further

### Debug Mode

Add verbose flags in functions for detailed output:
```python
result = analyze_single_subject(filepath, L, subject_id, group, verbose=True)
```

## 📝 Citation

If using this pipeline, please cite the relevant GP-VAR and GSP literature that inspired this implementation.