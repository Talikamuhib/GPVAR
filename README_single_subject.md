# Single Subject LTI vs TV Transfer Function Analysis

This analysis compares Linear Time-Invariant (LTI) and Time-Varying (TV) models for a single EEG subject by examining their frequency responses/transfer functions.

## Overview

The analysis performs the following steps:

1. **Load EEG Data**: Loads and preprocesses one subject's EEG recording
2. **Model Selection**: Finds optimal model order (P, K) using BIC
3. **LTI Model**: Fits a single GP-VAR model on the entire time series
4. **TV Models**: Fits separate GP-VAR models on overlapping time windows
5. **Transfer Functions**: Computes frequency responses G(ω,λ) for both models
6. **Visualization**: Creates detailed comparison plots

## Key Files

- `lti_tv_single_subject_analysis.py`: Main analysis script for single subject
- `run_single_subject_analysis.py`: Runner script to execute the analysis
- `README_single_subject.md`: This documentation

## Configuration

Edit the configuration section in `lti_tv_single_subject_analysis.py`:

```python
# Select ONE subject file to analyze
SUBJECT_FILE = '/path/to/your/eeg/file.set'

# Time-varying analysis parameters
WINDOW_LENGTH_SEC = 10.0  # Window length in seconds
WINDOW_OVERLAP = 0.5      # 50% overlap between windows

# Preprocessing
BAND = (0.5, 40.0)       # Bandpass filter range in Hz
TARGET_SFREQ = 100.0     # Target sampling rate
```

## Running the Analysis

```bash
python run_single_subject_analysis.py
```

Or directly:

```bash
python lti_tv_single_subject_analysis.py
```

## Output

The analysis creates the following in `./single_subject_lti_tv_analysis/`:

### 1. Detailed Transfer Function Comparison (`*_transfer_functions_detailed.png`)
- **Row 1**: 2D heatmaps comparing LTI and TV transfer functions
  - LTI |G(ω,λ)|: Transfer function magnitude for LTI model
  - TV Mean |G(ω,λ)|: Average transfer function across time windows
  - TV Mean - LTI: Difference between models
  - Variance Across Time: Time-varying behavior

- **Row 2**: Frequency and graph mode slices
  - Transfer functions at specific frequencies (5, 10, 20, 30 Hz)
  - Transfer functions at specific graph modes

- **Row 3**: Time evolution analysis
  - Evolution of transfer function at specific (ω,λ) points
  - Average response per graph mode and frequency
  - Mean squared difference (MSD) over time

### 2. 3D Surface Plots (`*_transfer_functions_3D.png`)
- 3D visualization of LTI and TV mean transfer functions
- Shows the relationship between temporal frequency, graph frequency, and magnitude

## Transfer Function Interpretation

The transfer function G(ω,λ) represents:
- **ω**: Temporal frequency (in radians or Hz)
- **λ**: Graph frequency (eigenvalues of the Laplacian)
- **|G(ω,λ)|**: Magnitude of the frequency response

### Key Metrics:

1. **Mean Squared Difference (MSD)**: Measures overall difference between LTI and TV models
2. **Coefficient of Variation**: Indicates degree of time-varying behavior
3. **Variance Across Windows**: Shows which frequency components vary most over time

## Determining Time-Varying Behavior

The system is considered time-varying if:
- Mean coefficient of variation > 0.1
- Significant MSD between LTI and TV models
- High variance across time windows in specific frequency bands

## Model Parameters

- **P**: AR model order (number of time lags)
- **K**: Graph filter order (polynomial degree in Laplacian)
- **Ridge parameter**: λ = 5e-3 for regularization

## Dependencies

```bash
pip install numpy scipy matplotlib seaborn mne pandas
```

## Troubleshooting

1. **Memory Issues**: Reduce `WINDOW_LENGTH_SEC` or increase `WINDOW_OVERLAP`
2. **Unstable Models**: Increase `RIDGE_LAMBDA` for more regularization
3. **Too Few Windows**: Ensure recording is long enough for desired window parameters

## Example Results Interpretation

- **Time-Invariant System**: 
  - LTI and TV transfer functions are similar
  - Low variance across windows
  - Small MSD values

- **Time-Varying System**:
  - Significant differences between LTI and TV
  - High variance in specific frequency bands
  - Large MSD values
  - Transfer functions evolve over time

## References

- GP-VAR: Gaussian Process Vector Autoregression with graph structure
- Transfer functions computed as: G(ω,λ) = 1 / (1 - Σ_p H_p(λ) e^{-iωp})
- Graph signal processing framework for spatiotemporal analysis