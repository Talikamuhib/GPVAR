# LTI vs Time-Varying GP-VAR Group Comparison Analysis

## Overview

This analysis compares Alzheimer's Disease (AD) and Healthy Control (HC) groups using both Linear Time-Invariant (LTI) and Time-Varying (TV) GP-VAR models.

## Model Selection for Thesis Analysis

### Model Selection Process

For each subject, the script performs automatic model selection by:

1. **Testing multiple P (AR order) and K (graph filter order) combinations**:
   - P range: [1, 2, 3, 5, 7, 10, 15, 20, 30]
   - K range: [1, 2, 3, 4]

2. **Evaluation criteria**:
   - Bayesian Information Criterion (BIC) - primary selection metric
   - R² (goodness of fit)
   - Mean Squared Error (MSE)
   - Spectral radius (ρ) - stability check (must be < 0.99)

3. **Selection rule**: Choose the model with **lowest BIC** among all stable models

### Output Files for Thesis Analysis

The script generates comprehensive model selection results suitable for thesis documentation:

#### 1. Model Selection Tables

**Location**: `./group_comparison_lti_tv_analysis/model_selection/`

- **Individual subject tables**: `{subject_id}_model_selection.csv`
  - Contains BIC, R², MSE, ρ, stability status for every P-K combination tested
  - Allows detailed analysis of model selection landscape per subject

- **Combined table**: `all_subjects_model_selection.csv`
  - All subjects' model selection results in one file
  - Enables group-level model selection analysis
  - Columns: `subject_id`, `group`, `P`, `K`, `BIC`, `R2`, `MSE`, `rho`, `stable`, `success`

#### 2. Model Selection Summary

**File**: `model_selection_summary.csv`

Contains the **selected model** for each subject:
- `subject_id`: Subject identifier
- `group`: AD or HC
- `selected_P`: Chosen AR order
- `selected_K`: Chosen graph filter order
- `selected_BIC`: BIC value of selected model

**Use for thesis**: 
- Report model order distributions (mean ± std)
- Compare model complexity between groups
- Statistical tests on P and K differences

#### 3. Model Selection Visualizations

**File**: `model_selection_analysis.png`

A comprehensive figure containing:

1. **P Distribution**: Histogram comparing selected P values (AD vs HC)
2. **K Distribution**: Histogram comparing selected K values (AD vs HC)
3. **P vs K Scatter**: Joint distribution showing model complexity
4. **P Comparison Boxplot**: Statistical comparison of AR orders between groups
5. **K Comparison Boxplot**: Statistical comparison of graph filter orders between groups
6. **Summary Statistics Table**: 
   - Mean ± std for P and K in each group
   - Mode (most common) values
   - p-values for group differences

#### 4. Main Results

**File**: `all_subjects_results.csv`

Complete results for each subject including:
- Model selection: `best_P`, `best_K`, `best_BIC`
- LTI model performance: `lti_R2`, `lti_rho`, `lti_BIC`
- TV model statistics: `tv_R2_mean`, `tv_R2_std`, `tv_rho_mean`, `tv_rho_std`
- Time-varying dynamics: `mean_msd`, `mean_cv`, `n_windows`

#### 5. Group Statistics

**File**: `group_statistics.csv`

Statistical comparisons between AD and HC for all metrics:
- Mean and std for each group
- t-statistics and p-values
- Cohen's d effect sizes
- Significance flags (p < 0.05)

#### 6. Transfer Function Comparisons

**File**: `group_comparison_transfer_functions.png`

Shows:
- Row 1: LTI transfer functions (AD, HC, difference)
- Row 2: TV transfer functions (AD, HC, difference)
- Row 3: Aggregate analyses (frequency response, graph mode response, time-varying dynamics)

#### 7. Mode-Averaged Frequency Responses ⭐ NEW

**File**: `mode_averaged_frequency_responses.png`

This is a KEY figure for thesis showing:
- **Row 1**: LTI frequency response averaged over all graph modes
  - AD vs HC comparison with standard error bars
  - Frequency bands highlighted (Delta, Theta, Alpha, Beta, Gamma)
- **Row 2**: TV frequency response averaged over all graph modes
  - Shows time-varying dynamics in frequency domain
- **Row 3**: 
  - LTI difference plot (AD - HC)
  - TV difference plot (AD - HC)
  - Statistical table with p-values per frequency band

**Interpretation**: Shows which temporal frequencies (Hz) are amplified/suppressed differently between groups, independent of spatial patterns.

#### 8. Individual Subject Frequency Responses

**File**: `individual_frequency_responses.png`

"Spaghetti plots" showing:
- All individual subject traces (thin lines)
- Group mean (thick line)
- Separate panels for AD/HC and LTI/TV
- Shows inter-subject variability

#### 9. Frequency Band Statistics

**File**: `frequency_band_statistics.csv`

Statistical comparisons for standard EEG bands:
- **Delta** (0.5-4 Hz): Slow waves, sleep, pathology
- **Theta** (4-8 Hz): Memory, drowsiness
- **Alpha** (8-13 Hz): Relaxed wakefulness, eyes closed
- **Beta** (13-30 Hz): Active thinking, focus
- **Gamma** (30-40 Hz): Cognitive processing

For each band:
- Mean magnitude (AD vs HC)
- t-statistic, p-value, Cohen's d
- Separate statistics for LTI and TV models

**Use for thesis**: Report specific frequency bands showing group differences

## Thesis Presentation Suggestions

### Model Selection Section

**Table: Model Selection Summary**
```
Group | n  | P (mean±std) | K (mean±std) | Mode P | Mode K | p-value
------|----|--------------|--------------| -------|--------|--------
AD    | XX | X.X ± X.X    | X.X ± X.X    |   X    |   X    |   
HC    | XX | X.X ± X.X    | X.X ± X.X    |   X    |   X    | p=X.XXX
```

**Figure: Model Selection Analysis**
- Use the 6-panel figure directly in thesis
- Caption: "Model selection results showing distribution of selected AR order (P) and graph filter order (K) across AD and HC groups. BIC was used as the selection criterion."

### Results Section

**Reporting model selection results**:
1. "Model selection was performed by testing P ∈ {1,2,3,5,7,10,15,20,30} and K ∈ {1,2,3,4}"
2. "The optimal model was selected based on minimum BIC among stable models (ρ < 0.99)"
3. "In the AD group (n=XX), the selected models had P=X.X±X.X and K=X.X±X.X"
4. "In the HC group (n=XX), the selected models had P=X.X±X.X and K=X.X±X.X"
5. "No significant difference was found in model complexity between groups (P: p=X.XX, K: p=X.XX)"
   OR
   "AD patients showed significantly higher/lower model complexity (P: p=X.XX, d=X.XX)"

### Mode-Averaged Frequency Response Section ⭐ NEW

**Figure Caption**:
"Mode-averaged frequency response |G(ω)| showing transfer function magnitude averaged over all graph modes (λ) for each temporal frequency. (A) LTI model comparison between AD and HC with shaded frequency bands. (B) Time-varying (TV) model comparison. (C-D) Difference plots highlighting frequencies where AD shows higher/lower response magnitude than HC. (E) Statistical comparison table showing p-values and effect sizes for standard EEG frequency bands."

**Reporting frequency band results** (use `frequency_band_statistics.csv`):

Example template:
```
Analysis of mode-averaged frequency responses revealed significant group 
differences in multiple frequency bands. In the LTI model, AD patients 
showed [increased/decreased] transfer function magnitude in the 
[delta/theta/alpha/beta/gamma] band (AD: X.XX±X.XX, HC: X.XX±X.XX, 
p=X.XXX, d=X.XX). Similar patterns were observed in the TV model for 
[band names] (p=X.XXX, d=X.XX).
```

**Interpretation examples**:
- **Higher delta/theta in AD**: "Elevated low-frequency amplification suggests increased slow-wave activity, consistent with cortical slowing in neurodegeneration"
- **Lower alpha in AD**: "Reduced alpha band response indicates disrupted thalamocortical rhythms"
- **Higher beta in AD**: "Increased beta activity may reflect compensatory mechanisms or hyperexcitability"

**Key advantage**: Mode-averaging removes spatial complexity, revealing pure frequency-domain differences independent of which brain regions or connectivity patterns are involved.

### Statistical Comparison

Use `group_statistics.csv` to report:

**Example**:
"Time-varying dynamics (mean coefficient of variation) were significantly higher in AD (CV=X.XX±X.XX) compared to HC (CV=X.XX±X.XX), t(XX)=X.XX, p=X.XXX, Cohen's d=X.XX"

## Running the Analysis

```bash
python lti_tv_group_comparison.py
```

**Requirements**:
- All subject EEG files must exist at specified paths
- Consensus Laplacian matrix must be available
- Dependencies: numpy, pandas, mne, scipy, matplotlib, seaborn, tqdm

**Expected Runtime**: ~1-3 hours (depends on number of subjects and data length)

## Output Directory Structure

```
./group_comparison_lti_tv_analysis/
├── model_selection/
│   ├── sub-30001_model_selection.csv      # Individual model selection tables
│   ├── sub-30002_model_selection.csv
│   ├── ...
│   └── all_subjects_model_selection.csv   # Combined all subjects
│
├── model_selection_summary.csv            # Selected P and K per subject
├── model_selection_analysis.png           # 6-panel model selection visualization
│
├── all_subjects_results.csv               # Main results table (all metrics)
├── group_statistics.csv                   # Statistical tests (overall metrics)
├── frequency_band_statistics.csv          # Statistical tests per frequency band ⭐ NEW
│
├── group_comparison_metrics.png           # Boxplots of all metrics
├── group_comparison_transfer_functions.png  # Full transfer function analysis
├── mode_averaged_frequency_responses.png  # Mode-averaged freq. response ⭐ NEW
└── individual_frequency_responses.png     # Spaghetti plots ⭐ NEW
```

## Key Differences: LTI vs TV

### LTI (Linear Time-Invariant) Model
- Single model fitted to entire time series
- Assumes stationary dynamics
- Lower variance but may miss temporal changes
- Metrics: `lti_R2`, `lti_rho`, `lti_BIC`

### TV (Time-Varying) Model
- Multiple models fitted to overlapping windows
- Captures non-stationary dynamics
- Higher variance reflects temporal evolution
- Metrics: `tv_R2_mean`, `tv_rho_mean`, `mean_cv` (coefficient of variation)
- `mean_msd`: Mean squared difference from LTI baseline

### Interpretation

**High CV and MSD**: System exhibits strong time-varying dynamics
**Low CV and MSD**: System is relatively time-invariant (stable over time)

**Clinical interpretation**:
- If AD shows higher CV/MSD than HC: AD brains have more unstable/variable connectivity
- If groups similar: Time-varying effects not disease-specific

## Contact

For questions about the analysis or outputs, refer to the main script: `lti_tv_group_comparison.py`

## Citation

If using this analysis framework in your thesis, cite:
- The GP-VAR methodology
- The consensus Laplacian approach
- This analysis framework

---

**Note**: All model selection results are automatically saved and formatted for easy inclusion in thesis chapters, supplementary materials, and presentations.
