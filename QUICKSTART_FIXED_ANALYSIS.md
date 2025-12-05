# Quick Start: Running the Fixed Analysis

## 1. Run the Analysis

```bash
python lti_tv_ad_hc_comparison_ALL_SUBJECTS_FIXED.py
```

Expected runtime: **2-4 hours** (processing ~70 subjects)

---

## 2. Monitor Progress

Watch the console output:
```
================================================================================
Processing AD Group
================================================================================

  Analyzing AD: sub-30018
    Loaded: 64 ch, 180.0s
    Model selection...
    Selected: P=8, K=2
    Fitting LTI...
    LTI: R²=0.652, ρ=0.893 ✓
    Fitting TV...
    TV: 15 windows ✓
    ✓ SUCCESS

  Analyzing AD: sub-30026
    Loaded: 64 ch, 180.0s
    Model selection...
    Selected: P=12, K=1
    Fitting LTI...
    REJECT: ρ=0.961 ≥ 0.95
```

### Status Indicators:
- `✓ SUCCESS` - Subject passed all QC
- `REJECT` - Failed quality thresholds (ρ, R², windows)
- `FAILED` - Technical error (file loading, fitting)

---

## 3. Check Results Immediately After Running

### Step 1: Quality Control Report (most important first!)

```bash
cat ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/quality_control_report.txt
```

Look for:
```
PROCESSING SUMMARY:
  AD: 28/35 successful (80.0%)
  HC: 25/31 successful (80.6%)

REJECTION REASONS:

  AD Group:
    Rejected: 5
      LTI_unstable: 3
      Low_R2: 1
      Too_few_stable_windows: 1
    Failed: 2
      EEG_loading_error: 2
```

**Action**: If success rate < 70%, investigate common rejection reasons

---

### Step 2: View QC Summary Plot

```bash
# Open in image viewer
eog ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/quality_control_summary.png

# Or if on remote server, copy to local machine:
scp user@server:ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/quality_control_summary.png .
```

**Check**:
- ✅ Spectral radius (ρ) histogram: All values < 0.95
- ✅ R² histogram: All values > 0.50
- ✅ Processing pie charts: Green (SUCCESS) should dominate

---

### Step 3: Statistical Significance

```bash
# View the statistical analysis plot
eog ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/statistical_significance_analysis.png
```

**Look for**:
- **Panel 1-2**: p-values (peaks above red line = significant)
- **Panel 3-4**: Effect sizes (Cohen's d > 0.5 = medium, > 0.8 = large)
- **Panel 5-6**: FDR-corrected significance (green = significant modes)

**Key question**: Are there FDR-corrected significant differences?

---

### Step 4: Detailed Subject Log

```bash
# View processing log
head -20 ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/subject_processing_log.csv
```

Shows:
```csv
subject_id,group,status,reason,detail_1,detail_2,detail_3
sub-30018,AD,SUCCESS,Completed,R2=0.652,rho=0.893,n_win=15
sub-30026,AD,REJECT,LTI_unstable,rho=0.9610,threshold=0.95,
sub-30011,AD,SUCCESS,Completed,R2=0.701,rho=0.875,n_win=18
sub-10002,HC,REJECT,Low_R2,R2=0.4850,threshold=0.50,
sub-10009,HC,FAILED,EEG_loading_error,OSError,File not found,
```

**Use this to**:
- Identify specific failed subjects
- See exact rejection values
- Debug systematic issues

---

## 4. Detailed Results Analysis

### Model Selection Summary

```bash
# View selected P and K for each subject
cat ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/model_selection_summary.csv
```

**Check for**:
- Boundary warnings (P at 1 or 20, K at 0 or 4)
- Group differences in selected orders

### Group Statistics

```bash
# Overall statistical comparisons
cat ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/group_statistics.csv
```

Shows:
```
metric       AD_mean  AD_std  HC_mean  HC_std  t_statistic  p_value  cohens_d  significant
lti_R2       0.652    0.089   0.678    0.095   -1.234       0.223    -0.283    False
lti_rho      0.892    0.024   0.875    0.031    2.567       0.013     0.601    True
best_P       9.82     3.45    7.56     2.89     2.891       0.006     0.699    True
best_K       1.75     0.84    1.28     0.74     2.456       0.018     0.597    True
n_windows    14.3     2.1     15.2     1.8     -1.876       0.067    -0.452    False
```

**Key findings**:
- Which metrics differ significantly? (p < 0.05)
- What are effect sizes? (d > 0.5 = meaningful)

### Pointwise Statistics

```bash
# Detailed per-mode statistics
head -10 ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/pointwise_statistics.csv
```

Shows which **specific graph modes** have significant AD vs HC differences.

---

## 5. Interpret Results

### A. Quality Control (QC)

**Good QC**:
- ✅ Success rate > 70%
- ✅ Rejections mostly for LTI_unstable or Too_few_windows
- ✅ Very few FAILED (technical errors)

**Poor QC**:
- ❌ Success rate < 60%
- ❌ Many FAILED entries
- ❌ Channel mismatch errors

**If poor QC**: Check file paths, Laplacian compatibility, preprocessing settings

---

### B. Model Selection

**Typical results**:
- P ∈ [5, 15] (AR order)
- K ∈ [0, 3] (graph filter order)
- AD may select higher P than HC (more complex dynamics)

**Red flags**:
- Many subjects at boundaries (P=1 or P=20)
- All K=0 (suggests graph structure not useful)

**Action**: If boundary hit, expand ranges and re-run

---

### C. Statistical Significance

**Strong findings**:
- Multiple FDR-corrected significant modes
- Large effect sizes (|d| > 0.8)
- Consistent patterns across LTI and TV

**Weak findings**:
- Few or no FDR-corrected significant modes
- Small effect sizes (|d| < 0.3)
- Inconsistent between LTI and TV

**Interpretation**:
- FDR-corrected significance = robust findings
- Effect size indicates practical importance
- Consistency suggests real biological difference

---

## 6. Common Issues & Solutions

### Issue 1: Low Success Rate (< 70%)

**Diagnosis**:
```bash
grep REJECT ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/subject_processing_log.csv | cut -d',' -f4 | sort | uniq -c
```

**Solutions**:
- If mostly `LTI_unstable`: Data may be noisy, consider preprocessing
- If mostly `Low_R2`: Models don't fit well, check data quality
- If mostly `Too_few_stable_windows`: Use longer recordings or reduce window size

---

### Issue 2: Channel Mismatch Errors

```bash
grep Channel_mismatch ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/subject_processing_log.csv
```

**Solution**: 
- Check Laplacian size matches EEG channels
- Verify consensus Laplacian path
- Ensure consistent channel count across subjects

---

### Issue 3: Many Boundary Warnings

**Example output**:
```
NOTE: Selected P=20 is at UPPER boundary - consider expanding range
NOTE: Selected K=4 is at UPPER boundary - consider expanding range
```

**Solution**:
Edit line 122-123:
```python
P_RANGE = [1, 2, ..., 25, 30]  # Extend if needed
K_RANGE = [0, 1, 2, 3, 4, 5]   # Extend if needed
```

---

### Issue 4: No Significant Differences

**Possible reasons**:
1. Small sample size (< 20 per group)
2. High inter-subject variability
3. No true group difference
4. Insufficient statistical power

**Check**:
```bash
# How many subjects passed QC?
grep SUCCESS ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/subject_processing_log.csv | wc -l
```

Need **≥15 per group** for adequate power.

---

## 7. Publication Checklist

Before publication, verify:

### Quality Control
- [ ] Success rate documented (report in methods)
- [ ] Rejection reasons summarized (table in supplement)
- [ ] QC metrics reported (mean ρ, R², n_windows)
- [ ] `quality_control_summary.png` in supplement

### Model Selection
- [ ] P and K ranges justified (cite pilot study or prior work)
- [ ] Boundary checks passed (no systematic boundary hits)
- [ ] Group differences in P/K tested and reported

### Statistical Analysis
- [ ] FDR correction applied and reported
- [ ] Effect sizes calculated (Cohen's d)
- [ ] Multiple comparisons addressed
- [ ] `statistical_significance_analysis.png` in main text

### Reproducibility
- [ ] All QC thresholds documented (ρ=0.95, R²=0.50, n_windows=10)
- [ ] Model selection ranges specified
- [ ] Preprocessing steps detailed
- [ ] Code and data availability statement

---

## 8. Next Steps

After initial run:

1. **Review QC** → Adjust thresholds if needed
2. **Check statistics** → Identify significant modes
3. **Biological interpretation** → Map modes to brain networks
4. **Sensitivity analysis** → Test robustness to threshold choices
5. **Visualization** → Create publication figures
6. **Write-up** → Document all findings

---

## 9. Getting Help

If issues persist:

1. **Check log files**:
   - `subject_processing_log.csv` - every subject
   - `quality_control_report.txt` - summary

2. **Verify paths**:
   ```python
   # Check if files exist
   import os
   for path in AD_PATHS[:5]:  # Check first 5
       print(f"{path}: {os.path.exists(path)}")
   ```

3. **Test on subset**:
   ```python
   # Edit script to use only 3 AD and 3 HC for testing
   AD_PATHS = AD_PATHS[:3]
   HC_PATHS = HC_PATHS[:3]
   ```

---

## 10. Expected Timeline

| Task | Time | Output |
|------|------|--------|
| Run analysis | 2-4 hours | All CSV and PNG files |
| Review QC | 10 minutes | Understanding of data quality |
| Interpret statistics | 30 minutes | Key findings identified |
| Create figures | 1 hour | Publication-ready plots |
| Write results | 2-3 hours | Methods and results sections |

**Total**: 1-2 days from running to draft results section

---

## Quick Command Reference

```bash
# Run analysis
python lti_tv_ad_hc_comparison_ALL_SUBJECTS_FIXED.py

# Check QC report
cat ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/quality_control_report.txt

# View QC plot
eog ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/quality_control_summary.png

# View statistical significance
eog ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/statistical_significance_analysis.png

# Count successful subjects
grep SUCCESS ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/subject_processing_log.csv | wc -l

# List rejection reasons
grep REJECT ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/subject_processing_log.csv | cut -d',' -f4 | sort | uniq -c

# View group statistics
cat ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED/group_statistics.csv | column -t -s,
```

---

**Ready to run?**

```bash
python lti_tv_ad_hc_comparison_ALL_SUBJECTS_FIXED.py
```

Good luck! 🚀
