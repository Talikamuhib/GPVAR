# Complete Guide: Output Files and How They Help Your Analysis

## 📁 Output Directory Structure

After running the analysis, you'll get:

```
./group_comparison_lti_tv_analysis/
│
├── 📊 FIGURES (5 PNG files at 300 DPI)
│   ├── mode_averaged_frequency_responses.png       ⭐ MOST IMPORTANT
│   ├── individual_frequency_responses.png
│   ├── model_selection_analysis.png
│   ├── group_comparison_metrics.png
│   └── group_comparison_transfer_functions.png
│
├── 📈 CSV FILES (4 data tables)
│   ├── frequency_band_statistics.csv               ⭐ MOST IMPORTANT
│   ├── all_subjects_results.csv
│   ├── model_selection_summary.csv
│   └── group_statistics.csv
│
└── 📂 model_selection/ (subfolder)
    ├── sub-30001_model_selection.csv
    ├── sub-30002_model_selection.csv
    ├── ...
    └── all_subjects_model_selection.csv
```

---

# 🖼️ IMAGES EXPLAINED (PNG Files)

## Image 1: mode_averaged_frequency_responses.png ⭐⭐⭐

### **What It Shows**

A comprehensive 6-panel figure (A-F) showing frequency-specific differences between AD and HC groups.

### **Panel-by-Panel Breakdown**

#### **Panel A: LTI Frequency Response**
- **X-axis**: Frequency in Hz (0.5 to 40)
- **Y-axis**: Transfer function magnitude |G(ω)|
- **Red line**: AD group average
- **Blue line**: HC group average
- **Shaded ribbons**: ±1 SEM (error bars)
- **Colored backgrounds**: Delta (gray), Theta (light blue), Alpha (light green), Beta (yellow), Gamma (pink)
- **Asterisks below**: Show which bands have significant differences

**How this helps your analysis:**
- ✅ Instantly see which frequencies differ between groups
- ✅ Identify clinical patterns (e.g., AD higher in delta = slowing)
- ✅ Compare to literature (maps directly to EEG bands)
- ✅ Support hypothesis about cortical slowing
- ✅ Main figure for thesis results section

#### **Panel B: TV Frequency Response**
- Same as Panel A but for time-varying model
- Dashed lines instead of solid

**How this helps:**
- ✅ Check if LTI findings hold in time-varying analysis
- ✅ Assess temporal stability of effects
- ✅ Show your analysis is robust across models

#### **Panel C: LTI Difference (AD - HC)**
- **Black line**: Actual difference
- **Red shading**: Where AD > HC
- **Blue shading**: Where HC > AD
- **Zero line**: No difference

**How this helps:**
- ✅ Visualize magnitude of differences
- ✅ Identify specific frequency ranges affected
- ✅ See continuous spectrum (not just bands)
- ✅ Highlight peaks and valleys for discussion

#### **Panel D: TV Difference (AD - HC)**
- Same as Panel C for time-varying model

**How this helps:**
- ✅ Compare LTI vs TV differences
- ✅ Show consistency or variability

#### **Panel E: Effect Size Bar Chart**
- **Bars**: Cohen's d for each frequency band
- **Positive**: AD > HC
- **Negative**: HC > AD
- **Reference lines**: Medium effect (±0.5), Large effect (±0.8)
- **Asterisks**: Significant findings

**How this helps:**
- ✅ Quick assessment of clinical importance
- ✅ Not just statistical (p-value) but practical significance
- ✅ Compare effect sizes across bands
- ✅ Identify strongest effects for discussion
- ✅ Report in thesis: "Large effect (d=0.87) in delta band"

#### **Panel F: Statistical Summary Table**
- **Rows**: Each frequency band × model type (10 rows total)
- **Columns**: Band, Frequency range, Model, AD mean±SD, HC mean±SD, Difference, p-value, Cohen's d, Significance
- **Color-coded**: Pink = AD elevated, Blue = HC elevated

**How this helps:**
- ✅ All statistics in one place
- ✅ Ready to copy into thesis text
- ✅ Easy comparison LTI vs TV
- ✅ Complete reporting (means, SDs, tests, effect sizes)

### **Overall Use of This Figure**

**For Thesis:**
- Main results figure (Figure 1)
- Reference in every results paragraph
- Shows your key findings at a glance

**For Analysis:**
- Identify which bands are significantly different
- Determine direction of effects (AD>HC or HC>AD)
- Assess clinical relevance via effect sizes
- Compare LTI and TV models

**For Presentations:**
- Single slide with all key results
- Self-explanatory to audience
- Professional quality

---

## Image 2: individual_frequency_responses.png

### **What It Shows**

4-panel "spaghetti plot" showing every individual subject's frequency response.

### **Panel Breakdown**

- **(A)** AD subjects, LTI model: Each thin red line = one patient
- **(B)** HC subjects, LTI model: Each thin blue line = one control
- **(C)** AD subjects, TV model
- **(D)** HC subjects, TV model
- **Thick line in each**: Group mean (bold, with white outline)

**What each element means:**
- **Tight bundle of lines**: Low variability, consistent pattern
- **Wide spread**: High variability, heterogeneous
- **Outlier lines**: Individual subjects very different from group

### **How This Helps Your Analysis**

#### **1. Data Quality Assessment**
```
Question: Are my results driven by outliers?
Answer: If most lines cluster around mean → NO, results are robust
         If lines are scattered → YES, questionable findings
```

#### **2. Justify Group Means**
```
Question: Can I trust the group average?
Answer: If individual traces follow mean shape → YES
         If individuals have different patterns → CAUTION
```

#### **3. Heterogeneity Analysis**
```
Question: Is AD group more variable than HC?
Answer: Compare spread of red lines (Panel A) vs blue lines (Panel B)
         Wider spread in AD → "Increased heterogeneity in AD"
```

#### **4. Subgroup Identification**
```
Question: Are there AD subtypes?
Answer: Look for clusters of lines
         Example: Some AD lines high in delta, others not → subtypes
```

### **Use in Thesis**

**Where**: Supplementary material or Methods section (Data Quality)

**Text Example**:
```
"Individual subject analysis (Supplementary Figure X) confirmed that group 
differences were not driven by outliers. While inter-subject variability 
was present, individual frequency responses clustered around their 
respective group means, with no extreme outliers observed (Panels A-D)."
```

**What It Proves**:
- Your data is good quality
- Group means are representative
- No cherry-picking of subjects
- Results are generalizable

---

## Image 3: model_selection_analysis.png

### **What It Shows**

6-panel figure explaining how you chose model parameters P (AR order) and K (graph filter order).

### **Panel Breakdown**

#### **Panels A-B: Histograms of Selected P and K**
- Show how often each value was selected
- Overlaid for AD (red) vs HC (blue)

**Example Interpretation**:
```
If P histogram peaks at 10 for both groups:
→ "Most subjects required AR order of 10"

If AD peaks at 15, HC at 7:
→ "AD patients required longer temporal history"
```

#### **Panel C: P vs K Scatter Plot**
- Each dot = one subject
- X-axis: Selected P
- Y-axis: Selected K

**What to look for**:
- Diagonal pattern → P and K correlated
- Random scatter → Independent selection
- Clusters → Distinct subgroups

#### **Panels D-E: Boxplots Comparing AD vs HC**
- Panel D: P values (AD vs HC)
- Panel E: K values (AD vs HC)
- P-value shown if groups differ

**Example Interpretation**:
```
If p < 0.05:
→ "AD required significantly higher AR order (p=0.03)"

If p > 0.05:
→ "Model complexity was comparable between groups (p=0.34)"
```

#### **Panel F: Summary Statistics Table**
- Mode (most common value)
- Mean ± SD for each group
- Statistical tests

### **How This Helps Your Analysis**

#### **1. Justify Model Choice**
```
Reviewer Question: "Why did you use P=10, K=3?"
Your Answer: "Model selection via BIC (Figure X, Panels A-B) showed 
              majority of subjects were best fit with P=7-10 and K=2-3."
```

#### **2. Compare Group Complexity**
```
Research Question: Do AD brains require more complex models?

Analysis: Look at Panel D
- If AD mean > HC mean (and p<0.05) → "AD requires longer history"
- If similar → "Comparable model complexity"

Interpretation:
- Higher P in AD → AD dynamics have longer temporal dependencies
- Higher K in AD → AD requires more complex spatial filters
- Similar → Structure intact, dynamics altered
```

#### **3. Check for Consistency**
```
Concern: Did all subjects converge to same model?
Check: Width of histogram (Panel A-B)
- Narrow peak → Consistent across subjects
- Flat distribution → Variable, less reliable
```

### **Use in Thesis**

**Where**: Methods section (Model Specification subsection)

**Text Example**:
```
"Model order selection was performed using BIC across P ∈ {1,2,3,5,7,10,
15,20,30} and K ∈ {1,2,3,4}. Selected models showed P=8.2±2.1 for AD and 
P=7.8±1.9 for HC (p=0.34, Figure X, Panel D), indicating comparable 
temporal complexity. Graph filter order was K=2.4±0.6 for AD and K=2.2±0.5 
for HC (p=0.18, Panel E), suggesting similar spatial complexity requirements. 
The most frequently selected model configuration was P=10, K=2 in both 
groups (Panels A-B)."
```

**What It Proves**:
- Systematic, data-driven model selection
- No arbitrary parameter choice
- Transparent methodology
- Comparable model complexity (if applicable)

---

## Image 4: group_comparison_metrics.png

### **What It Shows**

8 boxplots comparing AD vs HC on various model performance metrics.

### **Metrics Explained**

#### **1. lti_R2 (LTI Model Fit)**
- **Range**: 0 to 1
- **Higher = Better fit**
- **Interpretation**:
  - If AD < HC: "AD brains less predictable"
  - If AD > HC: "AD dynamics more regular/simpler"

#### **2. lti_rho (LTI Stability)**
- **Range**: 0 to 1
- **Closer to 1 = Less stable**
- **Interpretation**:
  - If AD > HC: "AD networks closer to instability"
  - Critical threshold is 1.0 (unstable)

#### **3. lti_BIC (Model Selection Criterion)**
- **Lower = Better balance of fit vs complexity**
- Usually not clinically interpreted

#### **4. tv_R2_mean (TV Model Fit)**
- Average R² across time windows
- Compare to lti_R2

#### **5. tv_rho_mean (TV Stability)**
- Average stability across windows
- Compare to lti_rho

#### **6. mean_msd (Mean Squared Difference)**
- How much TV differs from LTI
- **Higher = More time-varying**
- **Key metric for temporal dynamics**
- **Interpretation**:
  - If AD > HC: "AD shows more temporal variability"

#### **7. mean_cv (Coefficient of Variation)**
- Normalized measure of variability
- **Higher = Less stable over time**
- **Interpretation**:
  - If AD > HC: "AD dynamics less stable"
  - This is a **biomarker** candidate

#### **8. n_windows**
- Number of time windows analyzed
- Should be similar between groups

### **How This Helps Your Analysis**

#### **1. Overall Group Differences**
```
Question: Do AD and HC differ in general?
Check: How many boxplots show p < 0.05
Answer: 
- Many significant → Strong group differences
- Few significant → Subtle differences
- None → No overall difference (but may have frequency-specific)
```

#### **2. Identify Key Findings**
```
Look for:
- Largest p-values (most significant)
- Largest effect sizes (boxplots far apart)
- Consistent direction (AD always higher or lower)

Example:
If mean_cv shows p<0.001 with AD > HC:
→ "Primary finding: AD exhibits increased temporal instability"
```

#### **3. Model Validation**
```
Check lti_R2 and tv_R2:
- If both > 0.5 → Models fit well
- If < 0.3 → Poor fit, results questionable
- If TV > LTI → Time-varying model captures more dynamics
```

#### **4. Stability Analysis**
```
Check lti_rho and tv_rho:
- All < 0.99 → Models are stable (required)
- If AD ≈ 0.95, HC ≈ 0.85 → "AD closer to instability"
```

### **Use in Thesis**

**Where**: Results section (Overall Group Comparison subsection)

**Text Example**:
```
"Group comparison of model performance metrics (Figure X) revealed 
significantly higher temporal variability in AD patients. Coefficient of 
variation (CV) was elevated in AD (0.25±0.08) compared to HC (0.12±0.04), 
t(64)=7.23, p<0.001, d=0.92, indicating reduced temporal stability of 
network dynamics. Similarly, mean squared difference from the LTI baseline 
was higher in AD (0.15±0.05 vs 0.05±0.02, p<0.001), confirming increased 
time-varying behavior. Model fit quality (R²) was comparable between groups 
(AD: 0.68±0.12, HC: 0.71±0.10, p=0.18), indicating that group differences 
reflect genuine dynamics rather than poor model performance."
```

---

## Image 5: group_comparison_transfer_functions.png

### **What It Shows**

3×3 grid of heatmaps showing full 2D transfer functions G(ω,λ).

### **Understanding the Heatmaps**

**Axes**:
- X-axis: λ (lambda) = Graph frequency (eigenvalue of Laplacian)
- Y-axis: ω (omega) = Temporal frequency (Hz)
- Color: Magnitude |G(ω,λ)|

**What each position means**:
- **Bottom-left** (low ω, low λ): Global slow oscillations
- **Top-left** (high ω, low λ): Global fast oscillations  
- **Bottom-right** (low ω, high λ): Localized slow oscillations
- **Top-right** (high ω, high λ): Localized fast oscillations

### **Panel Layout**

**Row 1**: LTI Model
- Panel A: AD transfer function
- Panel B: HC transfer function
- Panel C: Difference (AD - HC)

**Row 2**: TV Model
- Panel D: AD transfer function
- Panel E: HC transfer function
- Panel F: Difference (AD - HC)

**Row 3**: Aggregate Analyses
- Panel G: Frequency response (averaged over λ)
- Panel H: Graph mode response (averaged over ω)
- Panel I: MSD scatter plot

### **How This Helps Your Analysis**

#### **1. Spatial-Temporal Coupling**
```
Question: Are frequency differences global or localized?
Check: Difference maps (Panels C, F)
- Red in left region (low λ) → Global effect
- Red in right region (high λ) → Localized effect
- Red spanning both → Both global and local
```

#### **2. Identify Resonances**
```
Look for: Bright spots in heatmaps (Panels A, B, D, E)
These are: Natural frequencies of the network
Compare: AD vs HC resonance patterns
Different: → "AD networks have altered resonance structure"
```

#### **3. Validate Mode-Averaging**
```
Check: Panel G should match main figure Panel A
If similar: → Mode-averaging preserves key features
If different: → Spatial structure important (discuss limitations)
```

### **Use in Thesis**

**Where**: 
- Technical results section (if emphasizing methodology)
- Supplementary material (if focusing on clinical findings)
- Appendix (for complete analysis)

**Text Example**:
```
"Full spectral-spatial analysis (Supplementary Figure X) revealed that 
group differences were predominantly in low graph frequency modes (λ < 0.5), 
indicating global network effects rather than localized alterations. The 
difference map (Panel C) showed elevated magnitude in the low-ω, low-λ 
quadrant for AD patients, corresponding to global slow-wave amplification."
```

---

# 📊 CSV FILES EXPLAINED

## CSV 1: frequency_band_statistics.csv ⭐⭐⭐

### **What It Contains**

Statistical results for 5 frequency bands × 2 models = 10 rows

### **Columns**:
1. **band**: delta, theta, alpha, beta, gamma
2. **freq_range**: Frequency range in Hz
3. **model_type**: LTI or TV
4. **AD_mean**: Average |G| in AD group
5. **AD_std**: Standard deviation in AD
6. **HC_mean**: Average |G| in HC group
7. **HC_std**: Standard deviation in HC
8. **t_statistic**: T-test result
9. **p_value**: Statistical significance
10. **cohens_d**: Effect size
11. **significant**: True/False (p < 0.05)

### **Example Row**:
```csv
band,freq_range,model_type,AD_mean,AD_std,HC_mean,HC_std,t_statistic,p_value,cohens_d,significant
alpha,8.0-13.0 Hz,LTI,1.65,0.23,2.34,0.28,-8.45,0.0001,-1.12,True
```

**Interpretation**: 
- Alpha band in LTI model
- AD mean is 1.65, HC mean is 2.34
- HC > AD (negative Cohen's d)
- Highly significant (p=0.0001)
- Large effect (|d| > 0.8)

### **How This Helps Your Analysis**

#### **1. Thesis Table Creation**
```
Use: Open in Excel/LibreOffice
Format: Sort by p_value (most significant first)
Copy: Directly into thesis table
```

**LaTeX Example**:
```latex
\begin{table}[h]
\caption{Frequency band analysis: AD vs HC}
\begin{tabular}{llrrrr}
\toprule
Band & Model & AD & HC & p-value & Cohen's d \\
\midrule
Alpha & LTI & 1.65±0.23 & 2.34±0.28 & <0.001*** & -1.12 \\
Delta & LTI & 2.45±0.21 & 1.89±0.18 & <0.001*** & +0.87 \\
...
\bottomrule
\end{tabular}
\end{table}
```

#### **2. Quick Significance Check**
```python
import pandas as pd
df = pd.read_csv('frequency_band_statistics.csv')

# Find significant bands
sig_bands = df[df['significant'] == True]
print(sig_bands[['band', 'model_type', 'p_value', 'cohens_d']])

# Find large effects
large_effects = df[abs(df['cohens_d']) > 0.8]
```

#### **3. Compare LTI vs TV**
```python
# For each band, compare LTI and TV
for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    lti = df[(df['band']==band) & (df['model_type']=='LTI')]
    tv = df[(df['band']==band) & (df['model_type']=='TV')]
    
    print(f"{band}: LTI d={lti['cohens_d'].values[0]:.2f}, "
          f"TV d={tv['cohens_d'].values[0]:.2f}")
```

#### **4. Report in Thesis**
```
"Analysis of standard EEG frequency bands (Table X) revealed significant 
group differences in three bands. Alpha band showed the largest effect 
(d=-1.12, p<0.001) with HC exhibiting higher transfer function magnitude 
(2.34±0.28) than AD (1.65±0.23). Delta band showed elevated magnitude 
in AD (2.45±0.21 vs 1.89±0.18, d=+0.87, p<0.001). Time-varying analysis 
(TV model) confirmed these patterns with similar effect sizes."
```

### **Clinical Interpretation Guide**

| Band | AD > HC | Interpretation |
|------|---------|----------------|
| Delta | ✓ | Cortical slowing, pathological |
| Theta | ✓ | Memory dysfunction, slowing |
| Alpha | ✗ (HC > AD) | Alpha suppression in AD, thalamocortical disruption |
| Beta | ? | Variable, may indicate compensation |
| Gamma | ✗ (HC > AD) | Reduced fast oscillations, cognitive impairment |

---

## CSV 2: all_subjects_results.csv

### **What It Contains**

One row per subject with ALL computed metrics.

### **Columns** (18 total):
1. **subject_id**: e.g., "sub-30001"
2. **group**: "AD" or "HC"
3. **n_channels**: Number of EEG channels
4. **duration**: Recording length in seconds
5. **best_P**: Selected AR order
6. **best_K**: Selected graph filter order
7. **best_BIC**: BIC of selected model
8. **lti_R2**: LTI model goodness of fit
9. **lti_rho**: LTI spectral radius
10. **lti_BIC**: LTI model BIC
11. **tv_R2_mean**: Mean TV model R²
12. **tv_R2_std**: Std of TV model R²
13. **tv_rho_mean**: Mean TV spectral radius
14. **tv_rho_std**: Std of TV spectral radius
15. **n_windows**: Number of time windows
16. **mean_msd**: Mean squared difference (TV vs LTI)
17. **mean_cv**: Coefficient of variation

### **Example Row**:
```csv
subject_id,group,n_channels,duration,best_P,best_K,best_BIC,lti_R2,lti_rho,lti_BIC,tv_R2_mean,tv_R2_std,tv_rho_mean,tv_rho_std,n_windows,mean_msd,mean_cv
sub-30001,AD,64,600.0,10,3,-45231.2,0.72,0.89,-45231.2,0.68,0.06,0.87,0.04,28,0.18,0.29
```

### **How This Helps Your Analysis**

#### **1. Individual Subject Lookup**
```python
# Find specific subject
df = pd.read_csv('all_subjects_results.csv')
subject = df[df['subject_id'] == 'sub-30001']
print(subject)

# Check if subject is outlier
ad_group = df[df['group'] == 'AD']
subject_cv = subject['mean_cv'].values[0]
group_cv_mean = ad_group['mean_cv'].mean()
group_cv_std = ad_group['mean_cv'].std()

if abs(subject_cv - group_cv_mean) > 2 * group_cv_std:
    print("Subject is an outlier!")
```

#### **2. Create Custom Plots**
```python
import matplotlib.pyplot as plt

ad = df[df['group'] == 'AD']
hc = df[df['group'] == 'HC']

plt.scatter(ad['lti_R2'], ad['mean_cv'], c='red', label='AD')
plt.scatter(hc['lti_R2'], hc['mean_cv'], c='blue', label='HC')
plt.xlabel('Model Fit (R²)')
plt.ylabel('Temporal Variability (CV)')
plt.legend()
plt.show()
```

#### **3. Correlation Analysis**
```python
# Within AD group, are higher P values associated with worse fit?
ad = df[df['group'] == 'AD']
correlation = ad['best_P'].corr(ad['lti_R2'])
print(f"Correlation between P and R²: {correlation:.3f}")
```

#### **4. Supplementary Table**
```
Use: Sort by any metric
Create: Tables showing "Top 10 most variable AD patients"
Or: "Subjects with unusual model selection"
```

### **Use in Thesis**

**Where**: Supplementary materials

**Example**:
```
"Complete subject-level results are provided in Supplementary Table X. 
Across all subjects (n=66), selected AR orders ranged from P=3 to P=30 
(median=10), with graph filter orders of K=1 to K=4 (median=2). Model 
fit quality (R²) ranged from 0.45 to 0.89 (mean=0.69±0.11), indicating 
generally good model performance."
```

---

## CSV 3: model_selection_summary.csv

### **What It Contains**

Selected model parameters for each subject.

### **Columns**:
1. **subject_id**: Subject identifier
2. **group**: AD or HC
3. **selected_P**: Chosen AR order
4. **selected_K**: Chosen graph filter order
5. **selected_BIC**: BIC value of selected model

### **Example**:
```csv
subject_id,group,selected_P,selected_K,selected_BIC
sub-30001,AD,10,3,-45231.2
sub-30002,AD,7,2,-38452.7
sub-10001,HC,10,2,-42103.5
```

### **How This Helps Your Analysis**

#### **1. Model Selection Statistics**
```python
df = pd.read_csv('model_selection_summary.csv')

# Summary by group
summary = df.groupby('group')[['selected_P', 'selected_K']].agg(['mean', 'std', 'min', 'max'])
print(summary)

# Most common model
from scipy import stats
ad_P_mode = stats.mode(df[df['group']=='AD']['selected_P'])[0]
print(f"Most common P in AD: {ad_P_mode}")
```

#### **2. Statistical Comparison**
```python
from scipy.stats import ttest_ind

ad_P = df[df['group']=='AD']['selected_P']
hc_P = df[df['group']=='HC']['selected_P']

t_stat, p_val = ttest_ind(ad_P, hc_P)
print(f"P comparison: t={t_stat:.2f}, p={p_val:.4f}")
```

#### **3. Create Methods Table**
```
Band | n | P (mean±SD) | K (mean±SD) | Mode P | Mode K
-----|---|-------------|-------------|--------|-------
AD   |35 | 9.2 ± 3.1   | 2.4 ± 0.7   | 10     | 2
HC   |31 | 8.7 ± 2.8   | 2.2 ± 0.6   | 10     | 2
```

### **Use in Thesis**

**Where**: Methods section (Model Selection subsection)

**Example**:
```
"Model selection via BIC resulted in AR orders of P=9.2±3.1 for AD and 
P=8.7±2.8 for HC (Table X), with no significant difference between groups 
(p=0.42). The most frequently selected configuration was P=10, K=2 for 
both groups, indicating consistent model complexity requirements."
```

---

## CSV 4: group_statistics.csv

### **What It Contains**

Statistical comparison results for overall metrics (not frequency-specific).

### **Columns**:
1. **metric**: Metric name (e.g., "lti_R2", "mean_cv")
2. **AD_mean**: Mean value in AD group
3. **AD_std**: Standard deviation in AD
4. **AD_n**: Sample size of AD group
5. **HC_mean**: Mean value in HC group
6. **HC_std**: Standard deviation in HC
7. **HC_n**: Sample size of HC group
8. **t_statistic**: T-test statistic
9. **p_value**: Statistical significance
10. **cohens_d**: Effect size
11. **significant**: True/False (p < 0.05)

### **Example Row**:
```csv
metric,AD_mean,AD_std,AD_n,HC_mean,HC_std,HC_n,t_statistic,p_value,cohens_d,significant
mean_cv,0.254,0.082,35,0.118,0.043,31,7.23,0.0001,0.92,True
```

### **How This Helps Your Analysis**

#### **1. Abstract/Summary Statistics**
```
Use: First paragraph of results
Copy: Key numbers directly

"AD patients exhibited significantly higher temporal variability 
(CV=0.254±0.082) compared to HC (0.118±0.043), p<0.001, d=0.92."
```

#### **2. Prioritize Findings**
```python
df = pd.read_csv('group_statistics.csv')

# Sort by effect size (largest first)
df_sorted = df.sort_values('cohens_d', key=abs, ascending=False)
print("Largest effects:")
print(df_sorted[['metric', 'cohens_d', 'p_value']].head(5))

# This tells you what to emphasize in thesis
```

#### **3. Results Table**
```
Metric          | AD           | HC           | p-value | Effect Size
----------------|--------------|--------------|---------|------------
Mean CV         | 0.25 ± 0.08  | 0.12 ± 0.04  | <0.001  | d = 0.92
LTI R²          | 0.68 ± 0.12  | 0.71 ± 0.10  | 0.18    | d = -0.27
TV R² (mean)    | 0.65 ± 0.14  | 0.69 ± 0.11  | 0.15    | d = -0.31
Mean MSD        | 0.15 ± 0.05  | 0.05 ± 0.02  | <0.001  | d = 1.15
```

### **Use in Thesis**

**Where**: Results section introduction

**Example**:
```
"Group comparisons revealed significant differences in temporal dynamics 
between AD and HC (Table X). AD patients showed elevated coefficient of 
variation (0.254±0.082 vs 0.118±0.043, t(64)=7.23, p<0.001, d=0.92) and 
mean squared difference from LTI baseline (0.15±0.05 vs 0.05±0.02, 
p<0.001, d=1.15), indicating increased time-varying behavior. Model fit 
quality (R²) was comparable between groups (p=0.18), suggesting that 
differences reflect genuine dynamics rather than poor model performance."
```

---

## CSV 5: Individual Model Selection Files

### **Location**: `model_selection/sub-XXXXX_model_selection.csv`

### **What Each Contains**

Complete model selection landscape for ONE subject.

### **Columns**:
1. **P**: Tested AR order
2. **K**: Tested graph filter order  
3. **BIC**: Bayesian Information Criterion
4. **R2**: Goodness of fit
5. **MSE**: Mean squared error
6. **rho**: Spectral radius (stability)
7. **stable**: True/False
8. **success**: True/False (fitting succeeded)

### **Example**:
```csv
P,K,BIC,R2,MSE,rho,stable,success
1,1,-12345.2,0.35,0.052,0.75,True,True
1,2,-13102.5,0.42,0.048,0.78,True,True
3,1,-14523.1,0.58,0.032,0.82,True,True
3,2,-15234.7,0.64,0.028,0.84,True,True  ← Selected (best BIC among stable)
```

### **How This Helps Your Analysis**

#### **1. Verify Model Selection**
```python
# Load one subject
df = pd.read_csv('model_selection/sub-30001_model_selection.csv')

# Show all stable models sorted by BIC
stable = df[df['stable'] == True]
sorted_models = stable.sort_values('BIC')
print("Top 5 models:")
print(sorted_models[['P', 'K', 'BIC', 'R2']].head(5))

# Best model should be row 1
```

#### **2. Check Alternative Models**
```python
# How much worse are other models?
best_bic = sorted_models.iloc[0]['BIC']
delta_bic = sorted_models['BIC'] - best_bic

print(f"Second best model is {delta_bic.iloc[1]:.1f} BIC units worse")
# If Δ BIC < 2 → Essentially equivalent
# If Δ BIC > 10 → Clearly worse
```

#### **3. Understand Why Model Chosen**
```python
# Compare P=5, K=2 vs P=10, K=2
model_a = df[(df['P']==5) & (df['K']==2)]
model_b = df[(df['P']==10) & (df['K']==2)]

print(f"P=5: BIC={model_a['BIC'].values[0]:.1f}, R²={model_a['R2'].values[0]:.3f}")
print(f"P=10: BIC={model_b['BIC'].values[0]:.1f}, R²={model_b['R2'].values[0]:.3f}")

# If P=10 has better R² but similar BIC → Justified increase in complexity
```

#### **4. Stability Analysis**
```python
# How many models were unstable?
n_unstable = (df['stable'] == False).sum()
n_failed = (df['success'] == False).sum()

print(f"Unstable: {n_unstable}/{len(df)}")
print(f"Failed: {n_failed}/{len(df)}")

# High failure rate might indicate data quality issues
```

### **Use in Thesis**

**Where**: Methods section or Supplementary materials

**Example**:
```
"For each subject, we evaluated 36 model configurations (9 values of P × 
4 values of K). Representative example (subject sub-30001, Supplementary 
Figure X) shows BIC ranging from -12,345 (P=1, K=1) to -15,234 (P=3, K=2, 
selected model), with R² ranging from 0.35 to 0.64. The selected model 
represented the best trade-off between fit quality and model complexity 
among stable configurations."
```

---

## CSV 6: all_subjects_model_selection.csv

### **Location**: `model_selection/all_subjects_model_selection.csv`

### **What It Contains**

COMBINED model selection results from all subjects.

### **Structure**:
- Same columns as individual files
- Plus: **subject_id** and **group** columns
- ~2,376 rows (66 subjects × 36 model configurations)

### **How This Helps Your Analysis**

#### **1. Group-Level Model Selection Patterns**
```python
df = pd.read_csv('model_selection/all_subjects_model_selection.csv')

# Filter to stable, successful models only
valid = df[(df['stable']==True) & (df['success']==True)]

# Average BIC for each P-K combination
pivot = valid.pivot_table(values='BIC', index='P', columns='K', aggfunc='mean')
print(pivot)

# This shows which P-K combinations generally work best
```

#### **2. Stability Analysis by Model**
```python
# What % of subjects had stable model for P=30, K=4?
p30k4 = df[(df['P']==30) & (df['K']==4)]
stability_rate = p30k4['stable'].mean()
print(f"P=30, K=4 was stable in {stability_rate*100:.1f}% of subjects")

# Low rate → Don't include this combination next time
```

#### **3. Group Comparison of Model Space**
```python
# Do AD patients have more unstable models overall?
ad = df[df['group']=='AD']
hc = df[df['group']=='HC']

ad_stable_rate = ad['stable'].mean()
hc_stable_rate = hc['stable'].mean()

print(f"AD: {ad_stable_rate*100:.1f}% stable")
print(f"HC: {hc_stable_rate*100:.1f}% stable")
```

#### **4. Identify Problematic Configurations**
```python
# Which P-K combinations failed most often?
failure_rate = df.groupby(['P', 'K'])['success'].apply(lambda x: 1 - x.mean())
worst = failure_rate.sort_values(ascending=False).head(5)
print("Most problematic configurations:")
print(worst)
```

### **Use in Thesis**

**Where**: Methods section or Technical appendix

**Example**:
```
"Across all 66 subjects, model configurations with P ≤ 10 achieved stability 
in >95% of cases, while P=30 configurations were stable in only 67% of 
subjects. This justified our conservative approach of limiting the model 
space to tested values. Complete model selection results are available in 
Supplementary Data File 1."
```

---

# 🎯 SUMMARY: How Files Work Together

## For Writing Results Section

**Use This Combination**:

1. **Figure**: `mode_averaged_frequency_responses.png` (visual)
2. **CSV**: `frequency_band_statistics.csv` (exact numbers)

**Workflow**:
```
1. Look at Figure Panel A → See alpha is different
2. Open frequency_band_statistics.csv
3. Find alpha row: p=0.0001, d=-1.12
4. Write: "Alpha band showed HC > AD (2.34±0.28 vs 1.65±0.23, 
   p<0.001, d=-1.12)"
5. Reference figure: "(Figure 3A)"
```

---

## For Methods Section

**Use This Combination**:

1. **Figure**: `model_selection_analysis.png` (justify choice)
2. **CSV**: `model_selection_summary.csv` (report statistics)

**Workflow**:
```
1. Open model_selection_summary.csv
2. Calculate: Mean P, mean K, mode, etc.
3. Reference figure: "Model selection (Figure 2) revealed..."
4. Report numbers from CSV
```

---

## For Supplementary Materials

**Provide These**:

1. **`all_subjects_results.csv`** - Complete subject data
2. **`model_selection/all_subjects_model_selection.csv`** - Full model selection
3. **`individual_frequency_responses.png`** - Data quality

**Purpose**: Transparency and reproducibility

---

## For Committee/Supervisor Meeting

**Show These First**:

1. **`mode_averaged_frequency_responses.png`** 
   - "Here's what we found"
2. **`frequency_band_statistics.csv`** (opened in Excel)
   - "Here are the exact statistics"
3. **`individual_frequency_responses.png`**
   - "Here's proof the data is good"

---

## For Revisions/Questions

**Reviewer asks**: "What about subject X?"

**Answer using**:
- `all_subjects_results.csv` → Find subject row
- `model_selection/sub-XXXXX_model_selection.csv` → Show their specific results

**Reviewer asks**: "Did you try different model orders?"

**Answer using**:
- `model_selection_analysis.png` → "Yes, we tested 9 P values and 4 K values"
- `model_selection_summary.csv` → Show distribution of selections

---

# 📋 Quick Reference Checklist

## ✅ For Thesis Results Chapter:

- [ ] Insert `mode_averaged_frequency_responses.png` as main figure
- [ ] Create table from `frequency_band_statistics.csv`
- [ ] Report key numbers from `group_statistics.csv`
- [ ] Reference `individual_frequency_responses.png` for data quality

## ✅ For Thesis Methods Chapter:

- [ ] Insert `model_selection_analysis.png`
- [ ] Report statistics from `model_selection_summary.csv`
- [ ] Describe model selection procedure

## ✅ For Thesis Supplementary Materials:

- [ ] Include `all_subjects_results.csv` as Supplementary Table 1
- [ ] Include `all_subjects_model_selection.csv` as Supplementary Data 1
- [ ] Include `group_comparison_transfer_functions.png` as Supplementary Figure

## ✅ For Defense Presentation:

- [ ] Slide 1: `mode_averaged_frequency_responses.png` (main results)
- [ ] Slide 2: `group_comparison_metrics.png` (overall differences)
- [ ] Backup: `individual_frequency_responses.png` (if asked about data quality)

---

**You now have a complete understanding of all outputs and how to use them for your thesis!** 🎓
