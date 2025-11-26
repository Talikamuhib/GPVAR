# How to Interpret Your Thesis Figures

This guide shows you exactly what to look for in each figure and how to interpret the results for your thesis.

---

## 🎯 Figure 1: Mode-Averaged Frequency Responses (THE KEY FIGURE)

### Panel A: LTI Frequency Response

**What you're looking at:**
- X-axis: Temporal frequency in Hz (0.5 to 40 Hz)
- Y-axis: Transfer function magnitude |G(ω)|
- Red line: AD group mean
- Blue line: HC group mean
- Shaded ribbons: ±1 SEM (standard error of mean)
- Colored backgrounds: EEG frequency bands
- Asterisks below: Significant differences

**How to interpret:**

1. **Higher magnitude = Stronger amplification**
   - If AD line is above HC: AD brains amplify this frequency more
   - If HC line is above AD: HC brains amplify this frequency more

2. **Look for crossovers**
   - Where lines cross = frequency where groups are similar
   - Large separations = strong group differences

3. **Check the frequency bands**
   - **Delta (0.5-4 Hz)**: If AD > HC → "Elevated slow-wave amplification"
   - **Theta (4-8 Hz)**: If AD > HC → "Increased theta activity"
   - **Alpha (8-13 Hz)**: If HC > AD → "Preserved thalamocortical rhythms in HC"
   - **Beta (13-30 Hz)**: Interpretations vary
   - **Gamma (30-40 Hz)**: If HC > AD → "Reduced fast oscillations in AD"

4. **Significance markers (asterisks below)**
   - *** = p < 0.001 (very strong evidence)
   - ** = p < 0.01 (strong evidence)
   - * = p < 0.05 (significant)
   - Look for which bands have asterisks

**Example interpretation:**

```
"The LTI model (Panel A) revealed significantly elevated transfer function 
magnitude in the delta band for AD patients (asterisk below), indicating 
increased amplification of slow-wave activity. This is consistent with 
cortical slowing, a hallmark of neurodegenerative disease. Conversely, 
healthy controls showed higher alpha band magnitude, suggesting preserved 
thalamocortical oscillations."
```

---

### Panel B: TV Frequency Response

**What's different from Panel A:**
- Dashed lines instead of solid
- Shows time-varying model results
- Should be similar to Panel A if dynamics are stable

**How to interpret:**

Compare Panel B to Panel A:

1. **If they look similar:**
   - "LTI and TV models show consistent patterns (Panels A-B)"
   - System is relatively stable over time
   - Group differences are persistent

2. **If Panel B is "smoother" or "flatter":**
   - "TV model shows reduced frequency selectivity"
   - Temporal averaging may obscure brief events
   - Consider windowing effects

3. **If Panel B has larger error bands:**
   - "Increased variability in TV model suggests time-varying dynamics"
   - System changes over time
   - AD may show more variability

**Example interpretation:**

```
"Time-varying analysis (Panel B) revealed qualitatively similar patterns 
to the LTI model, with [slightly larger/comparable] error bands in AD 
patients, suggesting [increased temporal variability/stable dynamics]."
```

---

### Panel C & D: Difference Plots

**What you're looking at:**
- X-axis: Frequency
- Y-axis: Difference in |G| (AD minus HC)
- Black line: The actual difference
- Red fill above zero: AD > HC
- Blue fill below zero: HC > AD
- Zero line: No difference

**How to interpret:**

1. **Peak above zero in delta/theta:**
   - "AD patients exhibit excess slow-wave amplification"
   - Pathological slowing

2. **Dip below zero in alpha:**
   - "Healthy controls show stronger alpha response"
   - Alpha suppression in AD

3. **Width of peaks/dips:**
   - Narrow peak = specific frequency affected
   - Broad peak = wide frequency range affected

4. **Height of peaks:**
   - Larger magnitude = bigger group difference
   - Compare to Panel E for effect size

**Example interpretation:**

```
"Difference analysis (Panel C) revealed a prominent peak in the delta-theta 
range (Δ|G| ≈ +0.5), indicating AD-specific elevation, and a pronounced 
dip in the alpha range (Δ|G| ≈ -0.7), reflecting HC-specific alpha 
amplification."
```

---

### Panel E: Effect Size Bar Chart

**What you're looking at:**
- X-axis: Five frequency bands
- Y-axis: Cohen's d (standardized effect size)
- Red bars: LTI model
- Blue bars: TV model
- Horizontal lines: Effect size thresholds
  - ±0.5 = medium effect
  - ±0.8 = large effect
- Asterisks above bars: Statistical significance

**How to interpret:**

1. **Positive bars (above zero):**
   - AD > HC for this band
   - Taller bar = larger difference

2. **Negative bars (below zero):**
   - HC > AD for this band
   - More negative = HC much stronger

3. **Effect size interpretation:**
   - d > 0.8 or d < -0.8: Large effect (clinically important)
   - 0.5 < |d| < 0.8: Medium effect (moderately important)
   - |d| < 0.5: Small effect (may not be clinically relevant)

4. **LTI vs TV comparison:**
   - If bars are similar height: Consistent across models
   - If LTI bar has asterisk but TV doesn't: Effect is time-averaged
   - If TV bar is taller: Effect is stronger in time-varying analysis

**Example interpretation:**

```
"Effect size analysis (Panel E) revealed large effects (|d| > 0.8) in 
the delta and alpha bands for both LTI and TV models. Delta showed 
d=+0.87 (AD > HC, LTI), indicating substantial slow-wave amplification 
in AD. Alpha showed d=-1.12 (HC > AD, LTI), reflecting marked alpha 
suppression in AD patients."
```

---

### Panel F: Statistical Table

**How to read the table:**

Each row shows:
1. **Band name & frequency range**
2. **Model type** (LTI or TV)
3. **AD Mean±SD**: Average magnitude in AD group
4. **HC Mean±SD**: Average magnitude in HC group
5. **Δ with arrow**: Difference (↑ means AD>HC, ↓ means HC>AD)
6. **p-value**: Statistical significance
7. **Cohen's d**: Effect size
8. **Sig**: Quick significance marker (**, *, ns)

**Color coding:**
- Pink/red rows: AD > HC
- Blue rows: HC > AD
- White rows: Not significant

**How to use this table:**

1. **Copy directly into thesis** (reformat as needed)
2. **Quick reference** for exact numbers
3. **Supplementary material** if figure is in main text

**Example interpretation:**

```
"Statistical analysis (Table/Panel F) confirmed significant group 
differences in three frequency bands (p<0.05). Delta band showed 
AD mean of 2.45±0.23 vs HC mean of 1.89±0.18 (p<0.001, d=0.87). 
Alpha band showed AD mean of 1.65±0.28 vs HC mean of 2.34±0.22 
(p<0.001, d=-1.12). [Continue for each significant band...]"
```

---

## 🔬 Figure 2: Individual Subject Traces

### What This Figure Shows

**Purpose**: Demonstrate that group differences are real and not driven by outliers.

**What to look for:**

1. **Spread of individual lines:**
   - Tight cluster = Low variability
   - Wide spread = High variability
   - Compare AD vs HC spread

2. **Outliers:**
   - Lines far from group mean
   - Should be few in number
   - If many outliers, questionable data quality

3. **Group separation:**
   - Do AD and HC clouds overlap?
   - Clear separation = Robust effect
   - Heavy overlap = Weak effect

4. **Shape consistency:**
   - Do individual lines follow same general shape as mean?
   - Yes = Good quality data
   - No = Heterogeneous population

**Example interpretation:**

```
"Individual subject analysis (Figure X) demonstrated consistent response 
patterns within each group. While inter-subject variability was present, 
individual traces clustered around their respective group means (Panel A-D), 
validating the reliability of group-level findings. No extreme outliers 
were observed, confirming data quality."
```

---

## 📊 Figure 3: Model Selection Analysis

### What This Figure Shows

**Purpose**: Justify why you chose the model parameters (P and K) you did.

### Panels A-B: P and K Distributions

**What to look for:**

1. **Mode (peak) of histogram:**
   - Most common selected value
   - "Most subjects required P=X"

2. **Overlap between AD and HC:**
   - Heavy overlap = Similar complexity
   - Separation = Different model needs

3. **Spread:**
   - Narrow peak = Consistent selection
   - Flat distribution = Variable needs

**Example interpretation:**

```
"Model selection analysis (Figure X, Panels A-B) revealed that most 
subjects were best fit with P=5-10 and K=2-3, with substantial overlap 
between AD and HC groups. This suggests comparable model complexity 
requirements despite group differences in dynamics."
```

### Panel C: P vs K Scatter

**What to look for:**

1. **Correlation:**
   - Diagonal trend = P and K related
   - No pattern = Independent selection

2. **Clustering:**
   - Distinct clusters = Subgroups
   - Diffuse scatter = Heterogeneous

**Example interpretation:**

```
"Joint parameter distribution (Panel C) showed weak correlation between 
P and K (r=0.23), indicating independent selection of temporal and spatial 
model complexity."
```

### Panels D-E: Statistical Comparisons

**What to look for:**

1. **P-value:**
   - p < 0.05 = Groups differ in complexity
   - p > 0.05 = Similar complexity

2. **Practical significance:**
   - Even if p < 0.05, is the difference large?
   - Mean difference > 2-3 lags = Meaningful
   - Mean difference < 1 lag = Negligible

**Example interpretation:**

If p > 0.05:
```
"No significant difference in selected AR order P (AD: 7.2±2.1, HC: 6.8±1.9, 
p=0.34) or graph filter order K (AD: 2.3±0.6, HC: 2.1±0.5, p=0.18) was 
observed, indicating that model complexity requirements were comparable 
between groups."
```

If p < 0.05:
```
"AD patients required significantly higher AR order P (AD: 9.5±2.3, 
HC: 6.2±1.8, p<0.01, d=0.76), suggesting that AD brain dynamics demand 
longer temporal history for accurate prediction."
```

---

## 📈 Figure 4: Group Comparison Metrics

### Each Boxplot Shows:

- Center line = median
- Box edges = 25th and 75th percentiles (IQR)
- Whiskers = 1.5 × IQR
- Dots = individual subjects
- P-value above = statistical test result

### How to Interpret Each Metric:

**LTI R²**:
- Higher = Better model fit
- If AD < HC: "AD brains less predictable"
- If AD > HC: "AD dynamics more regular"

**LTI ρ (spectral radius)**:
- Closer to 1 = Less stable
- If AD > HC: "AD networks closer to instability threshold"

**LTI BIC**:
- Lower = Better model (balances fit vs complexity)
- Usually not clinically interpreted directly

**TV R² mean**:
- Compare to LTI R²
- If TV > LTI: "Time-varying model captures additional dynamics"

**TV ρ mean**:
- Average stability across windows
- Compare to LTI ρ

**Mean MSD** (Mean Squared Difference):
- How much TV differs from LTI
- Higher = More time-varying
- If AD > HC: "AD shows more temporal variability"

**Mean CV** (Coefficient of Variation):
- Normalized variability measure
- If AD > HC: "AD dynamics less stable over time"

**Example interpretation:**

```
"Group comparison of model metrics (Figure X) revealed significantly 
higher coefficient of variation in AD patients (CV=0.25±0.08) compared 
to HC (CV=0.12±0.04), t(64)=7.23, p<0.001, d=0.92, indicating increased 
temporal instability of brain network dynamics in Alzheimer's disease."
```

---

## 🎨 Figure 5: Transfer Function Heatmaps

### What the Heatmaps Show:

**Dimensions:**
- X-axis: λ (graph frequency / eigenvalue of Laplacian)
- Y-axis: ω (temporal frequency in Hz)
- Color: |G(ω,λ)| magnitude

**How to interpret:**

1. **Bright regions:**
   - System amplifies these ω-λ combinations
   - Resonances of the network

2. **Dark regions:**
   - System suppresses these combinations
   - Filtering effects

3. **Patterns:**
   - Horizontal bands = Frequency-specific (all spatial modes)
   - Vertical bands = Spatial mode-specific (all frequencies)
   - Diagonal = Coupled spatial-temporal

4. **Difference maps (right column):**
   - Red = AD > HC (AD amplifies more)
   - Blue = HC > AD (HC amplifies more)
   - White = No difference

**Typical patterns:**

- **Low ω, low λ**: Global slow oscillations
- **High ω, low λ**: Global fast oscillations
- **Low ω, high λ**: Localized slow oscillations
- **High ω, high λ**: Localized fast oscillations

**Example interpretation:**

```
"Full spectral-spatial analysis (Figure X) revealed distinct group 
differences in the transfer function landscape. AD patients showed 
elevated magnitude in the low-frequency, low-λ region (bottom-left of 
difference map, Panel C), indicating excessive amplification of global 
slow-wave activity. Conversely, HC showed higher magnitude in the 
mid-frequency, mid-λ region (Panel C, center), corresponding to alpha 
rhythms in distributed cortical networks."
```

---

## ✍️ Writing Results Section

### Template Structure

```
## Results

### Model Selection

We performed model selection using BIC across P ∈ {1,2,3,5,7,10,15,20,30} 
and K ∈ {1,2,3,4} for each subject. [Describe Figure 3 results...]

### Frequency-Specific Group Differences

Mode-averaged frequency response analysis (Figure X) revealed significant 
group differences in [number] frequency bands...

#### Delta Band (0.5-4 Hz)
AD patients exhibited [significantly higher/lower] transfer function 
magnitude (AD: X.XX±X.XX, HC: X.XX±X.XX, p=X.XXX, d=X.XX). This indicates...

#### Theta Band (4-8 Hz)
[Describe theta results...]

#### Alpha Band (8-13 Hz)
[Describe alpha results - this is usually the most important...]

[Continue for each significant band...]

### Temporal Stability

Time-varying analysis revealed [describe Panel B and mean_cv results...]

### Model Performance

Group comparison of model fit metrics (Figure Y) showed [describe boxplots...]
```

---

## 🎓 Discussion Section Tips

### Connect to Literature

**For elevated delta/theta in AD:**
```
"Our findings of elevated delta/theta amplification in AD (Figure X, Panel A) 
align with extensive literature documenting cortical slowing in 
neurodegeneration [cite: 3-5 key papers]. However, our network-based 
transfer function analysis extends these findings by demonstrating this 
is a system-level property, not merely increased signal power."
```

**For alpha suppression in AD:**
```
"The marked reduction in alpha band magnitude (d=-1.12, Figure X, Panel E) 
corroborates prior EEG studies showing alpha suppression in AD [cite]. 
Our results further indicate this reflects altered network resonance 
properties, specifically reduced amplification at 8-13 Hz, potentially 
stemming from disrupted thalamocortical circuits [cite]."
```

### Clinical Implications

```
"The robust effect sizes observed in our frequency band analysis (|d| > 0.8, 
Figure X, Panel E) suggest potential utility as biomarkers for AD detection 
or monitoring. Specifically, the delta-to-alpha ratio in mode-averaged 
transfer functions may serve as a network-level diagnostic marker."
```

### Limitations

```
"While mode-averaging simplifies interpretation and connects to traditional 
EEG analysis, it necessarily discards spatial information. Future work should 
examine whether specific graph modes (spatial patterns) drive the observed 
frequency differences."
```

---

## 🔍 Common Patterns and Their Meanings

| Pattern | Interpretation | Clinical Relevance |
|---------|---------------|-------------------|
| AD > HC in delta/theta | Slow-wave amplification | Cortical slowing, neurodegeneration |
| HC > AD in alpha | Preserved alpha rhythms | Intact thalamocortical function |
| AD > HC in beta | Hyperexcitability | Compensatory or pathological |
| Higher CV in AD | Temporal instability | Loss of homeostatic regulation |
| Similar P and K | Comparable complexity | Structure intact, dynamics altered |
| AD P > HC P | Need longer history | Increased temporal dependencies |

---

## 💡 Final Tips

1. **Always report effect sizes** - p-values alone are not enough
2. **Show both LTI and TV** - Demonstrates thoroughness
3. **Use figure panels in order** - Tell a logical story (A→B→C)
4. **Reference specific panels** - "As shown in Figure 3C..."
5. **Quantify everything** - Provide actual numbers from table
6. **Connect to biology** - What does this mean for the brain?
7. **Be precise with language** - "Elevated" not "increased" (for transfer functions)

Your figures contain a wealth of information. Take time to understand each element, and your thesis writing will flow naturally from the visual results!
