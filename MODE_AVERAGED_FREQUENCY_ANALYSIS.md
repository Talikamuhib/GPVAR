# Mode-Averaged Frequency Response Analysis

## What is Mode-Averaging?

### The Challenge

The GP-VAR transfer function G(ω, λ) is a **2D function** with two independent variables:
- **ω (omega)**: Temporal frequency in radians/second (converts to Hz)
- **λ (lambda)**: Graph frequency (eigenvalues of the Laplacian)

This creates a complex landscape where it's hard to see clear patterns.

### The Solution: Mode-Averaging

**Mode-averaging** means computing the **mean over λ** for each ω:

```
G_avg(ω) = mean_over_λ [ |G(ω, λ)| ]
```

**Result**: A simple 1D curve showing how the system responds to different temporal frequencies, averaged across all spatial patterns.

---

## Why Mode-Averaging is Important for Your Thesis

### 1. **Clinical Interpretability**

Standard EEG analysis focuses on **temporal frequency bands**:
- Delta (0.5-4 Hz): Pathology, deep sleep
- Theta (4-8 Hz): Memory, meditation
- Alpha (8-13 Hz): Relaxed wakefulness
- Beta (13-30 Hz): Active cognition
- Gamma (30-40 Hz): Information processing

Mode-averaged responses directly map to these familiar EEG bands, making results clinically interpretable.

### 2. **Removes Spatial Complexity**

The full 2D transfer function G(ω, λ) mixes:
- Temporal dynamics (how fast things oscillate)
- Spatial patterns (which brain regions are involved)

By averaging over λ, you isolate **pure temporal effects**:
- "Do AD brains amplify slow frequencies more than HC?"
- "Is alpha suppression present in AD?"

These questions are independent of spatial connectivity patterns.

### 3. **Reduces Dimensionality**

Instead of comparing:
- 256 frequencies × 64 graph modes = **16,384 values** per subject

You compare:
- 256 frequencies = **256 values** per subject

This makes statistical testing more powerful and visualizations clearer.

### 4. **Connects to Existing Literature**

Most EEG/MEG studies report power spectra:
```
P(ω) = power at frequency ω
```

Your mode-averaged transfer function:
```
|G(ω)| = system response at frequency ω
```

These are conceptually related! Your analysis extends traditional spectral analysis by:
- Incorporating graph structure (via the GP-VAR model)
- Distinguishing LTI vs TV dynamics
- Revealing system-level resonances (not just raw power)

---

## Mathematical Details

### Full Transfer Function

The GP-VAR model transfer function is:

```
G(ω, λ) = 1 / [1 - Σ_p H_p(λ) e^{-iωp}]

Where:
H_p(λ) = Σ_k h_{p,k} λ^k  (graph filter at lag p)
```

- **ω**: Determines temporal oscillation e^{-iωp}
- **λ**: Determines spatial filtering λ^k

### Mode-Averaged Response

For each subject s and each frequency ω:

```
G_avg^(s)(ω) = (1/N_λ) Σ_{j=1}^{N_λ} |G(ω, λ_j)|

Where:
N_λ = number of graph modes (64 in your case)
λ_j = j-th eigenvalue of the Laplacian
```

### Group-Level Comparison

For each group (AD or HC):

```
G_group(ω) = mean over subjects [ G_avg^(s)(ω) ]
SE_group(ω) = std over subjects [ G_avg^(s)(ω) ] / sqrt(N_subjects)
```

Plot: G_AD(ω) vs G_HC(ω) with error bars ±SE

---

## Interpretation Guide

### High |G(ω)| at frequency f

**Meaning**: The brain network amplifies activity at frequency f.

**Mechanism**: 
- If input has frequency component at f
- Output will have LARGER magnitude at f
- System has a "resonance" at f

**Example**: High |G(10 Hz)| means alpha rhythms (10 Hz) are amplified by the network structure.

### Low |G(ω)| at frequency f

**Meaning**: The brain network suppresses/filters activity at frequency f.

**Mechanism**:
- Input at frequency f is attenuated
- System acts as a "notch filter" at f

**Example**: Low |G(30 Hz)| means beta rhythms are suppressed.

### AD vs HC Differences

**AD > HC at frequency f**:
- AD brains amplify frequency f more than HC
- Could indicate hyperexcitability or compensatory mechanisms
- Check if f corresponds to pathological rhythm (e.g., excessive delta/theta)

**HC > AD at frequency f**:
- HC brains amplify frequency f more than AD
- Could indicate loss of normal rhythm in AD
- Check if f corresponds to healthy rhythm (e.g., alpha)

---

## Frequency Band Statistics

Your analysis automatically computes statistics for each band:

### Example Output Table

| Band  | Freq Range | Model | AD Mean | HC Mean | p-value | Cohen's d | Significant |
|-------|-----------|-------|---------|---------|---------|-----------|-------------|
| Delta | 0.5-4 Hz  | LTI   | 2.45    | 1.89    | 0.003   | 0.87      | ✓           |
| Theta | 4-8 Hz    | LTI   | 2.12    | 1.95    | 0.234   | 0.31      | ✗           |
| Alpha | 8-13 Hz   | LTI   | 1.65    | 2.34    | 0.001   | -1.12     | ✓           |
| Beta  | 13-30 Hz  | LTI   | 1.88    | 1.76    | 0.456   | 0.18      | ✗           |
| Gamma | 30-40 Hz  | LTI   | 1.42    | 1.38    | 0.821   | 0.05      | ✗           |

### Interpretation of Example

**Delta (0.5-4 Hz)**:
- AD shows higher magnitude (2.45 vs 1.89)
- Large effect size (d=0.87)
- Interpretation: "AD patients exhibit elevated slow-wave amplification, consistent with cortical slowing"

**Alpha (8-13 Hz)**:
- HC shows higher magnitude (2.34 vs 1.65)
- Large effect size (d=-1.12, negative means HC > AD)
- Interpretation: "Healthy controls demonstrate stronger alpha rhythm amplification, suggesting intact thalamocortical oscillations. AD patients show alpha suppression."

---

## Relationship to Traditional Spectral Analysis

### Traditional Power Spectrum

Standard EEG analysis computes:
```
PSD(ω) = |FFT[X(t)]|²
```
This shows the **power** present at each frequency in the data.

### Your Mode-Averaged Transfer Function

Your analysis computes:
```
|G(ω)| = system gain at frequency ω
```
This shows how the **network dynamics amplify/suppress** each frequency.

### Key Difference

**PSD**: "What frequencies are in the data?"
**|G|**: "How does the brain network respond to frequencies?"

**Example**:
- High PSD at 10 Hz: Lots of alpha power in the recording
- High |G| at 10 Hz: The brain network amplifies alpha rhythms (system property)

Your transfer function reveals the **generative mechanism** behind the observed power spectrum.

---

## Advantages of Your Approach

### 1. Network-Based

Traditional spectral analysis ignores spatial structure. Your method:
- Uses the Laplacian to encode connectivity
- Transfer function reflects network resonances
- Captures how signals propagate through graph structure

### 2. Model-Based

Instead of just describing data (PSD), you fit a model that:
- Predicts future states from past states
- Has interpretable parameters
- Distinguishes stable vs unstable dynamics

### 3. Time-Varying Capable

By comparing LTI vs TV:
- Detect non-stationarity
- See how resonances change over time
- Quantify temporal stability (CV metric)

### 4. Statistically Rigorous

Proper group comparisons:
- t-tests per frequency band
- Effect sizes (Cohen's d)
- Multiple subjects per group
- Standard errors shown

---

## Thesis Writing Tips

### Results Section Template

```
We computed mode-averaged transfer function magnitudes |G(ω)| by 
averaging |G(ω,λ)| across all graph modes λ. This reveals the 
frequency-dependent gain of the brain network independent of 
spatial connectivity patterns.

Figure X shows the mode-averaged responses for LTI and TV models. 
AD patients exhibited significantly higher transfer function 
magnitude in the delta band (0.5-4 Hz) compared to HC (AD: 
X.XX±X.XX, HC: X.XX±X.XX, p<0.001, d=X.XX), suggesting elevated 
amplification of slow-wave activity. Conversely, HC showed higher 
alpha band (8-13 Hz) magnitude (HC: X.XX±X.XX, AD: X.XX±X.XX, 
p<0.001, d=X.XX), indicating preserved thalamocortical oscillations.

Time-varying analysis revealed similar patterns, with additional 
[increased/decreased] variability in AD patients for [specific bands].
```

### Figure Caption Template

```
Figure X: Mode-averaged frequency response analysis. 
(A) LTI model showing transfer function magnitude |G(ω)| averaged 
over all graph modes (λ) as a function of temporal frequency. 
Shaded regions indicate standard EEG frequency bands. AD patients 
(red) show elevated delta/theta response and reduced alpha response 
compared to HC (blue). Error bars: ±SEM. 
(B) Time-varying (TV) model showing qualitatively similar patterns. 
(C-D) Difference plots (AD - HC) highlighting frequency regions of 
significant group differences. 
(E) Statistical comparison table showing p-values and effect sizes 
for each frequency band. *p<0.05, **p<0.01, ***p<0.001.
```

### Discussion Points

1. **Consistency with literature**: "Our findings of elevated delta/theta in AD align with prior EEG studies showing cortical slowing [cite]"

2. **Novel contribution**: "However, our network-based transfer function analysis reveals this is a system-level property, not just increased power"

3. **Clinical relevance**: "The alpha suppression in AD may serve as a biomarker for thalamocortical dysfunction"

4. **Time-varying insights**: "TV analysis shows these frequency-specific alterations are [stable/variable] over time"

---

## Common Questions

### Q: Why not just use regular spectral analysis?

**A**: Regular spectral analysis (FFT/PSD) tells you what frequencies are in the data. Transfer functions tell you how the **system** (brain network) processes frequencies. It's like the difference between measuring what sounds come out of a speaker vs. understanding the frequency response of the speaker itself.

### Q: What does it mean if all frequencies show higher |G| in AD?

**A**: This could indicate:
1. Overall network hyperexcitability
2. Reduced inhibition
3. Loss of frequency selectivity (flatter response curve)

Compare the **shape** of the curves, not just magnitude. Look for shifts in peak frequencies or changes in bandwidth.

### Q: Should I interpret LTI or TV results?

**A**: Report both! 
- **LTI**: Average system behavior (more stable estimate)
- **TV**: Time-varying dynamics (reveals non-stationarity)

If LTI and TV show similar patterns → stable group differences
If they differ → temporal variability is group-specific

### Q: How do I know if results are clinically meaningful?

**A**: Check three things:
1. **Statistical significance**: p < 0.05
2. **Effect size**: |d| > 0.5 (medium) or > 0.8 (large)
3. **Literature support**: Does it match known EEG changes in AD?

All three → strong clinical relevance

---

## Summary

**Mode-averaged frequency response analysis**:
- ✓ Reduces complexity (2D → 1D)
- ✓ Clinically interpretable (maps to EEG bands)
- ✓ Statistically powerful (fewer comparisons)
- ✓ Network-based (incorporates graph structure)
- ✓ Connects to literature (extends spectral analysis)

**Key output for thesis**: `mode_averaged_frequency_responses.png` + `frequency_band_statistics.csv`

This is arguably your **most important result** for clinical interpretation!
