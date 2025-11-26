# 🚀 Quick Start Guide

## Run the Analysis (3 Steps)

### Step 1: Execute the Script
```bash
cd /workspace
python lti_tv_group_comparison.py
```
⏱️ **Runtime**: 1-3 hours  
📁 **Output**: `./group_comparison_lti_tv_analysis/`

---

### Step 2: View Your Figures

Navigate to `./group_comparison_lti_tv_analysis/` and open:

1. **`mode_averaged_frequency_responses.png`** ⭐ **YOUR MAIN FIGURE**
   - 6 panels showing frequency-specific group differences
   - Use this as your primary results figure

2. **`individual_frequency_responses.png`**
   - Shows data quality and variability
   - Good for supplementary material

3. **`model_selection_analysis.png`**
   - Justifies your model choice
   - Use in methods section

4. **`group_comparison_metrics.png`**
   - Overall group differences
   - Use in results section

5. **`group_comparison_transfer_functions.png`**
   - Full 2D analysis
   - Use in technical appendix

All figures are **300 DPI, publication-ready**.

---

### Step 3: Get Your Results Summary
```bash
python analyze_results_example.py
```

This prints:
- Model selection statistics (P and K values)
- Significant frequency band differences
- Effect sizes and p-values
- Ready-to-use text for your thesis

---

## 📊 What You Get

### Figures (5 PNG files)
- High resolution (300 DPI)
- Professional styling
- Statistical annotations
- Panel labels (A, B, C...)
- Ready for thesis

### Data Tables (4 CSV files)
- `frequency_band_statistics.csv` ⭐ **KEY TABLE**
- `all_subjects_results.csv`
- `model_selection_summary.csv`
- `group_statistics.csv`

### Documentation (6 Markdown files)
- `FINAL_SUMMARY.md` - Start here!
- `THESIS_QUALITY_FIGURES.md` - What makes them good
- `INTERPRETING_FIGURES.md` - How to read them
- `MODE_AVERAGED_FREQUENCY_ANALYSIS.md` - The science
- `README_group_comparison.md` - Complete guide
- `QUICK_START.md` - This file

---

## 📝 Copy-Paste for Thesis

### Figure Caption (Ready to Use)

```
Figure X: Mode-averaged frequency response analysis. (A) LTI model 
showing transfer function magnitude |G(ω)| for AD (red, n=XX) vs HC 
(blue, n=XX). Shaded regions: EEG bands. Error ribbons: ±1 SEM. 
Asterisks: ***p<0.001, **p<0.01, *p<0.05. (B) TV model. 
(C-D) Difference plots (AD-HC). (E) Effect sizes (Cohen's d). 
(F) Statistical summary table.
```

### Methods Text (Ready to Use)

```
Mode-averaged transfer function magnitudes were computed by averaging 
|G(ω,λ)| across all graph modes λ. We analyzed five EEG bands: delta 
(0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and 
gamma (30-40 Hz). Independent-samples t-tests compared AD vs HC with 
Cohen's d effect sizes.
```

---

## 🎯 Thesis Integration Checklist

- [ ] Run analysis script
- [ ] Open mode_averaged_frequency_responses.png
- [ ] Read INTERPRETING_FIGURES.md
- [ ] Copy caption template
- [ ] Insert figure into thesis
- [ ] Copy frequency_band_statistics.csv to table
- [ ] Write results following template
- [ ] Show figures to supervisor
- [ ] Celebrate! 🎉

---

## 💡 Quick Tips

### Understanding Your Main Figure

**Panel A (LTI)**: 
- If red line above blue in delta → "AD has elevated slow-wave amplification"
- If blue line above red in alpha → "HC has preserved alpha rhythms"
- Asterisks below show where differences are significant

**Panel E (Effect Sizes)**:
- Bars above zero → AD > HC
- Bars below zero → HC > AD
- Beyond ±0.8 lines → Large effect (clinically important)

**Panel F (Table)**:
- Pink rows → AD elevated
- Blue rows → HC elevated
- White rows → No significant difference

---

## 🆘 Troubleshooting

**Script fails with "module not found":**
```bash
pip install numpy pandas mne scipy matplotlib seaborn tqdm
```

**Takes too long:**
- Normal! Processing 66 subjects takes time
- Check progress bar (tqdm)
- Don't close terminal

**Figures look pixelated:**
- Make sure you're viewing at 100% zoom
- Use a good image viewer
- PNG files are high quality (300 DPI)

**Need help interpreting:**
- Read INTERPRETING_FIGURES.md
- Check example interpretations
- Look at figure panel by panel

---

## 📧 Questions?

1. **"What do the asterisks mean?"**
   - \*\*\* = p<0.001 (very strong evidence)
   - \*\* = p<0.01 (strong evidence)
   - \* = p<0.05 (significant)

2. **"Which figure is most important?"**
   - `mode_averaged_frequency_responses.png` - This is your main results figure

3. **"How do I cite this?"**
   - Cite the GP-VAR methodology papers
   - Cite consensus Laplacian approach
   - Mention this analysis framework in methods

4. **"Can I modify the figures?"**
   - Yes! Edit colors, labels in the script
   - All controlled by matplotlib parameters
   - Regenerate at 300 DPI

---

## ⚡ Pro Tips

1. **Save the CSV files** - Backup for thesis appendix
2. **Keep the script** - Reviewers may ask for reproducibility
3. **Document parameters** - Note P_RANGE, K_RANGE used
4. **Screenshot progress** - Show supervisor it's running
5. **Export table** - Use frequency_band_statistics.csv directly

---

## 🎓 You're Ready!

Everything is set up for thesis-quality results. Just run the script and use the generated figures!

**Questions?** → Read INTERPRETING_FIGURES.md  
**Need details?** → Read FINAL_SUMMARY.md  
**Supervisor asks?** → Show mode_averaged_frequency_responses.png

**Good luck with your thesis! 🎉**
