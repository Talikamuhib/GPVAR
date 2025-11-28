# Thesis Figures 4.4.1: LTI GP-VAR Model Selection and Fit

This folder contains the code and outputs for generating **thesis-quality figures** for Section 4.4.1.

## Generated Files

| File | Description |
|------|-------------|
| `Fig_4_4_1a_model_selection.png` | Model selection (P, K) distributions with bar plots, boxplots, and summary statistics |
| `Fig_4_4_1b_fit_metrics.png` | R² and MSE distributions with violin + jitter plots for AD vs HC |
| `Fig_4_4_1_combined_summary.png` | Combined single-row summary figure for quick reference |
| `thesis_text_4_4_1.txt` | Ready-to-use thesis text with methods, results, and figure captions |
| `statistics_4_4_1.json` | All statistics in JSON format for programmatic access |

## Usage

### With Demo Data (for testing)
```bash
python3 plot_model_selection_fit.py --use-demo-data
```

### With Real Analysis Results
Place your results in `../group_comparison_lti_tv_analysis/`:
- `model_selection_summary.csv` - Contains `subject_id`, `group`, `selected_P`, `selected_K`
- `all_subjects_results.csv` - Contains `subject_id`, `group`, `best_P`, `best_K`, `lti_R2`, `lti_MSE`

Then run:
```bash
python3 plot_model_selection_fit.py
```

## Figure Details

### Figure 4.4.1a: Model Selection (P, K)
- **(A)** Bar plot of P (AR order) selection counts by group
- **(B)** Bar plot of K (graph filter order) selection counts by group  
- **(C)** Scatter plot of P vs K with median markers
- **(D)** Boxplot of P by group with individual points
- **(E)** Boxplot of K by group with individual points
- **(F)** Summary statistics table

### Figure 4.4.1b: Model Fit Metrics
- **(A)** Violin + boxplot + jitter for R² (AD vs HC)
- **(B)** R² summary statistics
- **(C)** Violin + boxplot + jitter for MSE (AD vs HC)
- **(D)** MSE summary statistics

## Key Statistics Reported

### Model Selection
- Median (IQR) for P and K by group
- Range of selected values
- t-test p-value and Cohen's d

### Model Fit
- Median R² per group
- Mean ± SD for R² and MSE
- Statistical comparison (t-test, effect size)

## Dependencies
```
numpy
pandas
matplotlib
seaborn
scipy
```

## Thesis Text Output

The script automatically generates thesis-ready text including:
- Methods paragraph recalling the model selection procedure
- Results text with key numbers formatted consistently
- Interpretation based on statistical significance
- Figure captions

## Customization

Edit `plot_model_selection_fit.py` to modify:
- `P_RANGE` and `K_RANGE`: Model selection search ranges
- `COLOR_AD` and `COLOR_HC`: Group colors
- Figure dimensions and DPI settings
