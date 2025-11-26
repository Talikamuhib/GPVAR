"""
Example script showing how to read and analyze the group comparison results.
This demonstrates how to access the key findings for your thesis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set the results directory
RESULTS_DIR = Path("./group_comparison_lti_tv_analysis")

print("="*80)
print("GROUP COMPARISON RESULTS ANALYSIS")
print("="*80)

# ============================================================================
# 1. Model Selection Summary
# ============================================================================

print("\n" + "="*80)
print("1. MODEL SELECTION SUMMARY")
print("="*80)

model_summary = pd.read_csv(RESULTS_DIR / "model_selection_summary.csv")

ad_models = model_summary[model_summary['group'] == 'AD']
hc_models = model_summary[model_summary['group'] == 'HC']

print(f"\nAD Group (n={len(ad_models)}):")
print(f"  P (AR order): {ad_models['selected_P'].mean():.1f} ± {ad_models['selected_P'].std():.1f}")
print(f"    Range: [{ad_models['selected_P'].min()}, {ad_models['selected_P'].max()}]")
print(f"    Mode: {ad_models['selected_P'].mode().values[0]}")
print(f"  K (Graph filter order): {ad_models['selected_K'].mean():.1f} ± {ad_models['selected_K'].std():.1f}")
print(f"    Range: [{ad_models['selected_K'].min()}, {ad_models['selected_K'].max()}]")
print(f"    Mode: {ad_models['selected_K'].mode().values[0]}")

print(f"\nHC Group (n={len(hc_models)}):")
print(f"  P (AR order): {hc_models['selected_P'].mean():.1f} ± {hc_models['selected_P'].std():.1f}")
print(f"    Range: [{hc_models['selected_P'].min()}, {hc_models['selected_P'].max()}]")
print(f"    Mode: {hc_models['selected_P'].mode().values[0]}")
print(f"  K (Graph filter order): {hc_models['selected_K'].mean():.1f} ± {hc_models['selected_K'].std():.1f}")
print(f"    Range: [{hc_models['selected_K'].min()}, {hc_models['selected_K'].max()}]")
print(f"    Mode: {hc_models['selected_K'].mode().values[0]}")

# ============================================================================
# 2. Overall Group Statistics
# ============================================================================

print("\n" + "="*80)
print("2. OVERALL GROUP STATISTICS")
print("="*80)

group_stats = pd.read_csv(RESULTS_DIR / "group_statistics.csv")

print("\nSignificant differences (p < 0.05):")
sig_results = group_stats[group_stats['significant'] == True]

if len(sig_results) > 0:
    for _, row in sig_results.iterrows():
        direction = "AD > HC" if row['AD_mean'] > row['HC_mean'] else "HC > AD"
        effect_size = "small" if abs(row['cohens_d']) < 0.5 else "medium" if abs(row['cohens_d']) < 0.8 else "large"
        
        print(f"\n  {row['metric']}:")
        print(f"    AD: {row['AD_mean']:.4f} ± {row['AD_std']:.4f}")
        print(f"    HC: {row['HC_mean']:.4f} ± {row['HC_std']:.4f}")
        print(f"    Direction: {direction}")
        print(f"    p-value: {row['p_value']:.4f}")
        print(f"    Cohen's d: {row['cohens_d']:.3f} ({effect_size} effect)")
else:
    print("  No significant differences found in overall metrics")

print("\n" + "-"*80)
print("Non-significant trends (p < 0.10):")
trend_results = group_stats[(group_stats['significant'] == False) & (group_stats['p_value'] < 0.10)]

if len(trend_results) > 0:
    for _, row in trend_results.iterrows():
        print(f"  {row['metric']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
else:
    print("  No trends found")

# ============================================================================
# 3. Frequency Band Analysis
# ============================================================================

print("\n" + "="*80)
print("3. FREQUENCY BAND ANALYSIS (MODE-AVERAGED)")
print("="*80)

band_stats = pd.read_csv(RESULTS_DIR / "frequency_band_statistics.csv")

# Separate LTI and TV results
lti_bands = band_stats[band_stats['model_type'] == 'LTI']
tv_bands = band_stats[band_stats['model_type'] == 'TV']

print("\n--- LTI Model ---")
print("\nFrequency Band | AD Mean±SD | HC Mean±SD | p-value | Cohen's d | Sig")
print("-"*80)

for _, row in lti_bands.iterrows():
    sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
    print(f"{row['band'].capitalize():5s} {row['freq_range']:11s} | "
          f"{row['AD_mean']:5.3f}±{row['AD_std']:.3f} | "
          f"{row['HC_mean']:5.3f}±{row['HC_std']:.3f} | "
          f"{row['p_value']:7.4f} | "
          f"{row['cohens_d']:8.3f} | "
          f"{sig_marker:3s}")

print("\n--- TV Model ---")
print("\nFrequency Band | AD Mean±SD | HC Mean±SD | p-value | Cohen's d | Sig")
print("-"*80)

for _, row in tv_bands.iterrows():
    sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
    print(f"{row['band'].capitalize():5s} {row['freq_range']:11s} | "
          f"{row['AD_mean']:5.3f}±{row['AD_std']:.3f} | "
          f"{row['HC_mean']:5.3f}±{row['HC_std']:.3f} | "
          f"{row['p_value']:7.4f} | "
          f"{row['cohens_d']:8.3f} | "
          f"{sig_marker:3s}")

# Identify significant bands
sig_lti_bands = lti_bands[lti_bands['significant'] == True]
sig_tv_bands = tv_bands[tv_bands['significant'] == True]

if len(sig_lti_bands) > 0:
    print("\n" + "-"*80)
    print("SIGNIFICANT LTI BANDS:")
    for _, row in sig_lti_bands.iterrows():
        direction = "AD > HC" if row['AD_mean'] > row['HC_mean'] else "HC > AD"
        print(f"  {row['band'].upper()} ({row['freq_range']}): {direction}, p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")

if len(sig_tv_bands) > 0:
    print("\nSIGNIFICANT TV BANDS:")
    for _, row in sig_tv_bands.iterrows():
        direction = "AD > HC" if row['AD_mean'] > row['HC_mean'] else "HC > AD"
        print(f"  {row['band'].upper()} ({row['freq_range']}): {direction}, p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")

# ============================================================================
# 4. Individual Subject Results
# ============================================================================

print("\n" + "="*80)
print("4. INDIVIDUAL SUBJECT RESULTS PREVIEW")
print("="*80)

subject_results = pd.read_csv(RESULTS_DIR / "all_subjects_results.csv")

print(f"\nTotal subjects: {len(subject_results)}")
print(f"  AD: {len(subject_results[subject_results['group'] == 'AD'])}")
print(f"  HC: {len(subject_results[subject_results['group'] == 'HC'])}")

print("\nFirst 5 subjects:")
print(subject_results[['subject_id', 'group', 'best_P', 'best_K', 'lti_R2', 'mean_cv']].head())

# ============================================================================
# 5. Summary for Thesis
# ============================================================================

print("\n" + "="*80)
print("5. THESIS-READY SUMMARY")
print("="*80)

print("\n--- METHODS ---")
print(f"We analyzed {len(ad_models)} AD patients and {len(hc_models)} healthy controls.")
print(f"Model selection was performed using BIC, testing P ∈ {{1,2,3,5,7,10,15,20,30}} and K ∈ {{1,2,3,4}}.")
print(f"Both linear time-invariant (LTI) and time-varying (TV) models were fitted.")

print("\n--- RESULTS: Model Complexity ---")
if abs(ad_models['selected_P'].mean() - hc_models['selected_P'].mean()) > ad_models['selected_P'].std():
    print(f"AD patients required {'higher' if ad_models['selected_P'].mean() > hc_models['selected_P'].mean() else 'lower'} AR order "
          f"(P={ad_models['selected_P'].mean():.1f}±{ad_models['selected_P'].std():.1f}) "
          f"compared to HC (P={hc_models['selected_P'].mean():.1f}±{hc_models['selected_P'].std():.1f}).")
else:
    print(f"No substantial difference in model complexity between groups "
          f"(AD: P={ad_models['selected_P'].mean():.1f}±{ad_models['selected_P'].std():.1f}, "
          f"HC: P={hc_models['selected_P'].mean():.1f}±{hc_models['selected_P'].std():.1f}).")

print("\n--- RESULTS: Time-Varying Dynamics ---")
cv_row = group_stats[group_stats['metric'] == 'mean_cv'].iloc[0]
if cv_row['significant']:
    direction = "more" if cv_row['AD_mean'] > cv_row['HC_mean'] else "less"
    print(f"AD patients exhibited {direction} time-varying dynamics "
          f"(CV={cv_row['AD_mean']:.3f}±{cv_row['AD_std']:.3f}) "
          f"compared to HC (CV={cv_row['HC_mean']:.3f}±{cv_row['HC_std']:.3f}), "
          f"p={cv_row['p_value']:.4f}, d={cv_row['cohens_d']:.3f}.")
else:
    print(f"No significant difference in temporal variability between groups "
          f"(AD: CV={cv_row['AD_mean']:.3f}±{cv_row['AD_std']:.3f}, "
          f"HC: CV={cv_row['HC_mean']:.3f}±{cv_row['HC_std']:.3f}, p={cv_row['p_value']:.3f}).")

print("\n--- RESULTS: Frequency-Specific Effects ---")
if len(sig_lti_bands) > 0 or len(sig_tv_bands) > 0:
    print("Mode-averaged frequency response analysis revealed significant group differences:")
    
    if len(sig_lti_bands) > 0:
        for _, row in sig_lti_bands.iterrows():
            direction = "elevated" if row['AD_mean'] > row['HC_mean'] else "reduced"
            print(f"  • {row['band'].capitalize()} band ({row['freq_range']}): "
                  f"{direction} in AD (p={row['p_value']:.4f}, d={row['cohens_d']:.3f})")
else:
    print("No significant frequency-specific differences were observed.")

print("\n" + "="*80)
print("Analysis complete! Check the PNG files in the results directory for visualizations.")
print("="*80)
