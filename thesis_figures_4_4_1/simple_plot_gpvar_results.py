#!/usr/bin/env python3
"""
Simple Script: Plot GP-VAR Model Selection (P, K) and Fit Metrics
==================================================================
Section 4.4.1 - LTI GP-VAR Model Selection and Fit

This script creates:
1. Bar plots of P and K selection counts
2. Boxplots/violin plots of R² and MSE for AD vs HC
3. Summary statistics (median, IQR) for both groups
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors for groups
COLOR_AD = '#E74C3C'  # Red
COLOR_HC = '#2980B9'  # Blue

# Model selection ranges
P_RANGE = [1, 2, 3, 5, 7, 10, 15, 20, 30]
K_RANGE = [1, 2, 3, 4]

# Output directory
OUTPUT_DIR = Path("./")

# ============================================================================
# GENERATE DEMO DATA (replace with your real data loading)
# ============================================================================

def create_demo_data(n_ad=35, n_hc=31, seed=42):
    """
    Create synthetic demo data.
    
    REPLACE THIS with your actual data loading:
    df = pd.read_csv('your_results.csv')
    """
    np.random.seed(seed)
    
    data = []
    
    # AD subjects
    for i in range(n_ad):
        data.append({
            'subject_id': f'sub-AD{i+1:03d}',
            'group': 'AD',
            'P': np.random.choice([3, 5, 7, 10, 15], p=[0.15, 0.35, 0.25, 0.15, 0.10]),
            'K': np.random.choice([1, 2, 3, 4], p=[0.05, 0.45, 0.40, 0.10]),
            'R2': np.clip(np.random.normal(0.52, 0.08), 0.25, 0.75),
            'MSE': np.clip(np.random.normal(0.48, 0.06), 0.25, 0.75),
        })
    
    # HC subjects
    for i in range(n_hc):
        data.append({
            'subject_id': f'sub-HC{i+1:03d}',
            'group': 'HC',
            'P': np.random.choice([3, 5, 7, 10, 15], p=[0.10, 0.30, 0.30, 0.20, 0.10]),
            'K': np.random.choice([1, 2, 3, 4], p=[0.08, 0.42, 0.38, 0.12]),
            'R2': np.clip(np.random.normal(0.55, 0.07), 0.30, 0.80),
            'MSE': np.clip(np.random.normal(0.45, 0.05), 0.22, 0.70),
        })
    
    return pd.DataFrame(data)

# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

def get_stats(data):
    """Calculate summary statistics."""
    return {
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'median': np.median(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'min': np.min(data),
        'max': np.max(data),
    }

def compare_groups(ad_data, hc_data):
    """Statistical comparison between groups."""
    t_stat, p_val = stats.ttest_ind(ad_data, hc_data, equal_var=False)
    pooled_std = np.sqrt((np.var(ad_data, ddof=1) + np.var(hc_data, ddof=1)) / 2)
    cohens_d = (np.mean(ad_data) - np.mean(hc_data)) / (pooled_std + 1e-10)
    return {'t': t_stat, 'p': p_val, 'd': cohens_d}

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_model_selection(df, save_path):
    """
    Figure 1: Model Selection (P, K) Distribution
    - Bar plots of P and K counts
    - Boxplots by group
    """
    ad = df[df['group'] == 'AD']
    hc = df[df['group'] == 'HC']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ---- Panel A: P Distribution Bar Plot ----
    ax = axes[0, 0]
    P_vals = sorted(df['P'].unique())
    ad_counts = [sum(ad['P'] == p) for p in P_vals]
    hc_counts = [sum(hc['P'] == p) for p in P_vals]
    
    x = np.arange(len(P_vals))
    w = 0.35
    ax.bar(x - w/2, ad_counts, w, label=f"AD (n={len(ad)})", color=COLOR_AD, edgecolor='black')
    ax.bar(x + w/2, hc_counts, w, label=f"HC (n={len(hc)})", color=COLOR_HC, edgecolor='black')
    ax.set_xlabel('P (AR Order)', fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontweight='bold')
    ax.set_title('(A) Distribution of Selected P Values', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(P_vals)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Panel B: K Distribution Bar Plot ----
    ax = axes[0, 1]
    K_vals = sorted(df['K'].unique())
    ad_counts = [sum(ad['K'] == k) for k in K_vals]
    hc_counts = [sum(hc['K'] == k) for k in K_vals]
    
    x = np.arange(len(K_vals))
    ax.bar(x - w/2, ad_counts, w, label=f"AD", color=COLOR_AD, edgecolor='black')
    ax.bar(x + w/2, hc_counts, w, label=f"HC", color=COLOR_HC, edgecolor='black')
    ax.set_xlabel('K (Graph Filter Order)', fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontweight='bold')
    ax.set_title('(B) Distribution of Selected K Values', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(K_vals)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Panel C: P Boxplot ----
    ax = axes[1, 0]
    bp = ax.boxplot([ad['P'], hc['P']], positions=[1, 2], widths=0.5, 
                    patch_artist=True, showmeans=True,
                    meanprops={'marker': 'D', 'markerfacecolor': 'gold', 'markersize': 8})
    bp['boxes'][0].set_facecolor(COLOR_AD)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLOR_HC)
    bp['boxes'][1].set_alpha(0.6)
    
    # Add jitter points
    for i, (data, pos) in enumerate([(ad['P'], 1), (hc['P'], 2)]):
        x_jitter = np.random.normal(pos, 0.05, len(data))
        color = COLOR_AD if i == 0 else COLOR_HC
        ax.scatter(x_jitter, data, alpha=0.5, s=30, color=color, edgecolor='black', linewidth=0.5)
    
    # Statistics
    comp = compare_groups(ad['P'].values, hc['P'].values)
    sig = '***' if comp['p'] < 0.001 else '**' if comp['p'] < 0.01 else '*' if comp['p'] < 0.05 else 'ns'
    ax.text(1.5, max(df['P']) + 1, f"p={comp['p']:.3f} {sig}", ha='center', fontweight='bold')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['AD', 'HC'])
    ax.set_ylabel('P (AR Order)', fontweight='bold')
    ax.set_title('(C) P Selection by Group', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Panel D: K Boxplot ----
    ax = axes[1, 1]
    bp = ax.boxplot([ad['K'], hc['K']], positions=[1, 2], widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops={'marker': 'D', 'markerfacecolor': 'gold', 'markersize': 8})
    bp['boxes'][0].set_facecolor(COLOR_AD)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLOR_HC)
    bp['boxes'][1].set_alpha(0.6)
    
    for i, (data, pos) in enumerate([(ad['K'], 1), (hc['K'], 2)]):
        x_jitter = np.random.normal(pos, 0.05, len(data))
        color = COLOR_AD if i == 0 else COLOR_HC
        ax.scatter(x_jitter, data, alpha=0.5, s=30, color=color, edgecolor='black', linewidth=0.5)
    
    comp = compare_groups(ad['K'].values, hc['K'].values)
    sig = '***' if comp['p'] < 0.001 else '**' if comp['p'] < 0.01 else '*' if comp['p'] < 0.05 else 'ns'
    ax.text(1.5, max(df['K']) + 0.3, f"p={comp['p']:.3f} {sig}", ha='center', fontweight='bold')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['AD', 'HC'])
    ax.set_ylabel('K (Graph Filter Order)', fontweight='bold')
    ax.set_title('(D) K Selection by Group', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Figure 4.4.1a: LTI GP-VAR Model Selection (P, K)\n'
                 'BIC-Based Selection with Stability Constraints', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_fit_metrics(df, save_path):
    """
    Figure 2: Model Fit (R², MSE)
    - Violin + boxplot + jitter for R² and MSE
    """
    ad = df[df['group'] == 'AD']
    hc = df[df['group'] == 'HC']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---- Panel A: R² Distribution ----
    ax = axes[0]
    
    # Violin
    parts = ax.violinplot([ad['R2'], hc['R2']], positions=[1, 2], widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLOR_AD if i == 0 else COLOR_HC)
        pc.set_alpha(0.3)
    
    # Boxplot
    bp = ax.boxplot([ad['R2'], hc['R2']], positions=[1, 2], widths=0.15,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(COLOR_AD)
    bp['boxes'][1].set_facecolor(COLOR_HC)
    
    # Jitter points
    for i, (data, pos) in enumerate([(ad['R2'], 1), (hc['R2'], 2)]):
        x_jitter = np.random.normal(pos, 0.06, len(data))
        color = COLOR_AD if i == 0 else COLOR_HC
        ax.scatter(x_jitter, data, alpha=0.5, s=35, color=color, edgecolor='white', linewidth=0.5)
    
    # Statistics
    comp = compare_groups(ad['R2'].values, hc['R2'].values)
    sig = '***' if comp['p'] < 0.001 else '**' if comp['p'] < 0.01 else '*' if comp['p'] < 0.05 else 'ns'
    y_max = max(df['R2'].max(), 0.8)
    ax.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
    ax.text(1.5, y_max + 0.03, f"p={comp['p']:.3f} {sig}", ha='center', fontweight='bold')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'AD (n={len(ad)})', f'HC (n={len(hc)})'])
    ax.set_ylabel('R² (Coefficient of Determination)', fontweight='bold')
    ax.set_title('(A) LTI Model R² by Group', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.legend(loc='lower right')
    
    # ---- Panel B: MSE Distribution ----
    ax = axes[1]
    
    parts = ax.violinplot([ad['MSE'], hc['MSE']], positions=[1, 2], widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLOR_AD if i == 0 else COLOR_HC)
        pc.set_alpha(0.3)
    
    bp = ax.boxplot([ad['MSE'], hc['MSE']], positions=[1, 2], widths=0.15,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(COLOR_AD)
    bp['boxes'][1].set_facecolor(COLOR_HC)
    
    for i, (data, pos) in enumerate([(ad['MSE'], 1), (hc['MSE'], 2)]):
        x_jitter = np.random.normal(pos, 0.06, len(data))
        color = COLOR_AD if i == 0 else COLOR_HC
        ax.scatter(x_jitter, data, alpha=0.5, s=35, color=color, edgecolor='white', linewidth=0.5)
    
    comp = compare_groups(ad['MSE'].values, hc['MSE'].values)
    sig = '***' if comp['p'] < 0.001 else '**' if comp['p'] < 0.01 else '*' if comp['p'] < 0.05 else 'ns'
    y_max = df['MSE'].max()
    ax.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
    ax.text(1.5, y_max + 0.03, f"p={comp['p']:.3f} {sig}", ha='center', fontweight='bold')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'AD (n={len(ad)})', f'HC (n={len(hc)})'])
    ax.set_ylabel('MSE (Mean Squared Error)', fontweight='bold')
    ax.set_title('(B) LTI Model MSE by Group', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Figure 4.4.1b: LTI GP-VAR Model Fit Metrics\n'
                 'R² and MSE Distributions for AD vs HC', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def print_statistics(df):
    """Print summary statistics for thesis text."""
    ad = df[df['group'] == 'AD']
    hc = df[df['group'] == 'HC']
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS FOR THESIS SECTION 4.4.1")
    print("="*70)
    
    # Model Selection
    print("\n--- MODEL SELECTION (P, K) ---\n")
    
    for var in ['P', 'K']:
        ad_stats = get_stats(ad[var].values)
        hc_stats = get_stats(hc[var].values)
        comp = compare_groups(ad[var].values, hc[var].values)
        
        print(f"{var} (AR Order):" if var == 'P' else f"{var} (Graph Filter Order):")
        print(f"  AD (n={ad_stats['n']}): Median={ad_stats['median']:.0f}, "
              f"IQR=[{ad_stats['q1']:.0f}, {ad_stats['q3']:.0f}], "
              f"Range=[{ad_stats['min']:.0f}, {ad_stats['max']:.0f}]")
        print(f"  HC (n={hc_stats['n']}): Median={hc_stats['median']:.0f}, "
              f"IQR=[{hc_stats['q1']:.0f}, {hc_stats['q3']:.0f}], "
              f"Range=[{hc_stats['min']:.0f}, {hc_stats['max']:.0f}]")
        print(f"  t-test: t={comp['t']:.3f}, p={comp['p']:.4f}, Cohen's d={comp['d']:.3f}\n")
    
    # Fit Metrics
    print("--- MODEL FIT METRICS ---\n")
    
    for var, name in [('R2', 'R² (Coefficient of Determination)'), ('MSE', 'MSE (Mean Squared Error)')]:
        ad_stats = get_stats(ad[var].values)
        hc_stats = get_stats(hc[var].values)
        comp = compare_groups(ad[var].values, hc[var].values)
        
        print(f"{name}:")
        print(f"  AD (n={ad_stats['n']}): Median={ad_stats['median']:.3f}, "
              f"Mean={ad_stats['mean']:.3f}±{ad_stats['std']:.3f}, "
              f"IQR=[{ad_stats['q1']:.3f}, {ad_stats['q3']:.3f}]")
        print(f"  HC (n={hc_stats['n']}): Median={hc_stats['median']:.3f}, "
              f"Mean={hc_stats['mean']:.3f}±{hc_stats['std']:.3f}, "
              f"IQR=[{hc_stats['q1']:.3f}, {hc_stats['q3']:.3f}]")
        print(f"  t-test: t={comp['t']:.3f}, p={comp['p']:.4f}, Cohen's d={comp['d']:.3f}\n")
    
    # Generate thesis text
    print("="*70)
    print("READY-TO-USE THESIS TEXT")
    print("="*70)
    
    ad_P = get_stats(ad['P'].values)
    hc_P = get_stats(hc['P'].values)
    ad_K = get_stats(ad['K'].values)
    hc_K = get_stats(hc['K'].values)
    ad_R2 = get_stats(ad['R2'].values)
    hc_R2 = get_stats(hc['R2'].values)
    P_comp = compare_groups(ad['P'].values, hc['P'].values)
    R2_comp = compare_groups(ad['R2'].values, hc['R2'].values)
    
    print(f"""
METHODS:
We performed model selection over P ∈ {{{', '.join(map(str, P_RANGE))}}} and 
K ∈ {{{', '.join(map(str, K_RANGE))}}} using BIC with stability constraints.

RESULTS - Model Selection:
For the AD group (n={len(ad)}), the median selected P was {ad_P['median']:.0f} 
(IQR: [{ad_P['q1']:.0f}, {ad_P['q3']:.0f}]), while for the HC group (n={len(hc)}), 
the median P was {hc_P['median']:.0f} (IQR: [{hc_P['q1']:.0f}, {hc_P['q3']:.0f}]). 
This difference was {'significant' if P_comp['p'] < 0.05 else 'not significant'} 
(t={P_comp['t']:.3f}, p={P_comp['p']:.4f}, Cohen's d={P_comp['d']:.3f}).

For graph filter order K, AD showed median {ad_K['median']:.0f} (IQR: [{ad_K['q1']:.0f}, {ad_K['q3']:.0f}]) 
vs HC median {hc_K['median']:.0f} (IQR: [{hc_K['q1']:.0f}, {hc_K['q3']:.0f}]).

RESULTS - Model Fit:
The R² for AD was {ad_R2['median']:.3f} (median; IQR: [{ad_R2['q1']:.3f}, {ad_R2['q3']:.3f}]), 
compared to {hc_R2['median']:.3f} (IQR: [{hc_R2['q1']:.3f}, {hc_R2['q3']:.3f}]) for HC 
(t={R2_comp['t']:.3f}, p={R2_comp['p']:.4f}, Cohen's d={R2_comp['d']:.3f}).
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("GP-VAR Model Selection and Fit - Thesis Figures 4.4.1")
    print("="*70)
    
    # Load or create data
    # REPLACE THIS WITH YOUR ACTUAL DATA LOADING:
    # df = pd.read_csv('your_gpvar_results.csv')
    df = create_demo_data(n_ad=35, n_hc=31)
    print(f"\nLoaded data: {len(df)} subjects ({len(df[df['group']=='AD'])} AD, {len(df[df['group']=='HC'])} HC)")
    
    # Generate figures
    plot_model_selection(df, OUTPUT_DIR / 'Fig_4_4_1a_model_selection_simple.png')
    plot_fit_metrics(df, OUTPUT_DIR / 'Fig_4_4_1b_fit_metrics_simple.png')
    
    # Print statistics
    print_statistics(df)
    
    print("\n" + "="*70)
    print("DONE! Check the generated PNG files.")
    print("="*70)


if __name__ == "__main__":
    main()
