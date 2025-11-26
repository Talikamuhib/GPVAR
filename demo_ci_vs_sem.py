"""
Visual Demonstration: 95% CI vs SEM
====================================
This script shows the difference between plotting with SEM vs 95% CI.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Generate example data
np.random.seed(42)
n_subjects_ad = 35
n_subjects_hc = 31
n_freqs = 100

freqs = np.linspace(0.5, 40, n_freqs)

# Simulate frequency responses (similar to real data)
# AD group: slightly higher in low frequencies
ad_responses = np.zeros((n_subjects_ad, n_freqs))
for i in range(n_subjects_ad):
    base = 2.0 + 0.8 * np.exp(-freqs/10) + 0.1 * np.random.randn()
    noise = 0.15 * np.random.randn(n_freqs)
    ad_responses[i, :] = base + noise

# HC group: slightly lower
hc_responses = np.zeros((n_subjects_hc, n_freqs))
for i in range(n_subjects_hc):
    base = 1.8 + 0.7 * np.exp(-freqs/10) + 0.1 * np.random.randn()
    noise = 0.15 * np.random.randn(n_freqs)
    hc_responses[i, :] = base + noise

# Compute statistics
ad_mean = ad_responses.mean(axis=0)
ad_std = ad_responses.std(axis=0)
ad_sem = ad_std / np.sqrt(n_subjects_ad)

hc_mean = hc_responses.mean(axis=0)
hc_std = hc_responses.std(axis=0)
hc_sem = hc_std / np.sqrt(n_subjects_hc)

# Compute 95% CI using t-distribution
def compute_ci(data, confidence=0.95):
    n = data.shape[0]
    mean = data.mean(axis=0)
    sem = stats.sem(data, axis=0)
    df = n - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    ci_margin = t_crit * sem
    return mean, mean - ci_margin, mean + ci_margin, t_crit

ad_mean_ci, ad_ci_lower, ad_ci_upper, ad_t_crit = compute_ci(ad_responses)
hc_mean_ci, hc_ci_lower, hc_ci_upper, hc_t_crit = compute_ci(hc_responses)

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

color_ad = '#E74C3C'
color_hc = '#3498DB'

# ===================================================================
# Panel A: Using SEM (OLD METHOD)
# ===================================================================
ax = axes[0, 0]

# AD
line_ad = ax.plot(freqs, ad_mean, color=color_ad, linewidth=3, 
                  label=f'AD (n={n_subjects_ad})', zorder=3)[0]
ax.fill_between(freqs, 
                ad_mean - ad_sem, 
                ad_mean + ad_sem,
                color=color_ad, alpha=0.25, zorder=2)

# HC
line_hc = ax.plot(freqs, hc_mean, color=color_hc, linewidth=3, 
                  label=f'HC (n={n_subjects_hc})', zorder=3)[0]
ax.fill_between(freqs, 
                hc_mean - hc_sem, 
                hc_mean + hc_sem,
                color=color_hc, alpha=0.25, zorder=2)

ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('Transfer Function Magnitude', fontsize=12, fontweight='bold')
ax.set_title('(A) OLD METHOD: Mean ± SEM', fontsize=14, fontweight='bold', loc='left')
ax.legend([line_ad, line_hc], [f'AD Mean (n={n_subjects_ad})', f'HC Mean (n={n_subjects_hc})'], 
          fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([freqs.min(), freqs.max()])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add text annotation
ax.text(0.5, 0.05, 'Shaded region = ±SEM\n(Narrower bands)', 
        transform=ax.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ===================================================================
# Panel B: Using 95% CI (NEW METHOD)
# ===================================================================
ax = axes[0, 1]

# AD
line_ad = ax.plot(freqs, ad_mean, color=color_ad, linewidth=3, 
                  label=f'AD (n={n_subjects_ad})', zorder=3)[0]
ax.fill_between(freqs, 
                ad_ci_lower, 
                ad_ci_upper,
                color=color_ad, alpha=0.25, zorder=2, label='AD 95% CI')

# HC
line_hc = ax.plot(freqs, hc_mean, color=color_hc, linewidth=3, 
                  label=f'HC (n={n_subjects_hc})', zorder=3)[0]
ax.fill_between(freqs, 
                hc_ci_lower, 
                hc_ci_upper,
                color=color_hc, alpha=0.25, zorder=2, label='HC 95% CI')

ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('Transfer Function Magnitude', fontsize=12, fontweight='bold')
ax.set_title('(B) NEW METHOD: Mean with 95% CI', fontsize=14, fontweight='bold', loc='left')
ax.legend([line_ad, line_hc], [f'AD Mean (n={n_subjects_ad})', f'HC Mean (n={n_subjects_hc})'], 
          fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([freqs.min(), freqs.max()])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add text annotation
ax.text(0.5, 0.05, f'Shaded region = 95% CI\n(~{ad_t_crit:.2f}× wider than SEM)', 
        transform=ax.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ===================================================================
# Panel C: Direct Comparison (AD group only)
# ===================================================================
ax = axes[1, 0]

# Plot mean
ax.plot(freqs, ad_mean, color=color_ad, linewidth=3, label='Mean', zorder=5)

# Plot SEM
ax.fill_between(freqs, 
                ad_mean - ad_sem, 
                ad_mean + ad_sem,
                color='orange', alpha=0.4, label='±SEM', zorder=3)

# Plot 95% CI
ax.fill_between(freqs, 
                ad_ci_lower, 
                ad_ci_upper,
                color='blue', alpha=0.3, label='95% CI', zorder=2)

ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('Transfer Function Magnitude', fontsize=12, fontweight='bold')
ax.set_title(f'(C) Comparison: SEM vs 95% CI (AD only, n={n_subjects_ad})', 
             fontsize=14, fontweight='bold', loc='left')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([freqs.min(), freqs.max()])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===================================================================
# Panel D: Statistical Information
# ===================================================================
ax = axes[1, 1]
ax.axis('off')

info_text = f"""
COMPARISON: SEM vs 95% CONFIDENCE INTERVAL

AD Group Statistics (n = {n_subjects_ad}):
{'='*45}
Degrees of freedom:        df = {n_subjects_ad - 1}
t-critical value (95%):    t = {ad_t_crit:.4f}
Width multiplier:          95% CI ≈ {ad_t_crit:.2f} × SEM

HC Group Statistics (n = {n_subjects_hc}):
{'='*45}
Degrees of freedom:        df = {n_subjects_hc - 1}
t-critical value (95%):    t = {hc_t_crit:.4f}
Width multiplier:          95% CI ≈ {hc_t_crit:.2f} × SEM


KEY DIFFERENCES:
{'='*45}
Standard Error (SEM):
  • Formula: SEM = σ / √n
  • Represents: Uncertainty in mean estimate
  • Width: Narrower
  • Use: Describes sampling variability

95% Confidence Interval (CI):
  • Formula: CI = mean ± t_crit × SEM
  • Represents: 95% probability range for true mean
  • Width: ~2× wider than SEM
  • Use: Statistical inference about population

INTERPRETATION:
{'='*45}
We are 95% confident that the TRUE population mean
lies within the shaded confidence interval region.

This is the STANDARD for publication-quality figures!
"""

ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

ax.set_title('(D) Statistical Details', fontsize=14, fontweight='bold', loc='left')

# Overall title
fig.suptitle('Demonstration: Standard Error (SEM) vs 95% Confidence Interval (CI)\n' + 
             'Updated Analysis Now Uses 95% CI for All Visualizations',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
output_path = '/workspace/SEM_vs_CI_demonstration.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved demonstration figure: {output_path}")
print(f"\nKey findings:")
print(f"  - AD group (n={n_subjects_ad}): 95% CI is {ad_t_crit:.2f}× wider than SEM")
print(f"  - HC group (n={n_subjects_hc}): 95% CI is {hc_t_crit:.2f}× wider than SEM")
print(f"  - Visual difference: Shaded regions are approximately 2× wider")
print(f"  - Statistical advantage: Proper confidence statements about population")
print(f"\n🎓 This is the STANDARD for thesis-quality figures!")

plt.show()
