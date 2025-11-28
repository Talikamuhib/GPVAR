"""
Validation Script: Prove that Consensus Matrix is a Valid Consensus of ALL Subjects

This script demonstrates and PROVES that:
1. AD consensus = fraction of AD subjects with each edge
2. HC consensus = fraction of HC subjects with each edge  
3. OVERALL consensus = fraction of ALL subjects (AD + HC) with each edge

The OVERALL consensus is computed by:
1. Taking ALL individual subject binary matrices (AD subjects + HC subjects)
2. Computing consensus across ALL subjects together
3. This is NOT a weighted average of group consensus - it's a TRUE consensus

Mathematical Definition:
- C_overall[i,j] = (# of ALL subjects with edge i,j) / (total # of subjects)
                 = (k_AD + k_HC) / (n_AD + n_HC)

Author: Consensus Matrix Validation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')


class TrueConsensusValidator:
    """
    Validate that consensus matrices are computed correctly from individual subjects.
    
    The OVERALL consensus is the TRUE consensus of ALL subjects (AD + HC),
    computed directly from individual correlation matrices, NOT by averaging
    group consensus matrices.
    """
    
    def __init__(self):
        # Individual subject binary matrices
        self.ad_binary_matrices: Optional[np.ndarray] = None  # (n_ad, n_channels, n_channels)
        self.hc_binary_matrices: Optional[np.ndarray] = None  # (n_hc, n_channels, n_channels)
        
        # Individual subject weight matrices (correlation values)
        self.ad_weight_matrices: Optional[np.ndarray] = None
        self.hc_weight_matrices: Optional[np.ndarray] = None
        
        # Group consensus matrices
        self.ad_consensus: Optional[np.ndarray] = None
        self.hc_consensus: Optional[np.ndarray] = None
        
        # OVERALL consensus (true consensus of ALL subjects)
        self.overall_consensus: Optional[np.ndarray] = None
        self.overall_weights: Optional[np.ndarray] = None
        
        self.n_ad: int = 0
        self.n_hc: int = 0
        self.n_channels: int = 0
        
    def set_individual_matrices(self,
                                 ad_binary: np.ndarray,
                                 hc_binary: np.ndarray,
                                 ad_weights: Optional[np.ndarray] = None,
                                 hc_weights: Optional[np.ndarray] = None):
        """
        Set individual subject matrices.
        
        Parameters
        ----------
        ad_binary : np.ndarray
            Shape (n_ad_subjects, n_channels, n_channels) - binary connectivity for each AD subject
        hc_binary : np.ndarray
            Shape (n_hc_subjects, n_channels, n_channels) - binary connectivity for each HC subject
        ad_weights : np.ndarray, optional
            Shape (n_ad_subjects, n_channels, n_channels) - correlation values for each AD subject
        hc_weights : np.ndarray, optional
            Shape (n_hc_subjects, n_channels, n_channels) - correlation values for each HC subject
        """
        self.ad_binary_matrices = ad_binary
        self.hc_binary_matrices = hc_binary
        self.ad_weight_matrices = ad_weights if ad_weights is not None else ad_binary
        self.hc_weight_matrices = hc_weights if hc_weights is not None else hc_binary
        
        self.n_ad = ad_binary.shape[0]
        self.n_hc = hc_binary.shape[0]
        self.n_channels = ad_binary.shape[1]
        
        print(f"Loaded individual matrices:")
        print(f"  - AD: {self.n_ad} subjects × {self.n_channels} channels")
        print(f"  - HC: {self.n_hc} subjects × {self.n_channels} channels")
        
    def compute_group_consensus(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute consensus for each group separately.
        
        AD Consensus[i,j] = (# AD subjects with edge i,j) / n_AD
        HC Consensus[i,j] = (# HC subjects with edge i,j) / n_HC
        """
        # AD consensus: fraction of AD subjects with each edge
        self.ad_consensus = np.mean(self.ad_binary_matrices, axis=0)
        
        # HC consensus: fraction of HC subjects with each edge
        self.hc_consensus = np.mean(self.hc_binary_matrices, axis=0)
        
        print(f"\nGroup consensus computed:")
        print(f"  - AD consensus: mean={np.mean(self.ad_consensus):.4f}")
        print(f"  - HC consensus: mean={np.mean(self.hc_consensus):.4f}")
        
        return self.ad_consensus, self.hc_consensus
    
    def compute_overall_consensus(self) -> np.ndarray:
        """
        Compute TRUE OVERALL consensus from ALL individual subjects.
        
        This combines ALL binary matrices (AD + HC) and computes:
        Overall Consensus[i,j] = (# ALL subjects with edge i,j) / (n_AD + n_HC)
        
        This is the TRUE consensus, NOT a weighted average of group consensus.
        """
        # Stack all binary matrices together
        all_binary = np.concatenate([
            self.ad_binary_matrices,
            self.hc_binary_matrices
        ], axis=0)
        
        # Compute consensus across ALL subjects
        self.overall_consensus = np.mean(all_binary, axis=0)
        
        # For weights: use Fisher-z average across all subjects
        all_weights = np.concatenate([
            self.ad_weight_matrices,
            self.hc_weight_matrices
        ], axis=0)
        
        # Fisher-z transform, average, then back-transform
        z_all = np.arctanh(np.clip(all_weights, -0.999, 0.999))
        z_mean = np.mean(z_all, axis=0)
        self.overall_weights = np.abs(np.tanh(z_mean))
        np.fill_diagonal(self.overall_weights, 0)
        
        n_total = self.n_ad + self.n_hc
        print(f"\nOVERALL consensus computed from ALL {n_total} subjects:")
        print(f"  - Overall consensus mean: {np.mean(self.overall_consensus):.4f}")
        
        return self.overall_consensus
    
    def verify_consensus_definition(self) -> Dict:
        """
        Verify that consensus matrices follow the correct mathematical definition.
        
        Consensus[i,j] = (# subjects with edge) / (total subjects)
        
        This verifies:
        1. AD consensus = sum(AD binary) / n_AD
        2. HC consensus = sum(HC binary) / n_HC
        3. Overall consensus = sum(ALL binary) / (n_AD + n_HC)
        """
        n_total = self.n_ad + self.n_hc
        
        # Verify AD consensus
        expected_ad = np.sum(self.ad_binary_matrices, axis=0) / self.n_ad
        ad_error = np.max(np.abs(self.ad_consensus - expected_ad))
        
        # Verify HC consensus
        expected_hc = np.sum(self.hc_binary_matrices, axis=0) / self.n_hc
        hc_error = np.max(np.abs(self.hc_consensus - expected_hc))
        
        # Verify Overall consensus
        all_binary = np.concatenate([self.ad_binary_matrices, self.hc_binary_matrices], axis=0)
        expected_overall = np.sum(all_binary, axis=0) / n_total
        overall_error = np.max(np.abs(self.overall_consensus - expected_overall))
        
        # Also verify the relationship between group and overall consensus
        # Overall = (n_AD * C_AD + n_HC * C_HC) / (n_AD + n_HC)
        # This should be TRUE because it's mathematically equivalent
        weighted_avg = (self.n_ad * self.ad_consensus + self.n_hc * self.hc_consensus) / n_total
        equivalence_error = np.max(np.abs(self.overall_consensus - weighted_avg))
        
        verification = {
            'ad_consensus_correct': ad_error < 1e-10,
            'ad_max_error': float(ad_error),
            'hc_consensus_correct': hc_error < 1e-10,
            'hc_max_error': float(hc_error),
            'overall_consensus_correct': overall_error < 1e-10,
            'overall_max_error': float(overall_error),
            'weighted_avg_equivalent': equivalence_error < 1e-10,
            'equivalence_error': float(equivalence_error),
            'n_ad': self.n_ad,
            'n_hc': self.n_hc,
            'n_total': n_total
        }
        
        return verification
    
    def compute_statistics(self) -> Dict:
        """Compute comprehensive statistics."""
        triu_idx = np.triu_indices(self.n_channels, k=1)
        n_possible = len(triu_idx[0])
        
        def matrix_stats(C, name):
            values = C[triu_idx]
            return {
                f'{name}_mean': float(np.mean(values)),
                f'{name}_std': float(np.std(values)),
                f'{name}_min': float(np.min(values)),
                f'{name}_max': float(np.max(values)),
                f'{name}_median': float(np.median(values)),
                f'{name}_n_positive': int(np.sum(values > 0)),
                f'{name}_n_majority': int(np.sum(values > 0.5)),
                f'{name}_n_unanimous': int(np.sum(values >= 0.999)),
                f'{name}_sparsity_pct': float(np.sum(values > 0) / n_possible * 100)
            }
        
        stats_dict = {
            'n_channels': self.n_channels,
            'n_possible_edges': n_possible,
            'n_ad_subjects': self.n_ad,
            'n_hc_subjects': self.n_hc,
            'n_total_subjects': self.n_ad + self.n_hc,
            **matrix_stats(self.ad_consensus, 'AD'),
            **matrix_stats(self.hc_consensus, 'HC'),
            **matrix_stats(self.overall_consensus, 'Overall')
        }
        
        # Correlations
        ad_vals = self.ad_consensus[triu_idx]
        hc_vals = self.hc_consensus[triu_idx]
        overall_vals = self.overall_consensus[triu_idx]
        
        stats_dict['correlation_AD_HC'] = float(np.corrcoef(ad_vals, hc_vals)[0, 1])
        stats_dict['correlation_AD_Overall'] = float(np.corrcoef(ad_vals, overall_vals)[0, 1])
        stats_dict['correlation_HC_Overall'] = float(np.corrcoef(hc_vals, overall_vals)[0, 1])
        
        return stats_dict
    
    def analyze_subject_contribution(self) -> Dict:
        """
        Analyze how each subject contributes to the overall consensus.
        
        This shows that EVERY subject (AD and HC) contributes equally
        to the overall consensus.
        """
        n_total = self.n_ad + self.n_hc
        triu_idx = np.triu_indices(self.n_channels, k=1)
        
        # Count edges per subject
        ad_edges_per_subject = [np.sum(self.ad_binary_matrices[i][triu_idx]) 
                                for i in range(self.n_ad)]
        hc_edges_per_subject = [np.sum(self.hc_binary_matrices[i][triu_idx]) 
                                for i in range(self.n_hc)]
        
        analysis = {
            'ad_mean_edges': float(np.mean(ad_edges_per_subject)),
            'ad_std_edges': float(np.std(ad_edges_per_subject)),
            'hc_mean_edges': float(np.mean(hc_edges_per_subject)),
            'hc_std_edges': float(np.std(hc_edges_per_subject)),
            'ad_edges_per_subject': ad_edges_per_subject,
            'hc_edges_per_subject': hc_edges_per_subject,
            'weight_per_ad_subject': 1.0 / n_total,  # Each subject contributes equally
            'weight_per_hc_subject': 1.0 / n_total,
            'total_ad_weight': self.n_ad / n_total,
            'total_hc_weight': self.n_hc / n_total
        }
        
        return analysis
    
    def plot_full_validation(self,
                             output_path: str,
                             show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive validation figure showing TRUE consensus.
        """
        fig = plt.figure(figsize=(24, 20))
        
        triu_idx = np.triu_indices(self.n_channels, k=1)
        n_total = self.n_ad + self.n_hc
        
        # ===== ROW 1: Individual subject matrices (samples) =====
        # Show 3 AD subjects
        for i in range(3):
            ax = fig.add_subplot(5, 6, i+1)
            im = ax.imshow(self.ad_binary_matrices[i], cmap='Greys', vmin=0, vmax=1)
            ax.set_title(f'AD Subject {i+1}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Show 3 HC subjects
        for i in range(3):
            ax = fig.add_subplot(5, 6, i+4)
            im = ax.imshow(self.hc_binary_matrices[i], cmap='Greys', vmin=0, vmax=1)
            ax.set_title(f'HC Subject {i+1}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # ===== ROW 2: Group Consensus Matrices =====
        ax_ad = fig.add_subplot(5, 6, 7)
        im_ad = ax_ad.imshow(self.ad_consensus, cmap='hot', vmin=0, vmax=1)
        ax_ad.set_title(f'AD CONSENSUS\n(N={self.n_ad} subjects)', fontsize=12, fontweight='bold', color='red')
        plt.colorbar(im_ad, ax=ax_ad, fraction=0.046)
        
        ax_hc = fig.add_subplot(5, 6, 8)
        im_hc = ax_hc.imshow(self.hc_consensus, cmap='hot', vmin=0, vmax=1)
        ax_hc.set_title(f'HC CONSENSUS\n(N={self.n_hc} subjects)', fontsize=12, fontweight='bold', color='blue')
        plt.colorbar(im_hc, ax=ax_hc, fraction=0.046)
        
        ax_overall = fig.add_subplot(5, 6, 9)
        im_overall = ax_overall.imshow(self.overall_consensus, cmap='hot', vmin=0, vmax=1)
        ax_overall.set_title(f'OVERALL CONSENSUS\n(ALL {n_total} subjects)', 
                            fontsize=12, fontweight='bold', color='green')
        plt.colorbar(im_overall, ax=ax_overall, fraction=0.046)
        
        # Difference matrices
        ax_diff_adhc = fig.add_subplot(5, 6, 10)
        diff_adhc = self.ad_consensus - self.hc_consensus
        vmax = max(abs(diff_adhc.min()), abs(diff_adhc.max()))
        im_diff = ax_diff_adhc.imshow(diff_adhc, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_diff_adhc.set_title('AD - HC', fontsize=11)
        plt.colorbar(im_diff, ax=ax_diff_adhc, fraction=0.046)
        
        ax_diff_ad_overall = fig.add_subplot(5, 6, 11)
        diff_ad_overall = self.ad_consensus - self.overall_consensus
        im = ax_diff_ad_overall.imshow(diff_ad_overall, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax_diff_ad_overall.set_title('AD - Overall', fontsize=11)
        plt.colorbar(im, ax=ax_diff_ad_overall, fraction=0.046)
        
        ax_diff_hc_overall = fig.add_subplot(5, 6, 12)
        diff_hc_overall = self.hc_consensus - self.overall_consensus
        im = ax_diff_hc_overall.imshow(diff_hc_overall, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax_diff_hc_overall.set_title('HC - Overall', fontsize=11)
        plt.colorbar(im, ax=ax_diff_hc_overall, fraction=0.046)
        
        # ===== ROW 3: Verification Scatter Plots =====
        ad_vals = self.ad_consensus[triu_idx]
        hc_vals = self.hc_consensus[triu_idx]
        overall_vals = self.overall_consensus[triu_idx]
        
        ax_scatter1 = fig.add_subplot(5, 6, 13)
        ax_scatter1.scatter(ad_vals, overall_vals, alpha=0.3, s=5, c='red')
        ax_scatter1.plot([0, 1], [0, 1], 'k--', linewidth=2)
        r = np.corrcoef(ad_vals, overall_vals)[0, 1]
        ax_scatter1.set_xlabel('AD Consensus')
        ax_scatter1.set_ylabel('Overall Consensus')
        ax_scatter1.set_title(f'AD vs Overall\nr = {r:.4f}', fontsize=11)
        ax_scatter1.set_xlim(-0.05, 1.05)
        ax_scatter1.set_ylim(-0.05, 1.05)
        ax_scatter1.grid(True, alpha=0.3)
        
        ax_scatter2 = fig.add_subplot(5, 6, 14)
        ax_scatter2.scatter(hc_vals, overall_vals, alpha=0.3, s=5, c='blue')
        ax_scatter2.plot([0, 1], [0, 1], 'k--', linewidth=2)
        r = np.corrcoef(hc_vals, overall_vals)[0, 1]
        ax_scatter2.set_xlabel('HC Consensus')
        ax_scatter2.set_ylabel('Overall Consensus')
        ax_scatter2.set_title(f'HC vs Overall\nr = {r:.4f}', fontsize=11)
        ax_scatter2.set_xlim(-0.05, 1.05)
        ax_scatter2.set_ylim(-0.05, 1.05)
        ax_scatter2.grid(True, alpha=0.3)
        
        ax_scatter3 = fig.add_subplot(5, 6, 15)
        ax_scatter3.scatter(ad_vals, hc_vals, alpha=0.3, s=5, c='purple')
        ax_scatter3.plot([0, 1], [0, 1], 'k--', linewidth=2)
        r = np.corrcoef(ad_vals, hc_vals)[0, 1]
        ax_scatter3.set_xlabel('AD Consensus')
        ax_scatter3.set_ylabel('HC Consensus')
        ax_scatter3.set_title(f'AD vs HC\nr = {r:.4f}', fontsize=11)
        ax_scatter3.set_xlim(-0.05, 1.05)
        ax_scatter3.set_ylim(-0.05, 1.05)
        ax_scatter3.grid(True, alpha=0.3)
        
        # MATHEMATICAL VERIFICATION
        ax_verify = fig.add_subplot(5, 6, 16)
        # Overall should equal (n_AD * C_AD + n_HC * C_HC) / n_total
        expected = (self.n_ad * ad_vals + self.n_hc * hc_vals) / n_total
        ax_verify.scatter(expected, overall_vals, alpha=0.5, s=10, c='green')
        ax_verify.plot([0, 1], [0, 1], 'r-', linewidth=3)
        max_error = np.max(np.abs(expected - overall_vals))
        r = np.corrcoef(expected, overall_vals)[0, 1]
        ax_verify.set_xlabel('Expected from formula')
        ax_verify.set_ylabel('Actual Overall')
        ax_verify.set_title(f'PROOF: Formula = Reality\nr = {r:.10f}\nmax err = {max_error:.2e}', 
                           fontsize=11, fontweight='bold', color='green')
        ax_verify.set_xlim(-0.05, 1.05)
        ax_verify.set_ylim(-0.05, 1.05)
        ax_verify.grid(True, alpha=0.3)
        
        # Edge count histogram
        ax_hist1 = fig.add_subplot(5, 6, 17)
        analysis = self.analyze_subject_contribution()
        ax_hist1.hist(analysis['ad_edges_per_subject'], bins=20, alpha=0.7, 
                     color='red', label=f'AD (N={self.n_ad})')
        ax_hist1.hist(analysis['hc_edges_per_subject'], bins=20, alpha=0.7, 
                     color='blue', label=f'HC (N={self.n_hc})')
        ax_hist1.set_xlabel('# Edges per Subject')
        ax_hist1.set_ylabel('# Subjects')
        ax_hist1.set_title('Edges per Subject', fontsize=11)
        ax_hist1.legend()
        
        # Consensus distribution
        ax_hist2 = fig.add_subplot(5, 6, 18)
        ax_hist2.hist(ad_vals[ad_vals > 0], bins=50, alpha=0.6, color='red', label='AD')
        ax_hist2.hist(hc_vals[hc_vals > 0], bins=50, alpha=0.6, color='blue', label='HC')
        ax_hist2.hist(overall_vals[overall_vals > 0], bins=50, alpha=0.6, color='green', label='Overall')
        ax_hist2.set_xlabel('Consensus Value')
        ax_hist2.set_ylabel('Frequency')
        ax_hist2.set_title('Consensus Distributions', fontsize=11)
        ax_hist2.legend()
        
        # ===== ROW 4: Subject Contribution Visualization =====
        ax_contrib = fig.add_subplot(5, 3, 10)
        
        # Create bar showing contribution of each group
        bars = ax_contrib.bar(['AD Subjects', 'HC Subjects'], 
                             [self.n_ad, self.n_hc],
                             color=['red', 'blue'], alpha=0.7)
        ax_contrib.set_ylabel('# Subjects')
        ax_contrib.set_title(f'Subject Contribution to Overall Consensus\n'
                            f'Each subject contributes 1/{n_total} = {1/n_total:.4f}', fontsize=11)
        
        # Add weight annotation
        for bar, n in zip(bars, [self.n_ad, self.n_hc]):
            height = bar.get_height()
            ax_contrib.annotate(f'{n} subjects\n({n/n_total*100:.1f}% weight)',
                              xy=(bar.get_x() + bar.get_width()/2, height),
                              ha='center', va='bottom', fontsize=10)
        
        # Edge-by-edge comparison
        ax_edge = fig.add_subplot(5, 3, 11)
        np.random.seed(42)
        sample_idx = np.random.choice(len(ad_vals), 80, replace=False)
        sample_idx = np.sort(sample_idx)
        x = np.arange(len(sample_idx))
        
        ax_edge.plot(x, ad_vals[sample_idx], 'r-', alpha=0.7, linewidth=1, label='AD')
        ax_edge.plot(x, hc_vals[sample_idx], 'b-', alpha=0.7, linewidth=1, label='HC')
        ax_edge.plot(x, overall_vals[sample_idx], 'g-', alpha=0.9, linewidth=2, label='Overall')
        ax_edge.set_xlabel('Edge Index (sampled)')
        ax_edge.set_ylabel('Consensus')
        ax_edge.set_title('Edge-by-Edge: Overall is Between AD & HC', fontsize=11)
        ax_edge.legend()
        ax_edge.grid(True, alpha=0.3)
        
        # Summary text
        ax_summary = fig.add_subplot(5, 3, 12)
        ax_summary.axis('off')
        
        verification = self.verify_consensus_definition()
        stats = self.compute_statistics()
        
        summary_text = f"""
════════════════════════════════════════════════════════════════
             TRUE CONSENSUS VALIDATION - PROOF
════════════════════════════════════════════════════════════════

WHAT IS OVERALL CONSENSUS?
  It is the TRUE consensus computed from ALL individual
  subject correlation matrices (AD + HC together).

  Overall[i,j] = (# ALL subjects with edge i,j) / {n_total}

SAMPLE:
  • AD subjects: {self.n_ad}
  • HC subjects: {self.n_hc}  
  • TOTAL: {n_total}

MATHEMATICAL VERIFICATION:
  ✓ AD consensus correct: {verification['ad_consensus_correct']}
  ✓ HC consensus correct: {verification['hc_consensus_correct']}
  ✓ Overall consensus correct: {verification['overall_consensus_correct']}
  ✓ Max error: {verification['overall_max_error']:.2e}

EQUIVALENCE (proven mathematically):
  Overall = (n_AD × C_AD + n_HC × C_HC) / (n_AD + n_HC)
  Error: {verification['equivalence_error']:.2e}

NATURAL SPARSITY (NOT cut artificially):
  • AD: {stats['AD_sparsity_pct']:.2f}% edges present
  • HC: {stats['HC_sparsity_pct']:.2f}% edges present
  • Overall: {stats['Overall_sparsity_pct']:.2f}% edges present

CORRELATIONS:
  • AD vs HC: r = {stats['correlation_AD_HC']:.4f}
  • AD vs Overall: r = {stats['correlation_AD_Overall']:.4f}
  • HC vs Overall: r = {stats['correlation_HC_Overall']:.4f}

CONCLUSION:
  The Overall consensus correctly represents the
  connectivity pattern across ALL {n_total} subjects.
  Each subject contributes equally (1/{n_total}).

════════════════════════════════════════════════════════════════
"""
        ax_summary.text(0.02, 0.98, summary_text, transform=ax_summary.transAxes, fontsize=9,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # ===== ROW 5: Process Flow Diagram =====
        ax_flow = fig.add_subplot(5, 1, 5)
        ax_flow.axis('off')
        
        flow_text = """
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              HOW OVERALL CONSENSUS IS COMPUTED                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                                                      │
│   STEP 1: Individual Subject Correlation Matrices                                                                                                    │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐ ┌─────────┐ ┌─────────┐                                                                     │
│   │ AD #1   │ │ AD #2   │ │ AD #N   │       │ HC #1   │ │ HC #2   │ │ HC #M   │                                                                     │
│   │ Corr    │ │ Corr    │ │ Corr    │  ...  │ Corr    │ │ Corr    │ │ Corr    │                                                                     │
│   │ Matrix  │ │ Matrix  │ │ Matrix  │       │ Matrix  │ │ Matrix  │ │ Matrix  │                                                                     │
│   └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘ └────┬────┘ └────┬────┘                                                                     │
│        │           │           │                 │           │           │                                                                           │
│   STEP 2: Threshold to Binary (edge present = 1, absent = 0)                                                                                         │
│        │           │           │                 │           │           │                                                                           │
│        ▼           ▼           ▼                 ▼           ▼           ▼                                                                           │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐ ┌─────────┐ ┌─────────┐                                                                     │
│   │ Binary  │ │ Binary  │ │ Binary  │  ...  │ Binary  │ │ Binary  │ │ Binary  │                                                                     │
│   │   #1    │ │   #2    │ │   #N    │       │   #1    │ │   #2    │ │   #M    │                                                                     │
│   └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘ └────┬────┘ └────┬────┘                                                                     │
│        │           │           │                 │           │           │                                                                           │
│   STEP 3: Compute Consensus = mean(all binary matrices)                                                                                              │
│        │           │           │                 │           │           │                                                                           │
│        └───────────┴─────┬─────┴─────────────────┴───────────┴───────────┘                                                                           │
│                          │                                                                                                                           │
│                          ▼                                                                                                                           │
│                   ┌─────────────────────────────────────────┐                                                                                        │
│                   │     OVERALL CONSENSUS MATRIX            │                                                                                        │
│                   │                                         │                                                                                        │
│                   │  C[i,j] = (# subjects with edge i,j)    │                                                                                        │
│                   │           ─────────────────────────     │                                                                                        │
│                   │              (N + M) total subjects     │                                                                                        │
│                   │                                         │                                                                                        │
│                   │  Range: 0 (no subjects) to 1 (all)      │                                                                                        │
│                   └─────────────────────────────────────────┘                                                                                        │
│                                                                                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
        ax_flow.text(0.5, 0.5, flow_text, transform=ax_flow.transAxes, fontsize=8,
                    verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.suptitle('PROOF: Overall Consensus = True Consensus of ALL Subjects (AD + HC)',
                    fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nValidation figure saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_proof_report(self, output_path: str) -> str:
        """Generate detailed markdown proof report."""
        verification = self.verify_consensus_definition()
        stats = self.compute_statistics()
        analysis = self.analyze_subject_contribution()
        n_total = self.n_ad + self.n_hc
        
        triu_idx = np.triu_indices(self.n_channels, k=1)
        diff_ad_hc = (self.ad_consensus - self.hc_consensus)[triu_idx]
        
        lines = [
            "# True Consensus Matrix Validation Report",
            "",
            "## What is Overall Consensus?",
            "",
            "The **Overall Consensus** is the TRUE consensus computed from ALL individual",
            "subject correlation matrices. It is NOT simply averaging the group consensus",
            "matrices - it goes back to the individual subject level.",
            "",
            "### Definition",
            "",
            "```",
            "Overall Consensus[i,j] = (number of ALL subjects with edge i,j)",
            "                         ────────────────────────────────────",
            "                              (total number of subjects)",
            "```",
            "",
            "### How It's Computed",
            "",
            "1. **Start with individual subject correlation matrices**",
            f"   - {self.n_ad} AD subjects, each with a {self.n_channels}×{self.n_channels} correlation matrix",
            f"   - {self.n_hc} HC subjects, each with a {self.n_channels}×{self.n_channels} correlation matrix",
            "",
            "2. **Threshold each matrix to binary** (edge present = 1, absent = 0)",
            f"   - Results in {self.n_ad} AD binary matrices",
            f"   - Results in {self.n_hc} HC binary matrices",
            "",
            "3. **Stack ALL binary matrices together**",
            f"   - Total: {n_total} binary matrices",
            "",
            "4. **Compute mean across all subjects**",
            "   - Overall[i,j] = mean(all binary matrices at position i,j)",
            "   - This equals (count of subjects with edge) / (total subjects)",
            "",
            "## Mathematical Verification",
            "",
            "### Formula Equivalence (Proven)",
            "",
            "The overall consensus computed from individual matrices equals:",
            "",
            "```",
            f"Overall = (n_AD × C_AD + n_HC × C_HC) / (n_AD + n_HC)",
            f"        = ({self.n_ad} × C_AD + {self.n_hc} × C_HC) / {n_total}",
            "```",
            "",
            "**Proof:**",
            "- Let k_AD = count of AD subjects with edge (i,j)",
            "- Let k_HC = count of HC subjects with edge (i,j)",
            "- C_AD[i,j] = k_AD / n_AD",
            "- C_HC[i,j] = k_HC / n_HC",
            "",
            "Then:",
            "```",
            "Overall = (k_AD + k_HC) / (n_AD + n_HC)",
            "        = (n_AD × k_AD/n_AD + n_HC × k_HC/n_HC) / (n_AD + n_HC)",
            "        = (n_AD × C_AD + n_HC × C_HC) / (n_AD + n_HC)  ✓",
            "```",
            "",
            "### Numerical Verification Results",
            "",
            "| Check | Result | Max Error |",
            "|-------|--------|-----------|",
            f"| AD consensus formula | {'✓ PASS' if verification['ad_consensus_correct'] else '✗ FAIL'} | {verification['ad_max_error']:.2e} |",
            f"| HC consensus formula | {'✓ PASS' if verification['hc_consensus_correct'] else '✗ FAIL'} | {verification['hc_max_error']:.2e} |",
            f"| Overall consensus formula | {'✓ PASS' if verification['overall_consensus_correct'] else '✗ FAIL'} | {verification['overall_max_error']:.2e} |",
            f"| Weighted average equivalence | {'✓ PASS' if verification['weighted_avg_equivalent'] else '✗ FAIL'} | {verification['equivalence_error']:.2e} |",
            "",
            "## Sample Statistics",
            "",
            f"| Group | N Subjects | Mean Edges/Subject |",
            f"|-------|------------|-------------------|",
            f"| AD | {self.n_ad} | {analysis['ad_mean_edges']:.1f} ± {analysis['ad_std_edges']:.1f} |",
            f"| HC | {self.n_hc} | {analysis['hc_mean_edges']:.1f} ± {analysis['hc_std_edges']:.1f} |",
            f"| **Total** | **{n_total}** | - |",
            "",
            "## Consensus Matrix Statistics",
            "",
            "### Sparsity (Natural, NOT artificially cut)",
            "",
            "| Group | Edges Present | Sparsity | Majority (>50%) | Unanimous |",
            "|-------|--------------|----------|-----------------|-----------|",
            f"| AD | {stats['AD_n_positive']:,} | {stats['AD_sparsity_pct']:.2f}% | {stats['AD_n_majority']:,} | {stats['AD_n_unanimous']:,} |",
            f"| HC | {stats['HC_n_positive']:,} | {stats['HC_sparsity_pct']:.2f}% | {stats['HC_n_majority']:,} | {stats['HC_n_unanimous']:,} |",
            f"| **Overall** | **{stats['Overall_n_positive']:,}** | **{stats['Overall_sparsity_pct']:.2f}%** | **{stats['Overall_n_majority']:,}** | **{stats['Overall_n_unanimous']:,}** |",
            "",
            "### Consensus Value Distribution",
            "",
            "| Statistic | AD | HC | Overall |",
            "|-----------|----|----|---------|",
            f"| Mean | {stats['AD_mean']:.4f} | {stats['HC_mean']:.4f} | {stats['Overall_mean']:.4f} |",
            f"| Std | {stats['AD_std']:.4f} | {stats['HC_std']:.4f} | {stats['Overall_std']:.4f} |",
            f"| Median | {stats['AD_median']:.4f} | {stats['HC_median']:.4f} | {stats['Overall_median']:.4f} |",
            f"| Max | {stats['AD_max']:.4f} | {stats['HC_max']:.4f} | {stats['Overall_max']:.4f} |",
            "",
            "### Correlations Between Matrices",
            "",
            f"| Comparison | Correlation |",
            f"|------------|-------------|",
            f"| AD vs HC | r = {stats['correlation_AD_HC']:.4f} |",
            f"| AD vs Overall | r = {stats['correlation_AD_Overall']:.4f} |",
            f"| HC vs Overall | r = {stats['correlation_HC_Overall']:.4f} |",
            "",
            "## Group Differences (AD vs HC)",
            "",
            f"- Edges stronger in AD (diff > 0.1): **{int(np.sum(diff_ad_hc > 0.1)):,}**",
            f"- Edges stronger in HC (diff < -0.1): **{int(np.sum(diff_ad_hc < -0.1)):,}**",
            f"- Similar edges (|diff| ≤ 0.1): **{int(np.sum(np.abs(diff_ad_hc) <= 0.1)):,}**",
            f"- Mean difference (AD - HC): **{np.mean(diff_ad_hc):.4f}**",
            "",
            "## Interpretation for Alzheimer's Disease Research",
            "",
            "### What the Overall Consensus Shows",
            "",
            "The **Overall Consensus Matrix** represents the **general brain connectivity**",
            "pattern across the entire study population (both AD and HC). This is valuable for:",
            "",
            "1. **Identifying Core Network Structure**",
            "   - Edges with high overall consensus (>0.7) represent connections",
            "     present in most subjects regardless of disease status",
            "   - These form the stable \"backbone\" of brain connectivity",
            "",
            "2. **GP-VAR Analysis**",
            "   - The graph frequency spectrum from the overall consensus",
            "     captures variability across the entire population",
            "   - Natural sparsity is preserved (not cut at arbitrary threshold)",
            "",
            "3. **Comparing Groups to Overall**",
            "   - AD vs Overall: Shows how AD connectivity deviates from the norm",
            "   - HC vs Overall: Shows how healthy connectivity relates to overall",
            "   - If AD correlation with Overall is lower than HC, this indicates",
            "     AD patients show more deviation from the typical pattern",
            "",
            "### AD-Specific Findings",
            "",
            f"Based on the correlations:",
            f"- AD-Overall correlation: r = {stats['correlation_AD_Overall']:.4f}",
            f"- HC-Overall correlation: r = {stats['correlation_HC_Overall']:.4f}",
            "",
            "## Conclusion",
            "",
            f"The **Overall Consensus Matrix** is mathematically proven to be the true",
            f"consensus of ALL {n_total} subjects ({self.n_ad} AD + {self.n_hc} HC).",
            f"Each subject contributes equally (weight = 1/{n_total} = {1/n_total:.4f}).",
            "",
            "The consensus correctly captures:",
            "- Group-level connectivity patterns",
            "- Natural sparsity from the thresholding process",
            "- Valid graph structure for spectral analysis",
            ""
        ]
        
        Path(output_path).write_text("\n".join(lines))
        print(f"Proof report saved to: {output_path}")
        
        return output_path


def create_synthetic_demo(output_dir: str = "./consensus_validation"):
    """
    Create demonstration with synthetic data showing true consensus computation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRUE CONSENSUS VALIDATION DEMO")
    print("="*70)
    
    np.random.seed(42)
    
    # Parameters
    n_ad = 35
    n_hc = 31
    n_channels = 64  # Smaller for demo
    
    print(f"\nGenerating synthetic data:")
    print(f"  - {n_ad} AD subjects")
    print(f"  - {n_hc} HC subjects")
    print(f"  - {n_channels} channels")
    
    # Create base connectivity pattern
    base_prob = np.zeros((n_channels, n_channels))
    
    # Modular structure
    n_per_region = n_channels // 4
    for i in range(4):
        for j in range(4):
            r1_start, r1_end = i * n_per_region, (i+1) * n_per_region
            r2_start, r2_end = j * n_per_region, (j+1) * n_per_region
            
            if i == j:  # Intra-region (high probability)
                base_prob[r1_start:r1_end, r2_start:r2_end] = 0.7
            else:  # Inter-region (lower probability)
                base_prob[r1_start:r1_end, r2_start:r2_end] = 0.3
    
    base_prob = (base_prob + base_prob.T) / 2
    np.fill_diagonal(base_prob, 0)
    
    # Generate individual subject binary matrices
    # AD: reduced connectivity (lower probability of edges)
    ad_prob = np.clip(base_prob * 0.8, 0, 1)  # 20% reduction
    ad_binary = np.random.binomial(1, ad_prob, size=(n_ad, n_channels, n_channels))
    # Make symmetric
    for i in range(n_ad):
        ad_binary[i] = np.triu(ad_binary[i], k=1)
        ad_binary[i] = ad_binary[i] + ad_binary[i].T
    
    # HC: normal connectivity
    hc_prob = base_prob
    hc_binary = np.random.binomial(1, hc_prob, size=(n_hc, n_channels, n_channels))
    for i in range(n_hc):
        hc_binary[i] = np.triu(hc_binary[i], k=1)
        hc_binary[i] = hc_binary[i] + hc_binary[i].T
    
    # Generate weights (correlation-like values)
    ad_weights = ad_binary * (0.3 + 0.5 * np.random.rand(n_ad, n_channels, n_channels))
    hc_weights = hc_binary * (0.4 + 0.5 * np.random.rand(n_hc, n_channels, n_channels))
    
    # Make weights symmetric
    for i in range(n_ad):
        ad_weights[i] = (ad_weights[i] + ad_weights[i].T) / 2
        np.fill_diagonal(ad_weights[i], 0)
    for i in range(n_hc):
        hc_weights[i] = (hc_weights[i] + hc_weights[i].T) / 2
        np.fill_diagonal(hc_weights[i], 0)
    
    print("\nIndividual matrices generated:")
    print(f"  - AD binary shape: {ad_binary.shape}")
    print(f"  - HC binary shape: {hc_binary.shape}")
    
    # Create validator and compute consensus
    validator = TrueConsensusValidator()
    validator.set_individual_matrices(ad_binary, hc_binary, ad_weights, hc_weights)
    validator.compute_group_consensus()
    validator.compute_overall_consensus()
    
    # Verify
    verification = validator.verify_consensus_definition()
    
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"\n✓ AD consensus correct: {verification['ad_consensus_correct']} (error: {verification['ad_max_error']:.2e})")
    print(f"✓ HC consensus correct: {verification['hc_consensus_correct']} (error: {verification['hc_max_error']:.2e})")
    print(f"✓ Overall consensus correct: {verification['overall_consensus_correct']} (error: {verification['overall_max_error']:.2e})")
    print(f"✓ Weighted avg equivalent: {verification['weighted_avg_equivalent']} (error: {verification['equivalence_error']:.2e})")
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    validator.plot_full_validation(
        str(output_path / "true_consensus_validation.png"),
        show_plot=False
    )
    
    validator.generate_proof_report(
        str(output_path / "true_consensus_proof.md")
    )
    
    # Save matrices
    np.save(output_path / "AD_binary_matrices.npy", ad_binary)
    np.save(output_path / "HC_binary_matrices.npy", hc_binary)
    np.save(output_path / "AD_consensus.npy", validator.ad_consensus)
    np.save(output_path / "HC_consensus.npy", validator.hc_consensus)
    np.save(output_path / "Overall_consensus.npy", validator.overall_consensus)
    
    # Statistics
    stats = validator.compute_statistics()
    print(f"\nSparsity:")
    print(f"  - AD: {stats['AD_sparsity_pct']:.2f}%")
    print(f"  - HC: {stats['HC_sparsity_pct']:.2f}%")
    print(f"  - Overall: {stats['Overall_sparsity_pct']:.2f}%")
    
    print(f"\nCorrelations:")
    print(f"  - AD vs HC: r = {stats['correlation_AD_HC']:.4f}")
    print(f"  - AD vs Overall: r = {stats['correlation_AD_Overall']:.4f}")
    print(f"  - HC vs Overall: r = {stats['correlation_HC_Overall']:.4f}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*70)
    
    return validator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate True Consensus Matrix")
    parser.add_argument('--demo', action='store_true', 
                       help='Run demonstration with synthetic data')
    parser.add_argument('--output_dir', type=str, default='./consensus_validation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    create_synthetic_demo(args.output_dir)
