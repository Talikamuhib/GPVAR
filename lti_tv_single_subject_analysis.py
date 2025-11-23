"""
LTI vs Time-Varying GP-VAR Analysis for Single Subject
========================================================
Analyzes one EEG subject and plots frequency responses/transfer functions 
comparing LTI and TV models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import mne
from scipy import linalg, stats, signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_style('whitegrid')

# ============================================================================
# Configuration
# ============================================================================

# Select ONE subject file to analyze
SUBJECT_FILE = '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30018/eeg/s6_sub-30018_rs-hep_eeg.set'

# Consensus Laplacian path
CONSENSUS_LAPLACIAN_PATH = "/home/muhibt/project/filter_identification/Consensus matrix/group_consensus_laplacian/all_consensus_average.npy"

# Preprocessing
BAND = (0.5, 40.0)
TARGET_SFREQ = 100.0

# Model settings
USE_BIAS = False
RIDGE_LAMBDA = 5e-3

# Time-varying analysis
WINDOW_LENGTH_SEC = 10.0
WINDOW_OVERLAP = 0.5
MIN_WINDOWS = 5

# Output
OUT_DIR = Path("./single_subject_lti_tv_analysis")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_and_preprocess_eeg(filepath: str, band: Tuple[float,float], 
                           target_sfreq: float) -> Tuple[np.ndarray, List[str]]:
    """Load and preprocess EEG."""
    print(f"Loading EEG from: {filepath}")
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)
    raw.filter(l_freq=band[0], h_freq=band[1], method="fir", 
               fir_design="firwin", verbose=False)
    X = raw.get_data()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, raw.ch_names

def safe_zscore(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Z-score with robust handling."""
    mu = ref.mean(axis=1, keepdims=True)
    sd = ref.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
    Z = np.clip(Z, -10, 10)
    return Z

def load_consensus_laplacian(filepath: str) -> np.ndarray:
    """Load consensus Laplacian."""
    L = np.load(filepath)
    if not np.allclose(L, L.T):
        L = (L + L.T) / 2
    eigvals = np.linalg.eigvalsh(L)
    if eigvals.min() < -1e-8:
        L = L - eigvals.min() * np.eye(L.shape[0])
    return L

# ============================================================================
# GP-VAR Model Class
# ============================================================================

class GPVAR_SharedH:
    """GP-VAR with shared scalar coefficients."""
    
    def __init__(self, P: int, K: int, L: np.ndarray, lam: float = RIDGE_LAMBDA):
        self.P, self.K, self.L = P, K, L
        self.n = L.shape[0]
        self.lam = lam
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L)
        
        # Precompute L^k
        self.L_powers = [np.eye(self.n)]
        for k in range(1, K+1):
            self.L_powers.append(self.L_powers[-1] @ L)
    
    def _build_design(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build design matrix."""
        n, T = X.shape
        T_valid = T - self.P
        n_obs = n * T_valid
        n_feat = self.P * (self.K + 1)
        
        R = np.zeros((n_obs, n_feat), dtype=np.float32)
        Y = np.zeros(n_obs, dtype=np.float32)
        
        for t in range(self.P, T):
            t_idx = t - self.P
            filter_vals = []
            for p in range(1, self.P + 1):
                xlag = X[:, t - p]
                for k in range(self.K + 1):
                    filter_vals.append(self.L_powers[k] @ xlag)
            filter_vals = np.asarray(filter_vals, dtype=np.float32)
            
            for i in range(n):
                row_idx = t_idx * n + i
                R[row_idx, :] = filter_vals[:, i]
                Y[row_idx] = X[i, t]
        
        return R, Y
    
    def fit(self, Xtr: np.ndarray):
        """Fit model."""
        if Xtr.shape[1] <= self.P:
            raise ValueError(f"T must exceed P")
        
        R, Y = self._build_design(Xtr)
        RtR = R.T @ R
        RtY = R.T @ Y
        reg = self.lam * np.eye(RtR.shape[0], dtype=np.float32)
        theta = linalg.solve(RtR + reg, RtY, assume_a='sym')
        self.h = theta.astype(np.float64)
        self.b = np.zeros(self.n)
    
    def spectral_radius(self) -> float:
        """Compute spectral radius."""
        if not hasattr(self, 'h'):
            return np.nan
        
        A_mats = []
        idx = 0
        for p in range(self.P):
            A_p = np.zeros((self.n, self.n))
            for k in range(self.K + 1):
                A_p += self.h[idx] * self.L_powers[k]
                idx += 1
            A_mats.append(A_p)
        
        C = np.zeros((self.n * self.P, self.n * self.P))
        C[:self.n, :self.n*self.P] = np.hstack(A_mats)
        if self.P > 1:
            C[self.n:, :-self.n] = np.eye(self.n * (self.P - 1))
        
        vals = np.linalg.eigvals(C)
        vals = vals[np.isfinite(vals)]
        return float(np.max(np.abs(vals))) if vals.size else np.nan
    
    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate model (teacher-forcing)."""
        R, _ = self._build_design(X)
        Y_pred = (R @ self.h.astype(np.float32)).astype(np.float64)
        Y_pred = Y_pred.reshape(-1, self.n).T
        Y_true = X[:, self.P:]
        
        resid = Y_true - Y_pred
        RSS = float(np.sum(resid**2))
        TSS = float(np.sum((Y_true - Y_true.mean())**2))
        R2 = 1.0 - RSS/(TSS + 1e-10)
        MSE = RSS / Y_true.size
        
        n_obs = Y_true.size
        n_params = self.P * (self.K + 1)
        BIC = n_obs*np.log(MSE + 1e-10) + n_params*np.log(n_obs)
        
        return dict(RSS=RSS, R2=R2, MSE=MSE, BIC=BIC)
    
    def compute_transfer_function(self, omegas: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Compute AR transfer function G(ω, λ) with stability guard.
        """
        if not hasattr(self, 'h'):
            raise ValueError("Model not fitted")
        
        if omegas is None:
            omegas = np.linspace(0, np.pi, 256)
        
        lambdas = self.eigenvalues
        P, K = self.P, self.K
        
        # Compute H_p(λ) - coefficients as function of eigenvalues
        H_p = np.zeros((P, len(lambdas)), dtype=np.complex128)
        for p in range(P):
            for i, lam in enumerate(lambdas):
                val = 0.0
                for k in range(K + 1):
                    val += self.h[p*(K+1) + k] * (lam ** k)
                H_p[p, i] = val
        
        # Compute G(ω, λ) = 1 / (1 - Σ_p H_p(λ) e^{-iωp})
        G = np.zeros((len(omegas), len(lambdas)), dtype=np.complex128)
        for w_i, w in enumerate(omegas):
            z_terms = np.exp(-1j * w * np.arange(1, P+1))
            denom = 1.0 - (z_terms[:, None] * H_p).sum(axis=0)
            
            # Stability guard: prevent division by very small numbers
            denom = np.where(np.abs(denom) < 1e-3, denom + 1e-3, denom)
            
            G[w_i, :] = 1.0 / denom
        
        return {
            'omegas': omegas,
            'lambdas': lambdas,
            'G': G,
            'G_mag': np.abs(G),
            'G_phase': np.angle(G),
            'H_p': H_p
        }

# ============================================================================
# Model Selection
# ============================================================================

def find_best_model(X: np.ndarray, L: np.ndarray, 
                    P_range: List[int] = [1, 2, 3, 5, 7, 10],
                    K_range: List[int] = [1, 2, 3, 4]) -> Tuple[int, int]:
    """Find best P and K using BIC."""
    T = X.shape[1]
    T_train = int(0.70 * T)
    T_val = int(0.85 * T)
    
    X_train = X[:, :T_train]
    X_val = X[:, T_train:T_val]
    
    # Global standardization
    X_train = safe_zscore(X_train, X_train)
    X_val = safe_zscore(X_val, X_train)
    
    best_bic = np.inf
    best_P, best_K = None, None
    
    print("Finding best P and K...")
    for K in K_range:
        for P in P_range:
            try:
                m = GPVAR_SharedH(P=P, K=K, L=L)
                m.fit(X_train)
                
                # Check stability
                rho = m.spectral_radius()
                if not np.isfinite(rho) or rho >= 1.0:
                    continue
                
                metrics = m.evaluate(X_val)
                
                if metrics['BIC'] < best_bic:
                    best_bic = metrics['BIC']
                    best_P = P
                    best_K = K
                    print(f"  New best: P={P}, K={K}, BIC={metrics['BIC']:.2f}")
            except:
                continue
    
    if best_P is None:
        # Fallback
        best_P, best_K = 5, 2
    
    return best_P, best_K

# ============================================================================
# Time-Varying Analysis
# ============================================================================

def split_into_windows(X: np.ndarray, window_length_sec: float, 
                       overlap: float, fs: float) -> List[Tuple[int, int, np.ndarray]]:
    """Split data into overlapping windows."""
    window_samples = int(window_length_sec * fs)
    step_samples = int(window_samples * (1 - overlap))
    
    T = X.shape[1]
    windows = []
    
    start = 0
    while start + window_samples <= T:
        end = start + window_samples
        X_win = X[:, start:end]
        windows.append((start, end, X_win))
        start += step_samples
    
    return windows

def compute_tv_models(X_std: np.ndarray, L: np.ndarray, P: int, K: int,
                      window_length_sec: float, overlap: float, 
                      fs: float) -> List[Dict]:
    """Fit separate models for each time window."""
    windows = split_into_windows(X_std, window_length_sec, overlap, fs)
    
    tv_results = []
    
    print(f"Fitting {len(windows)} time-varying models...")
    for w_idx, (start_idx, end_idx, X_win) in enumerate(windows):
        try:
            model = GPVAR_SharedH(P=P, K=K, L=L)
            model.fit(X_win)
            
            # Check stability
            rho = model.spectral_radius()
            if not np.isfinite(rho) or rho >= 0.99:
                print(f"  Warning: Window {w_idx} unstable (ρ={rho:.3f}), skipping")
                continue
            
            metrics = model.evaluate(X_win)
            
            tv_results.append({
                'window_idx': w_idx,
                'start_time': start_idx / fs,
                'end_time': end_idx / fs,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'model': model,
                'metrics': metrics,
                'h': model.h.copy(),
                'rho': rho
            })
        except Exception as e:
            print(f"  Warning: Window {w_idx} failed: {e}")
            continue
    
    return tv_results

# ============================================================================
# Transfer Function Comparison and Visualization
# ============================================================================

def plot_transfer_function_comparison(lti_model: GPVAR_SharedH, 
                                     tv_results: List[Dict],
                                     subject_id: str,
                                     save_dir: Path):
    """Create detailed frequency response comparison plots."""
    
    print("\nCreating frequency response comparison plots...")
    
    # Compute transfer functions
    omegas = np.linspace(0, np.pi, 256)
    freqs_hz = omegas * TARGET_SFREQ / (2 * np.pi)  # Convert to Hz
    
    # LTI transfer function
    lti_tf = lti_model.compute_transfer_function(omegas)
    G_lti = lti_tf['G_mag']
    G_lti_phase = lti_tf['G_phase']
    lambdas = lti_tf['lambdas']
    
    # TV transfer functions for all windows
    n_windows = len(tv_results)
    n_omegas = len(omegas)
    n_lambdas = len(lambdas)
    
    G_tv_all = np.zeros((n_windows, n_omegas, n_lambdas))
    G_tv_phase_all = np.zeros((n_windows, n_omegas, n_lambdas))
    
    for w_idx, tv_res in enumerate(tv_results):
        tv_tf = tv_res['model'].compute_transfer_function(omegas)
        G_tv_all[w_idx, :, :] = tv_tf['G_mag']
        G_tv_phase_all[w_idx, :, :] = tv_tf['G_phase']
    
    # Statistics
    G_tv_mean = G_tv_all.mean(axis=0)
    G_tv_std = G_tv_all.std(axis=0)
    
    # =================================================================
    # Figure 1: Comprehensive Transfer Function Analysis
    # =================================================================
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)
    
    # --- Row 1: 2D Heatmaps of Transfer Functions ---
    
    # LTI Transfer Function
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(G_lti, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax1.set_title('LTI |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax1.set_xlabel('λ (Graph Frequency)')
    ax1.set_ylabel('f (Hz)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # TV Mean Transfer Function
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(G_tv_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax2.set_title('TV Mean |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax2.set_xlabel('λ (Graph Frequency)')
    ax2.set_ylabel('f (Hz)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Difference: TV - LTI
    ax3 = fig.add_subplot(gs[0, 2])
    diff = G_tv_mean - G_lti
    vmax = np.abs(diff).max()
    im3 = ax3.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()],
                     vmin=-vmax, vmax=vmax)
    ax3.set_title('TV Mean - LTI', fontsize=12, fontweight='bold')
    ax3.set_xlabel('λ (Graph Frequency)')
    ax3.set_ylabel('f (Hz)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Variance across windows
    ax4 = fig.add_subplot(gs[0, 3])
    var_across_windows = G_tv_all.var(axis=0)
    im4 = ax4.imshow(var_across_windows, aspect='auto', origin='lower', cmap='YlOrRd',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax4.set_title('TV Variance Across Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('λ (Graph Frequency)')
    ax4.set_ylabel('f (Hz)')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # --- Row 2: Frequency Slices ---
    
    # Select specific frequencies to plot
    freq_indices = [int(5 * 256/50), int(10 * 256/50), int(20 * 256/50), int(30 * 256/50)]  # 5, 10, 20, 30 Hz
    colors = ['blue', 'green', 'orange', 'red']
    
    # Magnitude response at different frequencies
    ax5 = fig.add_subplot(gs[1, :2])
    for f_idx, color in zip(freq_indices, colors):
        freq_val = freqs_hz[f_idx]
        ax5.plot(lambdas, G_lti[f_idx, :], '-', color=color, linewidth=2, 
                label=f'LTI @ {freq_val:.1f}Hz')
        ax5.plot(lambdas, G_tv_mean[f_idx, :], '--', color=color, linewidth=2,
                label=f'TV @ {freq_val:.1f}Hz')
        ax5.fill_between(lambdas,
                         G_tv_mean[f_idx, :] - G_tv_std[f_idx, :],
                         G_tv_mean[f_idx, :] + G_tv_std[f_idx, :],
                         alpha=0.2, color=color)
    ax5.set_xlabel('λ (Graph Frequency)')
    ax5.set_ylabel('Magnitude |G|')
    ax5.set_title('Transfer Function: Frequency Slices', fontsize=12, fontweight='bold')
    ax5.legend(ncol=2, fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Select specific graph modes to plot
    mode_indices = [0, len(lambdas)//4, len(lambdas)//2, 3*len(lambdas)//4]
    
    # Magnitude response at different graph modes
    ax6 = fig.add_subplot(gs[1, 2:])
    for m_idx, color in zip(mode_indices, colors):
        lambda_val = lambdas[m_idx]
        ax6.plot(freqs_hz, G_lti[:, m_idx], '-', color=color, linewidth=2,
                label=f'LTI λ={lambda_val:.2f}')
        ax6.plot(freqs_hz, G_tv_mean[:, m_idx], '--', color=color, linewidth=2,
                label=f'TV λ={lambda_val:.2f}')
        ax6.fill_between(freqs_hz,
                         G_tv_mean[:, m_idx] - G_tv_std[:, m_idx],
                         G_tv_mean[:, m_idx] + G_tv_std[:, m_idx],
                         alpha=0.2, color=color)
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude |G|')
    ax6.set_title('Transfer Function: Graph Mode Slices', fontsize=12, fontweight='bold')
    ax6.legend(ncol=2, fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # --- Row 3: Time Evolution of Transfer Functions ---
    
    # Pick a specific frequency and graph mode to show evolution
    f_show = int(10 * 256/50)  # 10 Hz
    m_show = len(lambdas)//2   # Middle graph mode
    
    ax7 = fig.add_subplot(gs[2, 0])
    window_times = [(r['start_time'] + r['end_time'])/2 for r in tv_results]
    tv_evolution = G_tv_all[:, f_show, m_show]
    ax7.plot(window_times, tv_evolution, 'o-', linewidth=2, markersize=6, color='red')
    ax7.axhline(G_lti[f_show, m_show], color='blue', linestyle='--', linewidth=2,
                label=f'LTI baseline')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Magnitude |G|')
    ax7.set_title(f'Time Evolution @ 10Hz, λ={lambdas[m_show]:.2f}', 
                  fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Average magnitude across all frequencies
    ax8 = fig.add_subplot(gs[2, 1])
    lti_avg_per_mode = G_lti.mean(axis=0)
    tv_avg_per_mode = G_tv_mean.mean(axis=0)
    tv_std_per_mode = G_tv_all.mean(axis=1).std(axis=0)
    
    ax8.plot(lambdas, lti_avg_per_mode, 'b-', linewidth=2.5, label='LTI')
    ax8.plot(lambdas, tv_avg_per_mode, 'r-', linewidth=2.5, label='TV Mean')
    ax8.fill_between(lambdas,
                     tv_avg_per_mode - tv_std_per_mode,
                     tv_avg_per_mode + tv_std_per_mode,
                     alpha=0.3, color='red', label='TV ±1σ')
    ax8.set_xlabel('λ (Graph Frequency)')
    ax8.set_ylabel('Average Magnitude')
    ax8.set_title('Average Response per Graph Mode', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Average magnitude across all graph modes
    ax9 = fig.add_subplot(gs[2, 2])
    lti_avg_per_freq = G_lti.mean(axis=1)
    tv_avg_per_freq = G_tv_mean.mean(axis=1)
    tv_std_per_freq = G_tv_all.mean(axis=2).std(axis=0)
    
    ax9.plot(freqs_hz, lti_avg_per_freq, 'b-', linewidth=2.5, label='LTI')
    ax9.plot(freqs_hz, tv_avg_per_freq, 'r-', linewidth=2.5, label='TV Mean')
    ax9.fill_between(freqs_hz,
                     tv_avg_per_freq - tv_std_per_freq,
                     tv_avg_per_freq + tv_std_per_freq,
                     alpha=0.3, color='red', label='TV ±1σ')
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('Average Magnitude')
    ax9.set_title('Average Response per Frequency', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # MSD per window
    ax10 = fig.add_subplot(gs[2, 3])
    msd_per_window = np.mean((G_tv_all - G_lti[None, :, :])**2, axis=(1, 2))
    ax10.plot(window_times, msd_per_window, 'o-', linewidth=2, markersize=6, color='purple')
    ax10.axhline(msd_per_window.mean(), color='red', linestyle='--', linewidth=2,
                 label=f'Mean MSD: {msd_per_window.mean():.4f}')
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('MSD')
    ax10.set_title('Mean Squared Difference vs LTI', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle(f'{subject_id}: LTI vs Time-Varying Transfer Functions', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    savepath = save_dir / f'{subject_id}_transfer_functions_detailed.png'
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath}")
    
    # =================================================================
    # Figure 2: 3D Surface Plots
    # =================================================================
    
    fig2 = plt.figure(figsize=(16, 8))
    
    # Create meshgrid for 3D plotting
    Lambda_grid, Omega_grid = np.meshgrid(lambdas, freqs_hz)
    
    # LTI 3D surface
    ax_3d1 = fig2.add_subplot(121, projection='3d')
    surf1 = ax_3d1.plot_surface(Lambda_grid, Omega_grid, G_lti, 
                                cmap='viridis', alpha=0.9)
    ax_3d1.set_xlabel('λ (Graph Freq)', fontsize=10)
    ax_3d1.set_ylabel('f (Hz)', fontsize=10)
    ax_3d1.set_zlabel('|G(ω,λ)|', fontsize=10)
    ax_3d1.set_title('LTI Transfer Function', fontsize=12, fontweight='bold')
    ax_3d1.view_init(elev=25, azim=45)
    fig2.colorbar(surf1, ax=ax_3d1, fraction=0.046, pad=0.1)
    
    # TV Mean 3D surface  
    ax_3d2 = fig2.add_subplot(122, projection='3d')
    surf2 = ax_3d2.plot_surface(Lambda_grid, Omega_grid, G_tv_mean,
                                cmap='viridis', alpha=0.9)
    ax_3d2.set_xlabel('λ (Graph Freq)', fontsize=10)
    ax_3d2.set_ylabel('f (Hz)', fontsize=10)
    ax_3d2.set_zlabel('|G(ω,λ)|', fontsize=10)
    ax_3d2.set_title('TV Mean Transfer Function', fontsize=12, fontweight='bold')
    ax_3d2.view_init(elev=25, azim=45)
    fig2.colorbar(surf2, ax=ax_3d2, fraction=0.046, pad=0.1)
    
    plt.suptitle(f'{subject_id}: 3D Transfer Function Surfaces', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    savepath2 = save_dir / f'{subject_id}_transfer_functions_3D.png'
    plt.savefig(savepath2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath2}")
    
    return {
        'G_lti': G_lti,
        'G_tv_mean': G_tv_mean,
        'G_tv_std': G_tv_std,
        'msd_per_window': msd_per_window,
        'freqs_hz': freqs_hz,
        'lambdas': lambdas
    }

# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_single_subject():
    """Complete LTI vs TV analysis for one subject."""
    
    print("="*80)
    print("SINGLE SUBJECT LTI vs TV TRANSFER FUNCTION ANALYSIS")
    print("="*80)
    
    # Extract subject ID from filename
    subject_id = Path(SUBJECT_FILE).stem.replace('s6_', '').replace('_rs-hep_eeg', '')
    print(f"\nAnalyzing subject: {subject_id}")
    print(f"File: {SUBJECT_FILE}")
    
    # Load consensus Laplacian
    print("\nLoading consensus Laplacian...")
    L = load_consensus_laplacian(CONSENSUS_LAPLACIAN_PATH)
    print(f"  Laplacian shape: {L.shape}")
    
    # Load and preprocess EEG
    print("\nLoading and preprocessing EEG...")
    X, ch_names = load_and_preprocess_eeg(SUBJECT_FILE, BAND, TARGET_SFREQ)
    n_channels, n_samples = X.shape
    duration = n_samples / TARGET_SFREQ
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Sampling rate: {TARGET_SFREQ} Hz")
    
    # Global standardization
    print("\nStandardizing data...")
    X_std = safe_zscore(X, X)
    
    # Find optimal model order
    print("\nFinding optimal model parameters...")
    best_P, best_K = find_best_model(X, L)
    print(f"  Selected: P={best_P}, K={best_K}")
    
    # Fit LTI model
    print("\nFitting LTI model on entire data...")
    lti_model = GPVAR_SharedH(P=best_P, K=best_K, L=L)
    lti_model.fit(X_std)
    lti_rho = lti_model.spectral_radius()
    lti_metrics = lti_model.evaluate(X_std)
    
    print(f"  LTI Model Performance:")
    print(f"    R²: {lti_metrics['R2']:.4f}")
    print(f"    BIC: {lti_metrics['BIC']:.2f}")
    print(f"    Spectral radius: {lti_rho:.3f}")
    
    if not np.isfinite(lti_rho) or lti_rho >= 1.0:
        print("ERROR: LTI model is unstable!")
        return None
    
    # Fit TV models
    print(f"\nFitting time-varying models...")
    print(f"  Window length: {WINDOW_LENGTH_SEC} seconds")
    print(f"  Overlap: {WINDOW_OVERLAP*100}%")
    
    tv_results = compute_tv_models(X_std, L, best_P, best_K, 
                                   WINDOW_LENGTH_SEC, WINDOW_OVERLAP, TARGET_SFREQ)
    
    print(f"  Successfully fitted {len(tv_results)} stable windows")
    
    if len(tv_results) < MIN_WINDOWS:
        print(f"ERROR: Too few windows ({len(tv_results)} < {MIN_WINDOWS})")
        return None
    
    # Print TV model statistics
    tv_r2s = [r['metrics']['R2'] for r in tv_results]
    tv_rhos = [r['rho'] for r in tv_results]
    print(f"\n  TV Models Performance:")
    print(f"    Mean R²: {np.mean(tv_r2s):.4f} ± {np.std(tv_r2s):.4f}")
    print(f"    Mean spectral radius: {np.mean(tv_rhos):.3f} ± {np.std(tv_rhos):.3f}")
    
    # Create visualizations
    tf_results = plot_transfer_function_comparison(lti_model, tv_results, 
                                                   subject_id, OUT_DIR)
    
    # Compute summary statistics
    print("\nTransfer Function Comparison Summary:")
    print(f"  Mean MSD between LTI and TV: {tf_results['msd_per_window'].mean():.6f}")
    print(f"  Max difference in magnitude: {np.abs(tf_results['G_tv_mean'] - tf_results['G_lti']).max():.4f}")
    
    # Check for time-varying behavior
    cv_across_windows = tf_results['G_tv_std'] / (tf_results['G_tv_mean'] + 1e-8)
    mean_cv = np.mean(cv_across_windows)
    print(f"  Mean coefficient of variation: {mean_cv:.4f}")
    
    if mean_cv > 0.1:
        print("\n  *** Evidence of TIME-VARYING dynamics detected ***")
    else:
        print("\n  *** System appears to be TIME-INVARIANT ***")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUT_DIR}")
    print("="*80)
    
    return {
        'subject_id': subject_id,
        'lti_model': lti_model,
        'tv_results': tv_results,
        'tf_comparison': tf_results,
        'best_P': best_P,
        'best_K': best_K,
        'n_channels': n_channels,
        'duration': duration
    }

if __name__ == "__main__":
    results = analyze_single_subject()