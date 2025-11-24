"""
Simplified LTI vs TV Transfer Function Comparison
==================================================
Clean line plots with 95% CI showing transfer function response
across graph frequencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import mne
from scipy import linalg, stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# ============================================================================
# Configuration
# ============================================================================

SUBJECT_FILE = '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30018/eeg/s6_sub-30018_rs-hep_eeg.set'
CONSENSUS_LAPLACIAN_PATH = "/home/muhibt/project/filter_identification/Consensus matrix/group_consensus_laplacian/all_consensus_average.npy"

BAND = (0.5, 40.0)
TARGET_SFREQ = 100.0
RIDGE_LAMBDA = 5e-3
WINDOW_LENGTH_SEC = 10.0
WINDOW_OVERLAP = 0.5
MIN_WINDOWS = 5

OUT_DIR = Path("./lti_vs_tv_simple")
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
    """Load and normalize consensus Laplacian."""
    print("Loading consensus matrix...")
    M = np.load(filepath)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.allclose(M, M.T, atol=1e-6):
        M = (M + M.T) / 2.0

    n = M.shape[0]
    row_sums = np.abs(M.sum(axis=1))
    diag_vals = np.diag(M)
    looks_like_laplacian = (row_sums.mean() < 1e-3 * np.abs(M).mean()) and (np.all(diag_vals >= -1e-6))

    if looks_like_laplacian:
        print("  ✓ Detected Laplacian input")
        L = M
        eigvals, U = np.linalg.eigh(L)
        if eigvals.min() < -1e-6:
            eigvals_clipped = np.clip(eigvals, 0.0, None)
            L = (U * eigvals_clipped) @ U.T
            L = (L + L.T) / 2.0
        d = np.diag(L).copy()
        d = np.where(d < 1e-10, 1e-10, d)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    else:
        print("  ✓ Detected adjacency matrix")
        A = M
        if np.any(A < 0):
            A = np.clip(A, 0.0, None)
        d = A.sum(axis=1)
        d = np.where(d < 1e-10, 1e-10, d)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    L_norm = (L_norm + L_norm.T) / 2.0
    eigs_norm = np.linalg.eigvalsh(L_norm)
    print(f"  ✓ Eigenvalue range: [{eigs_norm.min():.4f}, {eigs_norm.max():.4f}]")
    
    return L_norm

# ============================================================================
# GP-VAR Model Class
# ============================================================================

class GPVAR_SharedH:
    """GP-VAR with shared scalar coefficients."""
    
    def __init__(self, P: int, K: int, L_norm: np.ndarray = None, L: np.ndarray = None, 
                 lam: float = RIDGE_LAMBDA):
        if L_norm is None and L is None:
            raise ValueError("Must provide L_norm (or L)")
        if L_norm is None:
            L_norm = L
            
        self.P, self.K = P, K
        self.L = L_norm
        self.n = L_norm.shape[0]
        self.lam = lam
        
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L_norm)
        self.L_powers = [np.eye(self.n)]
        for k in range(1, K+1):
            self.L_powers.append(self.L_powers[-1] @ L_norm)
    
    def _build_design(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build design matrix."""
        n, T = X.shape
        if n != self.n:
            raise ValueError(f"Data has {n} channels but Laplacian has {self.n} nodes")
        
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
            raise ValueError(f"T={Xtr.shape[1]} must exceed P={self.P}")
        
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
        """Evaluate model."""
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
        """Compute transfer function G(ω, λ)."""
        if not hasattr(self, 'h'):
            raise ValueError("Model not fitted")
        
        if omegas is None:
            omegas = np.linspace(0, np.pi, 256)
        
        lambdas = self.eigenvalues  
        P, K = self.P, self.K
        
        H_p = np.zeros((P, len(lambdas)), dtype=np.complex128)
        for p in range(P):
            for i, lam in enumerate(lambdas):
                val = 0.0
                for k in range(K + 1):
                    val += self.h[p*(K+1) + k] * (lam ** k)
                H_p[p, i] = val
        
        G = np.zeros((len(omegas), len(lambdas)), dtype=np.complex128)
        for w_i, w in enumerate(omegas):
            z_terms = np.exp(-1j * w * np.arange(1, P+1))
            denom = 1.0 - (z_terms[:, None] * H_p).sum(axis=0)
            
            small_mask = np.abs(denom) < 1e-3
            if np.any(small_mask):
                denom[small_mask] = (denom[small_mask] / 
                                    (np.abs(denom[small_mask]) + 1e-10)) * 1e-3
            
            G[w_i, :] = 1.0 / denom
        
        return {
            'omegas': omegas,
            'lambdas': lambdas,
            'G': G,
            'G_mag': np.abs(G),
            'G_phase': np.angle(G)
        }

# ============================================================================
# Model Selection
# ============================================================================

def find_best_model(X: np.ndarray, L_norm: np.ndarray) -> Tuple[int, int]:
    """Find best P and K using BIC."""
    T = X.shape[1]
    T_train = int(0.70 * T)
    T_val = int(0.85 * T)
    
    X_train = X[:, :T_train]
    X_val = X[:, T_train:T_val]
    
    X_train = safe_zscore(X_train, X_train)
    X_val = safe_zscore(X_val, X_train)
    
    best_bic = np.inf
    best_P, best_K = None, None
    
    print("Finding best P and K...")
    for K in [1, 2, 3, 4]:
        for P in [1, 2, 3, 5, 7, 10]:
            try:
                m = GPVAR_SharedH(P=P, K=K, L_norm=L_norm)
                m.fit(X_train)
                
                rho = m.spectral_radius()
                if not np.isfinite(rho) or rho >= 0.99:
                    continue
                
                metrics = m.evaluate(X_val)
                
                if metrics['BIC'] < best_bic:
                    best_bic = metrics['BIC']
                    best_P = P
                    best_K = K
                    print(f"  New best: P={P}, K={K}, BIC={metrics['BIC']:.2f}, ρ={rho:.3f}")
            except Exception as e:
                print(f"  Failed P={P}, K={K}: {e}")
                continue
    
    if best_P is None:
        print("  WARNING: Using fallback P=5, K=2")
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

def compute_tv_models(X_std: np.ndarray, L_norm: np.ndarray, P: int, K: int,
                      window_length_sec: float, overlap: float, 
                      fs: float) -> List[Dict]:
    """Fit time-varying models."""
    windows = split_into_windows(X_std, window_length_sec, overlap, fs)
    tv_results = []
    
    print(f"Fitting {len(windows)} time-varying models...")
    for w_idx, (start_idx, end_idx, X_win) in enumerate(windows):
        try:
            model = GPVAR_SharedH(P=P, K=K, L_norm=L_norm)
            model.fit(X_win)
            
            rho = model.spectral_radius()
            if not np.isfinite(rho) or rho >= 0.99:
                print(f"  Window {w_idx}: unstable (ρ={rho:.3f}), skipping")
                continue
            
            metrics = model.evaluate(X_win)
            
            tv_results.append({
                'window_idx': w_idx,
                'start_time': start_idx / fs,
                'end_time': end_idx / fs,
                'model': model,
                'metrics': metrics,
                'rho': rho
            })
        except Exception as e:
            print(f"  Window {w_idx}: failed - {e}")
            continue
    
    return tv_results

# ============================================================================
# Simplified Plotting
# ============================================================================

def plot_lti_vs_tv_simple(lti_model: GPVAR_SharedH, 
                         tv_results: List[Dict],
                         subject_id: str,
                         save_dir: Path):
    """
    Simple line plot: Transfer function magnitude vs graph frequency index
    with 95% confidence intervals for TV models.
    """
    
    print("\nCreating LTI vs TV comparison plot...")
    
    # Compute transfer functions
    omegas = np.linspace(0, np.pi, 256)
    
    # LTI
    lti_tf = lti_model.compute_transfer_function(omegas)
    G_lti = lti_tf['G_mag']  # shape: (n_omegas, n_lambdas)
    lambdas = lti_tf['lambdas']
    
    # Average over temporal frequencies to get response per graph mode
    lti_response = G_lti.mean(axis=0)  # shape: (n_lambdas,)
    
    # TV models
    n_windows = len(tv_results)
    n_lambdas = len(lambdas)
    tv_responses = np.zeros((n_windows, n_lambdas))
    
    for w_idx, tv_res in enumerate(tv_results):
        tv_tf = tv_res['model'].compute_transfer_function(omegas)
        G_tv = tv_tf['G_mag']
        tv_responses[w_idx, :] = G_tv.mean(axis=0)
    
    # TV statistics
    tv_mean = tv_responses.mean(axis=0)
    tv_std = tv_responses.std(axis=0)
    tv_sem = tv_std / np.sqrt(n_windows)
    
    # 95% CI: mean ± 1.96 * SEM
    tv_ci_lower = tv_mean - 1.96 * tv_sem
    tv_ci_upper = tv_mean + 1.96 * tv_sem
    
    # Create graph frequency indices
    graph_freq_indices = np.arange(len(lambdas))
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # LTI line
    ax.plot(graph_freq_indices, lti_response, 
            'b-', linewidth=3, label='LTI', alpha=0.8)
    
    # TV mean line
    ax.plot(graph_freq_indices, tv_mean, 
            'r-', linewidth=3, label='TV Mean', alpha=0.8)
    
    # TV 95% CI shaded region
    ax.fill_between(graph_freq_indices, tv_ci_lower, tv_ci_upper,
                    alpha=0.3, color='red', label='TV 95% CI')
    
    # Styling
    ax.set_xlabel('Graph Frequency Index (sorted eigenvalue index)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Transfer Function Magnitude |G(ω,λ)|', fontsize=14, fontweight='bold')
    ax.set_title(f'{subject_id}: LTI vs Time-Varying Transfer Function Response\n' + 
                 f'(averaged over temporal frequencies, n={n_windows} windows)',
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add eigenvalue annotations on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Show eigenvalues at select positions
    n_ticks = 10
    tick_indices = np.linspace(0, len(lambdas)-1, n_ticks, dtype=int)
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([f'{lambdas[i]:.2f}' for i in tick_indices], fontsize=10)
    ax2.set_xlabel('Graph Frequency λ (eigenvalue of normalized Laplacian)', 
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    savepath = save_dir / f'{subject_id}_lti_vs_tv_simple.png'
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {savepath}")
    
    # Print statistics
    print(f"\nTransfer Function Statistics:")
    print(f"  LTI mean magnitude: {lti_response.mean():.4f}")
    print(f"  TV mean magnitude: {tv_mean.mean():.4f}")
    print(f"  Mean absolute difference: {np.abs(tv_mean - lti_response).mean():.4f}")
    print(f"  Max absolute difference: {np.abs(tv_mean - lti_response).max():.4f}")
    
    # Check if CI includes LTI
    lti_in_ci = np.sum((lti_response >= tv_ci_lower) & (lti_response <= tv_ci_upper))
    print(f"  LTI within TV 95% CI: {lti_in_ci}/{len(lambdas)} points ({100*lti_in_ci/len(lambdas):.1f}%)")
    
    return {
        'lti_response': lti_response,
        'tv_mean': tv_mean,
        'tv_ci_lower': tv_ci_lower,
        'tv_ci_upper': tv_ci_upper,
        'lambdas': lambdas,
        'graph_freq_indices': graph_freq_indices
    }

# ============================================================================
# Main Analysis
# ============================================================================

def analyze_single_subject():
    """Complete analysis."""
    
    print("="*80)
    print("LTI vs TV TRANSFER FUNCTION ANALYSIS (SIMPLIFIED)")
    print("="*80)
    
    subject_id = Path(SUBJECT_FILE).stem.replace('s6_', '').replace('_rs-hep_eeg', '')
    print(f"\nSubject: {subject_id}")
    
    # Load Laplacian
    print("\n" + "="*80)
    L_norm = load_consensus_laplacian(CONSENSUS_LAPLACIAN_PATH)
    print("="*80)
    
    # Load EEG
    print("\nLoading EEG...")
    X, ch_names = load_and_preprocess_eeg(SUBJECT_FILE, BAND, TARGET_SFREQ)
    n_channels, n_samples = X.shape
    duration = n_samples / TARGET_SFREQ
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {duration:.1f}s")
    
    # Check compatibility
    if L_norm.shape[0] != n_channels:
        raise ValueError(
            f"Laplacian size ({L_norm.shape[0]}) ≠ EEG channels ({n_channels})"
        )
    print(f"  ✓ Laplacian size matches")
    
    # Standardize
    X_std = safe_zscore(X, X)
    
    # Model selection
    print("\n" + "="*80)
    best_P, best_K = find_best_model(X, L_norm)
    print(f"="*80)
    print(f"Selected: P={best_P}, K={best_K}")
    
    # Fit LTI
    print("\nFitting LTI model...")
    lti_model = GPVAR_SharedH(P=best_P, K=best_K, L_norm=L_norm)
    lti_model.fit(X_std)
    lti_rho = lti_model.spectral_radius()
    lti_metrics = lti_model.evaluate(X_std)
    
    print(f"  R²: {lti_metrics['R2']:.4f}")
    print(f"  Spectral radius: {lti_rho:.3f}")
    
    if not np.isfinite(lti_rho) or lti_rho >= 1.0:
        raise ValueError(f"LTI model unstable (ρ={lti_rho:.3f})")
    
    # Fit TV
    print("\n" + "="*80)
    tv_results = compute_tv_models(X_std, L_norm, best_P, best_K, 
                                   WINDOW_LENGTH_SEC, WINDOW_OVERLAP, TARGET_SFREQ)
    print("="*80)
    print(f"Successfully fitted {len(tv_results)} windows")
    
    if len(tv_results) < MIN_WINDOWS:
        raise ValueError(f"Too few windows ({len(tv_results)} < {MIN_WINDOWS})")
    
    tv_r2s = [r['metrics']['R2'] for r in tv_results]
    tv_rhos = [r['rho'] for r in tv_results]
    print(f"  Mean R²: {np.mean(tv_r2s):.4f} ± {np.std(tv_r2s):.4f}")
    print(f"  Mean ρ: {np.mean(tv_rhos):.3f} ± {np.std(tv_rhos):.3f}")
    
    # Plot
    print("\n" + "="*80)
    plot_results = plot_lti_vs_tv_simple(lti_model, tv_results, subject_id, OUT_DIR)
    print("="*80)
    
    print("\n" + "="*80)
    print(f"COMPLETE - Results in: {OUT_DIR}")
    print("="*80)
    
    return {
        'subject_id': subject_id,
        'lti_model': lti_model,
        'tv_results': tv_results,
        'plot_results': plot_results,
        'best_P': best_P,
        'best_K': best_K
    }

if __name__ == "__main__":
    results = analyze_single_subject()
