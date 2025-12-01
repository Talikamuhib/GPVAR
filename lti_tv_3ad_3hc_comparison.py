"""
LTI vs Time-Varying GP-VAR Analysis: 3 AD vs 3 HC Comparison
==============================================================
Analyzes 3 AD and 3 HC subjects, comparing LTI and TV models between groups.

Features:
- Individual subject analysis with model selection
- Model selection heatmaps (BIC grid) for each subject
- Model selection results saved as CSV
- AD vs HC comparison plots for both LTI and TV models
- Statistical comparisons between groups
- Publication-quality visualizations
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
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.patches as mpatches

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

sns.set_style('whitegrid')
sns.set_palette("Set2")

# ============================================================================
# Configuration
# ============================================================================

# Select 3 AD and 3 HC subjects
AD_PATHS = [
    '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30018/eeg/s6_sub-30018_rs-hep_eeg.set',
    '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30026/eeg/s6_sub-30026_rs-hep_eeg.set',
    '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30011/eeg/s6_sub-30011_rs-hep_eeg.set',
]

HC_PATHS = [
    "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10002/eeg/s6_sub-10002_rs_eeg.set",
    "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10009/eeg/s6_sub-10009_rs_eeg.set",
    "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100012/eeg/s6_sub-100012_rs_eeg.set",
]

# Consensus Laplacian path
CONSENSUS_LAPLACIAN_PATH = "/home/muhibt/project/filter_identification/Consensus matrix/group_consensus_laplacian/all_consensus_average.npy"

# Preprocessing
BAND = (0.5, 40.0)
TARGET_SFREQ = 100.0

# Model settings
RIDGE_LAMBDA = 5e-3

# Model selection ranges
P_RANGE = [1, 2, 3, 5, 7, 10, 15, 20]
K_RANGE = [1, 2, 3, 4]

# Time-varying analysis
WINDOW_LENGTH_SEC = 10.0
WINDOW_OVERLAP = 0.5
MIN_WINDOWS = 5

# Output
OUT_DIR = Path("./3ad_3hc_lti_tv_comparison")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Helper Functions
# ============================================================================

def compute_confidence_interval(data: np.ndarray, confidence=0.95, axis=0):
    """Compute confidence interval using t-distribution."""
    n = data.shape[axis]
    mean = np.mean(data, axis=axis)
    sem = stats.sem(data, axis=axis)
    df = n - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    ci_margin = t_critical * sem
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin
    return mean, ci_lower, ci_upper, ci_margin

def load_and_preprocess_eeg(filepath: str, band: Tuple[float,float], 
                           target_sfreq: float) -> Tuple[np.ndarray, List[str]]:
    """Load and preprocess EEG."""
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
    """Robust loader that handles both adjacency and Laplacian matrices."""
    M = np.load(filepath)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.allclose(M, M.T, atol=1e-6):
        M = (M + M.T) / 2.0

    n = M.shape[0]

    row_sums = np.abs(M.sum(axis=1))
    diag_vals = np.diag(M)
    looks_like_laplacian = (row_sums.mean() < 1e-3 * np.abs(M).mean()) and (np.all(diag_vals >= -1e-6))

    if looks_like_laplacian:
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
        A = M
        if np.any(A < 0):
            A = np.clip(A, 0.0, None)
        d = A.sum(axis=1)
        d = np.where(d < 1e-10, 1e-10, d)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    L_norm = (L_norm + L_norm.T) / 2.0
    return L_norm

# ============================================================================
# GP-VAR Model Class
# ============================================================================

class GPVAR_SharedH:
    """GP-VAR with shared scalar coefficients using NORMALIZED Laplacian."""
    
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
        """Build design matrix for regression."""
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
        """Fit model via ridge regression."""
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
        """Compute spectral radius of the companion matrix."""
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
        """Compute AR transfer function G(ω, λ) in the graph spectral domain."""
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
            'G_phase': np.angle(G),
            'H_p': H_p
        }

# ============================================================================
# Model Selection with Detailed Results
# ============================================================================

def find_best_model_with_grid(X: np.ndarray, L_norm: np.ndarray, 
                               P_range: List[int] = P_RANGE,
                               K_range: List[int] = K_RANGE) -> Dict:
    """
    Find best P and K using BIC.
    Returns detailed model selection results including full grid for heatmap.
    """
    T = X.shape[1]
    T_train = int(0.70 * T)
    T_val = int(0.85 * T)
    
    X_train = X[:, :T_train]
    X_val = X[:, T_train:T_val]
    
    X_train = safe_zscore(X_train, X_train)
    X_val = safe_zscore(X_val, X_train)
    
    best_bic = np.inf
    best_P, best_K = None, None
    
    # Initialize grid for heatmap
    bic_grid = np.full((len(K_range), len(P_range)), np.nan)
    r2_grid = np.full((len(K_range), len(P_range)), np.nan)
    stable_grid = np.full((len(K_range), len(P_range)), False)
    
    # Store all model selection results
    model_selection_results = []
    
    for k_idx, K in enumerate(K_range):
        for p_idx, P in enumerate(P_range):
            try:
                m = GPVAR_SharedH(P=P, K=K, L_norm=L_norm)
                m.fit(X_train)
                
                rho = m.spectral_radius()
                stable = np.isfinite(rho) and rho < 0.99
                
                metrics = m.evaluate(X_val)
                
                bic_grid[k_idx, p_idx] = metrics['BIC']
                r2_grid[k_idx, p_idx] = metrics['R2']
                stable_grid[k_idx, p_idx] = stable
                
                model_selection_results.append({
                    'P': P,
                    'K': K,
                    'BIC': metrics['BIC'],
                    'R2': metrics['R2'],
                    'MSE': metrics['MSE'],
                    'rho': rho,
                    'stable': stable,
                    'success': True
                })
                
                if stable and metrics['BIC'] < best_bic:
                    best_bic = metrics['BIC']
                    best_P = P
                    best_K = K
            
            except Exception as e:
                model_selection_results.append({
                    'P': P,
                    'K': K,
                    'BIC': np.nan,
                    'R2': np.nan,
                    'MSE': np.nan,
                    'rho': np.nan,
                    'stable': False,
                    'success': False
                })
                continue
    
    if best_P is None:
        best_P, best_K = 5, 2
        print(f"    WARNING: All models failed or unstable, using fallback P={best_P}, K={best_K}")
    
    return {
        'best_P': best_P,
        'best_K': best_K,
        'best_BIC': best_bic,
        'model_selection_table': pd.DataFrame(model_selection_results),
        'bic_grid': bic_grid,
        'r2_grid': r2_grid,
        'stable_grid': stable_grid,
        'P_range': P_range,
        'K_range': K_range
    }

def plot_model_selection_heatmap(model_selection: Dict, subject_id: str, 
                                  group: str, save_dir: Path):
    """Create model selection heatmap for a single subject."""
    
    bic_grid = model_selection['bic_grid']
    r2_grid = model_selection['r2_grid']
    stable_grid = model_selection['stable_grid']
    P_range = model_selection['P_range']
    K_range = model_selection['K_range']
    best_P = model_selection['best_P']
    best_K = model_selection['best_K']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BIC Heatmap
    ax = axes[0]
    # Mask unstable models
    bic_masked = np.ma.masked_where(~stable_grid, bic_grid)
    im1 = ax.imshow(bic_masked, aspect='auto', cmap='viridis_r', origin='lower')
    
    # Mark unstable cells with X
    for k_idx in range(len(K_range)):
        for p_idx in range(len(P_range)):
            if not stable_grid[k_idx, p_idx]:
                ax.text(p_idx, k_idx, '✗', ha='center', va='center', 
                       fontsize=12, color='red', fontweight='bold')
            elif not np.isnan(bic_grid[k_idx, p_idx]):
                ax.text(p_idx, k_idx, f'{bic_grid[k_idx, p_idx]:.0f}', 
                       ha='center', va='center', fontsize=8, color='white')
    
    # Mark best model
    best_p_idx = P_range.index(best_P)
    best_k_idx = K_range.index(best_K)
    rect = plt.Rectangle((best_p_idx - 0.5, best_k_idx - 0.5), 1, 1, 
                         fill=False, edgecolor='lime', linewidth=3)
    ax.add_patch(rect)
    
    ax.set_xticks(range(len(P_range)))
    ax.set_xticklabels(P_range)
    ax.set_yticks(range(len(K_range)))
    ax.set_yticklabels(K_range)
    ax.set_xlabel('P (AR Order)', fontsize=12, fontweight='bold')
    ax.set_ylabel('K (Graph Filter Order)', fontsize=12, fontweight='bold')
    ax.set_title(f'BIC Grid (lower = better)\nBest: P={best_P}, K={best_K}', 
                fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax, label='BIC')
    
    # R² Heatmap
    ax = axes[1]
    r2_masked = np.ma.masked_where(~stable_grid, r2_grid)
    im2 = ax.imshow(r2_masked, aspect='auto', cmap='RdYlGn', origin='lower',
                   vmin=0, vmax=1)
    
    for k_idx in range(len(K_range)):
        for p_idx in range(len(P_range)):
            if not stable_grid[k_idx, p_idx]:
                ax.text(p_idx, k_idx, '✗', ha='center', va='center', 
                       fontsize=12, color='red', fontweight='bold')
            elif not np.isnan(r2_grid[k_idx, p_idx]):
                ax.text(p_idx, k_idx, f'{r2_grid[k_idx, p_idx]:.3f}', 
                       ha='center', va='center', fontsize=8, 
                       color='black' if r2_grid[k_idx, p_idx] > 0.5 else 'white')
    
    # Mark best model
    rect = plt.Rectangle((best_p_idx - 0.5, best_k_idx - 0.5), 1, 1, 
                         fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)
    
    ax.set_xticks(range(len(P_range)))
    ax.set_xticklabels(P_range)
    ax.set_yticks(range(len(K_range)))
    ax.set_yticklabels(K_range)
    ax.set_xlabel('P (AR Order)', fontsize=12, fontweight='bold')
    ax.set_ylabel('K (Graph Filter Order)', fontsize=12, fontweight='bold')
    ax.set_title(f'R² Grid (higher = better)\nBest: P={best_P}, K={best_K}', 
                fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='R²')
    
    plt.suptitle(f'{group} Subject: {subject_id}\nModel Selection Grid (✗ = unstable)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / f'{group}_{subject_id}_model_selection_heatmap.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved heatmap: {savepath}")
    
    return savepath

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
    """Fit separate models for each time window."""
    windows = split_into_windows(X_std, window_length_sec, overlap, fs)
    
    tv_results = []
    
    for w_idx, (start_idx, end_idx, X_win) in enumerate(windows):
        try:
            model = GPVAR_SharedH(P=P, K=K, L_norm=L_norm)
            model.fit(X_win)
            
            rho = model.spectral_radius()
            if not np.isfinite(rho) or rho >= 0.99:
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
        
        except Exception:
            continue
    
    return tv_results

# ============================================================================
# Single Subject Analysis
# ============================================================================

def analyze_single_subject(filepath: str, L_norm: np.ndarray, group: str,
                           save_dir: Path) -> Optional[Dict]:
    """
    Analyze a single subject. Returns results dict or None if failed.
    """
    try:
        subject_id = Path(filepath).stem.replace('s6_', '').replace('_rs-hep_eeg', '').replace('_rs_eeg', '')
        print(f"\n  Analyzing {group} subject: {subject_id}")
        
        # Load EEG
        X, ch_names = load_and_preprocess_eeg(filepath, BAND, TARGET_SFREQ)
        n_channels, n_samples = X.shape
        print(f"    Loaded: {n_channels} channels, {n_samples/TARGET_SFREQ:.1f}s duration")
        
        # Check size compatibility
        if L_norm.shape[0] != n_channels:
            print(f"    SKIP: Channel mismatch (Laplacian={L_norm.shape[0]}, EEG={n_channels})")
            return None
        
        # Standardize
        X_std = safe_zscore(X, X)
        
        # Model selection with grid
        print(f"    Running model selection...")
        model_selection = find_best_model_with_grid(X, L_norm)
        best_P = model_selection['best_P']
        best_K = model_selection['best_K']
        print(f"    Selected: P={best_P}, K={best_K}, BIC={model_selection['best_BIC']:.2f}")
        
        # Save model selection heatmap
        plot_model_selection_heatmap(model_selection, subject_id, group, save_dir)
        
        # Fit LTI
        print(f"    Fitting LTI model...")
        lti_model = GPVAR_SharedH(P=best_P, K=best_K, L_norm=L_norm)
        lti_model.fit(X_std)
        lti_rho = lti_model.spectral_radius()
        lti_metrics = lti_model.evaluate(X_std)
        
        if not np.isfinite(lti_rho) or lti_rho >= 1.0:
            print(f"    SKIP: LTI unstable (ρ={lti_rho:.3f})")
            return None
        
        print(f"    LTI: R²={lti_metrics['R2']:.4f}, ρ={lti_rho:.3f}")
        
        # Fit TV models
        print(f"    Fitting TV models...")
        tv_results = compute_tv_models(X_std, L_norm, best_P, best_K, 
                                       WINDOW_LENGTH_SEC, WINDOW_OVERLAP, TARGET_SFREQ)
        
        if len(tv_results) < MIN_WINDOWS:
            print(f"    SKIP: Too few windows ({len(tv_results)} < {MIN_WINDOWS})")
            return None
        
        print(f"    TV: {len(tv_results)} stable windows")
        
        # Compute transfer functions
        omegas = np.linspace(0, np.pi, 256)
        lti_tf = lti_model.compute_transfer_function(omegas)
        
        n_windows = len(tv_results)
        n_omegas = len(omegas)
        n_lambdas = len(lti_tf['lambdas'])
        
        G_tv_all = np.zeros((n_windows, n_omegas, n_lambdas))
        for w_idx, tv_res in enumerate(tv_results):
            tv_tf = tv_res['model'].compute_transfer_function(omegas)
            G_tv_all[w_idx, :, :] = tv_tf['G_mag']
        
        G_tv_mean = G_tv_all.mean(axis=0)
        G_tv_std = G_tv_all.std(axis=0)
        
        # Compute summary metrics
        msd_per_window = np.mean((G_tv_all - lti_tf['G_mag'][None, :, :])**2, axis=(1, 2))
        mean_msd = msd_per_window.mean()
        
        cv = G_tv_std / (G_tv_mean + 1e-8)
        mean_cv = np.mean(cv)
        
        print(f"    MSD={mean_msd:.6f}, CV={mean_cv:.4f}")
        
        return {
            'subject_id': subject_id,
            'group': group,
            'filepath': filepath,
            'n_channels': n_channels,
            'n_samples': n_samples,
            'duration': n_samples / TARGET_SFREQ,
            'best_P': best_P,
            'best_K': best_K,
            'best_BIC': model_selection['best_BIC'],
            'model_selection': model_selection,
            'lti_model': lti_model,
            'lti_metrics': lti_metrics,
            'lti_rho': lti_rho,
            'lti_R2': lti_metrics['R2'],
            'lti_BIC': lti_metrics['BIC'],
            'tv_results': tv_results,
            'n_windows': len(tv_results),
            'tv_R2_mean': np.mean([r['metrics']['R2'] for r in tv_results]),
            'tv_R2_std': np.std([r['metrics']['R2'] for r in tv_results]),
            'tv_rho_mean': np.mean([r['rho'] for r in tv_results]),
            'tv_rho_std': np.std([r['rho'] for r in tv_results]),
            'G_lti': lti_tf['G_mag'],
            'G_tv_mean': G_tv_mean,
            'G_tv_std': G_tv_std,
            'G_tv_all': G_tv_all,
            'omegas': omegas,
            'freqs_hz': omegas * TARGET_SFREQ / (2 * np.pi),
            'lambdas': lti_tf['lambdas'],
            'msd_per_window': msd_per_window,
            'mean_msd': mean_msd,
            'mean_cv': mean_cv,
        }
    
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Group Visualization and Comparison
# ============================================================================

def save_model_selection_csv(ad_results: List[Dict], hc_results: List[Dict], 
                             save_dir: Path):
    """Save all model selection results to CSV."""
    
    print("\nSaving model selection results to CSV...")
    
    # Individual subject tables
    all_tables = []
    for result in ad_results + hc_results:
        ms_table = result['model_selection']['model_selection_table'].copy()
        ms_table['subject_id'] = result['subject_id']
        ms_table['group'] = result['group']
        all_tables.append(ms_table)
    
    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_csv = save_dir / "all_subjects_model_selection.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"  Saved: {combined_csv}")
    
    # Summary table
    summary_data = []
    for result in ad_results + hc_results:
        summary_data.append({
            'subject_id': result['subject_id'],
            'group': result['group'],
            'selected_P': result['best_P'],
            'selected_K': result['best_K'],
            'selected_BIC': result['best_BIC'],
            'lti_R2': result['lti_R2'],
            'lti_rho': result['lti_rho'],
            'tv_R2_mean': result['tv_R2_mean'],
            'n_windows': result['n_windows'],
            'mean_msd': result['mean_msd'],
            'mean_cv': result['mean_cv']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = save_dir / "model_selection_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved: {summary_csv}")
    
    return summary_df

def plot_group_model_selection_summary(ad_results: List[Dict], hc_results: List[Dict],
                                       save_dir: Path):
    """Create summary plot of model selection across groups."""
    
    print("\nCreating model selection summary plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data
    ad_P = [r['best_P'] for r in ad_results]
    hc_P = [r['best_P'] for r in hc_results]
    ad_K = [r['best_K'] for r in ad_results]
    hc_K = [r['best_K'] for r in hc_results]
    ad_ids = [r['subject_id'] for r in ad_results]
    hc_ids = [r['subject_id'] for r in hc_results]
    
    # Plot 1: P values comparison
    ax = axes[0, 0]
    x = np.arange(max(len(ad_P), len(hc_P)))
    width = 0.35
    if len(ad_P) > 0:
        ax.bar(x[:len(ad_P)] - width/2, ad_P, width, label='AD', color='red', alpha=0.7)
    if len(hc_P) > 0:
        ax.bar(x[:len(hc_P)] + width/2, hc_P, width, label='HC', color='blue', alpha=0.7)
    ax.set_xlabel('Subject Index')
    ax.set_ylabel('Selected P (AR Order)')
    ax.set_title('Selected P Values by Subject', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: K values comparison
    ax = axes[0, 1]
    if len(ad_K) > 0:
        ax.bar(x[:len(ad_K)] - width/2, ad_K, width, label='AD', color='red', alpha=0.7)
    if len(hc_K) > 0:
        ax.bar(x[:len(hc_K)] + width/2, hc_K, width, label='HC', color='blue', alpha=0.7)
    ax.set_xlabel('Subject Index')
    ax.set_ylabel('Selected K (Graph Filter Order)')
    ax.set_title('Selected K Values by Subject', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: P vs K scatter
    ax = axes[0, 2]
    if len(ad_P) > 0:
        for i, (p, k, sid) in enumerate(zip(ad_P, ad_K, ad_ids)):
            ax.scatter(p, k, s=150, color='red', alpha=0.7, edgecolors='black', zorder=3)
            ax.annotate(sid, (p, k), xytext=(5, 5), textcoords='offset points', fontsize=8)
    if len(hc_P) > 0:
        for i, (p, k, sid) in enumerate(zip(hc_P, hc_K, hc_ids)):
            ax.scatter(p, k, s=150, color='blue', alpha=0.7, edgecolors='black', zorder=3)
            ax.annotate(sid, (p, k), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add legend handles
    ax.scatter([], [], s=100, color='red', alpha=0.7, edgecolors='black', label='AD')
    ax.scatter([], [], s=100, color='blue', alpha=0.7, edgecolors='black', label='HC')
    ax.set_xlabel('P (AR Order)')
    ax.set_ylabel('K (Graph Filter Order)')
    ax.set_title('P vs K Selection per Subject', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: LTI R² comparison
    ax = axes[1, 0]
    ad_r2 = [r['lti_R2'] for r in ad_results]
    hc_r2 = [r['lti_R2'] for r in hc_results]
    if len(ad_r2) > 0:
        ax.bar(x[:len(ad_r2)] - width/2, ad_r2, width, label='AD', color='red', alpha=0.7)
    if len(hc_r2) > 0:
        ax.bar(x[:len(hc_r2)] + width/2, hc_r2, width, label='HC', color='blue', alpha=0.7)
    ax.set_xlabel('Subject Index')
    ax.set_ylabel('LTI R²')
    ax.set_title('LTI Model Fit (R²) by Subject', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 5: TV R² comparison
    ax = axes[1, 1]
    ad_tv_r2 = [r['tv_R2_mean'] for r in ad_results]
    hc_tv_r2 = [r['tv_R2_mean'] for r in hc_results]
    if len(ad_tv_r2) > 0:
        ax.bar(x[:len(ad_tv_r2)] - width/2, ad_tv_r2, width, label='AD', color='red', alpha=0.7)
    if len(hc_tv_r2) > 0:
        ax.bar(x[:len(hc_tv_r2)] + width/2, hc_tv_r2, width, label='HC', color='blue', alpha=0.7)
    ax.set_xlabel('Subject Index')
    ax.set_ylabel('TV Mean R²')
    ax.set_title('TV Model Fit (Mean R²) by Subject', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = [
        "MODEL SELECTION SUMMARY",
        "=" * 40,
        "",
        f"AD Group (n={len(ad_results)})",
    ]
    if len(ad_P) > 0:
        summary_text.extend([
            f"  P: {np.mean(ad_P):.1f} ± {np.std(ad_P):.1f}",
            f"  K: {np.mean(ad_K):.1f} ± {np.std(ad_K):.1f}",
            f"  LTI R²: {np.mean(ad_r2):.3f} ± {np.std(ad_r2):.3f}",
            f"  TV R²: {np.mean(ad_tv_r2):.3f} ± {np.std(ad_tv_r2):.3f}",
        ])
    summary_text.append("")
    summary_text.append(f"HC Group (n={len(hc_results)})")
    if len(hc_P) > 0:
        summary_text.extend([
            f"  P: {np.mean(hc_P):.1f} ± {np.std(hc_P):.1f}",
            f"  K: {np.mean(hc_K):.1f} ± {np.std(hc_K):.1f}",
            f"  LTI R²: {np.mean(hc_r2):.3f} ± {np.std(hc_r2):.3f}",
            f"  TV R²: {np.mean(hc_tv_r2):.3f} ± {np.std(hc_tv_r2):.3f}",
        ])
    
    ax.text(0.1, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Model Selection Summary: 3 AD vs 3 HC\nBIC-Based Hyperparameter Selection', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / 'model_selection_summary.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath}")

def plot_ad_vs_hc_comparison(ad_results: List[Dict], hc_results: List[Dict],
                             save_dir: Path):
    """Create comprehensive AD vs HC comparison plots for LTI and TV models."""
    
    print("\nCreating AD vs HC comparison plots...")
    
    if len(ad_results) == 0 or len(hc_results) == 0:
        print("  WARNING: Not enough subjects for comparison")
        return
    
    # Extract common frequency and lambda arrays
    freqs_hz = ad_results[0]['freqs_hz']
    lambdas = ad_results[0]['lambdas']
    
    # Compute group averages
    ad_G_lti_all = np.array([r['G_lti'] for r in ad_results])
    hc_G_lti_all = np.array([r['G_lti'] for r in hc_results])
    ad_G_tv_all = np.array([r['G_tv_mean'] for r in ad_results])
    hc_G_tv_all = np.array([r['G_tv_mean'] for r in hc_results])
    
    # Mean across subjects
    ad_G_lti_mean = ad_G_lti_all.mean(axis=0)
    hc_G_lti_mean = hc_G_lti_all.mean(axis=0)
    ad_G_tv_mean = ad_G_tv_all.mean(axis=0)
    hc_G_tv_mean = hc_G_tv_all.mean(axis=0)
    
    # Std for error bands
    ad_G_lti_std = ad_G_lti_all.std(axis=0)
    hc_G_lti_std = hc_G_lti_all.std(axis=0)
    ad_G_tv_std = ad_G_tv_all.std(axis=0)
    hc_G_tv_std = hc_G_tv_all.std(axis=0)
    
    # Mode-averaged frequency response
    ad_lti_freq = ad_G_lti_mean.mean(axis=1)
    hc_lti_freq = hc_G_lti_mean.mean(axis=1)
    ad_tv_freq = ad_G_tv_mean.mean(axis=1)
    hc_tv_freq = hc_G_tv_mean.mean(axis=1)
    
    # Frequency-averaged mode response
    ad_lti_mode = ad_G_lti_mean.mean(axis=0)
    hc_lti_mode = hc_G_lti_mean.mean(axis=0)
    ad_tv_mode = ad_G_tv_mean.mean(axis=0)
    hc_tv_mode = hc_G_tv_mean.mean(axis=0)
    
    # Colors
    color_ad = '#E74C3C'
    color_hc = '#3498DB'
    
    # =================================================================
    # Figure 1: Transfer Function Heatmaps
    # =================================================================
    
    fig1 = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig1, hspace=0.35, wspace=0.3)
    
    # Row 1: LTI Transfer Functions
    ax1 = fig1.add_subplot(gs[0, 0])
    im1 = ax1.imshow(ad_G_lti_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax1.set_title('AD LTI |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax1.set_xlabel('λ (Graph Frequency)')
    ax1.set_ylabel('f (Hz)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig1.add_subplot(gs[0, 1])
    im2 = ax2.imshow(hc_G_lti_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax2.set_title('HC LTI |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax2.set_xlabel('λ (Graph Frequency)')
    ax2.set_ylabel('f (Hz)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig1.add_subplot(gs[0, 2])
    diff_lti = ad_G_lti_mean - hc_G_lti_mean
    vmax_lti = np.abs(diff_lti).max()
    im3 = ax3.imshow(diff_lti, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()],
                     vmin=-vmax_lti, vmax=vmax_lti)
    ax3.set_title('LTI Difference (AD - HC)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('λ (Graph Frequency)')
    ax3.set_ylabel('f (Hz)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Row 2: TV Transfer Functions  
    ax4 = fig1.add_subplot(gs[1, 0])
    im4 = ax4.imshow(ad_G_tv_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax4.set_title('AD TV |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax4.set_xlabel('λ (Graph Frequency)')
    ax4.set_ylabel('f (Hz)')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig1.add_subplot(gs[1, 1])
    im5 = ax5.imshow(hc_G_tv_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax5.set_title('HC TV |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax5.set_xlabel('λ (Graph Frequency)')
    ax5.set_ylabel('f (Hz)')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = fig1.add_subplot(gs[1, 2])
    diff_tv = ad_G_tv_mean - hc_G_tv_mean
    vmax_tv = np.abs(diff_tv).max()
    im6 = ax6.imshow(diff_tv, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()],
                     vmin=-vmax_tv, vmax=vmax_tv)
    ax6.set_title('TV Difference (AD - HC)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('λ (Graph Frequency)')
    ax6.set_ylabel('f (Hz)')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # Column 4: Variance maps
    ax7 = fig1.add_subplot(gs[0, 3])
    ad_var = ad_G_lti_all.var(axis=0)
    im7 = ax7.imshow(ad_var, aspect='auto', origin='lower', cmap='YlOrRd',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax7.set_title('AD LTI Variance (across subjects)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('λ')
    ax7.set_ylabel('f (Hz)')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    ax8 = fig1.add_subplot(gs[1, 3])
    hc_var = hc_G_lti_all.var(axis=0)
    im8 = ax8.imshow(hc_var, aspect='auto', origin='lower', cmap='YlOrRd',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax8.set_title('HC LTI Variance (across subjects)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('λ')
    ax8.set_ylabel('f (Hz)')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # Row 3: Line plots
    ax9 = fig1.add_subplot(gs[2, :2])
    ax9.plot(freqs_hz, ad_lti_freq, color=color_ad, linewidth=2.5, label='AD LTI')
    ax9.plot(freqs_hz, hc_lti_freq, color=color_hc, linewidth=2.5, label='HC LTI')
    ax9.plot(freqs_hz, ad_tv_freq, '--', color=color_ad, linewidth=2.5, label='AD TV')
    ax9.plot(freqs_hz, hc_tv_freq, '--', color=color_hc, linewidth=2.5, label='HC TV')
    ax9.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax9.set_title('Mode-Averaged Frequency Response', fontsize=12, fontweight='bold')
    ax9.legend(ncol=2, fontsize=10)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    ax10 = fig1.add_subplot(gs[2, 2:])
    ax10.plot(lambdas, ad_lti_mode, color=color_ad, linewidth=2.5, label='AD LTI')
    ax10.plot(lambdas, hc_lti_mode, color=color_hc, linewidth=2.5, label='HC LTI')
    ax10.plot(lambdas, ad_tv_mode, '--', color=color_ad, linewidth=2.5, label='AD TV')
    ax10.plot(lambdas, hc_tv_mode, '--', color=color_hc, linewidth=2.5, label='HC TV')
    ax10.set_xlabel('λ (Graph Frequency)', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Frequency-Averaged |G|', fontsize=12, fontweight='bold')
    ax10.set_title('Frequency-Averaged Graph Mode Response', fontsize=12, fontweight='bold')
    ax10.legend(ncol=2, fontsize=10)
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle('AD vs HC: Transfer Function Comparison (LTI and TV)\n3 AD vs 3 HC Subjects', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    savepath1 = save_dir / 'ad_vs_hc_transfer_functions.png'
    plt.savefig(savepath1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath1}")
    
    # =================================================================
    # Figure 2: Detailed Frequency Response Comparison
    # =================================================================
    
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define frequency bands for annotation
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
             'Beta': (13, 30), 'Gamma': (30, 40)}
    band_colors = {'Delta': '#E8E8E8', 'Theta': '#D5E5F5', 'Alpha': '#D5F5E5',
                   'Beta': '#FFF8DC', 'Gamma': '#FFE5E5'}
    
    # Plot 1: LTI Mode-Averaged Frequency Response
    ax = axes[0, 0]
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.15, color=band_colors[band_name])
    
    # Compute individual subject curves for error bands
    ad_lti_freq_all = np.array([r['G_lti'].mean(axis=1) for r in ad_results])
    hc_lti_freq_all = np.array([r['G_lti'].mean(axis=1) for r in hc_results])
    
    ad_mean_freq = ad_lti_freq_all.mean(axis=0)
    hc_mean_freq = hc_lti_freq_all.mean(axis=0)
    ad_std_freq = ad_lti_freq_all.std(axis=0)
    hc_std_freq = hc_lti_freq_all.std(axis=0)
    
    ax.plot(freqs_hz, ad_mean_freq, color=color_ad, linewidth=3, 
           label=f'AD (n={len(ad_results)})')
    ax.fill_between(freqs_hz, ad_mean_freq - ad_std_freq, ad_mean_freq + ad_std_freq,
                   color=color_ad, alpha=0.2)
    ax.plot(freqs_hz, hc_mean_freq, color=color_hc, linewidth=3, 
           label=f'HC (n={len(hc_results)})')
    ax.fill_between(freqs_hz, hc_mean_freq - hc_std_freq, hc_mean_freq + hc_std_freq,
                   color=color_hc, alpha=0.2)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title('(A) LTI: Mode-Averaged Frequency Response', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    # Plot 2: TV Mode-Averaged Frequency Response
    ax = axes[0, 1]
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.15, color=band_colors[band_name])
    
    ad_tv_freq_all = np.array([r['G_tv_mean'].mean(axis=1) for r in ad_results])
    hc_tv_freq_all = np.array([r['G_tv_mean'].mean(axis=1) for r in hc_results])
    
    ad_tv_mean_freq = ad_tv_freq_all.mean(axis=0)
    hc_tv_mean_freq = hc_tv_freq_all.mean(axis=0)
    ad_tv_std_freq = ad_tv_freq_all.std(axis=0)
    hc_tv_std_freq = hc_tv_freq_all.std(axis=0)
    
    ax.plot(freqs_hz, ad_tv_mean_freq, '--', color=color_ad, linewidth=3,
           label=f'AD (n={len(ad_results)})')
    ax.fill_between(freqs_hz, ad_tv_mean_freq - ad_tv_std_freq, 
                   ad_tv_mean_freq + ad_tv_std_freq, color=color_ad, alpha=0.2)
    ax.plot(freqs_hz, hc_tv_mean_freq, '--', color=color_hc, linewidth=3,
           label=f'HC (n={len(hc_results)})')
    ax.fill_between(freqs_hz, hc_tv_mean_freq - hc_tv_std_freq, 
                   hc_tv_mean_freq + hc_tv_std_freq, color=color_hc, alpha=0.2)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title('(B) TV: Mode-Averaged Frequency Response', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    # Plot 3: LTI Graph Mode Response
    ax = axes[1, 0]
    ad_lti_mode_all = np.array([r['G_lti'].mean(axis=0) for r in ad_results])
    hc_lti_mode_all = np.array([r['G_lti'].mean(axis=0) for r in hc_results])
    
    ad_mean_mode = ad_lti_mode_all.mean(axis=0)
    hc_mean_mode = hc_lti_mode_all.mean(axis=0)
    ad_std_mode = ad_lti_mode_all.std(axis=0)
    hc_std_mode = hc_lti_mode_all.std(axis=0)
    
    ax.plot(lambdas, ad_mean_mode, color=color_ad, linewidth=3,
           label=f'AD (n={len(ad_results)})')
    ax.fill_between(lambdas, ad_mean_mode - ad_std_mode, ad_mean_mode + ad_std_mode,
                   color=color_ad, alpha=0.2)
    ax.plot(lambdas, hc_mean_mode, color=color_hc, linewidth=3,
           label=f'HC (n={len(hc_results)})')
    ax.fill_between(lambdas, hc_mean_mode - hc_std_mode, hc_mean_mode + hc_std_mode,
                   color=color_hc, alpha=0.2)
    
    ax.set_xlabel('λ (Graph Frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title('(C) LTI: Graph Mode Response', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: TV Graph Mode Response
    ax = axes[1, 1]
    ad_tv_mode_all = np.array([r['G_tv_mean'].mean(axis=0) for r in ad_results])
    hc_tv_mode_all = np.array([r['G_tv_mean'].mean(axis=0) for r in hc_results])
    
    ad_tv_mean_mode = ad_tv_mode_all.mean(axis=0)
    hc_tv_mean_mode = hc_tv_mode_all.mean(axis=0)
    ad_tv_std_mode = ad_tv_mode_all.std(axis=0)
    hc_tv_std_mode = hc_tv_mode_all.std(axis=0)
    
    ax.plot(lambdas, ad_tv_mean_mode, '--', color=color_ad, linewidth=3,
           label=f'AD (n={len(ad_results)})')
    ax.fill_between(lambdas, ad_tv_mean_mode - ad_tv_std_mode, 
                   ad_tv_mean_mode + ad_tv_std_mode, color=color_ad, alpha=0.2)
    ax.plot(lambdas, hc_tv_mean_mode, '--', color=color_hc, linewidth=3,
           label=f'HC (n={len(hc_results)})')
    ax.fill_between(lambdas, hc_tv_mean_mode - hc_tv_std_mode, 
                   hc_tv_mean_mode + hc_tv_std_mode, color=color_hc, alpha=0.2)
    
    ax.set_xlabel('λ (Graph Frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title('(D) TV: Graph Mode Response', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('AD vs HC: Detailed Transfer Function Comparison\nShaded regions: ±1 SD across subjects', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    savepath2 = save_dir / 'ad_vs_hc_detailed_comparison.png'
    plt.savefig(savepath2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath2}")
    
    # =================================================================
    # Figure 3: Individual Subject Traces
    # =================================================================
    
    fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # AD LTI individual traces
    ax = axes[0, 0]
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.1, color=band_colors[band_name])
    for r in ad_results:
        ax.plot(freqs_hz, r['G_lti'].mean(axis=1), color=color_ad, alpha=0.5, linewidth=1.5)
    ax.plot(freqs_hz, ad_mean_freq, color='darkred', linewidth=4, label='Group Mean')
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title(f'(A) AD LTI Individual Subjects (n={len(ad_results)})', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    # HC LTI individual traces
    ax = axes[0, 1]
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.1, color=band_colors[band_name])
    for r in hc_results:
        ax.plot(freqs_hz, r['G_lti'].mean(axis=1), color=color_hc, alpha=0.5, linewidth=1.5)
    ax.plot(freqs_hz, hc_mean_freq, color='darkblue', linewidth=4, label='Group Mean')
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title(f'(B) HC LTI Individual Subjects (n={len(hc_results)})', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    # AD TV individual traces
    ax = axes[1, 0]
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.1, color=band_colors[band_name])
    for r in ad_results:
        ax.plot(freqs_hz, r['G_tv_mean'].mean(axis=1), '--', color=color_ad, alpha=0.5, linewidth=1.5)
    ax.plot(freqs_hz, ad_tv_mean_freq, '--', color='darkred', linewidth=4, label='Group Mean')
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title(f'(C) AD TV Individual Subjects (n={len(ad_results)})', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    # HC TV individual traces
    ax = axes[1, 1]
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.1, color=band_colors[band_name])
    for r in hc_results:
        ax.plot(freqs_hz, r['G_tv_mean'].mean(axis=1), '--', color=color_hc, alpha=0.5, linewidth=1.5)
    ax.plot(freqs_hz, hc_tv_mean_freq, '--', color='darkblue', linewidth=4, label='Group Mean')
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mode-Averaged |G|', fontsize=12, fontweight='bold')
    ax.set_title(f'(D) HC TV Individual Subjects (n={len(hc_results)})', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([freqs_hz.min(), freqs_hz.max()])
    
    plt.suptitle('AD vs HC: Individual Subject Frequency Responses', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    savepath3 = save_dir / 'ad_vs_hc_individual_traces.png'
    plt.savefig(savepath3, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath3}")

def compute_and_save_statistics(ad_results: List[Dict], hc_results: List[Dict],
                                save_dir: Path):
    """Compute and save statistical comparisons."""
    
    print("\nComputing statistics...")
    
    if len(ad_results) < 2 or len(hc_results) < 2:
        print("  WARNING: Not enough subjects for statistical tests")
        return
    
    # Metrics to compare
    metrics = ['lti_R2', 'lti_rho', 'lti_BIC', 'tv_R2_mean', 'tv_rho_mean', 
               'mean_msd', 'mean_cv', 'best_P', 'best_K', 'n_windows']
    
    stats_data = []
    for metric in metrics:
        ad_vals = np.array([r[metric] for r in ad_results])
        hc_vals = np.array([r[metric] for r in hc_results])
        
        # T-test
        t_stat, p_val = stats.ttest_ind(ad_vals, hc_vals, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((ad_vals.std()**2 + hc_vals.std()**2) / 2)
        cohens_d = (ad_vals.mean() - hc_vals.mean()) / (pooled_std + 1e-10)
        
        stats_data.append({
            'metric': metric,
            'AD_mean': ad_vals.mean(),
            'AD_std': ad_vals.std(),
            'HC_mean': hc_vals.mean(),
            'HC_std': hc_vals.std(),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < 0.05
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_csv = save_dir / 'group_statistics.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")
    
    # Print summary
    print("\n  Statistical Summary:")
    print(stats_df.to_string(index=False))
    
    return stats_df

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Main analysis: 3 AD vs 3 HC comparison."""
    
    print("="*80)
    print("LTI vs TV GP-VAR ANALYSIS: 3 AD vs 3 HC COMPARISON")
    print("="*80)
    
    # Load consensus Laplacian
    print("\nLoading consensus Laplacian...")
    L_norm = load_consensus_laplacian(CONSENSUS_LAPLACIAN_PATH)
    print(f"Laplacian shape: {L_norm.shape}")
    
    # Process AD subjects
    print("\n" + "="*80)
    print("Processing AD Group")
    print("="*80)
    ad_results = []
    for filepath in AD_PATHS:
        result = analyze_single_subject(filepath, L_norm, 'AD', OUT_DIR)
        if result is not None:
            ad_results.append(result)
    
    # Process HC subjects
    print("\n" + "="*80)
    print("Processing HC Group")
    print("="*80)
    hc_results = []
    for filepath in HC_PATHS:
        result = analyze_single_subject(filepath, L_norm, 'HC', OUT_DIR)
        if result is not None:
            hc_results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"AD: {len(ad_results)}/{len(AD_PATHS)} subjects successful")
    print(f"HC: {len(hc_results)}/{len(HC_PATHS)} subjects successful")
    
    if len(ad_results) == 0 or len(hc_results) == 0:
        print("\nERROR: Not enough subjects for comparison")
        return
    
    # Save model selection CSV
    summary_df = save_model_selection_csv(ad_results, hc_results, OUT_DIR)
    
    # Create summary plots
    plot_group_model_selection_summary(ad_results, hc_results, OUT_DIR)
    
    # Create AD vs HC comparison plots
    plot_ad_vs_hc_comparison(ad_results, hc_results, OUT_DIR)
    
    # Compute statistics
    stats_df = compute_and_save_statistics(ad_results, hc_results, OUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to: {OUT_DIR}")
    print("\nGenerated files:")
    print("  - Individual subject model selection heatmaps (*.png)")
    print("  - all_subjects_model_selection.csv")
    print("  - model_selection_summary.csv")
    print("  - model_selection_summary.png")
    print("  - ad_vs_hc_transfer_functions.png")
    print("  - ad_vs_hc_detailed_comparison.png")
    print("  - ad_vs_hc_individual_traces.png")
    print("  - group_statistics.csv")
    print("="*80)

if __name__ == "__main__":
    main()
