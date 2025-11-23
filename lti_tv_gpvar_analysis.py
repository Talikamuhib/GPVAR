"""
LTI vs Time-Varying GP-VAR Analysis Pipeline - CORRECTED VERSION
==================================================================

CRITICAL FIXES APPLIED:
1. Global z-scoring (no fake time-variation from window scaling)
2. Proper surrogate-based null distribution
3. Stability guards in transfer function computation
4. Direct coefficient variation metrics
5. Fixed group comparisons

Goal: Determine if EEG dynamics are time-varying by comparing:
  - LTI model (single fit on entire data)
  - TV models (separate fits on time windows)
  - Statistical testing with proper null distribution
  - Multi-subject analysis
  - AD vs HC group comparison
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
from tqdm import tqdm
import json

# ============================================================================
# Configuration
# ============================================================================

# Paths
CONSENSUS_LAPLACIAN_PATH = "/home/muhibt/project/filter_identification/Consensus matrix/group_consensus_laplacian/all_consensus_average.npy"

# List of subject files - FILL THIS IN
SUBJECT_FILES = {
    'AD': [
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30018/eeg/s6_sub-30018_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30026/eeg/s6_sub-30026_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30011/eeg/s6_sub-30011_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30009/eeg/s6_sub-30009_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30012/eeg/s6_sub-30012_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30002/eeg/s6_sub-30002_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30017/eeg/s6_sub-30017_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30001/eeg/s6_sub-30001_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30029/eeg/s6_sub-30029_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30015/eeg/s6_sub-30015_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30013/eeg/s6_sub-30013_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30008/eeg/s6_sub-30008_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30031/eeg/s6_sub-30031_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30022/eeg/s6_sub-30022_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30020/eeg/s6_sub-30020_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/AR/sub-30004/eeg/s6_sub-30004_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30003/eeg/s6_sub-30003_rs-hep_eeg.set',
        '/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30007/eeg/s6_sub-30007_rs-hep_eeg.set',
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30005/eeg/s6_sub-30005_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30006/eeg/s6_sub-30006_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30010/eeg/s6_sub-30010_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30014/eeg/s6_sub-30014_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30016/eeg/s6_sub-30016_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30019/eeg/s6_sub-30019_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30021/eeg/s6_sub-30021_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30023/eeg/s6_sub-30023_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30024/eeg/s6_sub-30024_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30025/eeg/s6_sub-30025_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30027/eeg/s6_sub-30027_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30028/eeg/s6_sub-30028_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30030/eeg/s6_sub-30030_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30032/eeg/s6_sub-30032_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30033/eeg/s6_sub-30033_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30034/eeg/s6_sub-30034_rs-hep_eeg.set",
        "/home/muhibt/project/filter_identification/data/synapse_data/1_AD/CL/sub-30035/eeg/s6_sub-30035_rs-hep_eeg.set",
    ],
    'HC': [
        "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10002/eeg/s6_sub-10002_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10009/eeg/s6_sub-10009_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100012/eeg/s6_sub-100012_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100015/eeg/s6_sub-100015_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100020/eeg/s6_sub-100020_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100035/eeg/s6_sub-100035_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100028/eeg/s6_sub-100028_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10006/eeg/s6_sub-10006_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10007/eeg/s6_sub-10007_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100033/eeg/s6_sub-100033_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100022/eeg/s6_sub-100022_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100031/eeg/s6_sub-100031_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10003/eeg/s6_sub-10003_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100026/eeg/s6_sub-100026_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100030/eeg/s6_sub-100030_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100018/eeg/s6_sub-100018_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100024/eeg/s6_sub-100024_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-100038/eeg/s6_sub-100038_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/AR/sub-10004/eeg/s6_sub-10004_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10001/eeg/s6_sub-10001_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10005/eeg/s6_sub-10005_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-10008/eeg/s6_sub-10008_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100010/eeg/s6_sub-100010_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100011/eeg/s6_sub-100011_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100014/eeg/s6_sub-100014_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100017/eeg/s6_sub-100017_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100021/eeg/s6_sub-100021_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100029/eeg/s6_sub-100029_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100034/eeg/s6_sub-100034_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100037/eeg/s6_sub-100037_rs_eeg.set",
            "/home/muhibt/project/filter_identification/data/synapse_data/5_HC/CL/sub-100043/eeg/s6_sub-100043_rs_eeg.set",
    ]
}

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

# Statistical testing
N_SURROGATES = 200
ALPHA = 0.05

# Output
OUT_DIR = Path("./lti_vs_tv_analysis_corrected")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Helper Functions
# ============================================================================

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
    """Load consensus Laplacian."""
    L = np.load(filepath)
    if not np.allclose(L, L.T):
        L = (L + L.T) / 2
    eigvals = np.linalg.eigvalsh(L)
    if eigvals.min() < -1e-8:
        L = L - eigvals.min() * np.eye(L.shape[0])
    return L


def circular_shift_surrogate(X_std: np.ndarray, rng) -> np.ndarray:
    """
    Create surrogate by circular shifting each channel independently.
    Preserves per-channel spectrum but destroys aligned time structure.
    """
    n, T = X_std.shape
    Xs = np.empty_like(X_std)
    for i in range(n):
        shift = rng.integers(0, T)
        Xs[i] = np.roll(X_std[i], shift)
    return Xs

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
            omegas = np.linspace(0, np.pi, 128)
        
        lambdas = self.eigenvalues
        P, K = self.P, self.K
        
        # Compute H_p(λ)
        H_p = np.zeros((P, len(lambdas)), dtype=np.complex128)
        for p in range(P):
            for i, lam in enumerate(lambdas):
                val = 0.0
                for k in range(K + 1):
                    val += self.h[p*(K+1) + k] * (lam ** k)
                H_p[p, i] = val
        
        # Compute G(ω, λ) with stability guard
        G = np.zeros((len(omegas), len(lambdas)), dtype=np.complex128)
        for w_i, w in enumerate(omegas):
            z_terms = np.exp(-1j * w * np.arange(1, P+1))
            denom = 1.0 - (z_terms[:, None] * H_p).sum(axis=0)
            
            # STABILITY GUARD: prevent division by very small numbers
            denom = np.where(np.abs(denom) < 1e-3, denom + 1e-3, denom)
            
            G[w_i, :] = 1.0 / denom
        
        return {
            'omegas': omegas,
            'lambdas': lambdas,
            'G': G,
            'G_mag': np.abs(G),
            'H_p': H_p
        }

# ============================================================================
# Model Selection
# ============================================================================

def find_best_model(X: np.ndarray, L: np.ndarray, 
                    P_range: List[int] = [1, 2, 3, 5, 7, 10, 15, 20],
                    K_range: List[int] = [1, 2, 3, 4]) -> Tuple[int, int]:
    """Find best P and K using BIC."""
    T = X.shape[1]
    T_train = int(0.70 * T)
    T_val = int(0.85 * T)
    
    X_train = X[:, :T_train]
    X_val = X[:, T_train:T_val]
    
    # CRITICAL: Single global standardization
    X_train = safe_zscore(X_train, X_train)
    X_val = safe_zscore(X_val, X_train)
    
    best_bic = np.inf
    best_P, best_K = None, None
    
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
            except:
                continue
    
    if best_P is None:
        # Fallback
        best_P, best_K = 5, 2
    
    return best_P, best_K

# ============================================================================
# LTI vs TV Analysis (CORRECTED)
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
    """
    Fit separate models for each time window.
    
    CRITICAL FIX: X_std is already globally standardized.
    No per-window z-scoring!
    """
    windows = split_into_windows(X_std, window_length_sec, overlap, fs)
    
    tv_results = []
    
    for w_idx, (start_idx, end_idx, X_win) in enumerate(windows):
        try:
            model = GPVAR_SharedH(P=P, K=K, L=L)
            model.fit(X_win)
            
            # Check stability
            rho = model.spectral_radius()
            if not np.isfinite(rho) or rho >= 0.99:
                print(f"    Warning: Window {w_idx} unstable (ρ={rho:.3f}), skipping")
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
                'h': model.h.copy(),  # Store coefficients
                'rho': rho
            })
        except Exception as e:
            print(f"    Warning: Window {w_idx} failed: {e}")
            continue
    
    return tv_results


def coeff_variation_stats(tv_results: List[Dict]) -> Dict:
    """
    Direct coefficient variation analysis.
    Measures how much h_{k,p} varies across windows.
    """
    H = np.stack([r["h"] for r in tv_results], axis=0)  # [n_windows, P(K+1)]
    
    h_mean = H.mean(axis=0)
    h_std = H.std(axis=0)
    h_cv = h_std / (np.abs(h_mean) + 1e-8)
    
    global_coeff_cv = h_cv.mean()
    
    return {
        "H_windows": H,
        "h_mean": h_mean,
        "h_std": h_std,
        "h_cv": h_cv,
        "global_coeff_cv": float(global_coeff_cv),
    }


def compare_transfer_functions(lti_model: GPVAR_SharedH, 
                               tv_results: List[Dict],
                               omegas: np.ndarray = None) -> Dict:
    """Compare LTI vs TV transfer functions."""
    if omegas is None:
        omegas = np.linspace(0, np.pi, 128)
    
    # Compute LTI transfer function
    lti_tf = lti_model.compute_transfer_function(omegas)
    G_lti = lti_tf['G_mag']
    
    # Compute TV transfer functions
    n_windows = len(tv_results)
    n_omegas = len(omegas)
    n_lambdas = len(lti_tf['lambdas'])
    
    G_tv_all = np.zeros((n_windows, n_omegas, n_lambdas))
    
    for w_idx, tv_res in enumerate(tv_results):
        tv_tf = tv_res['model'].compute_transfer_function(omegas)
        G_tv_all[w_idx, :, :] = tv_tf['G_mag']
    
    # Statistics
    G_tv_mean = G_tv_all.mean(axis=0)
    G_tv_std = G_tv_all.std(axis=0)
    
    # MSD per window
    msd_per_window = np.zeros(n_windows)
    for w_idx in range(n_windows):
        diff = G_tv_all[w_idx, :, :] - G_lti
        msd_per_window[w_idx] = np.mean(diff ** 2)
    
    # Variance across windows
    variance_across_windows = G_tv_all.var(axis=0)
    
    # Global metrics
    global_msd = np.mean(msd_per_window)
    global_variance = variance_across_windows.mean()
    
    # Per-mode and per-frequency statistics
    G_lti_per_mode = G_lti.mean(axis=0)
    G_tv_mean_per_mode = G_tv_mean.mean(axis=0)
    G_tv_std_per_mode = G_tv_all.mean(axis=1).std(axis=0)
    
    G_lti_per_freq = G_lti.mean(axis=1)
    G_tv_mean_per_freq = G_tv_mean.mean(axis=1)
    G_tv_std_per_freq = G_tv_all.mean(axis=2).std(axis=0)
    
    return {
        'omegas': omegas,
        'lambdas': lti_tf['lambdas'],
        'G_lti': G_lti,
        'G_tv_all': G_tv_all,
        'G_tv_mean': G_tv_mean,
        'G_tv_std': G_tv_std,
        'variance_across_windows': variance_across_windows,
        'msd_per_window': msd_per_window,
        'global_msd': global_msd,
        'global_variance': global_variance,
        'G_lti_per_mode': G_lti_per_mode,
        'G_tv_mean_per_mode': G_tv_mean_per_mode,
        'G_tv_std_per_mode': G_tv_std_per_mode,
        'G_lti_per_freq': G_lti_per_freq,
        'G_tv_mean_per_freq': G_tv_mean_per_freq,
        'G_tv_std_per_freq': G_tv_std_per_freq,
        'n_windows': n_windows
    }


def surrogate_msd_null(X_std: np.ndarray, L: np.ndarray, P: int, K: int, 
                       fs: float, n_surr: int = N_SURROGATES, 
                       rng_seed: int = 0) -> np.ndarray:
    """
    Build null distribution of MSD using circular shift surrogates.
    
    CRITICAL FIX: Proper null that preserves spectrum but destroys time structure.
    """
    rng = np.random.default_rng(rng_seed)
    msd_null = []
    
    for i in range(n_surr):
        if i % 50 == 0:
            print(f"    Surrogate {i+1}/{n_surr}...")
        
        # Create surrogate
        Xs = circular_shift_surrogate(X_std, rng)
        
        try:
            # Fit LTI on surrogate
            lti_s = GPVAR_SharedH(P=P, K=K, L=L)
            lti_s.fit(Xs)
            
            # Check stability
            rho_lti = lti_s.spectral_radius()
            if not np.isfinite(rho_lti) or rho_lti >= 1.0:
                continue
            
            # Fit TV on surrogate
            tv_s = compute_tv_models(Xs, L, P, K,
                                     WINDOW_LENGTH_SEC, WINDOW_OVERLAP, fs)
            
            if len(tv_s) < MIN_WINDOWS:
                continue
            
            # Compute MSD
            comp_s = compare_transfer_functions(lti_s, tv_s)
            msd_null.append(comp_s["global_msd"])
            
        except Exception as e:
            continue
    
    return np.array(msd_null)


def statistical_test_time_varying(comparison: Dict, X_std: np.ndarray, 
                                  L: np.ndarray, P: int, K: int, fs: float,
                                  alpha: float = ALPHA) -> Dict:
    """
    Statistical hypothesis test with proper surrogate null.
    
    CRITICAL FIXES:
    1. Proper surrogate-based null for MSD
    2. 95% CI test on transfer functions
    """
    print("  Running statistical tests...")
    
    observed_msd = comparison["global_msd"]
    
    # Build surrogate null
    print("  Building surrogate null distribution...")
    msd_null = surrogate_msd_null(X_std, L, P, K, fs, n_surr=N_SURROGATES)
    
    if len(msd_null) < 50:
        print(f"    WARNING: Only {len(msd_null)} valid surrogates")
        p_value_msd = np.nan
    else:
        p_value_msd = (msd_null >= observed_msd).mean()
        print(f"    P-value (MSD): {p_value_msd:.4f}")
    
    # 95% CI test
    G_lti_flat = comparison['G_lti'].ravel()
    G_tv_flat = comparison['G_tv_all'].reshape(comparison['n_windows'], -1)
    
    G_tv_mean = G_tv_flat.mean(axis=0)
    G_tv_se = G_tv_flat.std(axis=0) / np.sqrt(comparison['n_windows'])
    ci_lower = G_tv_mean - 1.96 * G_tv_se
    ci_upper = G_tv_mean + 1.96 * G_tv_se
    
    outside_ci = ((G_lti_flat < ci_lower) | (G_lti_flat > ci_upper)).mean()
    print(f"    Outside CI: {outside_ci*100:.1f}%")
    
    # Combined decision
    is_tv = False
    if not np.isnan(p_value_msd):
        is_tv = (p_value_msd < alpha) or (outside_ci > 0.05)
    else:
        is_tv = (outside_ci > 0.10)  # More conservative if no surrogate test
    
    return {
        "is_time_varying": bool(is_tv),
        "p_value_msd": float(p_value_msd) if not np.isnan(p_value_msd) else None,
        "outside_ci_fraction": float(outside_ci),
        "observed_msd": float(observed_msd),
        "msd_null": msd_null,
        "alpha": alpha,
        "ci_lower": ci_lower.reshape(comparison["G_lti"].shape),
        "ci_upper": ci_upper.reshape(comparison["G_lti"].shape)
    }

# ============================================================================
# Single-Subject Analysis
# ============================================================================

def analyze_single_subject(filepath: str, L: np.ndarray, 
                           subject_id: str, group: str) -> Optional[Dict]:
    """Complete LTI vs TV analysis for one subject."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {subject_id} ({group})")
    print(f"{'='*80}")
    
    # Load data
    print("Loading data...")
    X, ch_names = load_and_preprocess_eeg(filepath, BAND, TARGET_SFREQ)
    n, T = X.shape
    duration = T / TARGET_SFREQ
    print(f"  Channels: {n}, Duration: {duration:.1f}s")
    
    # CRITICAL: Single global standardization
    print("  Standardizing globally...")
    X_std = safe_zscore(X, X)
    
    # Step 1: Find best P and K
    print("\nStep 1: Finding optimal P and K...")
    best_P, best_K = find_best_model(X, L)
    print(f"  Selected: P={best_P}, K={best_K}")
    
    # Step 2: Fit LTI model on globally standardized data
    print("\nStep 2: Fitting LTI model...")
    lti_model = GPVAR_SharedH(P=best_P, K=best_K, L=L)
    lti_model.fit(X_std)
    lti_rho = lti_model.spectral_radius()
    lti_metrics = lti_model.evaluate(X_std)
    print(f"  LTI R²: {lti_metrics['R2']:.4f}, BIC: {lti_metrics['BIC']:.2f}, ρ: {lti_rho:.3f}")
    
    if not np.isfinite(lti_rho) or lti_rho >= 1.0:
        print(f"  ERROR: LTI model unstable")
        return None
    
    # Step 3: Fit TV models (no per-window z-scoring!)
    print("\nStep 3: Fitting time-varying models...")
    tv_results = compute_tv_models(X_std, L, best_P, best_K, 
                                   WINDOW_LENGTH_SEC, WINDOW_OVERLAP, TARGET_SFREQ)
    print(f"  Fitted {len(tv_results)} stable windows")
    
    if len(tv_results) < MIN_WINDOWS:
        print(f"  ERROR: Too few windows ({len(tv_results)} < {MIN_WINDOWS})")
        return None
    
    # Step 4: Coefficient variation analysis
    print("\nStep 4: Coefficient variation analysis...")
    coeff_stats = coeff_variation_stats(tv_results)
    print(f"  Global coefficient CV: {coeff_stats['global_coeff_cv']:.4f}")
    
    # Step 5: Compare transfer functions
    print("\nStep 5: Comparing transfer functions...")
    comparison = compare_transfer_functions(lti_model, tv_results)
    print(f"  Global MSD: {comparison['global_msd']:.6f}")
    print(f"  Global variance: {comparison['global_variance']:.6f}")
    
    # Step 6: Statistical testing with surrogate null
    print("\nStep 6: Statistical hypothesis testing...")
    test_results = statistical_test_time_varying(comparison, X_std, L, 
                                                 best_P, best_K, TARGET_SFREQ)
    
    if test_results['p_value_msd'] is not None:
        print(f"  P-value (MSD): {test_results['p_value_msd']:.4f}")
    print(f"  Outside CI: {test_results['outside_ci_fraction']*100:.1f}%")
    print(f"  Decision: {'TIME-VARYING' if test_results['is_time_varying'] else 'TIME-INVARIANT'}")
    
    return {
        'subject_id': subject_id,
        'group': group,
        'n_channels': n,
        'duration': duration,
        'P': best_P,
        'K': best_K,
        'lti_model': lti_model,
        'lti_metrics': lti_metrics,
        'lti_rho': lti_rho,
        'tv_results': tv_results,
        'coeff_stats': coeff_stats,
        'comparison': comparison,
        'test_results': test_results
    }

# ============================================================================
# Visualization (IMPROVED)
# ============================================================================

def plot_subject_analysis(results: Dict, save_dir: Path):
    """Comprehensive visualization."""
    sid = results["subject_id"]
    comp = results["comparison"]
    test = results["test_results"]
    coeff = results["coeff_stats"]
    
    omegas = comp["omegas"]
    lambdas = comp["lambdas"]
    G_lti = comp["G_lti"]
    G_tv_mean = comp["G_tv_mean"]
    var_w = comp["variance_across_windows"]
    msd_w = comp["msd_per_window"]
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # ========================================================================
    # Row 1: Transfer function heatmaps
    # ========================================================================
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(G_lti, aspect="auto", origin="lower", cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), omegas.min(), omegas.max()])
    ax1.set_title("LTI |G(ω,λ)|", fontsize=11, fontweight='bold')
    ax1.set_xlabel("λ (Graph Freq)")
    ax1.set_ylabel("ω (Temporal Freq)")
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(G_tv_mean, aspect="auto", origin="lower", cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), omegas.min(), omegas.max()])
    ax2.set_title("TV Mean |G(ω,λ)|", fontsize=11, fontweight='bold')
    ax2.set_xlabel("λ (Graph Freq)")
    ax2.set_ylabel("ω (Temporal Freq)")
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(var_w, aspect="auto", origin="lower", cmap='YlOrRd',
                     extent=[lambdas.min(), lambdas.max(), omegas.min(), omegas.max()])
    ax3.set_title("Variance Across Windows", fontsize=11, fontweight='bold')
    ax3.set_xlabel("λ (Graph Freq)")
    ax3.set_ylabel("ω (Temporal Freq)")
    fig.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    diff = G_tv_mean - G_lti
    vmax = np.abs(diff).max()
    im4 = ax4.imshow(diff, aspect="auto", origin="lower", cmap='RdBu_r',
                     extent=[lambdas.min(), lambdas.max(), omegas.min(), omegas.max()],
                     vmin=-vmax, vmax=vmax)
    ax4.set_title("TV Mean − LTI", fontsize=11, fontweight='bold')
    ax4.set_xlabel("λ (Graph Freq)")
    ax4.set_ylabel("ω (Temporal Freq)")
    fig.colorbar(im4, ax=ax4, fraction=0.046)
    
    # ========================================================================
    # Row 2: Temporal variation and per-mode comparison
    # ========================================================================
    
    ax5 = fig.add_subplot(gs[1, :2])
    mid_times = [(r["start_time"] + r["end_time"]) / 2 for r in results["tv_results"]]
    ax5.plot(mid_times, msd_w, "o-", linewidth=2, markersize=6)
    ax5.axhline(comp["global_msd"], color='r', linestyle='--', linewidth=2,
                label=f'Mean: {comp["global_msd"]:.4f}')
    ax5.set_title("MSD per Window vs LTI", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("MSD")
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.plot(lambdas, comp["G_lti_per_mode"], 'b-', linewidth=2, label="LTI")
    ax6.fill_between(lambdas,
                     comp["G_tv_mean_per_mode"] - 1.96*comp["G_tv_std_per_mode"],
                     comp["G_tv_mean_per_mode"] + 1.96*comp["G_tv_std_per_mode"],
                     alpha=0.25, color='red', label="TV 95% CI")
    ax6.plot(lambdas, comp["G_tv_mean_per_mode"], 'r-', linewidth=2, label="TV Mean")
    ax6.set_title("Per-Mode Response (avg over ω)", fontsize=11, fontweight='bold')
    ax6.set_xlabel("λ (Graph Frequency)")
    ax6.set_ylabel("Mean |G|")
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # ========================================================================
    # Row 3: Statistical testing results
    # ========================================================================
    
    ax7 = fig.add_subplot(gs[2, :2])
    outside_map = ((G_lti < test["ci_lower"]) | (G_lti > test["ci_upper"])).astype(float)
    im7 = ax7.imshow(outside_map, aspect="auto", origin="lower", cmap='RdYlGn_r',
                     extent=[lambdas.min(), lambdas.max(), omegas.min(), omegas.max()])
    ax7.set_title(f"LTI Outside TV 95% CI ({test['outside_ci_fraction']*100:.1f}%)",
                  fontsize=11, fontweight='bold')
    ax7.set_xlabel("λ")
    ax7.set_ylabel("ω")
    fig.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[2, 2:])
    if test["msd_null"] is not None and len(test["msd_null"]) > 0:
        ax8.hist(test["msd_null"], bins=30, alpha=0.7, edgecolor='black')
        ax8.axvline(test["observed_msd"], color='red', linestyle='--', linewidth=2.5,
                    label=f'Observed: {test["observed_msd"]:.4f}')
        p_str = f'{test["p_value_msd"]:.3f}' if test["p_value_msd"] is not None else 'N/A'
        ax8.set_title(f"Surrogate MSD Null (p = {p_str})", 
                      fontsize=11, fontweight='bold')
        ax8.legend(fontsize=10)
    else:
        ax8.text(0.5, 0.5, "No surrogate null", ha="center", va="center", fontsize=12)
    ax8.set_xlabel("MSD")
    ax8.set_ylabel("Count")
    ax8.grid(alpha=0.3, axis='y')
    
    # ========================================================================
    # Row 4: Coefficient variation
    # ========================================================================
    
    ax9 = fig.add_subplot(gs[3, :2])
    H_windows = coeff["H_windows"]
    for w_idx in range(min(10, H_windows.shape[0])):  # Show first 10 windows
        ax9.plot(H_windows[w_idx, :], alpha=0.5, linewidth=1)
    ax9.plot(coeff["h_mean"], 'k-', linewidth=3, label='Mean across windows')
    ax9.set_title("Coefficient Trajectories Across Windows", fontsize=11, fontweight='bold')
    ax9.set_xlabel("Coefficient Index")
    ax9.set_ylabel("Coefficient Value")
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.bar(range(len(coeff["h_cv"])), coeff["h_cv"], alpha=0.7, edgecolor='black')
    ax10.axhline(coeff["global_coeff_cv"], color='r', linestyle='--', linewidth=2,
                 label=f'Global CV: {coeff["global_coeff_cv"]:.3f}')
    ax10.set_title("Coefficient Variation (CV)", fontsize=11, fontweight='bold')
    ax10.set_xlabel("Coefficient Index")
    ax10.set_ylabel("CV = σ / |μ|")
    ax10.legend()
    ax10.grid(alpha=0.3, axis='y')
    
    # ========================================================================
    # Overall title
    # ========================================================================
    
    tv_status = 'TIME-VARYING' if test['is_time_varying'] else 'TIME-INVARIANT'
    fig.suptitle(f"{sid} ({results['group']}) | P={results['P']}, K={results['K']} | "
                 f"ρ={results['lti_rho']:.3f} | {tv_status}", 
                 fontsize=14, fontweight='bold')
    
    outpath = save_dir / f"{sid}_lti_vs_tv.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved plot → {outpath}")

# ============================================================================
# Multi-Subject Analysis
# ============================================================================

def run_all_subjects(subject_files: Dict[str, List[str]], L: np.ndarray):
    """Run analysis on all subjects."""
    all_results = []
    
    for group, files in subject_files.items():
        print(f"\n{'#'*80}")
        print(f"Processing {group} group ({len(files)} subjects)")
        print(f"{'#'*80}")
        
        for file_idx, filepath in enumerate(files):
            # Extract subject ID from path
            subject_id = Path(filepath).stem.replace('.set', '')
            if not subject_id:
                subject_id = f"{group}_{file_idx+1:02d}"
            
            try:
                result = analyze_single_subject(filepath, L, subject_id, group)
                
                if result is not None:
                    all_results.append(result)
                    
                    # Save visualization
                    subject_dir = OUT_DIR / group
                    subject_dir.mkdir(exist_ok=True, parents=True)
                    plot_subject_analysis(result, subject_dir)
                    
            except Exception as e:
                print(f"ERROR processing {subject_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return all_results


def create_summary_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame."""
    rows = []
    
    for r in all_results:
        rows.append({
            "subject_id": r["subject_id"],
            "group": r["group"],
            "P": r["P"],
            "K": r["K"],
            "duration": r["duration"],
            "lti_R2": r["lti_metrics"]["R2"],
            "lti_BIC": r["lti_metrics"]["BIC"],
            "lti_rho": r["lti_rho"],
            "n_windows": r["comparison"]["n_windows"],
            "global_msd": r["comparison"]["global_msd"],
            "global_variance": r["comparison"]["global_variance"],
            "global_coeff_cv": r["coeff_stats"]["global_coeff_cv"],
            "outside_ci_fraction": r["test_results"]["outside_ci_fraction"],
            "p_value_msd": r["test_results"]["p_value_msd"],
            "is_time_varying": r["test_results"]["is_time_varying"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "subject_summary.csv", index=False)
    print(f"\n✓ Saved summary → {OUT_DIR / 'subject_summary.csv'}")
    
    return df

# ============================================================================
# Group Comparison
# ============================================================================

def compare_groups(df: pd.DataFrame):
    """Statistical comparison between AD and HC."""
    print(f"\n{'='*80}")
    print("GROUP COMPARISON: AD vs HC")
    print(f"{'='*80}")
    
    ad_data = df[df['group'] == 'AD']
    hc_data = df[df['group'] == 'HC']
    
    print(f"\nSample sizes:")
    print(f"  AD: {len(ad_data)}")
    print(f"  HC: {len(hc_data)}")
    
    if len(ad_data) == 0 or len(hc_data) == 0:
        print("ERROR: Need both AD and HC subjects")
        return None
    
    # Metrics to compare
    metrics = {
        'global_msd': 'Global MSD',
        'global_variance': 'Global Variance',
        'global_coeff_cv': 'Coefficient CV',
        'outside_ci_fraction': 'Outside CI Fraction'
    }
    
    results = {}
    
    for metric, label in metrics.items():
        ad_vals = ad_data[metric].dropna().values
        hc_vals = hc_data[metric].dropna().values
        
        if len(ad_vals) < 2 or len(hc_vals) < 2:
            continue
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_val = stats.mannwhitneyu(ad_vals, hc_vals, alternative='two-sided')
        
        # Effect size (r = Z / sqrt(N))
        n_total = len(ad_vals) + len(hc_vals)
        z_score = stats.norm.ppf(1 - p_val/2) * np.sign(ad_vals.mean() - hc_vals.mean())
        effect_r = z_score / np.sqrt(n_total)
        
        results[metric] = {
            'ad_mean': ad_vals.mean(),
            'ad_std': ad_vals.std(),
            'hc_mean': hc_vals.mean(),
            'hc_std': hc_vals.std(),
            'u_stat': u_stat,
            'p_value': p_val,
            'effect_r': effect_r
        }
        
        print(f"\n{label}:")
        print(f"  AD: {ad_vals.mean():.4f} ± {ad_vals.std():.4f}")
        print(f"  HC: {hc_vals.mean():.4f} ± {hc_vals.std():.4f}")
        print(f"  U = {u_stat:.1f}, p = {p_val:.4f}, r = {effect_r:.3f}")
        if p_val < 0.05:
            print(f"  *** SIGNIFICANT ***")
    
    # Chi-square for TV classification
    ad_tv = (ad_data['is_time_varying'] == True).sum()
    hc_tv = (hc_data['is_time_varying'] == True).sum()
    ad_lti = len(ad_data) - ad_tv
    hc_lti = len(hc_data) - hc_tv
    
    contingency = np.array([[ad_tv, ad_lti], [hc_tv, hc_lti]])
    chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
    
    print(f"\nTime-Varying Classification:")
    print(f"  AD: {ad_tv}/{len(ad_data)} ({100*ad_tv/len(ad_data):.1f}%) time-varying")
    print(f"  HC: {hc_tv}/{len(hc_data)} ({100*hc_tv/len(hc_data):.1f}%) time-varying")
    print(f"  χ² = {chi2:.3f}, p = {p_chi:.4f}")
    if p_chi < 0.05:
        print(f"  *** SIGNIFICANT ***")
    
    # Save results
    with open(OUT_DIR / 'group_comparison_stats.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    plot_group_comparison(df, results, OUT_DIR)
    
    return results


def plot_group_comparison(df: pd.DataFrame, stats_results: Dict, save_dir: Path):
    """Visualize group comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    metrics = [
        ('global_msd', 'Global MSD'),
        ('global_variance', 'Global Variance'),
        ('global_coeff_cv', 'Coefficient CV'),
        ('outside_ci_fraction', 'Outside CI Fraction'),
    ]
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        
        ad_vals = df[df['group'] == 'AD'][metric].dropna().values
        hc_vals = df[df['group'] == 'HC'][metric].dropna().values
        
        # Box plot
        bp = ax.boxplot([ad_vals, hc_vals], labels=['AD', 'HC'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('salmon')
        bp['boxes'][1].set_facecolor('lightblue')
        
        # Scatter
        x_ad = np.random.normal(1, 0.04, size=len(ad_vals))
        x_hc = np.random.normal(2, 0.04, size=len(hc_vals))
        ax.scatter(x_ad, ad_vals, alpha=0.5, color='darkred', s=40)
        ax.scatter(x_hc, hc_vals, alpha=0.5, color='darkblue', s=40)
        
        ax.set_ylabel(label, fontsize=11)
        
        if metric in stats_results:
            p_val = stats_results[metric]['p_value']
            ax.set_title(f'{label}\np = {p_val:.4f}', fontsize=11, fontweight='bold')
            
            # Significance marker
            if p_val < 0.05:
                y_max = max(ad_vals.max(), hc_vals.max())
                ax.plot([1, 2], [y_max*1.1, y_max*1.1], 'k-', linewidth=2)
                sig_marker = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
                ax.text(1.5, y_max*1.15, sig_marker, ha='center', fontsize=16)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # Proportion TV
    ax = axes[4]
    ad_tv_prop = (df[df['group'] == 'AD']['is_time_varying'] == True).mean()
    hc_tv_prop = (df[df['group'] == 'HC']['is_time_varying'] == True).mean()
    
    bars = ax.bar(['AD', 'HC'], [ad_tv_prop, hc_tv_prop], 
                  color=['salmon', 'lightblue'], width=0.6, edgecolor='black', linewidth=2)
    ax.set_ylabel('Proportion Time-Varying', fontsize=11)
    ax.set_title('Time-Varying Classification', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (group, prop) in enumerate([('AD', ad_tv_prop), ('HC', hc_tv_prop)]):
        ax.text(i, prop + 0.05, f'{prop*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle('AD vs HC: Time-Varying Characteristics', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'group_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved group comparison → {save_dir / 'group_comparison.png'}")

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Complete LTI vs TV analysis pipeline."""
    print("="*80)
    print("LTI vs TIME-VARYING GP-VAR ANALYSIS [CORRECTED VERSION]")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Window length: {WINDOW_LENGTH_SEC}s")
    print(f"  Window overlap: {WINDOW_OVERLAP*100}%")
    print(f"  N surrogates: {N_SURROGATES}")
    print(f"  Significance level: {ALPHA}")
    
    print(f"\nCRITICAL FIXES APPLIED:")
    print(f"  ✓ Global z-scoring (no fake time-variation)")
    print(f"  ✓ Proper surrogate null distribution")
    print(f"  ✓ Stability guards in transfer functions")
    print(f"  ✓ Direct coefficient variation metrics")
    
    # Load consensus Laplacian
    print(f"\nLoading consensus Laplacian...")
    L = load_consensus_laplacian(CONSENSUS_LAPLACIAN_PATH)
    print(f"  Shape: {L.shape}")
    
    # Analyze all subjects
    print(f"\nStarting multi-subject analysis...")
    all_results = run_all_subjects(SUBJECT_FILES, L)
    
    if len(all_results) == 0:
        print("ERROR: No subjects successfully analyzed")
        return None
    
    print(f"\nCompleted {len(all_results)} subjects successfully")
    
    # Create summary
    df = create_summary_dataframe(all_results)
    
    # Group comparison
    if len(df[df['group'] == 'AD']) > 0 and len(df[df['group'] == 'HC']) > 0:
        group_results = compare_groups(df)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {OUT_DIR}")
    print(f"  - subject_summary.csv: Summary metrics for all subjects")
    print(f"  - group_comparison.png: AD vs HC visualization")
    print(f"  - group_comparison_stats.json: Statistical test results")
    print(f"  - [group]/[subject_id]_lti_vs_tv.png: Individual analyses")
    
    return df, all_results


if __name__ == "__main__":
    results = main()