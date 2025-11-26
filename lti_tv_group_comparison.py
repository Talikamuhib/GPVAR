"""
LTI vs Time-Varying GP-VAR Analysis - GROUP COMPARISON (AD vs HC)
=================================================================
Compares:
1. AD Time-Varying vs HC Time-Varying
2. AD LTI vs HC LTI

Features:
- Processes all subjects in both groups
- Robust error handling per subject
- Group-level statistical comparisons
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
sns.set_style('whitegrid')
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Subject file paths
AD_PATHS = [
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
]

HC_PATHS = [
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

# Consensus Laplacian path
CONSENSUS_LAPLACIAN_PATH = "/home/muhibt/project/filter_identification/Consensus matrix/group_consensus_laplacian/all_consensus_average.npy"

# Preprocessing
BAND = (0.5, 40.0)
TARGET_SFREQ = 100.0

# Model settings
RIDGE_LAMBDA = 5e-3

# Time-varying analysis
WINDOW_LENGTH_SEC = 10.0
WINDOW_OVERLAP = 0.5
MIN_WINDOWS = 5

# Model selection ranges
P_RANGE = [1, 2, 3, 5, 7, 10]
K_RANGE = [1, 2, 3, 4]

# Output
OUT_DIR = Path("./group_comparison_lti_tv_analysis")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Helper Functions (from single-subject script)
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
    """Robust loader that handles both adjacency and Laplacian matrices."""
    M = np.load(filepath)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # Symmetrize
    if not np.allclose(M, M.T, atol=1e-6):
        M = (M + M.T) / 2.0

    n = M.shape[0]

    # Heuristic: Laplacian should have row sums ≈ 0 and diagonal ≥ 0
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
# Model Selection & Time-Varying Analysis
# ============================================================================

def find_best_model(X: np.ndarray, L_norm: np.ndarray, 
                    P_range: List[int] = P_RANGE,
                    K_range: List[int] = K_RANGE) -> Tuple[int, int]:
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
    
    for K in K_range:
        for P in P_range:
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
            
            except Exception:
                continue
    
    if best_P is None:
        best_P, best_K = 5, 2
    
    return best_P, best_K

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

def analyze_single_subject(filepath: str, L_norm: np.ndarray, subject_id: str = None) -> Optional[Dict]:
    """
    Analyze a single subject. Returns results dict or None if failed.
    """
    try:
        if subject_id is None:
            subject_id = Path(filepath).stem.replace('s6_', '').replace('_rs-hep_eeg', '').replace('_rs_eeg', '')
        
        # Load EEG
        X, ch_names = load_and_preprocess_eeg(filepath, BAND, TARGET_SFREQ)
        n_channels, n_samples = X.shape
        
        # Check size compatibility
        if L_norm.shape[0] != n_channels:
            print(f"  SKIP: Channel mismatch (Laplacian={L_norm.shape[0]}, EEG={n_channels})")
            return None
        
        # Standardize
        X_std = safe_zscore(X, X)
        
        # Model selection
        best_P, best_K = find_best_model(X, L_norm)
        
        # Fit LTI
        lti_model = GPVAR_SharedH(P=best_P, K=best_K, L_norm=L_norm)
        lti_model.fit(X_std)
        lti_rho = lti_model.spectral_radius()
        lti_metrics = lti_model.evaluate(X_std)
        
        if not np.isfinite(lti_rho) or lti_rho >= 1.0:
            print(f"  SKIP: LTI unstable (ρ={lti_rho:.3f})")
            return None
        
        # Fit TV models
        tv_results = compute_tv_models(X_std, L_norm, best_P, best_K, 
                                       WINDOW_LENGTH_SEC, WINDOW_OVERLAP, TARGET_SFREQ)
        
        if len(tv_results) < MIN_WINDOWS:
            print(f"  SKIP: Too few windows ({len(tv_results)} < {MIN_WINDOWS})")
            return None
        
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
        
        return {
            'subject_id': subject_id,
            'filepath': filepath,
            'n_channels': n_channels,
            'n_samples': n_samples,
            'duration': n_samples / TARGET_SFREQ,
            'best_P': best_P,
            'best_K': best_K,
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
        print(f"  ERROR: {e}")
        return None

# ============================================================================
# Group Analysis
# ============================================================================

def process_group(filepaths: List[str], group_name: str, L_norm: np.ndarray) -> List[Dict]:
    """Process all subjects in a group."""
    print(f"\n{'='*80}")
    print(f"Processing {group_name} group ({len(filepaths)} subjects)")
    print(f"{'='*80}")
    
    results = []
    
    for idx, filepath in enumerate(tqdm(filepaths, desc=group_name)):
        subject_id = Path(filepath).stem.replace('s6_', '').replace('_rs-hep_eeg', '').replace('_rs_eeg', '')
        print(f"\n[{idx+1}/{len(filepaths)}] {subject_id}")
        
        result = analyze_single_subject(filepath, L_norm, subject_id)
        
        if result is not None:
            results.append(result)
            print(f"  ✓ Success: R²={result['lti_R2']:.3f}, MSD={result['mean_msd']:.6f}, CV={result['mean_cv']:.3f}")
        else:
            print(f"  ✗ Failed")
    
    print(f"\n{group_name}: {len(results)}/{len(filepaths)} successful")
    return results

# ============================================================================
# Statistical Comparisons
# ============================================================================

def compute_group_statistics(ad_results: List[Dict], hc_results: List[Dict]) -> pd.DataFrame:
    """Compute statistical comparisons between groups."""
    
    metrics = [
        'lti_R2', 'lti_rho', 'lti_BIC',
        'tv_R2_mean', 'tv_rho_mean',
        'mean_msd', 'mean_cv'
    ]
    
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
            'AD_n': len(ad_vals),
            'HC_mean': hc_vals.mean(),
            'HC_std': hc_vals.std(),
            'HC_n': len(hc_vals),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < 0.05
        })
    
    df = pd.DataFrame(stats_data)
    return df

# ============================================================================
# Visualization
# ============================================================================

def plot_group_comparison(ad_results: List[Dict], hc_results: List[Dict], 
                         stats_df: pd.DataFrame, save_dir: Path):
    """Create comprehensive group comparison plots."""
    
    print("\nCreating group comparison visualizations...")
    
    # =================================================================
    # Figure 1: Metric Distributions
    # =================================================================
    
    fig1, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    metrics = ['lti_R2', 'lti_rho', 'tv_R2_mean', 'tv_rho_mean', 
               'mean_msd', 'mean_cv', 'lti_BIC', 'n_windows']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        ad_vals = [r[metric] for r in ad_results]
        hc_vals = [r[metric] for r in hc_results]
        
        positions = [1, 2]
        bp = ax.boxplot([ad_vals, hc_vals], positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True)
        
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        
        # Add individual points
        for i, vals in enumerate([ad_vals, hc_vals], 1):
            x = np.random.normal(i, 0.04, len(vals))
            ax.scatter(x, vals, alpha=0.3, s=30, color='black')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['AD', 'HC'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add p-value if available
        if metric in stats_df['metric'].values:
            row = stats_df[stats_df['metric'] == metric].iloc[0]
            p_val = row['p_value']
            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            y_max = max(max(ad_vals), max(hc_vals))
            ax.text(1.5, y_max * 1.05, f'p={p_val:.3f} {sig_marker}', 
                   ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('AD vs HC: Metric Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    savepath1 = save_dir / 'group_comparison_metrics.png'
    plt.savefig(savepath1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath1}")
    
    # =================================================================
    # Figure 2: Transfer Function Comparison
    # =================================================================
    
    # Average transfer functions across subjects
    ad_G_lti_all = np.array([r['G_lti'] for r in ad_results])
    hc_G_lti_all = np.array([r['G_lti'] for r in hc_results])
    
    ad_G_tv_all = np.array([r['G_tv_mean'] for r in ad_results])
    hc_G_tv_all = np.array([r['G_tv_mean'] for r in hc_results])
    
    ad_G_lti_mean = ad_G_lti_all.mean(axis=0)
    hc_G_lti_mean = hc_G_lti_all.mean(axis=0)
    
    ad_G_tv_mean = ad_G_tv_all.mean(axis=0)
    hc_G_tv_mean = hc_G_tv_all.mean(axis=0)
    
    freqs_hz = ad_results[0]['freqs_hz']
    lambdas = ad_results[0]['lambdas']
    
    fig2 = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig2, hspace=0.3, wspace=0.3)
    
    # Row 1: LTI Transfer Functions
    ax1 = fig2.add_subplot(gs[0, 0])
    im1 = ax1.imshow(ad_G_lti_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax1.set_title('AD LTI |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax1.set_xlabel('λ (Graph Frequency)')
    ax1.set_ylabel('f (Hz)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig2.add_subplot(gs[0, 1])
    im2 = ax2.imshow(hc_G_lti_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax2.set_title('HC LTI |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax2.set_xlabel('λ (Graph Frequency)')
    ax2.set_ylabel('f (Hz)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig2.add_subplot(gs[0, 2])
    diff_lti = ad_G_lti_mean - hc_G_lti_mean
    vmax = np.abs(diff_lti).max()
    im3 = ax3.imshow(diff_lti, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()],
                     vmin=-vmax, vmax=vmax)
    ax3.set_title('AD LTI - HC LTI', fontsize=12, fontweight='bold')
    ax3.set_xlabel('λ (Graph Frequency)')
    ax3.set_ylabel('f (Hz)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Row 2: TV Transfer Functions
    ax4 = fig2.add_subplot(gs[1, 0])
    im4 = ax4.imshow(ad_G_tv_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax4.set_title('AD TV |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax4.set_xlabel('λ (Graph Frequency)')
    ax4.set_ylabel('f (Hz)')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig2.add_subplot(gs[1, 1])
    im5 = ax5.imshow(hc_G_tv_mean, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax5.set_title('HC TV |G(ω,λ)|', fontsize=12, fontweight='bold')
    ax5.set_xlabel('λ (Graph Frequency)')
    ax5.set_ylabel('f (Hz)')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = fig2.add_subplot(gs[1, 2])
    diff_tv = ad_G_tv_mean - hc_G_tv_mean
    vmax2 = np.abs(diff_tv).max()
    im6 = ax6.imshow(diff_tv, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()],
                     vmin=-vmax2, vmax=vmax2)
    ax6.set_title('AD TV - HC TV', fontsize=12, fontweight='bold')
    ax6.set_xlabel('λ (Graph Frequency)')
    ax6.set_ylabel('f (Hz)')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # Row 3: Aggregate comparisons
    ax7 = fig2.add_subplot(gs[2, 0])
    ad_avg_freq = ad_G_lti_mean.mean(axis=1)
    hc_avg_freq = hc_G_lti_mean.mean(axis=1)
    ad_avg_freq_tv = ad_G_tv_mean.mean(axis=1)
    hc_avg_freq_tv = hc_G_tv_mean.mean(axis=1)
    
    ax7.plot(freqs_hz, ad_avg_freq, 'r-', linewidth=2.5, label='AD LTI')
    ax7.plot(freqs_hz, hc_avg_freq, 'b-', linewidth=2.5, label='HC LTI')
    ax7.plot(freqs_hz, ad_avg_freq_tv, 'r--', linewidth=2.5, label='AD TV')
    ax7.plot(freqs_hz, hc_avg_freq_tv, 'b--', linewidth=2.5, label='HC TV')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Average |G|')
    ax7.set_title('Frequency Response', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig2.add_subplot(gs[2, 1])
    ad_avg_mode = ad_G_lti_mean.mean(axis=0)
    hc_avg_mode = hc_G_lti_mean.mean(axis=0)
    ad_avg_mode_tv = ad_G_tv_mean.mean(axis=0)
    hc_avg_mode_tv = hc_G_tv_mean.mean(axis=0)
    
    ax8.plot(lambdas, ad_avg_mode, 'r-', linewidth=2.5, label='AD LTI')
    ax8.plot(lambdas, hc_avg_mode, 'b-', linewidth=2.5, label='HC LTI')
    ax8.plot(lambdas, ad_avg_mode_tv, 'r--', linewidth=2.5, label='AD TV')
    ax8.plot(lambdas, hc_avg_mode_tv, 'b--', linewidth=2.5, label='HC TV')
    ax8.set_xlabel('λ (Graph Frequency)')
    ax8.set_ylabel('Average |G|')
    ax8.set_title('Graph Mode Response', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig2.add_subplot(gs[2, 2])
    ad_msd = [r['mean_msd'] for r in ad_results]
    hc_msd = [r['mean_msd'] for r in hc_results]
    ad_cv = [r['mean_cv'] for r in ad_results]
    hc_cv = [r['mean_cv'] for r in hc_results]
    
    ax9.scatter(ad_msd, ad_cv, s=80, alpha=0.6, color='red', label='AD', edgecolors='black')
    ax9.scatter(hc_msd, hc_cv, s=80, alpha=0.6, color='blue', label='HC', edgecolors='black')
    ax9.set_xlabel('Mean Squared Difference (MSD)')
    ax9.set_ylabel('Coefficient of Variation (CV)')
    ax9.set_title('Time-Varying Dynamics', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('AD vs HC: Transfer Function Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    savepath2 = save_dir / 'group_comparison_transfer_functions.png'
    plt.savefig(savepath2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath2}")

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Main group comparison analysis."""
    
    print("="*80)
    print("GROUP COMPARISON: AD vs HC (LTI and Time-Varying GP-VAR)")
    print("="*80)
    
    # Load consensus Laplacian
    print("\nLoading consensus Laplacian...")
    L_norm = load_consensus_laplacian(CONSENSUS_LAPLACIAN_PATH)
    print(f"Laplacian shape: {L_norm.shape}")
    
    # Process AD group
    ad_results = process_group(AD_PATHS, "AD", L_norm)
    
    # Process HC group
    hc_results = process_group(HC_PATHS, "HC", L_norm)
    
    # Check if we have enough subjects
    if len(ad_results) < 3 or len(hc_results) < 3:
        print(f"\nERROR: Insufficient subjects (AD={len(ad_results)}, HC={len(hc_results)})")
        return
    
    # Statistical comparison
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    stats_df = compute_group_statistics(ad_results, hc_results)
    print(stats_df.to_string(index=False))
    
    # Save statistics
    stats_csv = OUT_DIR / 'group_statistics.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nSaved statistics: {stats_csv}")
    
    # Visualizations
    plot_group_comparison(ad_results, hc_results, stats_df, OUT_DIR)
    
    # Save individual subject results
    print("\nSaving individual subject results...")
    subject_data = []
    for r in ad_results:
        subject_data.append({
            'subject_id': r['subject_id'],
            'group': 'AD',
            'n_channels': r['n_channels'],
            'duration': r['duration'],
            'best_P': r['best_P'],
            'best_K': r['best_K'],
            'lti_R2': r['lti_R2'],
            'lti_rho': r['lti_rho'],
            'lti_BIC': r['lti_BIC'],
            'tv_R2_mean': r['tv_R2_mean'],
            'tv_rho_mean': r['tv_rho_mean'],
            'n_windows': r['n_windows'],
            'mean_msd': r['mean_msd'],
            'mean_cv': r['mean_cv'],
        })
    
    for r in hc_results:
        subject_data.append({
            'subject_id': r['subject_id'],
            'group': 'HC',
            'n_channels': r['n_channels'],
            'duration': r['duration'],
            'best_P': r['best_P'],
            'best_K': r['best_K'],
            'lti_R2': r['lti_R2'],
            'lti_rho': r['lti_rho'],
            'lti_BIC': r['lti_BIC'],
            'tv_R2_mean': r['tv_R2_mean'],
            'tv_rho_mean': r['tv_rho_mean'],
            'n_windows': r['n_windows'],
            'mean_msd': r['mean_msd'],
            'mean_cv': r['mean_cv'],
        })
    
    subjects_df = pd.DataFrame(subject_data)
    subjects_csv = OUT_DIR / 'all_subjects_results.csv'
    subjects_df.to_csv(subjects_csv, index=False)
    print(f"Saved subject results: {subjects_csv}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"AD subjects: {len(ad_results)}/{len(AD_PATHS)} successful")
    print(f"HC subjects: {len(hc_results)}/{len(HC_PATHS)} successful")
    print(f"\nSignificant differences (p < 0.05):")
    sig_metrics = stats_df[stats_df['significant']]
    if len(sig_metrics) > 0:
        for _, row in sig_metrics.iterrows():
            print(f"  - {row['metric']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
    else:
        print("  None")
    
    print(f"\nResults saved to: {OUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
