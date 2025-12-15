"""
LTI vs Time-Varying GP-VAR Analysis: ALL AD vs ALL HC Comparison (FIXED VERSION)
================================================================================
Comprehensive analysis with all quality control fixes:
- Fix #1: Tighter stability thresholds (ρ < 0.95)
- Fix #2: Expanded model selection ranges (P: 1-20, K: 0-4)
- Fix #3: Comprehensive failure logging
- Fix #4: Statistical significance with FDR correction
- Fix #5: Removed unstable transfer function clipping
- Fix #6: Quality control reports

Features:
- Individual subject analysis with BIC-based model selection
- Model selection heatmaps for each subject
- LTI and Time-Varying model comparison
- Statistical significance testing with multiple comparisons correction
- Publication-quality visualizations
- Comprehensive quality control and logging
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

# ALL AD subjects
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

# ALL HC subjects
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
CONSENSUS_LAPLACIAN_PATH = "/home/muhibt/GPVAR/consensus_results/consensus_results/ALL_Files/consensus_distance_graph.npy"

# Preprocessing
BAND = (0.5, 40.0)
TARGET_SFREQ = 100.0
EEG_REFERENCE = 'average'

# Model settings
RIDGE_LAMBDA = 5e-3

# ============================================================================
# FIX #1 & #2: Quality Control Thresholds and Expanded Model Selection
# ============================================================================

# Stability thresholds (FIX #1: Tightened from 0.99 to 0.95)
RHO_THRESHOLD_LTI = 0.95  # LTI model must have ρ < 0.95
RHO_THRESHOLD_TV = 0.95   # Each TV window must have ρ < 0.95

# Model fit quality
R2_THRESHOLD = 0.50  # Minimum R² for acceptable fit

# Minimum requirements
MIN_STABLE_WINDOWS = 10  # Need at least 10 stable TV windows

# FIX #2: Expanded model selection ranges
P_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
K_RANGE = [0, 1, 2, 3, 4]  # K=0 means standard VAR (no graph filtering)

# Time-varying analysis
WINDOW_LENGTH_SEC = 10.0
WINDOW_OVERLAP = 0.5

# Output
OUT_DIR = Path("./ad_hc_lti_tv_comparison_ALL_SUBJECTS_FIXED")
OUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("LTI vs TV GP-VAR ANALYSIS: ALL AD vs ALL HC (FIXED VERSION)")
print("="*80)
print(f"\nQuality Control Settings:")
print(f"  ρ threshold (LTI): {RHO_THRESHOLD_LTI}")
print(f"  ρ threshold (TV): {RHO_THRESHOLD_TV}")
print(f"  R² threshold: {R2_THRESHOLD}")
print(f"  Minimum stable windows: {MIN_STABLE_WINDOWS}")
print(f"  P range: {min(P_RANGE)} to {max(P_RANGE)}")
print(f"  K range: {min(K_RANGE)} to {max(K_RANGE)}")
print(f"\nDataset:")
print(f"  AD subjects: {len(AD_PATHS)}")
print(f"  HC subjects: {len(HC_PATHS)}")

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

# FIX #3: Add comprehensive logging function
def log_subject_status(subject_id: str, group: str, status: str, 
                       details: Dict = None, save_dir: Path = OUT_DIR):
    """
    Log subject processing status.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    group : str
        'AD' or 'HC'
    status : str
        'SUCCESS', 'REJECT', 'FAILED'
    details : Dict
        Additional details about the status
    """
    log_file = save_dir / 'subject_processing_log.csv'
    
    # Create header if file doesn't exist
    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write("subject_id,group,status,reason,detail_1,detail_2,detail_3\n")
    
    # Prepare log entry
    reason = details.get('reason', '') if details else ''
    detail_1 = details.get('detail_1', '') if details else ''
    detail_2 = details.get('detail_2', '') if details else ''
    detail_3 = details.get('detail_3', '') if details else ''
    
    # Write to log
    with open(log_file, 'a') as f:
        f.write(f"{subject_id},{group},{status},{reason},{detail_1},{detail_2},{detail_3}\n")

def load_and_preprocess_eeg(filepath: str, band: Tuple[float,float], 
                           target_sfreq: float,
                           reference: str = 'average') -> Tuple[np.ndarray, List[str]]:
    """Load and preprocess EEG with re-referencing."""
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)
    
    raw.filter(l_freq=band[0], h_freq=band[1], method="fir", 
               fir_design="firwin", verbose=False)
    
    if reference == 'average':
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
    
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

def add_eigenvalue_secondary_axis(ax, lambdas, n_ticks=6):
    """Add a secondary x-axis showing eigenvalue (λ) values."""
    ax2 = ax.twiny()
    xlim = ax.get_xlim()
    ax2.set_xlim(xlim)
    n_lambdas = len(lambdas)
    tick_indices = np.linspace(0, n_lambdas - 1, n_ticks, dtype=int)
    ax2.set_xticks(tick_indices)
    tick_labels = [f'{lambdas[i]:.2f}' for i in tick_indices]
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Graph Frequency λ (eigenvalue)', fontsize=10, fontweight='bold')
    return ax2

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
        
        # K=0 case handled correctly (only L^0 = I)
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
        """
        FIX #5: Compute AR transfer function with stability check instead of clipping.
        """
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
            
            # FIX #5: Check for resonances instead of clipping
            min_denom = np.abs(denom).min()
            max_gain = 1.0 / (min_denom + 1e-10)
            
            if min_denom < 1e-6 or max_gain > 1000:
                raise ValueError(f"Transfer function unstable: min|denom|={min_denom:.2e}, max|G|={max_gain:.1f}")
            
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
    """Find best P and K using BIC with boundary detection."""
    T = X.shape[1]
    T_train = int(0.70 * T)
    T_val = int(0.85 * T)
    
    X_train = X[:, :T_train]
    X_val = X[:, T_train:T_val]
    
    X_train = safe_zscore(X_train, X_train)
    X_val = safe_zscore(X_val, X_train)
    
    best_bic = np.inf
    best_P, best_K = None, None
    
    bic_grid = np.full((len(K_range), len(P_range)), np.nan)
    r2_grid = np.full((len(K_range), len(P_range)), np.nan)
    stable_grid = np.full((len(K_range), len(P_range)), False)
    
    model_selection_results = []
    
    for k_idx, K in enumerate(K_range):
        for p_idx, P in enumerate(P_range):
            try:
                m = GPVAR_SharedH(P=P, K=K, L_norm=L_norm)
                m.fit(X_train)
                
                rho = m.spectral_radius()
                # Use tightened threshold
                stable = np.isfinite(rho) and rho < RHO_THRESHOLD_LTI
                
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
            
            except Exception:
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
    
    # Report boundary hits
    if best_P is not None:
        if best_P == min(P_range):
            print(f"    NOTE: Selected P={best_P} is at LOWER boundary")
        if best_P == max(P_range):
            print(f"    NOTE: Selected P={best_P} is at UPPER boundary - consider expanding range")
        if best_K == min(K_range):
            print(f"    NOTE: Selected K={best_K} is at LOWER boundary")
        if best_K == max(K_range):
            print(f"    NOTE: Selected K={best_K} is at UPPER boundary - consider expanding range")
    
    if best_P is None:
        best_P, best_K = 5, 1
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
    bic_masked = np.ma.masked_where(~stable_grid, bic_grid)
    im1 = ax.imshow(bic_masked, aspect='auto', cmap='viridis_r', origin='lower')
    
    for k_idx in range(len(K_range)):
        for p_idx in range(len(P_range)):
            if not stable_grid[k_idx, p_idx]:
                ax.text(p_idx, k_idx, '✗', ha='center', va='center', 
                       fontsize=8, color='red', fontweight='bold')
    
    best_p_idx = P_range.index(best_P)
    best_k_idx = K_range.index(best_K)
    rect = plt.Rectangle((best_p_idx - 0.5, best_k_idx - 0.5), 1, 1, 
                         fill=False, edgecolor='lime', linewidth=3)
    ax.add_patch(rect)
    
    ax.set_xticks(range(0, len(P_range), 2))
    ax.set_xticklabels([P_range[i] for i in range(0, len(P_range), 2)])
    ax.set_yticks(range(len(K_range)))
    ax.set_yticklabels(K_range)
    ax.set_xlabel('P (AR Order)', fontsize=12, fontweight='bold')
    ax.set_ylabel('K (Graph Filter Order)', fontsize=12, fontweight='bold')
    ax.set_title(f'BIC Grid\nBest: P={best_P}, K={best_K}', fontsize=12, fontweight='bold')
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
                       fontsize=8, color='red', fontweight='bold')
    
    rect = plt.Rectangle((best_p_idx - 0.5, best_k_idx - 0.5), 1, 1, 
                         fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)
    
    ax.set_xticks(range(0, len(P_range), 2))
    ax.set_xticklabels([P_range[i] for i in range(0, len(P_range), 2)])
    ax.set_yticks(range(len(K_range)))
    ax.set_yticklabels(K_range)
    ax.set_xlabel('P (AR Order)', fontsize=12, fontweight='bold')
    ax.set_ylabel('K (Graph Filter Order)', fontsize=12, fontweight='bold')
    ax.set_title(f'R² Grid\nBest: P={best_P}, K={best_K}', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='R²')
    
    plt.suptitle(f'{group}: {subject_id}\nModel Selection (✗ = unstable)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / f'{group}_{subject_id}_model_selection.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

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
    """Fit separate models for each time window with tightened stability check."""
    windows = split_into_windows(X_std, window_length_sec, overlap, fs)
    
    tv_results = []
    
    for w_idx, (start_idx, end_idx, X_win) in enumerate(windows):
        try:
            model = GPVAR_SharedH(P=P, K=K, L_norm=L_norm)
            model.fit(X_win)
            
            rho = model.spectral_radius()
            # Use tightened threshold
            if not np.isfinite(rho) or rho >= RHO_THRESHOLD_TV:
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
# Single Subject Analysis (with comprehensive logging)
# ============================================================================

def analyze_single_subject(filepath: str, L_norm: np.ndarray, group: str,
                           save_dir: Path) -> Optional[Dict]:
    """FIX #3: Analyze single subject with comprehensive logging."""
    
    subject_id = None
    
    try:
        subject_id = Path(filepath).stem.replace('s6_', '').replace('_rs-hep_eeg', '').replace('_rs_eeg', '')
        print(f"\n  Analyzing {group}: {subject_id}")
        
        # Load EEG
        try:
            X, ch_names = load_and_preprocess_eeg(filepath, BAND, TARGET_SFREQ, EEG_REFERENCE)
            n_channels, n_samples = X.shape
            print(f"    Loaded: {n_channels} ch, {n_samples/TARGET_SFREQ:.1f}s")
        except Exception as e:
            log_subject_status(subject_id, group, 'FAILED', {
                'reason': 'EEG_loading_error',
                'detail_1': str(type(e).__name__),
                'detail_2': str(e)[:50]
            })
            print(f"    FAILED: EEG loading - {e}")
            return None
        
        # Check channel compatibility
        if L_norm.shape[0] != n_channels:
            log_subject_status(subject_id, group, 'REJECT', {
                'reason': 'Channel_mismatch',
                'detail_1': f'L={L_norm.shape[0]}',
                'detail_2': f'EEG={n_channels}'
            })
            print(f"    REJECT: Channel mismatch (L={L_norm.shape[0]}, EEG={n_channels})")
            return None
        
        X_std = safe_zscore(X, X)
        
        # Model selection
        print(f"    Model selection...")
        try:
            model_selection = find_best_model_with_grid(X, L_norm)
            best_P = model_selection['best_P']
            best_K = model_selection['best_K']
            print(f"    Selected: P={best_P}, K={best_K}")
        except Exception as e:
            log_subject_status(subject_id, group, 'FAILED', {
                'reason': 'Model_selection_error',
                'detail_1': str(type(e).__name__),
                'detail_2': str(e)[:50]
            })
            print(f"    FAILED: Model selection - {e}")
            return None
        
        # Save heatmap
        try:
            plot_model_selection_heatmap(model_selection, subject_id, group, save_dir)
        except Exception:
            pass
        
        # Fit LTI
        print(f"    Fitting LTI...")
        try:
            lti_model = GPVAR_SharedH(P=best_P, K=best_K, L_norm=L_norm)
            lti_model.fit(X_std)
            lti_rho = lti_model.spectral_radius()
            lti_metrics = lti_model.evaluate(X_std)
        except Exception as e:
            log_subject_status(subject_id, group, 'FAILED', {
                'reason': 'LTI_fitting_error',
                'detail_1': str(type(e).__name__),
                'detail_2': str(e)[:50]
            })
            print(f"    FAILED: LTI fitting - {e}")
            return None
        
        # Check LTI stability (tightened threshold)
        if not np.isfinite(lti_rho) or lti_rho >= RHO_THRESHOLD_LTI:
            log_subject_status(subject_id, group, 'REJECT', {
                'reason': 'LTI_unstable',
                'detail_1': f'rho={lti_rho:.4f}',
                'detail_2': f'threshold={RHO_THRESHOLD_LTI}'
            })
            print(f"    REJECT: ρ={lti_rho:.3f} ≥ {RHO_THRESHOLD_LTI}")
            return None
        
        # Check R² quality
        if lti_metrics['R2'] < R2_THRESHOLD:
            log_subject_status(subject_id, group, 'REJECT', {
                'reason': 'Low_R2',
                'detail_1': f'R2={lti_metrics["R2"]:.4f}',
                'detail_2': f'threshold={R2_THRESHOLD}'
            })
            print(f"    REJECT: R²={lti_metrics['R2']:.3f} < {R2_THRESHOLD}")
            return None
        
        print(f"    LTI: R²={lti_metrics['R2']:.3f}, ρ={lti_rho:.3f} ✓")
        
        # Fit TV
        print(f"    Fitting TV...")
        try:
            tv_results = compute_tv_models(X_std, L_norm, best_P, best_K, 
                                          WINDOW_LENGTH_SEC, WINDOW_OVERLAP, TARGET_SFREQ)
        except Exception as e:
            log_subject_status(subject_id, group, 'FAILED', {
                'reason': 'TV_fitting_error',
                'detail_1': str(type(e).__name__),
                'detail_2': str(e)[:50]
            })
            print(f"    FAILED: TV fitting - {e}")
            return None
        
        # Check minimum windows
        if len(tv_results) < MIN_STABLE_WINDOWS:
            log_subject_status(subject_id, group, 'REJECT', {
                'reason': 'Too_few_stable_windows',
                'detail_1': f'n={len(tv_results)}',
                'detail_2': f'threshold={MIN_STABLE_WINDOWS}'
            })
            print(f"    REJECT: {len(tv_results)} windows < {MIN_STABLE_WINDOWS}")
            return None
        
        print(f"    TV: {len(tv_results)} windows ✓")
        
        # Compute transfer functions
        try:
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
        except Exception as e:
            log_subject_status(subject_id, group, 'FAILED', {
                'reason': 'Transfer_function_error',
                'detail_1': str(type(e).__name__),
                'detail_2': str(e)[:50]
            })
            print(f"    FAILED: Transfer functions - {e}")
            return None
        
        # Log success
        log_subject_status(subject_id, group, 'SUCCESS', {
            'reason': 'Completed',
            'detail_1': f'R2={lti_metrics["R2"]:.3f}',
            'detail_2': f'rho={lti_rho:.3f}',
            'detail_3': f'n_win={len(tv_results)}'
        })
        
        print(f"    ✓ SUCCESS")
        
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
            'mean_msd': np.mean((G_tv_all - lti_tf['G_mag'][None, :, :])**2),
            'mean_cv': np.mean(G_tv_std / (G_tv_mean + 1e-8)),
        }
    
    except Exception as e:
        if subject_id:
            log_subject_status(subject_id, group, 'FAILED', {
                'reason': 'Unexpected_error',
                'detail_1': str(type(e).__name__),
                'detail_2': str(e)[:50]
            })
        print(f"    FAILED: {e}")
        return None

# ============================================================================
# FIX #4: Statistical Significance Functions
# ============================================================================

def compute_pointwise_statistics(ad_data: np.ndarray, hc_data: np.ndarray,
                                 alpha: float = 0.05) -> Dict:
    """
    Compute pointwise t-tests with FDR correction.
    
    Parameters:
    -----------
    ad_data : np.ndarray
        Shape (n_ad_subjects, n_points)
    hc_data : np.ndarray
        Shape (n_hc_subjects, n_points)
    alpha : float
        Significance level
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("WARNING: statsmodels not available, installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'statsmodels'])
        from statsmodels.stats.multitest import multipletests
    
    n_points = ad_data.shape[1]
    
    p_values = np.zeros(n_points)
    t_stats = np.zeros(n_points)
    cohens_d = np.zeros(n_points)
    
    for i in range(n_points):
        ad_vals = ad_data[:, i]
        hc_vals = hc_data[:, i]
        
        t, p = stats.ttest_ind(ad_vals, hc_vals, equal_var=False)
        
        pooled_std = np.sqrt((ad_vals.std()**2 + hc_vals.std()**2) / 2)
        d = (ad_vals.mean() - hc_vals.mean()) / (pooled_std + 1e-10)
        
        p_values[i] = p
        t_stats[i] = t
        cohens_d[i] = d
    
    # FDR correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    return {
        'p_values': p_values,
        'p_corrected': p_corrected,
        't_stats': t_stats,
        'cohens_d': cohens_d,
        'significant_uncorrected': p_values < alpha,
        'significant_corrected': reject,
        'n_sig_uncorrected': (p_values < alpha).sum(),
        'n_sig_corrected': reject.sum()
    }

def plot_statistical_significance(ad_results: List[Dict], hc_results: List[Dict],
                                   save_dir: Path):
    """FIX #4: Create detailed statistical significance plots."""
    print("\nCreating statistical significance plots...")
    
    lambdas = ad_results[0]['lambdas']
    n_lambdas = len(lambdas)
    graph_freq_indices = np.arange(n_lambdas)
    
    # Collect data
    ad_lti_mode_all = np.array([r['G_lti'].mean(axis=0) for r in ad_results])
    hc_lti_mode_all = np.array([r['G_lti'].mean(axis=0) for r in hc_results])
    ad_tv_mode_all = np.array([r['G_tv_mean'].mean(axis=0) for r in ad_results])
    hc_tv_mode_all = np.array([r['G_tv_mean'].mean(axis=0) for r in hc_results])
    
    # Compute statistics
    stats_lti = compute_pointwise_statistics(ad_lti_mode_all, hc_lti_mode_all)
    stats_tv = compute_pointwise_statistics(ad_tv_mode_all, hc_tv_mode_all)
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # Panel 1: LTI p-values
    ax = axes[0, 0]
    ax.plot(graph_freq_indices, -np.log10(stats_lti['p_values']), 'o-', 
           color='blue', linewidth=2, markersize=3)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax.axhline(-np.log10(0.01), color='darkred', linestyle='--', linewidth=2, label='p=0.01')
    ax.set_xlabel('Graph Frequency Index')
    ax.set_ylabel('-log₁₀(p-value)')
    ax.set_title('LTI: Statistical Significance (uncorrected)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_eigenvalue_secondary_axis(ax, lambdas, n_ticks=6)
    
    # Panel 2: TV p-values
    ax = axes[0, 1]
    ax.plot(graph_freq_indices, -np.log10(stats_tv['p_values']), 'o-', 
           color='red', linewidth=2, markersize=3)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax.axhline(-np.log10(0.01), color='darkred', linestyle='--', linewidth=2, label='p=0.01')
    ax.set_xlabel('Graph Frequency Index')
    ax.set_ylabel('-log₁₀(p-value)')
    ax.set_title('TV: Statistical Significance (uncorrected)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_eigenvalue_secondary_axis(ax, lambdas, n_ticks=6)
    
    # Panel 3: Effect sizes (Cohen's d) - LTI
    ax = axes[1, 0]
    colors = ['red' if d > 0 else 'blue' for d in stats_lti['cohens_d']]
    ax.bar(graph_freq_indices, stats_lti['cohens_d'], color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
    ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-0.8, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Graph Frequency Index')
    ax.set_ylabel("Cohen's d (AD - HC)")
    ax.set_title("LTI: Effect Sizes", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    add_eigenvalue_secondary_axis(ax, lambdas, n_ticks=6)
    
    # Panel 4: Effect sizes - TV
    ax = axes[1, 1]
    colors = ['red' if d > 0 else 'blue' for d in stats_tv['cohens_d']]
    ax.bar(graph_freq_indices, stats_tv['cohens_d'], color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
    ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-0.8, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Graph Frequency Index')
    ax.set_ylabel("Cohen's d (AD - HC)")
    ax.set_title("TV: Effect Sizes", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    add_eigenvalue_secondary_axis(ax, lambdas, n_ticks=6)
    
    # Panel 5: Significance masks - LTI
    ax = axes[2, 0]
    sig_matrix = np.vstack([
        stats_lti['significant_uncorrected'].astype(int),
        stats_lti['significant_corrected'].astype(int)
    ])
    im = ax.imshow(sig_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[0, n_lambdas-1, 0, 2])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Uncorrected\np<0.05', 'FDR\ncorrected'])
    ax.set_xlabel('Graph Frequency Index')
    ax.set_title(f'LTI: Significance (FDR: {stats_lti["n_sig_corrected"]}/{n_lambdas})', fontweight='bold')
    plt.colorbar(im, ax=ax, ticks=[0, 1], label='Significant')
    
    # Panel 6: Significance masks - TV
    ax = axes[2, 1]
    sig_matrix_tv = np.vstack([
        stats_tv['significant_uncorrected'].astype(int),
        stats_tv['significant_corrected'].astype(int)
    ])
    im = ax.imshow(sig_matrix_tv, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[0, n_lambdas-1, 0, 2])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Uncorrected\np<0.05', 'FDR\ncorrected'])
    ax.set_xlabel('Graph Frequency Index')
    ax.set_title(f'TV: Significance (FDR: {stats_tv["n_sig_corrected"]}/{n_lambdas})', fontweight='bold')
    plt.colorbar(im, ax=ax, ticks=[0, 1], label='Significant')
    
    plt.suptitle('AD vs HC: Statistical Significance Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / 'statistical_significance_analysis.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath}")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'graph_freq_index': graph_freq_indices,
        'eigenvalue': lambdas,
        'lti_p_value': stats_lti['p_values'],
        'lti_p_corrected': stats_lti['p_corrected'],
        'lti_cohens_d': stats_lti['cohens_d'],
        'lti_significant': stats_lti['significant_corrected'],
        'tv_p_value': stats_tv['p_values'],
        'tv_p_corrected': stats_tv['p_corrected'],
        'tv_cohens_d': stats_tv['cohens_d'],
        'tv_significant': stats_tv['significant_corrected']
    })
    stats_csv = save_dir / 'pointwise_statistics.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Saved: {stats_csv}")

# ============================================================================
# FIX #6: Quality Control Report
# ============================================================================

def create_qc_report(ad_results: List[Dict], hc_results: List[Dict], 
                     save_dir: Path):
    """FIX #6: Create comprehensive quality control report."""
    print("\nCreating QC report...")
    
    log_file = save_dir / 'subject_processing_log.csv'
    if not log_file.exists():
        print("  WARNING: No processing log found")
        return
    
    log_df = pd.read_csv(log_file)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("QUALITY CONTROL REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    total_ad = len(AD_PATHS)
    total_hc = len(HC_PATHS)
    success_ad = len(ad_results)
    success_hc = len(hc_results)
    
    report_lines.append(f"PROCESSING SUMMARY:")
    report_lines.append(f"  AD: {success_ad}/{total_ad} successful ({100*success_ad/total_ad:.1f}%)")
    report_lines.append(f"  HC: {success_hc}/{total_hc} successful ({100*success_hc/total_hc:.1f}%)")
    report_lines.append("")
    
    report_lines.append("REJECTION REASONS:")
    
    for group in ['AD', 'HC']:
        group_log = log_df[log_df['group'] == group]
        rejected = group_log[group_log['status'] == 'REJECT']
        failed = group_log[group_log['status'] == 'FAILED']
        
        report_lines.append(f"\n  {group} Group:")
        
        if len(rejected) > 0:
            report_lines.append(f"    Rejected: {len(rejected)}")
            for reason, count in rejected['reason'].value_counts().items():
                report_lines.append(f"      {reason}: {count}")
        
        if len(failed) > 0:
            report_lines.append(f"    Failed: {len(failed)}")
            for reason, count in failed['reason'].value_counts().items():
                report_lines.append(f"      {reason}: {count}")
    
    report_lines.append("")
    report_lines.append("QUALITY METRICS (Accepted Subjects):")
    
    for group, results in [('AD', ad_results), ('HC', hc_results)]:
        if len(results) == 0:
            continue
        
        rhos = np.array([r['lti_rho'] for r in results])
        r2s = np.array([r['lti_R2'] for r in results])
        n_wins = np.array([r['n_windows'] for r in results])
        
        report_lines.append(f"\n  {group} Group (n={len(results)}):")
        report_lines.append(f"    Spectral Radius (ρ):")
        report_lines.append(f"      Mean: {rhos.mean():.3f}")
        report_lines.append(f"      Std:  {rhos.std():.3f}")
        report_lines.append(f"      Range: [{rhos.min():.3f}, {rhos.max():.3f}]")
        report_lines.append(f"    R² (model fit):")
        report_lines.append(f"      Mean: {r2s.mean():.3f}")
        report_lines.append(f"      Std:  {r2s.std():.3f}")
        report_lines.append(f"      Range: [{r2s.min():.3f}, {r2s.max():.3f}]")
        report_lines.append(f"    Stable Windows:")
        report_lines.append(f"      Mean: {n_wins.mean():.1f}")
        report_lines.append(f"      Range: [{n_wins.min()}, {n_wins.max()}]")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_file = save_dir / 'quality_control_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n  Saved: {report_file}")
    
    # Create QC visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Spectral radius distribution
    ax = axes[0, 0]
    ad_rhos = [r['lti_rho'] for r in ad_results]
    hc_rhos = [r['lti_rho'] for r in hc_results]
    ax.hist(ad_rhos, bins=20, alpha=0.6, color='red', label=f'AD (n={len(ad_results)})')
    ax.hist(hc_rhos, bins=20, alpha=0.6, color='blue', label=f'HC (n={len(hc_results)})')
    ax.axvline(RHO_THRESHOLD_LTI, color='black', linestyle='--', linewidth=2, 
              label=f'Threshold ({RHO_THRESHOLD_LTI})')
    ax.set_xlabel('Spectral Radius (ρ)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Spectral Radius', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: R² distribution
    ax = axes[0, 1]
    ad_r2s = [r['lti_R2'] for r in ad_results]
    hc_r2s = [r['lti_R2'] for r in hc_results]
    ax.hist(ad_r2s, bins=20, alpha=0.6, color='red', label='AD')
    ax.hist(hc_r2s, bins=20, alpha=0.6, color='blue', label='HC')
    ax.axvline(R2_THRESHOLD, color='black', linestyle='--', linewidth=2,
              label=f'Threshold ({R2_THRESHOLD})')
    ax.set_xlabel('R² (Model Fit)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Model Fit Quality', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Number of stable windows
    ax = axes[0, 2]
    ad_nwin = [r['n_windows'] for r in ad_results]
    hc_nwin = [r['n_windows'] for r in hc_results]
    ax.hist(ad_nwin, bins=20, alpha=0.6, color='red', label='AD')
    ax.hist(hc_nwin, bins=20, alpha=0.6, color='blue', label='HC')
    ax.axvline(MIN_STABLE_WINDOWS, color='black', linestyle='--', linewidth=2,
              label=f'Threshold ({MIN_STABLE_WINDOWS})')
    ax.set_xlabel('Number of Stable Windows')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Stable TV Windows', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: ρ vs R² scatter
    ax = axes[1, 0]
    ax.scatter(ad_rhos, ad_r2s, s=100, alpha=0.7, color='red', edgecolors='black', label='AD')
    ax.scatter(hc_rhos, hc_r2s, s=100, alpha=0.7, color='blue', edgecolors='black', label='HC')
    ax.axvline(RHO_THRESHOLD_LTI, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(R2_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectral Radius (ρ)')
    ax.set_ylabel('R²')
    ax.set_title('Model Stability vs Fit Quality', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Processing status pie chart - AD
    ax = axes[1, 1]
    ad_log = log_df[log_df['group'] == 'AD']
    ad_counts = ad_log['status'].value_counts()
    colors_map = {'SUCCESS': 'green', 'REJECT': 'orange', 'FAILED': 'red'}
    pie_colors = [colors_map.get(status, 'gray') for status in ad_counts.index]
    ax.pie(ad_counts.values, labels=ad_counts.index, autopct='%1.1f%%',
          colors=pie_colors, startangle=90)
    ax.set_title(f'AD Processing Status (n={total_ad})', fontweight='bold')
    
    # Plot 6: Processing status pie chart - HC
    ax = axes[1, 2]
    hc_log = log_df[log_df['group'] == 'HC']
    hc_counts = hc_log['status'].value_counts()
    pie_colors = [colors_map.get(status, 'gray') for status in hc_counts.index]
    ax.pie(hc_counts.values, labels=hc_counts.index, autopct='%1.1f%%',
          colors=pie_colors, startangle=90)
    ax.set_title(f'HC Processing Status (n={total_hc})', fontweight='bold')
    
    plt.suptitle('Quality Control Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    qc_plot = save_dir / 'quality_control_summary.png'
    plt.savefig(qc_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {qc_plot}")

# ============================================================================
# Simplified Plotting Functions
# ============================================================================

def save_model_selection_csv(ad_results: List[Dict], hc_results: List[Dict], 
                             save_dir: Path):
    """Save model selection results to CSV."""
    print("\nSaving model selection results...")
    
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
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = save_dir / "model_selection_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved: {summary_csv}")
    
    return summary_df

def compute_and_save_statistics(ad_results: List[Dict], hc_results: List[Dict],
                                save_dir: Path):
    """Compute group statistics."""
    print("\nComputing statistics...")
    
    if len(ad_results) < 2 or len(hc_results) < 2:
        print("  WARNING: Not enough subjects for statistical tests")
        return None
    
    metrics = ['lti_R2', 'lti_rho', 'best_P', 'best_K', 'n_windows']
    
    stats_data = []
    for metric in metrics:
        ad_vals = np.array([r[metric] for r in ad_results])
        hc_vals = np.array([r[metric] for r in hc_results])
        
        t_stat, p_val = stats.ttest_ind(ad_vals, hc_vals, equal_var=False)
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
    print("\n" + stats_df.to_string(index=False))
    
    return stats_df

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Main analysis with all fixes applied."""
    
    # Load consensus Laplacian
    print("\n" + "="*80)
    print("Loading consensus Laplacian...")
    print("="*80)
    L_norm = load_consensus_laplacian(CONSENSUS_LAPLACIAN_PATH)
    print(f"  Shape: {L_norm.shape}")
    
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
    print(f"AD: {len(ad_results)}/{len(AD_PATHS)} successful ({100*len(ad_results)/len(AD_PATHS):.1f}%)")
    print(f"HC: {len(hc_results)}/{len(HC_PATHS)} successful ({100*len(hc_results)/len(HC_PATHS):.1f}%)")
    
    if len(ad_results) == 0 or len(hc_results) == 0:
        print("\nERROR: Not enough subjects for comparison")
        return
    
    # Save results
    save_model_selection_csv(ad_results, hc_results, OUT_DIR)
    
    # Compute statistics
    stats_df = compute_and_save_statistics(ad_results, hc_results, OUT_DIR)
    
    # FIX #4: Statistical significance plots
    plot_statistical_significance(ad_results, hc_results, OUT_DIR)
    
    # FIX #6: QC report
    create_qc_report(ad_results, hc_results, OUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUT_DIR}")
    print("\nGenerated files:")
    print("  - subject_processing_log.csv (comprehensive logging)")
    print("  - quality_control_report.txt (FIX #6)")
    print("  - quality_control_summary.png (FIX #6)")
    print("  - model_selection_summary.csv")
    print("  - [group]_[subject]_model_selection.png (per subject)")
    print("  - statistical_significance_analysis.png (FIX #4)")
    print("  - pointwise_statistics.csv (FIX #4)")
    print("  - group_statistics.csv")
    print("="*80)

if __name__ == "__main__":
    main()
