"""
=============================================================================
FIND BEST GP-VAR MODEL FOR SYSTEM DYNAMICS IDENTIFICATION
=============================================================================

After finding the consensus matrix (graph Laplacian), this script:
1. Loads the consensus Laplacian (the brain network structure)
2. Performs model selection to find optimal P (AR order) and K (graph filter order)
3. Fits the GP-VAR model to identify the system dynamics
4. Analyzes the transfer function G(ω, λ) to understand frequency responses
5. Provides comprehensive visualizations and interpretations

The GP-VAR model equation:
    x_t = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} L^k x_{t-p} + e_t

Where:
- x_t: EEG signal at time t (N channels)
- P: AR order (temporal memory)
- K: Graph filter order (spatial smoothness)
- h_{p,k}: Scalar coefficients (shared across channels)
- L: Graph Laplacian (from consensus matrix)
- L^k: k-th power of Laplacian (k-hop neighborhood)

=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import linalg, stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# ============================================================================

# Path to consensus Laplacian (from your consensus matrix analysis)
# This should be the output from your consensus matrix construction
CONSENSUS_LAPLACIAN_PATH = "./consensus_laplacian.npy"  # CHANGE THIS

# Path to EEG data file (for model fitting)
EEG_DATA_PATH = "./eeg_data.set"  # CHANGE THIS or use synthetic data

# Use synthetic data for demonstration?
USE_SYNTHETIC_DATA = True  # Set to False when using real data

# Preprocessing parameters
BAND = (0.5, 40.0)  # Bandpass filter range (Hz)
TARGET_SFREQ = 100.0  # Target sampling frequency (Hz)

# Model selection parameters
P_RANGE = [1, 2, 3, 5, 7, 10, 15, 20]  # AR orders to test
K_RANGE = [1, 2, 3, 4]  # Graph filter orders to test

# Model fitting parameters
RIDGE_LAMBDA = 5e-3  # Ridge regularization strength

# Output directory
OUT_DIR = Path("./gpvar_model_selection_results")
OUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("GP-VAR MODEL SELECTION FOR SYSTEM DYNAMICS IDENTIFICATION")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_zscore(X: np.ndarray, ref: np.ndarray = None) -> np.ndarray:
    """Z-score normalization with robust handling."""
    if ref is None:
        ref = X
    mu = ref.mean(axis=1, keepdims=True)
    sd = ref.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
    Z = np.clip(Z, -10, 10)
    return Z


def create_consensus_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Create normalized Laplacian from adjacency matrix.
    
    L = D - A  (combinatorial Laplacian)
    or
    L = I - D^{-1/2} A D^{-1/2}  (normalized Laplacian)
    """
    # Ensure symmetric
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    
    # Degree matrix
    D = np.diag(A.sum(axis=1))
    
    # Combinatorial Laplacian
    L = D - A
    
    # Ensure symmetric and positive semi-definite
    L = (L + L.T) / 2
    eigvals = np.linalg.eigvalsh(L)
    if eigvals.min() < -1e-8:
        L = L - eigvals.min() * np.eye(L.shape[0])
    
    return L


def generate_synthetic_data(n_channels: int = 64, duration_sec: float = 300.0,
                           fs: float = 100.0, L: np.ndarray = None) -> np.ndarray:
    """
    Generate synthetic EEG-like data with graph structure.
    
    This creates data that follows a GP-VAR model with known parameters,
    useful for validating the model fitting procedure.
    """
    print("\nGenerating synthetic EEG data...")
    
    np.random.seed(42)  # For reproducibility
    n_samples = int(duration_sec * fs)
    
    if L is None:
        # Create random graph Laplacian
        A = np.random.rand(n_channels, n_channels) * 0.5
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        L = create_consensus_laplacian(A)
    
    # Normalize Laplacian eigenvalues to [0, 2] for stability
    eigvals = np.linalg.eigvalsh(L)
    L_normalized = L / (eigvals.max() / 2 + 1e-8)
    
    # True model parameters (known for validation)
    true_P = 5
    true_K = 2
    
    # Generate h coefficients (simulating brain dynamics)
    # Make them decay with lag and emphasize lower-order graph filters
    n_coef = true_P * (true_K + 1)
    h_true = np.zeros(n_coef)
    
    idx = 0
    for p in range(1, true_P + 1):
        for k in range(true_K + 1):
            # Decay with lag, prefer L^0 and L^1
            base = 0.3 * np.exp(-0.3 * (p - 1)) * np.exp(-0.5 * k)
            h_true[idx] = base * (1 + 0.3 * np.random.randn())
            idx += 1
    
    # Ensure overall stability by scaling
    h_true = h_true * 0.6
    
    # Precompute L^k
    L_powers = [np.eye(n_channels)]
    for k in range(1, true_K + 1):
        L_powers.append(L_powers[-1] @ L_normalized)
    
    # Generate data with more realistic noise
    X = np.zeros((n_channels, n_samples))
    
    # Initialize with realistic EEG-like startup
    for t in range(true_P):
        X[:, t] = np.random.randn(n_channels) * 0.5
    
    # Generate time series
    for t in range(true_P, n_samples):
        prediction = np.zeros(n_channels)
        idx = 0
        for p in range(1, true_P + 1):
            for k in range(true_K + 1):
                prediction += h_true[idx] * (L_powers[k] @ X[:, t - p])
                idx += 1
        
        # Add structured noise (1/f like EEG)
        noise = np.random.randn(n_channels) * 1.0
        X[:, t] = prediction + noise
    
    # Add some realistic EEG-like oscillations
    time = np.arange(n_samples) / fs
    for ch in range(n_channels):
        # Add alpha rhythm (8-12 Hz)
        alpha_freq = 10 + np.random.randn() * 1.5
        alpha_amp = 0.3 + np.random.rand() * 0.2
        X[ch, :] += alpha_amp * np.sin(2 * np.pi * alpha_freq * time + np.random.rand() * 2 * np.pi)
        
        # Add some delta (1-4 Hz)
        delta_freq = 2 + np.random.rand() * 2
        delta_amp = 0.2 + np.random.rand() * 0.1
        X[ch, :] += delta_amp * np.sin(2 * np.pi * delta_freq * time + np.random.rand() * 2 * np.pi)
    
    print(f"  Generated: {n_channels} channels × {n_samples} samples")
    print(f"  True model: P={true_P}, K={true_K}")
    print(f"  Duration: {duration_sec:.1f} seconds")
    print(f"  Signal-to-noise characteristics: realistic EEG-like")
    
    return X, L, {'P': true_P, 'K': true_K, 'h': h_true}


# ============================================================================
# GP-VAR MODEL CLASS
# ============================================================================

class GPVAR_Model:
    """
    Graph Polynomial Vector Autoregression (GP-VAR) Model
    
    This model combines:
    - Graph Signal Processing: Respects brain network topology
    - Autoregressive Modeling: Captures temporal dynamics
    - Polynomial Graph Filters: Flexible spatial filtering
    
    Model equation:
        x_t = Σ_{p=1}^P Σ_{k=0}^K h_{p,k} L^k x_{t-p} + e_t
    
    Transfer function in graph Fourier domain:
        G(ω, λ) = 1 / (1 - Σ_p H_p(λ) e^{-jωp})
    
    Where H_p(λ) = Σ_k h_{p,k} λ^k
    """
    
    def __init__(self, P: int, K: int, L: np.ndarray, ridge_lambda: float = RIDGE_LAMBDA):
        """
        Initialize GP-VAR model.
        
        Parameters:
        -----------
        P : int
            AR order (number of time lags)
        K : int
            Graph filter order (polynomial degree in Laplacian)
        L : np.ndarray
            Graph Laplacian matrix (N × N)
        ridge_lambda : float
            Ridge regularization parameter
        """
        self.P = P
        self.K = K
        self.L = L
        self.n = L.shape[0]  # Number of channels
        self.ridge_lambda = ridge_lambda
        
        # Eigendecomposition of Laplacian
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L)
        
        # Precompute L^k for k = 0, 1, ..., K
        self.L_powers = [np.eye(self.n)]
        for k in range(1, K + 1):
            self.L_powers.append(self.L_powers[-1] @ L)
        
        # Model parameters (to be learned)
        self.h = None  # Coefficients
        self.fitted = False
    
    def _build_design_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build design matrix for ridge regression.
        
        For each time t > P:
            y = X[:, t]  (target)
            features = [L^0 x_{t-1}, L^1 x_{t-1}, ..., L^K x_{t-1},
                       L^0 x_{t-2}, ..., L^K x_{t-P}]
        """
        n, T = X.shape
        T_valid = T - self.P
        n_obs = n * T_valid
        n_feat = self.P * (self.K + 1)
        
        R = np.zeros((n_obs, n_feat), dtype=np.float64)
        Y = np.zeros(n_obs, dtype=np.float64)
        
        for t in range(self.P, T):
            t_idx = t - self.P
            
            # Compute filter outputs for all lags and orders
            filter_vals = []
            for p in range(1, self.P + 1):
                x_lag = X[:, t - p]
                for k in range(self.K + 1):
                    filter_vals.append(self.L_powers[k] @ x_lag)
            filter_vals = np.asarray(filter_vals, dtype=np.float64)
            
            # Fill design matrix
            for i in range(n):
                row_idx = t_idx * n + i
                R[row_idx, :] = filter_vals[:, i]
                Y[row_idx] = X[i, t]
        
        return R, Y
    
    def fit(self, X: np.ndarray) -> 'GPVAR_Model':
        """
        Fit the GP-VAR model using ridge regression.
        
        Parameters:
        -----------
        X : np.ndarray
            EEG data matrix (N channels × T time samples)
        
        Returns:
        --------
        self : GPVAR_Model
            Fitted model
        """
        if X.shape[1] <= self.P:
            raise ValueError(f"Time series length ({X.shape[1]}) must exceed P ({self.P})")
        
        # Build design matrix
        R, Y = self._build_design_matrix(X)
        
        # Ridge regression: h = (R'R + λI)^{-1} R'Y
        RtR = R.T @ R
        RtY = R.T @ Y
        reg = self.ridge_lambda * np.eye(RtR.shape[0])
        self.h = linalg.solve(RtR + reg, RtY, assume_a='sym')
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next time step (teacher forcing).
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        R, _ = self._build_design_matrix(X)
        Y_pred = R @ self.h
        return Y_pred.reshape(-1, self.n).T
    
    def spectral_radius(self) -> float:
        """
        Compute spectral radius of the model (stability measure).
        
        The model is stable if spectral radius < 1.
        """
        if not self.fitted:
            return np.nan
        
        # Build companion matrix
        A_mats = []
        idx = 0
        for p in range(self.P):
            A_p = np.zeros((self.n, self.n))
            for k in range(self.K + 1):
                A_p += self.h[idx] * self.L_powers[k]
                idx += 1
            A_mats.append(A_p)
        
        # Companion form
        C = np.zeros((self.n * self.P, self.n * self.P))
        C[:self.n, :self.n * self.P] = np.hstack(A_mats)
        if self.P > 1:
            C[self.n:, :-self.n] = np.eye(self.n * (self.P - 1))
        
        # Compute eigenvalues
        vals = np.linalg.eigvals(C)
        vals = vals[np.isfinite(vals)]
        
        return float(np.max(np.abs(vals))) if vals.size else np.nan
    
    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
        --------
        dict with keys:
            - R2: Coefficient of determination
            - MSE: Mean squared error
            - BIC: Bayesian Information Criterion
            - RSS: Residual sum of squares
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        Y_pred = self.predict(X)
        Y_true = X[:, self.P:]
        
        # Residuals
        resid = Y_true - Y_pred
        RSS = float(np.sum(resid ** 2))
        TSS = float(np.sum((Y_true - Y_true.mean()) ** 2))
        
        # Metrics
        R2 = 1.0 - RSS / (TSS + 1e-10)
        MSE = RSS / Y_true.size
        
        # BIC
        n_obs = Y_true.size
        n_params = self.P * (self.K + 1)
        BIC = n_obs * np.log(MSE + 1e-10) + n_params * np.log(n_obs)
        
        return {
            'R2': R2,
            'MSE': MSE,
            'BIC': BIC,
            'RSS': RSS,
            'n_params': n_params
        }
    
    def compute_transfer_function(self, omegas: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Compute the transfer function G(ω, λ).
        
        G(ω, λ) = 1 / (1 - Σ_p H_p(λ) e^{-jωp})
        
        where H_p(λ) = Σ_k h_{p,k} λ^k
        
        Parameters:
        -----------
        omegas : np.ndarray
            Temporal frequencies (radians). Default: linspace(0, π, 256)
        
        Returns:
        --------
        dict with keys:
            - omegas: Temporal frequencies
            - lambdas: Graph eigenvalues (graph frequencies)
            - G: Complex transfer function
            - G_mag: Magnitude |G(ω, λ)|
            - G_phase: Phase ∠G(ω, λ)
            - H_p: Polynomial coefficients H_p(λ)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        if omegas is None:
            omegas = np.linspace(0, np.pi, 256)
        
        lambdas = self.eigenvalues
        P, K = self.P, self.K
        
        # Compute H_p(λ) = Σ_k h_{p,k} λ^k for each p
        H_p = np.zeros((P, len(lambdas)), dtype=np.complex128)
        for p in range(P):
            for i, lam in enumerate(lambdas):
                val = 0.0
                for k in range(K + 1):
                    val += self.h[p * (K + 1) + k] * (lam ** k)
                H_p[p, i] = val
        
        # Compute G(ω, λ) = 1 / (1 - Σ_p H_p(λ) e^{-jωp})
        G = np.zeros((len(omegas), len(lambdas)), dtype=np.complex128)
        for w_i, w in enumerate(omegas):
            z_terms = np.exp(-1j * w * np.arange(1, P + 1))
            denom = 1.0 - (z_terms[:, None] * H_p).sum(axis=0)
            
            # Stability guard
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
    
    def get_coefficients_interpretation(self) -> pd.DataFrame:
        """
        Get interpretable coefficient table.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        rows = []
        idx = 0
        for p in range(1, self.P + 1):
            for k in range(self.K + 1):
                rows.append({
                    'p (lag)': p,
                    'k (L^k)': k,
                    'h_{p,k}': self.h[idx],
                    'interpretation': f"Weight of L^{k} at lag {p}"
                })
                idx += 1
        
        return pd.DataFrame(rows)


# ============================================================================
# MODEL SELECTION
# ============================================================================

def model_selection(X: np.ndarray, L: np.ndarray,
                   P_range: List[int] = P_RANGE,
                   K_range: List[int] = K_RANGE,
                   validation_split: float = 0.15) -> Dict:
    """
    Find optimal model order (P, K) using BIC on validation set.
    
    Parameters:
    -----------
    X : np.ndarray
        EEG data (N × T)
    L : np.ndarray
        Graph Laplacian
    P_range : list
        AR orders to test
    K_range : list
        Graph filter orders to test
    validation_split : float
        Fraction of data for validation
    
    Returns:
    --------
    dict with best model info and all results
    """
    print("\n" + "=" * 80)
    print("MODEL SELECTION: Finding Optimal P and K")
    print("=" * 80)
    
    T = X.shape[1]
    T_train = int(0.70 * T)
    T_val = int((1 - validation_split) * T)
    
    X_train = X[:, :T_train]
    X_val = X[:, T_train:T_val]
    
    # Standardize
    X_train_std = safe_zscore(X_train)
    X_val_std = safe_zscore(X_val, X_train)  # Use training stats
    
    print(f"\nData split:")
    print(f"  Training:   {T_train} samples ({T_train/T*100:.1f}%)")
    print(f"  Validation: {T_val - T_train} samples ({(T_val-T_train)/T*100:.1f}%)")
    print(f"  Test:       {T - T_val} samples ({(T-T_val)/T*100:.1f}%)")
    
    print(f"\nTesting P ∈ {P_range} and K ∈ {K_range}...")
    print(f"Total configurations: {len(P_range) * len(K_range)}")
    
    results = []
    best_bic = np.inf
    best_P, best_K = None, None
    
    print("\n" + "-" * 70)
    print(f"{'P':>5} {'K':>5} {'BIC':>15} {'R²':>10} {'ρ':>8} {'Stable':>8}")
    print("-" * 70)
    
    for P in P_range:
        for K in K_range:
            try:
                # Fit model
                model = GPVAR_Model(P=P, K=K, L=L)
                model.fit(X_train_std)
                
                # Check stability
                rho = model.spectral_radius()
                stable = np.isfinite(rho) and rho < 1.0
                
                if not stable:
                    print(f"{P:>5} {K:>5} {'---':>15} {'---':>10} {rho:>8.3f} {'No':>8}")
                    results.append({
                        'P': P, 'K': K, 'BIC': np.inf, 'R2': np.nan,
                        'rho': rho, 'stable': False, 'success': True
                    })
                    continue
                
                # Evaluate on validation set
                metrics = model.evaluate(X_val_std)
                
                results.append({
                    'P': P, 'K': K,
                    'BIC': metrics['BIC'],
                    'R2': metrics['R2'],
                    'MSE': metrics['MSE'],
                    'rho': rho,
                    'stable': True,
                    'success': True,
                    'n_params': metrics['n_params']
                })
                
                # Check if best
                marker = ""
                if metrics['BIC'] < best_bic:
                    best_bic = metrics['BIC']
                    best_P = P
                    best_K = K
                    marker = " ← BEST"
                
                print(f"{P:>5} {K:>5} {metrics['BIC']:>15.2f} {metrics['R2']:>10.4f} "
                      f"{rho:>8.3f} {'Yes':>8}{marker}")
                
            except Exception as e:
                print(f"{P:>5} {K:>5} {'ERROR':>15} {'---':>10} {'---':>8} {'---':>8}")
                results.append({
                    'P': P, 'K': K, 'BIC': np.inf, 'R2': np.nan,
                    'rho': np.nan, 'stable': False, 'success': False,
                    'error': str(e)
                })
    
    print("-" * 70)
    
    # Fallback
    if best_P is None:
        print("\nWarning: No stable model found. Using fallback P=5, K=2")
        best_P, best_K = 5, 2
    
    print(f"\n✓ BEST MODEL: P = {best_P}, K = {best_K}")
    print(f"  BIC = {best_bic:.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "model_selection_results.csv", index=False)
    print(f"\n✓ Results saved to: {OUT_DIR / 'model_selection_results.csv'}")
    
    return {
        'best_P': best_P,
        'best_K': best_K,
        'best_BIC': best_bic,
        'results': results_df
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_model_selection(results_df: pd.DataFrame, save_dir: Path):
    """Create model selection visualization."""
    
    print("\nCreating model selection visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: BIC heatmap
    ax1 = axes[0, 0]
    pivot_bic = results_df.pivot(index='K', columns='P', values='BIC')
    
    # Replace inf with NaN for visualization
    pivot_bic = pivot_bic.replace([np.inf, -np.inf], np.nan)
    
    sns.heatmap(pivot_bic, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax1,
                cbar_kws={'label': 'BIC'})
    ax1.set_title('A) BIC (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('P (AR order)')
    ax1.set_ylabel('K (Graph filter order)')
    
    # Mark best model
    best_row = results_df[results_df['BIC'] == results_df['BIC'].min()].iloc[0]
    ax1.scatter([list(pivot_bic.columns).index(best_row['P']) + 0.5],
                [list(pivot_bic.index).index(best_row['K']) + 0.5],
                marker='*', s=500, color='gold', edgecolor='black', linewidth=2,
                zorder=10, label='Best')
    ax1.legend(loc='upper right')
    
    # Panel B: R² heatmap
    ax2 = axes[0, 1]
    pivot_r2 = results_df.pivot(index='K', columns='P', values='R2')
    
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2,
                cbar_kws={'label': 'R²'})
    ax2.set_title('B) R² (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('P (AR order)')
    ax2.set_ylabel('K (Graph filter order)')
    
    # Panel C: Spectral radius
    ax3 = axes[1, 0]
    pivot_rho = results_df.pivot(index='K', columns='P', values='rho')
    
    sns.heatmap(pivot_rho, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax3,
                vmin=0, vmax=1.1, cbar_kws={'label': 'Spectral Radius'})
    ax3.set_title('C) Spectral Radius (must be < 1 for stability)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('P (AR order)')
    ax3.set_ylabel('K (Graph filter order)')
    
    # Panel D: BIC vs P for each K
    ax4 = axes[1, 1]
    for K in sorted(results_df['K'].unique()):
        subset = results_df[results_df['K'] == K]
        subset = subset[subset['BIC'] < np.inf]  # Remove unstable
        ax4.plot(subset['P'], subset['BIC'], 'o-', linewidth=2, markersize=8, label=f'K={K}')
    
    ax4.axhline(results_df['BIC'].min(), color='red', linestyle='--', linewidth=2,
                label=f'Best BIC = {results_df["BIC"].min():.1f}')
    ax4.set_xlabel('P (AR order)', fontsize=11)
    ax4.set_ylabel('BIC', fontsize=11)
    ax4.set_title('D) BIC vs Model Order', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('GP-VAR Model Selection Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / 'model_selection_summary.png'
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {savepath}")


def visualize_transfer_function(model: GPVAR_Model, fs: float, save_dir: Path):
    """Create comprehensive transfer function visualization."""
    
    print("\nCreating transfer function visualization...")
    
    # Compute transfer function
    omegas = np.linspace(0, np.pi, 256)
    freqs_hz = omegas * fs / (2 * np.pi)
    
    tf = model.compute_transfer_function(omegas)
    G_mag = tf['G_mag']
    G_phase = tf['G_phase']
    lambdas = tf['lambdas']
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # ─────────────────────────────────────────────────────────────────
    # Panel A: 2D Magnitude heatmap
    # ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(G_mag, aspect='auto', origin='lower', cmap='hot',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax1.set_title('A) Transfer Function |G(ω, λ)|', fontsize=12, fontweight='bold')
    ax1.set_xlabel('λ (Graph Frequency)')
    ax1.set_ylabel('f (Hz)')
    plt.colorbar(im1, ax=ax1, label='Magnitude')
    
    # ─────────────────────────────────────────────────────────────────
    # Panel B: 2D Phase heatmap
    # ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(G_phase, aspect='auto', origin='lower', cmap='twilight',
                     extent=[lambdas.min(), lambdas.max(), 0, freqs_hz.max()])
    ax2.set_title('B) Phase Response ∠G(ω, λ)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('λ (Graph Frequency)')
    ax2.set_ylabel('f (Hz)')
    plt.colorbar(im2, ax=ax2, label='Phase (rad)')
    
    # ─────────────────────────────────────────────────────────────────
    # Panel C: Average over graph modes (frequency response)
    # ─────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    G_avg_modes = G_mag.mean(axis=1)
    ax3.plot(freqs_hz, G_avg_modes, 'b-', linewidth=2)
    ax3.fill_between(freqs_hz, 0, G_avg_modes, alpha=0.3)
    
    # Mark EEG bands
    bands = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 40)}
    colors = ['gray', 'cyan', 'green', 'orange', 'red']
    for (name, (f1, f2)), color in zip(bands.items(), colors):
        ax3.axvspan(f1, f2, alpha=0.15, color=color, label=name)
    
    ax3.set_xlabel('Frequency (Hz)', fontsize=11)
    ax3.set_ylabel('Average |G|', fontsize=11)
    ax3.set_title('C) Mode-Averaged Frequency Response', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', ncol=5, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 45])
    
    # ─────────────────────────────────────────────────────────────────
    # Panel D: Average over frequencies (graph mode response)
    # ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    G_avg_freq = G_mag.mean(axis=0)
    ax4.plot(lambdas, G_avg_freq, 'r-', linewidth=2)
    ax4.fill_between(lambdas, 0, G_avg_freq, alpha=0.3, color='red')
    ax4.set_xlabel('λ (Graph Frequency)', fontsize=11)
    ax4.set_ylabel('Average |G|', fontsize=11)
    ax4.set_title('D) Frequency-Averaged Graph Mode Response', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ─────────────────────────────────────────────────────────────────
    # Panel E: Frequency slices
    # ─────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    freq_indices = [int(f * len(freqs_hz) / freqs_hz.max()) for f in [5, 10, 20, 30]]
    colors = ['blue', 'green', 'orange', 'red']
    for f_idx, color in zip(freq_indices, colors):
        if f_idx < len(freqs_hz):
            ax5.plot(lambdas, G_mag[f_idx, :], '-', color=color, linewidth=2,
                    label=f'{freqs_hz[f_idx]:.0f} Hz')
    ax5.set_xlabel('λ (Graph Frequency)', fontsize=11)
    ax5.set_ylabel('|G|', fontsize=11)
    ax5.set_title('E) Transfer Function at Different Frequencies', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ─────────────────────────────────────────────────────────────────
    # Panel F: Graph mode slices
    # ─────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    mode_indices = [0, len(lambdas)//4, len(lambdas)//2, 3*len(lambdas)//4]
    for m_idx, color in zip(mode_indices, colors):
        ax6.plot(freqs_hz, G_mag[:, m_idx], '-', color=color, linewidth=2,
                label=f'λ={lambdas[m_idx]:.2f}')
    ax6.set_xlabel('Frequency (Hz)', fontsize=11)
    ax6.set_ylabel('|G|', fontsize=11)
    ax6.set_title('F) Transfer Function at Different Graph Modes', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 45])
    
    # ─────────────────────────────────────────────────────────────────
    # Panel G: Model coefficients
    # ─────────────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    coef_df = model.get_coefficients_interpretation()
    
    # Reshape for visualization
    h_matrix = model.h.reshape(model.P, model.K + 1)
    im7 = ax7.imshow(h_matrix.T, aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(model.h).max(), vmax=np.abs(model.h).max())
    ax7.set_xlabel('p (lag)')
    ax7.set_ylabel('k (L^k)')
    ax7.set_title('G) Model Coefficients h_{p,k}', fontsize=12, fontweight='bold')
    ax7.set_xticks(range(model.P))
    ax7.set_xticklabels(range(1, model.P + 1))
    ax7.set_yticks(range(model.K + 1))
    plt.colorbar(im7, ax=ax7, label='Coefficient')
    
    # ─────────────────────────────────────────────────────────────────
    # Panel H: Coefficient bar plot
    # ─────────────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    coef_labels = [f'h_{{{p},{k}}}' for p in range(1, model.P+1) for k in range(model.K+1)]
    colors = ['red' if h < 0 else 'blue' for h in model.h]
    ax8.bar(range(len(model.h)), model.h, color=colors, alpha=0.7, edgecolor='black')
    ax8.axhline(0, color='black', linewidth=1)
    ax8.set_xlabel('Coefficient Index', fontsize=11)
    ax8.set_ylabel('Value', fontsize=11)
    ax8.set_title('H) Coefficient Values', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ─────────────────────────────────────────────────────────────────
    # Panel I: Summary statistics
    # ─────────────────────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
╔═══════════════════════════════════════════════════╗
║            GP-VAR MODEL SUMMARY                   ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  MODEL PARAMETERS                                 ║
║  ────────────────                                 ║
║    P (AR order):        {model.P}                        ║
║    K (graph filter):    {model.K}                        ║
║    Total coefficients:  {len(model.h)}                       ║
║    Ridge λ:             {model.ridge_lambda}                   ║
║                                                   ║
║  STABILITY                                        ║
║  ─────────                                        ║
║    Spectral radius:     {model.spectral_radius():.4f}                 ║
║    Status:              {'STABLE ✓' if model.spectral_radius() < 1 else 'UNSTABLE ✗'}                ║
║                                                   ║
║  TRANSFER FUNCTION                                ║
║  ─────────────────                                ║
║    Peak magnitude:      {G_mag.max():.4f}                 ║
║    Mean magnitude:      {G_mag.mean():.4f}                 ║
║    Low-freq peak (λ):   {lambdas[G_mag[:10].mean(axis=0).argmax()]:.4f}                 ║
║                                                   ║
║  INTERPRETATION                                   ║
║  ──────────────                                   ║
║    • Low λ: Global brain modes                    ║
║    • High λ: Localized brain modes                ║
║    • Low ω: Slow oscillations                     ║
║    • High ω: Fast oscillations                    ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""
    ax9.text(0.02, 0.98, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    
    plt.suptitle(f'GP-VAR System Dynamics: P={model.P}, K={model.K}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / 'transfer_function_analysis.png'
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {savepath}")
    
    return tf


def visualize_3d_transfer_function(model: GPVAR_Model, fs: float, save_dir: Path):
    """Create 3D surface plot of transfer function."""
    
    print("Creating 3D transfer function visualization...")
    
    omegas = np.linspace(0, np.pi, 128)
    freqs_hz = omegas * fs / (2 * np.pi)
    
    tf = model.compute_transfer_function(omegas)
    G_mag = tf['G_mag']
    lambdas = tf['lambdas']
    
    # Downsample for smoother 3D plot
    Lambda_grid, Freq_grid = np.meshgrid(lambdas[::2], freqs_hz[::2])
    G_plot = G_mag[::2, ::2]
    
    fig = plt.figure(figsize=(16, 7))
    
    # 3D Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(Lambda_grid, Freq_grid, G_plot, 
                           cmap='viridis', alpha=0.9, antialiased=True)
    ax1.set_xlabel('λ (Graph Frequency)', fontsize=10)
    ax1.set_ylabel('f (Hz)', fontsize=10)
    ax1.set_zlabel('|G(ω, λ)|', fontsize=10)
    ax1.set_title('3D Transfer Function Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.6, label='Magnitude')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(Lambda_grid, Freq_grid, G_plot, levels=20, cmap='viridis')
    ax2.set_xlabel('λ (Graph Frequency)', fontsize=11)
    ax2.set_ylabel('f (Hz)', fontsize=11)
    ax2.set_title('Contour Plot of |G(ω, λ)|', fontsize=12, fontweight='bold')
    fig.colorbar(contour, ax=ax2, label='Magnitude')
    
    plt.suptitle('GP-VAR Transfer Function: Joint Graph-Temporal Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    savepath = save_dir / 'transfer_function_3d.png'
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {savepath}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Main analysis pipeline:
    1. Load/create consensus Laplacian
    2. Load/generate EEG data
    3. Perform model selection
    4. Fit best model
    5. Analyze and visualize results
    """
    
    print("\n" + "=" * 80)
    print("STARTING GP-VAR MODEL IDENTIFICATION")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────
    # Step 1: Load or create consensus Laplacian
    # ─────────────────────────────────────────────────────────────────
    print("\nStep 1: Loading consensus Laplacian...")
    
    if Path(CONSENSUS_LAPLACIAN_PATH).exists() and not USE_SYNTHETIC_DATA:
        L = np.load(CONSENSUS_LAPLACIAN_PATH)
        print(f"  Loaded from: {CONSENSUS_LAPLACIAN_PATH}")
    else:
        print("  Creating synthetic Laplacian for demonstration...")
        n_channels = 64
        
        # Create synthetic adjacency (simulating brain network)
        A = np.random.rand(n_channels, n_channels) * 0.5
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        # Add some structure (local connectivity)
        for i in range(n_channels):
            for j in range(max(0, i-5), min(n_channels, i+6)):
                if i != j:
                    A[i, j] += 0.3
        
        L = create_consensus_laplacian(A)
    
    n_channels = L.shape[0]
    print(f"  Laplacian shape: {L.shape}")
    print(f"  Eigenvalue range: [{np.linalg.eigvalsh(L).min():.4f}, {np.linalg.eigvalsh(L).max():.4f}]")
    
    # Save Laplacian
    np.save(OUT_DIR / "consensus_laplacian.npy", L)
    
    # ─────────────────────────────────────────────────────────────────
    # Step 2: Load or generate EEG data
    # ─────────────────────────────────────────────────────────────────
    print("\nStep 2: Loading/Generating EEG data...")
    
    if USE_SYNTHETIC_DATA:
        X, L, true_params = generate_synthetic_data(n_channels=n_channels, 
                                                    duration_sec=300.0, 
                                                    fs=TARGET_SFREQ, L=L)
    else:
        try:
            import mne
            raw = mne.io.read_raw_eeglab(EEG_DATA_PATH, preload=True, verbose=False)
            if raw.info["sfreq"] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)
            raw.filter(l_freq=BAND[0], h_freq=BAND[1], verbose=False)
            X = raw.get_data()
            X = np.nan_to_num(X, nan=0.0)
            true_params = None
        except Exception as e:
            print(f"  Warning: Could not load EEG data ({e})")
            print("  Falling back to synthetic data...")
            X, L, true_params = generate_synthetic_data(n_channels=n_channels,
                                                        duration_sec=300.0,
                                                        fs=TARGET_SFREQ, L=L)
    
    print(f"  Data shape: {X.shape}")
    print(f"  Duration: {X.shape[1] / TARGET_SFREQ:.1f} seconds")
    
    # Standardize
    X_std = safe_zscore(X)
    
    # ─────────────────────────────────────────────────────────────────
    # Step 3: Model Selection
    # ─────────────────────────────────────────────────────────────────
    selection_results = model_selection(X_std, L, P_RANGE, K_RANGE)
    
    best_P = selection_results['best_P']
    best_K = selection_results['best_K']
    
    # Visualize model selection
    visualize_model_selection(selection_results['results'], OUT_DIR)
    
    # ─────────────────────────────────────────────────────────────────
    # Step 4: Fit Best Model on Full Data
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FITTING BEST GP-VAR MODEL")
    print("=" * 80)
    
    print(f"\nFitting model with P={best_P}, K={best_K}...")
    
    best_model = GPVAR_Model(P=best_P, K=best_K, L=L)
    best_model.fit(X_std)
    
    # Evaluate
    metrics = best_model.evaluate(X_std)
    rho = best_model.spectral_radius()
    
    print(f"\nModel Performance:")
    print(f"  R²:              {metrics['R2']:.4f}")
    print(f"  MSE:             {metrics['MSE']:.6f}")
    print(f"  BIC:             {metrics['BIC']:.2f}")
    print(f"  Spectral radius: {rho:.4f}")
    print(f"  Stable:          {'Yes ✓' if rho < 1 else 'No ✗'}")
    
    if true_params is not None:
        print(f"\n  (True model was P={true_params['P']}, K={true_params['K']})")
    
    # ─────────────────────────────────────────────────────────────────
    # Step 5: Analyze and Visualize
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYZING SYSTEM DYNAMICS")
    print("=" * 80)
    
    # Transfer function visualization
    tf_results = visualize_transfer_function(best_model, TARGET_SFREQ, OUT_DIR)
    
    # 3D visualization
    visualize_3d_transfer_function(best_model, TARGET_SFREQ, OUT_DIR)
    
    # Save coefficients
    coef_df = best_model.get_coefficients_interpretation()
    coef_df.to_csv(OUT_DIR / "model_coefficients.csv", index=False)
    print(f"\n✓ Coefficients saved to: {OUT_DIR / 'model_coefficients.csv'}")
    
    # Save model info
    model_info = {
        'P': best_P,
        'K': best_K,
        'n_channels': n_channels,
        'n_coefficients': len(best_model.h),
        'spectral_radius': rho,
        'R2': metrics['R2'],
        'MSE': metrics['MSE'],
        'BIC': metrics['BIC'],
        'stable': rho < 1
    }
    
    import json
    with open(OUT_DIR / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✓ Model info saved to: {OUT_DIR / 'model_info.json'}")
    
    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           FINAL SUMMARY                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  BEST MODEL FOUND                                                             ║
║  ────────────────                                                             ║
║    • AR order (P):           {best_P}                                              ║
║    • Graph filter order (K): {best_K}                                              ║
║    • Total parameters:       {len(best_model.h)} coefficients                              ║
║                                                                               ║
║  MODEL PERFORMANCE                                                            ║
║  ─────────────────                                                            ║
║    • R² (variance explained): {metrics['R2']:.4f}                                     ║
║    • Spectral radius:         {rho:.4f} {'(STABLE)' if rho < 1 else '(UNSTABLE)'}                              ║
║    • BIC:                     {metrics['BIC']:.2f}                                   ║
║                                                                               ║
║  WHAT THIS MEANS                                                              ║
║  ───────────────                                                              ║
║    The GP-VAR model captures the system dynamics by:                          ║
║    • P={best_P} means the system has memory of {best_P} past time steps               ║
║    • K={best_K} means graph filtering uses up to {best_K}-hop neighbors               ║
║                                                                               ║
║    The transfer function G(ω,λ) reveals:                                      ║
║    • How temporal frequencies (ω) interact with graph modes (λ)               ║
║    • Which brain rhythms are amplified/attenuated                             ║
║    • The joint spectral properties of the system                              ║
║                                                                               ║
║  OUTPUT FILES                                                                 ║
║  ────────────                                                                 ║
║    {OUT_DIR}/                                                   ║
║    ├── model_selection_results.csv  (all P,K combinations)                    ║
║    ├── model_selection_summary.png  (selection visualization)                 ║
║    ├── model_coefficients.csv       (fitted h_{{p,k}} values)                   ║
║    ├── model_info.json              (summary)                                 ║
║    ├── transfer_function_analysis.png                                         ║
║    └── transfer_function_3d.png                                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    
    return best_model, metrics, tf_results


if __name__ == "__main__":
    model, metrics, tf_results = main()
