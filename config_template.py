"""
Configuration Template for LTI vs TV GP-VAR Analysis
=====================================================

Copy this file and modify the paths and parameters for your dataset.
"""

from pathlib import Path

# ============================================================================
# DATA PATHS - MODIFY THESE FOR YOUR DATA
# ============================================================================

# Path to your consensus Laplacian matrix (numpy .npy file)
# This should be an NxN symmetric positive semi-definite matrix
# representing the brain network structure
CONSENSUS_LAPLACIAN_PATH = "/path/to/your/consensus_laplacian.npy"

# Dictionary of subject files organized by group
# Each key is a group name, each value is a list of file paths
SUBJECT_FILES = {
    'Group1': [
        '/path/to/group1/subject1.set',
        '/path/to/group1/subject2.set',
        # Add more subjects...
    ],
    'Group2': [
        '/path/to/group2/subject1.set',
        '/path/to/group2/subject2.set',
        # Add more subjects...
    ],
    # Add more groups as needed...
}

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Frequency band for band-pass filtering (Hz)
BAND = (0.5, 40.0)  # Default: 0.5-40 Hz (typical EEG range)

# Target sampling frequency after resampling (Hz)
TARGET_SFREQ = 100.0  # Default: 100 Hz

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Whether to include bias term in the model
USE_BIAS = False  # Default: False (no bias for GP-VAR)

# Ridge regularization parameter
RIDGE_LAMBDA = 5e-3  # Default: 5e-3 (adjust based on data SNR)

# Search ranges for model selection
P_RANGE = [1, 2, 3, 5, 7, 10, 15, 20]  # Temporal AR orders to test
K_RANGE = [1, 2, 3, 4]  # Graph polynomial orders to test

# ============================================================================
# TIME-VARYING ANALYSIS PARAMETERS
# ============================================================================

# Window length in seconds for time-varying analysis
WINDOW_LENGTH_SEC = 10.0  # Default: 10 seconds

# Window overlap fraction (0 = no overlap, 0.5 = 50% overlap)
WINDOW_OVERLAP = 0.5  # Default: 50% overlap

# Minimum number of windows required for TV analysis
MIN_WINDOWS = 5  # Default: at least 5 windows

# ============================================================================
# STATISTICAL TESTING PARAMETERS
# ============================================================================

# Number of surrogate datasets for null distribution
N_SURROGATES = 200  # Default: 200 (more = better statistics but slower)

# Significance level for hypothesis testing
ALPHA = 0.05  # Default: 0.05 (5% significance level)

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output directory for results
OUT_DIR = Path("./my_analysis_results")

# Figure settings
FIG_DPI = 200  # DPI for saved figures
FIG_FORMAT = 'png'  # Format for saved figures ('png', 'pdf', 'svg')

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Stability threshold for model acceptance
MAX_SPECTRAL_RADIUS = 0.99  # Models with ρ ≥ this are rejected

# Numerical stability parameters
MIN_STD = 1e-8  # Minimum standard deviation for z-scoring
MAX_ZSCORE = 10.0  # Maximum absolute z-score value (clipping)

# Convergence criteria for optimization
CONVERGENCE_TOL = 1e-6  # Tolerance for iterative algorithms

# Parallel processing (if implementing)
N_JOBS = -1  # Number of parallel jobs (-1 = use all cores)

# ============================================================================
# PLOTTING PREFERENCES
# ============================================================================

# Colormap choices
COLORMAP_TF = 'hot'  # Colormap for transfer functions
COLORMAP_DIFF = 'RdBu_r'  # Colormap for differences
COLORMAP_VAR = 'YlOrRd'  # Colormap for variance

# Font sizes
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

# Minimum recording duration in seconds
MIN_DURATION = 60.0  # Default: 60 seconds minimum

# Maximum proportion of bad channels allowed
MAX_BAD_CHANNELS = 0.2  # Default: 20% max bad channels

# ============================================================================
# NOTES FOR USERS
# ============================================================================

"""
IMPORTANT CONSIDERATIONS:

1. Laplacian Matrix:
   - Must be symmetric positive semi-definite
   - Typically derived from connectivity analysis or electrode distances
   - Should match the number of channels in your EEG data

2. Data Format:
   - Currently supports .set files (EEGLAB format)
   - Modify load_and_preprocess_eeg() function for other formats

3. Group Names:
   - Use meaningful group names (e.g., 'AD', 'HC', 'Control', 'Patient')
   - These will be used in output files and visualizations

4. Window Length:
   - Should be long enough to capture dynamics (typically 5-20 seconds)
   - Shorter windows = more time resolution but less stable estimates
   - Must fit at least MIN_WINDOWS in your shortest recording

5. Model Order Selection:
   - P (temporal order): Higher = more past dependence, risk of overfitting
   - K (graph order): Higher = more complex spatial patterns, computational cost

6. Computational Time:
   - Scales with: N_subjects × N_surrogates × Recording_length
   - Typical: 2-5 minutes per subject
   - Can be reduced by decreasing N_SURROGATES or downsampling

7. Memory Requirements:
   - Approximately: 2-4 GB per subject
   - Increases with recording length and number of channels

8. Interpretation:
   - TIME-INVARIANT: Consistent dynamics, simpler model sufficient
   - TIME-VARYING: Evolving dynamics, may indicate state changes

9. Quality Control:
   - Check spectral radius (should be < 1 for stability)
   - Verify sufficient R² (typically > 0.3 for good fit)
   - Ensure adequate number of windows for TV analysis
"""

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_config():
    """Validate configuration parameters."""
    import numpy as np
    
    errors = []
    warnings = []
    
    # Check paths exist
    if not Path(CONSENSUS_LAPLACIAN_PATH).exists():
        errors.append(f"Laplacian file not found: {CONSENSUS_LAPLACIAN_PATH}")
    
    # Check subject files
    total_subjects = 0
    for group, files in SUBJECT_FILES.items():
        for f in files:
            if not Path(f).exists():
                warnings.append(f"Subject file not found: {f}")
            else:
                total_subjects += 1
    
    if total_subjects == 0:
        errors.append("No valid subject files found")
    
    # Check parameters
    if WINDOW_LENGTH_SEC <= 0:
        errors.append("Window length must be positive")
    
    if not 0 <= WINDOW_OVERLAP < 1:
        errors.append("Window overlap must be in [0, 1)")
    
    if N_SURROGATES < 100:
        warnings.append("N_SURROGATES < 100 may give unreliable p-values")
    
    if not 0 < ALPHA < 1:
        errors.append("Alpha must be in (0, 1)")
    
    # Report
    print("Configuration Validation")
    print("=" * 50)
    
    if errors:
        print("❌ ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✓ No errors found")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    print(f"\n📊 Summary:")
    print(f"  - Groups: {len(SUBJECT_FILES)}")
    print(f"  - Total subjects: {total_subjects}")
    print(f"  - Window length: {WINDOW_LENGTH_SEC}s")
    print(f"  - Surrogates: {N_SURROGATES}")
    
    return len(errors) == 0


if __name__ == "__main__":
    # Run validation when this config file is executed directly
    if validate_config():
        print("\n✅ Configuration is valid and ready to use!")
    else:
        print("\n❌ Please fix errors before running analysis")