#!/usr/bin/env python3
"""
Script to run the single subject LTI vs TV transfer function analysis.
This script can be executed to analyze one EEG subject and compare
the frequency responses of LTI and TV models.
"""

import sys
import os
from pathlib import Path

# Add the workspace to the path if needed
sys.path.insert(0, '/workspace')

def main():
    """Run the single subject analysis."""
    
    print("Starting single subject LTI vs TV analysis...")
    print("-" * 60)
    
    # Check if we can import the module
    try:
        from lti_tv_single_subject_analysis import analyze_single_subject
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install numpy scipy matplotlib seaborn mne pandas")
        return 1
    
    # Run the analysis
    try:
        results = analyze_single_subject()
        
        if results is not None:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\nSubject ID: {results['subject_id']}")
            print(f"Model Order: P={results['best_P']}, K={results['best_K']}")
            print(f"Duration: {results['duration']:.1f} seconds")
            print(f"Number of channels: {results['n_channels']}")
            print(f"Number of TV windows: {len(results['tv_results'])}")
            
            # Summary of transfer function comparison
            tf_comp = results['tf_comparison']
            print(f"\nTransfer Function Analysis:")
            print(f"  Mean MSD: {tf_comp['msd_per_window'].mean():.6f}")
            print(f"  Frequency range: 0 - {tf_comp['freqs_hz'].max():.1f} Hz")
            print(f"  Number of graph modes: {len(tf_comp['lambdas'])}")
            
            print(f"\nResults saved in: ./single_subject_lti_tv_analysis/")
            print("  - Transfer function comparison plots")
            print("  - 3D surface visualizations")
            
            return 0
        else:
            print("\nAnalysis failed. Please check the error messages above.")
            return 1
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())