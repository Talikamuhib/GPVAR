"""
Process EEG files to create consensus matrices for AD and HC groups.
This script uses the full list of files provided and demonstrates the consensus matrix analysis.
"""

import numpy as np
import logging
from pathlib import Path
from consensus_matrix_eeg import ConsensusMatrix, process_eeg_files
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define all file paths
FILE_PATHS = {
    'AD_AR': [
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
    ],
    'AD_CL': [
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
    'HC_AR': [
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
    ],
    'HC_CL': [
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


def analyze_consensus_properties(C, W, G, group_name):
    """
    Analyze and report properties of consensus matrices.
    
    Parameters
    ----------
    C : np.ndarray
        Consensus matrix
    W : np.ndarray
        Weight matrix
    G : np.ndarray
        Final group graph
    group_name : str
        Name of the group for reporting
    """
    print(f"\n{'='*60}")
    print(f"Analysis for {group_name}")
    print(f"{'='*60}")
    
    # Consensus matrix statistics
    n_nodes = C.shape[0]
    triu_idx = np.triu_indices(n_nodes, k=1)
    consensus_values = C[triu_idx]
    
    print("\nConsensus Matrix Statistics:")
    print(f"  - Shape: {C.shape}")
    print(f"  - Min consensus: {consensus_values.min():.3f}")
    print(f"  - Max consensus: {consensus_values.max():.3f}")
    print(f"  - Mean consensus: {consensus_values.mean():.3f}")
    print(f"  - Median consensus: {np.median(consensus_values):.3f}")
    print(f"  - Edges with C > 0: {np.sum(consensus_values > 0)} / {len(consensus_values)}")
    print(f"  - Edges with C > 0.5: {np.sum(consensus_values > 0.5)}")
    print(f"  - Edges with C = 1.0: {np.sum(consensus_values == 1.0)}")
    
    # Weight matrix statistics
    weight_values = W[triu_idx]
    weight_values_nonzero = weight_values[weight_values > 0]
    
    print("\nWeight Matrix Statistics:")
    print(f"  - Min weight (non-zero): {weight_values_nonzero.min():.3f}")
    print(f"  - Max weight: {weight_values_nonzero.max():.3f}")
    print(f"  - Mean weight (non-zero): {weight_values_nonzero.mean():.3f}")
    print(f"  - Median weight (non-zero): {np.median(weight_values_nonzero):.3f}")
    
    # Final graph statistics
    graph_values = G[triu_idx]
    n_edges_final = np.sum(graph_values > 0)
    sparsity_final = n_edges_final / len(graph_values)
    
    print("\nFinal Graph Statistics:")
    print(f"  - Number of edges: {n_edges_final}")
    print(f"  - Sparsity: {sparsity_final:.3f}")
    print(f"  - Min edge weight: {graph_values[graph_values > 0].min():.3f}")
    print(f"  - Max edge weight: {graph_values[graph_values > 0].max():.3f}")
    print(f"  - Mean edge weight: {graph_values[graph_values > 0].mean():.3f}")
    
    # Degree distribution
    degrees = np.sum(G > 0, axis=0)
    print(f"\nDegree Distribution:")
    print(f"  - Min degree: {degrees.min()}")
    print(f"  - Max degree: {degrees.max()}")
    print(f"  - Mean degree: {degrees.mean():.2f}")
    print(f"  - Std degree: {degrees.std():.2f}")


def visualize_group_comparison(results_dict, output_dir="./consensus_results"):
    """
    Create comparison visualizations across groups.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing results for each group
    output_dir : str
        Directory to save visualizations
    """
    n_groups = len(results_dict)
    fig, axes = plt.subplots(3, n_groups, figsize=(5*n_groups, 15))
    
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (group_name, results) in enumerate(results_dict.items()):
        C = results['consensus_matrix']
        W = results['weight_matrix']
        G = results['final_graph']
        
        # Plot consensus matrix
        im1 = axes[0, idx].imshow(C, cmap='hot', vmin=0, vmax=1)
        axes[0, idx].set_title(f'{group_name}\nConsensus Matrix')
        axes[0, idx].set_xlabel('Channel')
        axes[0, idx].set_ylabel('Channel')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046)
        
        # Plot weight matrix
        im2 = axes[1, idx].imshow(W, cmap='viridis', vmin=0)
        axes[1, idx].set_title(f'Weight Matrix')
        axes[1, idx].set_xlabel('Channel')
        axes[1, idx].set_ylabel('Channel')
        plt.colorbar(im2, ax=axes[1, idx], fraction=0.046)
        
        # Plot final graph
        im3 = axes[2, idx].imshow(G, cmap='plasma', vmin=0)
        axes[2, idx].set_title(f'Final Graph')
        axes[2, idx].set_xlabel('Channel')
        axes[2, idx].set_ylabel('Channel')
        plt.colorbar(im3, ax=axes[2, idx], fraction=0.046)
    
    plt.suptitle('Consensus Matrix Analysis Across Groups', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "group_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create histogram comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
    
    for idx, (group_name, results) in enumerate(results_dict.items()):
        C = results['consensus_matrix']
        n_nodes = C.shape[0]
        triu_idx = np.triu_indices(n_nodes, k=1)
        consensus_values = C[triu_idx]
        
        # Histogram of consensus values
        axes[idx].hist(consensus_values[consensus_values > 0], 
                      bins=50, alpha=0.7, color=colors[idx], 
                      edgecolor='black', linewidth=1.2)
        axes[idx].set_xlabel('Consensus Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{group_name} - Consensus Distribution')
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = consensus_values[consensus_values > 0].mean()
        median_val = np.median(consensus_values[consensus_values > 0])
        axes[idx].axvline(mean_val, color='red', linestyle='--', 
                         label=f'Mean: {mean_val:.3f}')
        axes[idx].axvline(median_val, color='blue', linestyle='--', 
                         label=f'Median: {median_val:.3f}')
        axes[idx].legend()
    
    plt.suptitle('Consensus Value Distributions', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "consensus_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to process all EEG files and create consensus matrices.
    """
    # Parameters
    SPARSITY_BINARIZE = 0.15  # Sparsity for initial binarization (15% edges kept)
    SPARSITY_FINAL = 0.10      # Target sparsity for final graph (10% edges)
    USE_DISTANCE = True        # Use distance-dependent consensus
    OUTPUT_DIR = "./consensus_results"
    
    # Store all results
    all_results = {}
    
    # Process each group
    for group_name, file_list in FILE_PATHS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {group_name} group ({len(file_list)} files)")
        logger.info(f"{'='*60}")
        
        try:
            # Process files
            results = process_eeg_files(
                file_paths=file_list,
                sparsity_binarize=SPARSITY_BINARIZE,
                sparsity_final=SPARSITY_FINAL,
                method='distance' if USE_DISTANCE else 'uniform',
                output_dir=f"{OUTPUT_DIR}/{group_name}"
            )
            
            all_results[group_name] = results
            
            # Analyze properties
            analyze_consensus_properties(
                results['consensus_matrix'],
                results['weight_matrix'],
                results['final_graph'],
                group_name
            )
            
        except Exception as e:
            logger.error(f"Failed to process {group_name}: {e}")
            continue
    
    # Create comparison visualizations
    if all_results:
        logger.info("\nCreating group comparison visualizations...")
        visualize_group_comparison(all_results, OUTPUT_DIR)
        
        # Save combined results
        logger.info("\nSaving combined results...")
        np.savez(f"{OUTPUT_DIR}/all_consensus_results.npz", **all_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed {len(all_results)} groups:")
        for group_name in all_results.keys():
            n_subjects = len(all_results[group_name]['valid_files'])
            print(f"  - {group_name}: {n_subjects} subjects")
        print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    # Run the main analysis
    results = main()