import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_all_burst_data(folder_path):
    """Load all CSV files from the individual_csvs folder and combine them."""
    all_files = glob.glob(os.path.join(folder_path, "burst_data_*.csv"))
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df = df.dropna(subset=['Length'])
            dfs.append(df)
        except Exception as e:
            pass
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    return pd.DataFrame()

def compute_cdf(data):
    """Compute the Cumulative Distribution Function (CDF)."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf

def main():
    # Path to the individual CSV files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(script_dir, '..', 'individual_csvs')
    output_dir = script_dir
    
    print(f"Loading data from: {csv_folder}")
    
    # Load all burst data
    df = load_all_burst_data(csv_folder)
    
    if df.empty:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(df)} burst records")
    
    # Get burst durations (Length column, in ms)
    durations = df['Length'].values
    durations = durations[durations > 0]  # Filter out zero durations
    
    print(f"Burst duration range: [{durations.min():.2f}, {durations.max():.2f}] ms")
    print(f"Median duration: {np.median(durations):.2f} ms")
    print(f"Mean duration: {np.mean(durations):.2f} ms")
    
    # Compute CDF
    x, cdf = compute_cdf(durations)
    
    # Plot CDF - side by side: normal and log x-axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Normal x-axis
    axes[0].plot(x, cdf, 'b-', linewidth=2)
    axes[0].set_xlabel('Burst Duration (ms)', fontsize=12)
    axes[0].set_ylabel('CDF', fontsize=12)
    axes[0].set_title('Burst Duration CDF (Linear Scale)', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Right plot: Log x-axis
    axes[1].plot(x, cdf, 'b-', linewidth=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Burst Duration (ms)', fontsize=12)
    axes[1].set_ylabel('CDF', fontsize=12)
    axes[1].set_title('Burst Duration CDF (Log Scale)', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.5, which='both')
    
    # Add statistics to the right plot
    stats_text = (f'Total bursts: {len(durations):,}\n'
                  f'Min: {durations.min():.1f} ms\n'
                  f'Median: {np.median(durations):.1f} ms\n'
                  f'Mean: {np.mean(durations):.1f} ms\n'
                  f'Max: {durations.max():.1f} ms')
    axes[1].text(0.98, 0.02, stats_text, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'burst_duration_cdf.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

if __name__ == '__main__':
    main()
