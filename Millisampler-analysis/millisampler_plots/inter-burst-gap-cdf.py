import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_all_burst_data(folder_path):
    """Load all CSV files from the individual_csvs folder and combine them."""
    all_files = glob.glob(os.path.join(folder_path, "burst_data_*.csv"))
    
    print(f"Found {len(all_files)} files to process...")
    
    all_data = []
    for i, file in enumerate(all_files):
        if i % 500 == 0:
            print(f"  Processing file {i}/{len(all_files)}...")
        try:
            df = pd.read_csv(file, usecols=['Position', 'Length'])
            # Filter out rows that don't have burst data (session summary rows)
            df = df.dropna(subset=['Position', 'Length'])
            df['file_idx'] = i  # Use index instead of filename for grouping
            all_data.append(df)
        except Exception as e:
            pass  # Skip problematic files silently
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return pd.DataFrame()

def compute_inter_burst_gaps(df):
    """Compute inter-burst gaps for each host."""
    gaps = []
    
    # Group by host (file_idx) to compute gaps within each host's burst sequence
    grouped = df.groupby('file_idx')
    total_groups = len(grouped)
    
    for idx, (file_idx, group) in enumerate(grouped):
        if idx % 500 == 0:
            print(f"  Computing gaps for host {idx}/{total_groups}...")
        
        # Sort by position within the session
        positions = group['Position'].values
        lengths = group['Length'].values
        
        # Sort by position
        sort_idx = np.argsort(positions)
        positions = positions[sort_idx]
        lengths = lengths[sort_idx]
        
        # Inter-burst gap = start of next burst - (start of current burst + length of current burst)
        if len(positions) > 1:
            burst_ends = positions[:-1] + lengths[:-1]
            host_gaps = positions[1:] - burst_ends
            gaps.extend(host_gaps[host_gaps >= 0])
    
    return np.array(gaps)

def compute_cdf(data):
    """Compute the Cumulative Distribution Function (CDF)."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf

def plot_cdf(gaps, output_file):
    """Plot CDF on linear scale."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter out zero gaps
    gaps_nonzero = gaps[gaps > 0]
    
    print(f"Total inter-burst gaps: {len(gaps)}")
    print(f"Non-zero gaps: {len(gaps_nonzero)}")
    print(f"Gap range: [{gaps_nonzero.min():.2f}, {gaps_nonzero.max():.2f}] ms")
    print(f"Median gap: {np.median(gaps_nonzero):.2f} ms")
    print(f"Mean gap: {np.mean(gaps_nonzero):.2f} ms")
    
    # Compute CDF
    x, cdf = compute_cdf(gaps_nonzero)
    
    # Plot CDF
    ax.plot(x, cdf, 'b-', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Inter-burst Gap (ms)', fontsize=12)
    ax.set_ylabel('CDF (P[X ≤ x])', fontsize=12)
    ax.set_title('Inter-burst Gap Distribution (CDF)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add statistics annotation
    stats_text = (f'Total gaps: {len(gaps_nonzero):,}\n'
                  f'Min: {gaps_nonzero.min():.1f} ms\n'
                  f'Median: {np.median(gaps_nonzero):.1f} ms\n'
                  f'Mean: {np.mean(gaps_nonzero):.1f} ms\n'
                  f'Max: {gaps_nonzero.max():.1f} ms')
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

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
    
    print(f"Loaded {len(df)} burst records from {df['file_idx'].nunique()} hosts")
    
    # Compute inter-burst gaps
    print("\nComputing inter-burst gaps...")
    gaps = compute_inter_burst_gaps(df)
    
    if len(gaps) == 0:
        print("No gaps computed!")
        return
    
    # Plot CDF
    plot_cdf(
        gaps=gaps,
        output_file=os.path.join(output_dir, 'inter_burst_gap_cdf.png')
    )

if __name__ == '__main__':
    main()
