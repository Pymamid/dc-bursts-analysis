import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Link rate for normalization (in bytes per millisecond)
# 12.5 Gbps = 12.5e9 bits/s = 1.5625e9 bytes/s = 1.5625e6 bytes/ms
LINK_RATE_BYTES_PER_MS = 1.5625e6  # 12.5 Gbps in bytes per millisecond

def load_all_burst_data(folder_path):
    """Load all CSV files from the individual_csvs folder and combine them."""
    all_files = glob.glob(os.path.join(folder_path, "burst_data_*.csv"))
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # Filter out rows that don't have burst data (session summary rows)
            df = df.dropna(subset=['Length', 'ingressMax', 'maxConnections'])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    return pd.DataFrame()

def plot_hexbin_all(x, y, ecn_data, retx_data, xlabel, ylabel, output_file, 
                     link_rate=LINK_RATE_BYTES_PER_MS):
    """Create three separate hexbin plots: count, ECN fraction, and Retx fraction."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Normalize y by link rate
    y_normalized = y / link_rate
    
    # Filter out invalid values
    mask = np.isfinite(x) & np.isfinite(y_normalized) & np.isfinite(ecn_data) & np.isfinite(retx_data)
    x_clean = x[mask]
    y_clean = y_normalized[mask]
    ecn_clean = ecn_data[mask]
    retx_clean = retx_data[mask]
    
    print(f"Total data points: {len(x_clean)}")
    print(f"X range: [{x_clean.min():.2f}, {x_clean.max():.2f}]")
    print(f"Y range (normalized): [{y_clean.min():.4f}, {y_clean.max():.4f}]")
    
    # Convert to binary
    has_ecn = (ecn_clean > 0).astype(float)
    has_retx = (retx_clean > 0).astype(float)
    
    # Plot 1: Count (density)
    hb1 = ax1.hexbin(x_clean, y_clean, gridsize=50, cmap='YlOrRd', 
                     mincnt=1, bins='log')
    cb1 = fig.colorbar(hb1, ax=ax1, label='Count (log scale)')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title('Burst Count', fontsize=14)
    ax1.text(0.02, 0.98, f'Total: {len(x_clean):,}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: ECN fraction
    hb2 = ax2.hexbin(x_clean, y_clean, C=has_ecn, gridsize=50, cmap='YlOrRd', 
                     reduce_C_function=np.mean, mincnt=1, vmin=0, vmax=1)
    cb2 = fig.colorbar(hb2, ax=ax2, label='Fraction with ECN')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title('ECN Presence', fontsize=14)
    ecn_count = (has_ecn > 0).sum()
    no_ecn_count = (has_ecn == 0).sum()
    ax2.text(0.02, 0.98, f'No ECN: {no_ecn_count:,}\nHas ECN: {ecn_count:,}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Retx fraction
    hb3 = ax3.hexbin(x_clean, y_clean, C=has_retx, gridsize=50, cmap='YlOrRd', 
                     reduce_C_function=np.mean, mincnt=1, vmin=0, vmax=1)
    cb3 = fig.colorbar(hb3, ax=ax3, label='Fraction with Retx')
    ax3.set_xlabel(xlabel, fontsize=12)
    ax3.set_ylabel(ylabel, fontsize=12)
    ax3.set_title('Retransmission Presence', fontsize=14)
    retx_count = (has_retx > 0).sum()
    no_retx_count = (has_retx == 0).sum()
    ax3.text(0.02, 0.98, f'No Retx: {no_retx_count:,}\nHas Retx: {retx_count:,}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Burst Concurrency vs Intensity', fontsize=14, y=1.02)
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
    
    print(f"Loaded {len(df)} burst records")
    
    # Extract relevant columns
    max_connections = df['maxConnections'].values
    avg_connections = df['avgConnections'].values
    ingress_max = df['ingressMax'].values
    ingress_vol = df['ingressVol'].values
    ecn_vol = df['ecnVol'].values
    retx_vol = df['ingressRetxVol'].values
    
    # Plot all three: count, ECN, and Retx
    plot_hexbin_all(
        x=max_connections,
        y=ingress_max,
        ecn_data=ecn_vol,
        retx_data=retx_vol,
        xlabel='Max Connections',
        ylabel='Ingress Max (normalized by link rate)',
        output_file=os.path.join(output_dir, 'concurrency_vs_intensity.png')
    )

if __name__ == '__main__':
    main()
