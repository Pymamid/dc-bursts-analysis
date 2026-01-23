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
            # These rows have NaN in Length column
            df = df.dropna(subset=['Length', 'ingressHeightP95', 'ingressMax'])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    return pd.DataFrame()

def plot_2d_hexbin(x, y, xlabel, ylabel, title, output_file, link_rate=LINK_RATE_BYTES_PER_MS):
    """Create a 2D hexbin plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize y by link rate
    y_normalized = y / link_rate
    
    # Filter out invalid values
    mask = (x > 0) & (y_normalized > 0) & np.isfinite(x) & np.isfinite(y_normalized)
    x_clean = x[mask]
    y_clean = y_normalized[mask]
    
    print(f"Total data points: {len(x_clean)}")
    print(f"X range: [{x_clean.min():.2f}, {x_clean.max():.2f}] ms")
    print(f"Y range (normalized): [{y_clean.min():.4f}, {y_clean.max():.4f}]")
    
    # Hexbin plot
    hb = ax.hexbin(x_clean, y_clean, gridsize=50, cmap='viridis', 
                   mincnt=1, bins='log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    cb = plt.colorbar(hb, ax=ax, label='Log10(count)')
    
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
    burst_length = df['Length'].values  # in ms
    ingress_max = df['ingressMax'].values
    
    # Plot: Burst Length vs ingressMax
    plot_2d_hexbin(
        x=burst_length,
        y=ingress_max,
        xlabel='Burst Length (ms)',
        ylabel='Ingress Max (normalized by link rate)',
        title='Burst Height vs Duration',
        output_file=os.path.join(output_dir, 'burst_height_vs_duration.png')
    )

if __name__ == '__main__':
    main()
