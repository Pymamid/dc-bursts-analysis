#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

#cd /home/pragna/work/DC_bursts/Analysis-scripts/ns3-analysis/ns3-millisampler-type-plots && python3 burst_height_duration_plot.py /home/pragna/work/DC_bursts/Analysis-scripts/ns3-analysis/ns3-millisampler-type-output/k=8DCTCPburstaware.txt --output burst_height_vs_duration_ns3.png --title "DCTCP k=8 (Burst-Aware)"

# Link rate for normalization (in bytes per millisecond)
# 25 Gbps = 25e9 bits/s = 3.125e9 bytes/s = 3.125e6 bytes/ms
LINK_RATE_BYTES_PER_MS = 3.125e6  # 25 Gbps in bytes per millisecond

def load_burst_data(file_path):
    """Load burst data from NS3 analysis output file."""
    try:
        # Read the file, skipping comment lines
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        burst_length = float(parts[0])  # BurstLength(ms)
                        burst_start = float(parts[1])   # BurstStart(s)
                        ingress_max = float(parts[2])   # IngressMax(Bytes)
                        max_connections = int(parts[3]) # MaxConnections
                        data.append({
                            'BurstLength(ms)': burst_length,
                            'BurstStart(s)': burst_start,
                            'IngressMax(Bytes)': ingress_max,
                            'MaxConnections': max_connections
                        })
                    except ValueError:
                        continue
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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
    if len(x_clean) > 0:
        print(f"X range: [{x_clean.min():.2f}, {x_clean.max():.2f}] ms")
        print(f"Y range (normalized): [{y_clean.min():.4f}, {y_clean.max():.4f}]")
        
        # Hexbin plot
        hb = ax.hexbin(x_clean, y_clean, gridsize=50, cmap='viridis', 
                       mincnt=1, bins='log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        cb = plt.colorbar(hb, ax=ax, label='Log10(count)')
    else:
        ax.text(0.5, 0.5, 'No valid data points to plot', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create burst height vs duration plot from NS3 burst data')
    parser.add_argument('input_file', help='Path to the burst data text file')
    parser.add_argument('--output', '-o', help='Output plot file path', 
                        default='burst_height_vs_duration.png')
    parser.add_argument('--title', '-t', help='Plot title', 
                        default='Burst Height vs Duration')
    parser.add_argument('--link-rate', type=float, default=LINK_RATE_BYTES_PER_MS,
                        help=f'Link rate in bytes/ms for normalization (default: {LINK_RATE_BYTES_PER_MS})')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    print(f"Loading data from: {args.input_file}")
    
    # Load burst data
    df = load_burst_data(args.input_file)
    
    if df.empty:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(df)} burst records")
    
    # Extract relevant columns
    burst_length = df['BurstLength(ms)'].values  # in ms
    ingress_max = df['IngressMax(Bytes)'].values  # in bytes
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot: Burst Length vs IngressMax
    plot_2d_hexbin(
        x=burst_length,
        y=ingress_max,
        xlabel='Burst Length (ms)',
        ylabel='Ingress Max (normalized by link rate)',
        title=args.title,
        output_file=args.output,
        link_rate=args.link_rate
    )
    
    # Print some statistics
    print(f"\nBurst Statistics:")
    print(f"  Total bursts: {len(df)}")
    print(f"  Burst length range: {burst_length.min():.1f} - {burst_length.max():.1f} ms")
    print(f"  Ingress max range: {ingress_max.min():,} - {ingress_max.max():,} bytes")
    print(f"  Normalized ingress max range: {(ingress_max/args.link_rate).min():.4f} - {(ingress_max/args.link_rate).max():.4f}")
    print(f"  Max connections range: {df['MaxConnections'].min()} - {df['MaxConnections'].max()}")

if __name__ == '__main__':
    main()