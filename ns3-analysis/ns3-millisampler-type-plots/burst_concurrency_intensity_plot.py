#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


#cd /home/pragna/work/DC_bursts/Analysis-scripts/ns3-analysis/ns3-millisampler-type-plots && python3 burst_concurrency_intensity_plot.py /home/pragna/work/DC_bursts/Analysis-scripts/ns3-analysis/ns3-millisampler-type-output/k=8DCTCPburstaware.txt --output burst_concurrency_intensity_ns3.png --title "DCTCP k=8 (Burst-Aware)"

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

def plot_hexbin_concurrency_intensity(x, y, xlabel, ylabel, title, output_file, 
                                      link_rate=LINK_RATE_BYTES_PER_MS):
    """Create a hexbin plot for burst concurrency vs intensity."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize y by link rate
    y_normalized = y / link_rate
    
    # Filter out invalid values
    mask = (x > 0) & (y_normalized > 0) & np.isfinite(x) & np.isfinite(y_normalized)
    x_clean = x[mask]
    y_clean = y_normalized[mask]
    
    print(f"Total data points: {len(x_clean)}")
    if len(x_clean) > 0:
        print(f"X range: [{x_clean.min():.0f}, {x_clean.max():.0f}] connections")
        print(f"Y range (normalized): [{y_clean.min():.4f}, {y_clean.max():.4f}]")
        
        # Hexbin plot with count (density)
        hb = ax.hexbin(x_clean, y_clean, gridsize=40, cmap='YlOrRd', 
                       mincnt=1, bins='log')
        cb = fig.colorbar(hb, ax=ax, label='Count (log scale)')
        
        # Add statistics text box
        stats_text = f'Total Bursts: {len(x_clean):,}\n'
        stats_text += f'Max Connections: {x_clean.min():.0f} - {x_clean.max():.0f}\n'
        stats_text += f'Intensity Range: {y_clean.min():.3f} - {y_clean.max():.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
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

def plot_duration_analysis(df, title, output_file, link_rate=LINK_RATE_BYTES_PER_MS):
    """Create additional analysis plots showing burst characteristics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    burst_length = df['BurstLength(ms)'].values
    ingress_max = df['IngressMax(Bytes)'].values
    max_connections = df['MaxConnections'].values
    ingress_normalized = ingress_max / link_rate
    
    # Filter valid data
    valid_mask = (burst_length > 0) & (ingress_max > 0) & (max_connections > 0)
    burst_length_clean = burst_length[valid_mask]
    ingress_normalized_clean = ingress_normalized[valid_mask]
    max_connections_clean = max_connections[valid_mask]
    
    if len(burst_length_clean) == 0:
        fig.suptitle(f"{title} - No Valid Data", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    # Plot 1: Concurrency vs Intensity (same as main plot)
    hb1 = ax1.hexbin(max_connections_clean, ingress_normalized_clean, 
                     gridsize=30, cmap='YlOrRd', mincnt=1, bins='log')
    cb1 = fig.colorbar(hb1, ax=ax1, label='Count')
    ax1.set_xlabel('Max Connections', fontsize=11)
    ax1.set_ylabel('Intensity (norm. by link rate)', fontsize=11)
    ax1.set_title('Concurrency vs Intensity', fontsize=12)
    
    # Plot 2: Duration vs Intensity
    hb2 = ax2.hexbin(burst_length_clean, ingress_normalized_clean, 
                     gridsize=30, cmap='plasma', mincnt=1, bins='log')
    cb2 = fig.colorbar(hb2, ax=ax2, label='Count')
    ax2.set_xlabel('Burst Length (ms)', fontsize=11)
    ax2.set_ylabel('Intensity (norm. by link rate)', fontsize=11)
    ax2.set_title('Duration vs Intensity', fontsize=12)
    
    # Plot 3: Duration vs Concurrency
    hb3 = ax3.hexbin(burst_length_clean, max_connections_clean, 
                     gridsize=30, cmap='viridis', mincnt=1, bins='log')
    cb3 = fig.colorbar(hb3, ax=ax3, label='Count')
    ax3.set_xlabel('Burst Length (ms)', fontsize=11)
    ax3.set_ylabel('Max Connections', fontsize=11)
    ax3.set_title('Duration vs Concurrency', fontsize=12)
    
    # Plot 4: Burst characteristics over time
    burst_start = df['BurstStart(s)'].values[valid_mask]
    sc = ax4.scatter(burst_start, ingress_normalized_clean, 
                     c=max_connections_clean, cmap='coolwarm', 
                     alpha=0.6, s=20)
    cb4 = fig.colorbar(sc, ax=ax4, label='Max Connections')
    ax4.set_xlabel('Burst Start Time (s)', fontsize=11)
    ax4.set_ylabel('Intensity (norm. by link rate)', fontsize=11)
    ax4.set_title('Burst Evolution Over Time', fontsize=12)
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create burst concurrency vs intensity plot from NS3 burst data')
    parser.add_argument('input_file', help='Path to the burst data text file')
    parser.add_argument('--output', '-o', help='Output plot file path', 
                        default='burst_concurrency_vs_intensity.png')
    parser.add_argument('--analysis-output', help='Output file for detailed analysis plots',
                        default='burst_analysis_detailed.png')
    parser.add_argument('--title', '-t', help='Plot title', 
                        default='Burst Concurrency vs Intensity')
    parser.add_argument('--link-rate', type=float, default=LINK_RATE_BYTES_PER_MS,
                        help=f'Link rate in bytes/ms for normalization (default: {LINK_RATE_BYTES_PER_MS})')
    parser.add_argument('--detailed', action='store_true',
                        help='Also create detailed analysis plots')
    
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
    max_connections = df['MaxConnections'].values
    ingress_max = df['IngressMax(Bytes)'].values
    
    # Create output directories if they don't exist
    for output_file in [args.output, args.analysis_output]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Main plot: Concurrency vs Intensity
    plot_hexbin_concurrency_intensity(
        x=max_connections,
        y=ingress_max,
        xlabel='Max Connections',
        ylabel='Ingress Max (normalized by link rate)',
        title=args.title,
        output_file=args.output,
        link_rate=args.link_rate
    )
    
    # Detailed analysis plots if requested
    if args.detailed:
        plot_duration_analysis(
            df=df,
            title=f"{args.title} - Detailed Analysis",
            output_file=args.analysis_output,
            link_rate=args.link_rate
        )
    
    # Print some statistics
    print(f"\nBurst Statistics:")
    print(f"  Total bursts: {len(df)}")
    print(f"  Burst length range: {df['BurstLength(ms)'].min():.1f} - {df['BurstLength(ms)'].max():.1f} ms")
    print(f"  Ingress max range: {ingress_max.min():,} - {ingress_max.max():,} bytes")
    print(f"  Normalized intensity range: {(ingress_max/args.link_rate).min():.4f} - {(ingress_max/args.link_rate).max():.4f}")
    print(f"  Max connections range: {max_connections.min()} - {max_connections.max()}")
    print(f"  Average burst length: {df['BurstLength(ms)'].mean():.1f} ms")
    print(f"  Average max connections: {max_connections.mean():.1f}")

if __name__ == '__main__':
    main()