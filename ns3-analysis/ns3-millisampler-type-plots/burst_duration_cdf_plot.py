#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

## With detailed analysis
# python3 burst_duration_cdf_plot.py ../ns3-millisampler-type-output/k=8DCTCPburstaware.txt --output burst_duration_cdf_ns3.png --detailed --title "Burst Duration Analysis (DCTCP k=8)"

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

def compute_cdf(data):
    """Compute the Cumulative Distribution Function (CDF)."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf

def plot_duration_cdf(durations, title, output_file):
    """Plot burst duration CDF on both linear and log scales."""
    if len(durations) == 0:
        print("No duration data to plot!")
        return
    
    print(f"Burst duration range: [{durations.min():.2f}, {durations.max():.2f}] ms")
    print(f"Median duration: {np.median(durations):.2f} ms")
    print(f"Mean duration: {np.mean(durations):.2f} ms")
    
    # Compute CDF
    x, cdf = compute_cdf(durations)
    
    # Plot CDF - side by side: normal and log x-axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Normal x-axis
    axes[0].plot(x, cdf, 'b-', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Burst Duration (ms)', fontsize=12)
    axes[0].set_ylabel('CDF (P[X ≤ x])', fontsize=12)
    axes[0].set_title('Burst Duration CDF (Linear Scale)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Add percentile markers to left plot
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(durations, p)
        axes[0].axvline(val, color='red', linestyle=':', alpha=0.6)
        axes[0].text(val, 0.1, f'P{p}\n{val:.1f}ms', rotation=90, 
                     verticalalignment='bottom', horizontalalignment='right',
                     fontsize=8, bbox=dict(boxstyle='round,pad=0.1', 
                     facecolor='white', alpha=0.7))
    
    # Right plot: Log x-axis
    axes[1].plot(x, cdf, 'b-', linewidth=2, alpha=0.8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Burst Duration (ms)', fontsize=12)
    axes[1].set_ylabel('CDF (P[X ≤ x])', fontsize=12)
    axes[1].set_title('Burst Duration CDF (Log Scale)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5, which='both')
    
    # Add percentile markers to right plot
    for p in percentiles:
        val = np.percentile(durations, p)
        axes[1].axvline(val, color='red', linestyle=':', alpha=0.6)
    
    # Add statistics to the right plot
    stats_text = (f'Total bursts: {len(durations):,}\n'
                  f'Min: {durations.min():.1f} ms\n'
                  f'Median: {np.median(durations):.1f} ms\n'
                  f'Mean: {np.mean(durations):.1f} ms\n'
                  f'Max: {durations.max():.1f} ms\n'
                  f'Std: {np.std(durations):.1f} ms')
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add overall title
    fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

def plot_duration_analysis(df, title, output_file):
    """Create additional analysis plots for burst durations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    durations = df['BurstLength(ms)'].values
    durations = durations[durations > 0]  # Filter positive durations
    
    if len(durations) == 0:
        fig.suptitle(f"{title} - No Duration Data", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    # Plot 1: CDF (linear scale)
    x, cdf = compute_cdf(durations)
    ax1.plot(x, cdf, 'b-', linewidth=2)
    ax1.set_xlabel('Burst Duration (ms)', fontsize=11)
    ax1.set_ylabel('CDF', fontsize=11)
    ax1.set_title('Duration CDF (Linear)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CDF (log scale)
    ax2.plot(x, cdf, 'b-', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Burst Duration (ms)', fontsize=11)
    ax2.set_ylabel('CDF', fontsize=11)
    ax2.set_title('Duration CDF (Log)', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Histogram
    ax3.hist(durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Burst Duration (ms)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Duration Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Duration vs Time (if we have timing data)
    if not df.empty:
        burst_starts = df['BurstStart(s)'].values
        valid_mask = (durations > 0) & (len(burst_starts) == len(durations))
        if np.any(valid_mask) and len(burst_starts) == len(durations):
            ax4.scatter(burst_starts, durations, alpha=0.6, s=20, color='orange')
            ax4.set_xlabel('Burst Start Time (s)', fontsize=11)
            ax4.set_ylabel('Burst Duration (ms)', fontsize=11)
            ax4.set_title('Duration vs Time', fontsize=12)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Duration-Time\nmismatch', 
                     transform=ax4.transAxes, ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 'No timing data', 
                 transform=ax4.transAxes, ha='center', va='center')
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create burst duration CDF from NS3 burst data')
    parser.add_argument('input_file', help='Path to the burst data text file')
    parser.add_argument('--output', '-o', help='Output plot file path', 
                        default='burst_duration_cdf.png')
    parser.add_argument('--analysis-output', help='Output file for detailed analysis plots',
                        default='burst_duration_analysis.png')
    parser.add_argument('--title', '-t', help='Plot title', 
                        default='Burst Duration Distribution')
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
    
    # Get burst durations (BurstLength column, in ms)
    durations = df['BurstLength(ms)'].values
    durations = durations[durations > 0]  # Filter out zero durations
    
    if len(durations) == 0:
        print("No valid durations found!")
        return
    
    # Create output directories if they don't exist
    for output_file in [args.output, args.analysis_output]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Main duration CDF plot
    plot_duration_cdf(
        durations=durations,
        title=args.title,
        output_file=args.output
    )
    
    # Detailed analysis plots if requested
    if args.detailed:
        plot_duration_analysis(
            df=df,
            title=f"{args.title} - Detailed Analysis",
            output_file=args.analysis_output
        )
    
    # Print detailed statistics
    print(f"\nDetailed Duration Statistics:")
    print(f"  Total bursts: {len(durations)}")
    print(f"  Duration range: {durations.min():.2f} - {durations.max():.2f} ms")
    print(f"  Duration statistics:")
    print(f"    Mean: {np.mean(durations):.2f} ms")
    print(f"    Median: {np.median(durations):.2f} ms")
    print(f"    Std: {np.std(durations):.2f} ms")
    print(f"    CV: {np.std(durations)/np.mean(durations):.3f}")
    print(f"  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p}: {np.percentile(durations, p):.2f} ms")

if __name__ == '__main__':
    main()