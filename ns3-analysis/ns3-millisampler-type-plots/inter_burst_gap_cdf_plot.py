#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Basic CDF plot
#python3 inter_burst_gap_cdf_plot.py ../ns3-millisampler-type-output/k=8DCTCPburstaware.txt --output inter_burst_gaps_ns3.png --title "Inter-burst Gap CDF (DCTCP k=8)"

# With logarithmic scale
#python3 inter_burst_gap_cdf_plot.py ../ns3-millisampler-type-output/k=8DCTCPburstaware.txt --output inter_burst_gaps_ns3_log.png --log-scale --title "Inter-burst Gap CDF - Log Scale (DCTCP k=8)"

# With detailed analysis
#python3 inter_burst_gap_cdf_plot.py ../ns3-millisampler-type-output/k=8DCTCPburstaware.txt --output inter_burst_gaps_ns3.png --detailed --title "Inter-burst Gap Analysis (DCTCP k=8)"

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
                            'MaxConnections': max_connections,
                            'BurstEnd(s)': burst_start + burst_length / 1000.0  # Convert ms to s
                        })
                    except ValueError:
                        continue
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def compute_inter_burst_gaps(df):
    """Compute inter-burst gaps from burst data."""
    if len(df) <= 1:
        return np.array([])
    
    # Sort bursts by start time
    df_sorted = df.sort_values('BurstStart(s)')
    
    # Compute gaps
    gaps = []
    burst_ends = df_sorted['BurstEnd(s)'].values[:-1]
    next_burst_starts = df_sorted['BurstStart(s)'].values[1:]
    
    # Inter-burst gap = start of next burst - end of current burst
    raw_gaps = next_burst_starts - burst_ends
    
    # Only keep positive gaps (no overlap)
    positive_gaps = raw_gaps[raw_gaps > 0]
    
    # Convert to milliseconds for consistency with burst length units
    gaps_ms = positive_gaps * 1000.0
    
    return gaps_ms

def compute_cdf(data):
    """Compute the Cumulative Distribution Function (CDF)."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf

def plot_cdf(gaps, title, output_file, log_scale=False):
    """Plot CDF on linear or log scale."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if len(gaps) == 0:
        ax.text(0.5, 0.5, 'No inter-burst gaps to plot', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_xlabel('Inter-burst Gap (ms)', fontsize=12)
        ax.set_ylabel('CDF (P[X ≤ x])', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    print(f"Total inter-burst gaps: {len(gaps)}")
    print(f"Gap range: [{gaps.min():.2f}, {gaps.max():.2f}] ms")
    print(f"Median gap: {np.median(gaps):.2f} ms")
    print(f"Mean gap: {np.mean(gaps):.2f} ms")
    
    # Compute CDF
    x, cdf = compute_cdf(gaps)
    
    # Plot CDF
    ax.plot(x, cdf, 'b-', linewidth=2, alpha=0.8, label='Inter-burst gaps')
    
    ax.set_xlabel('Inter-burst Gap (ms)', fontsize=12)
    ax.set_ylabel('CDF (P[X ≤ x])', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if log_scale:
        ax.set_xscale('log')
    
    # Add percentile lines
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(gaps, p)
        ax.axvline(val, color='red', linestyle='--', alpha=0.6)
        ax.text(val, 0.1, f'P{p}\n{val:.1f}ms', rotation=90, 
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))
    
    # Add statistics annotation
    stats_text = (f'Total gaps: {len(gaps):,}\n'
                  f'Min: {gaps.min():.1f} ms\n'
                  f'Median: {np.median(gaps):.1f} ms\n'
                  f'Mean: {np.mean(gaps):.1f} ms\n'
                  f'Max: {gaps.max():.1f} ms\n'
                  f'Std: {np.std(gaps):.1f} ms')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

def plot_gap_analysis(gaps, burst_data, title, output_file):
    """Create additional analysis plots for inter-burst gaps."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    if len(gaps) == 0:
        fig.suptitle(f"{title} - No Gap Data", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    # Plot 1: CDF (same as main plot)
    x, cdf = compute_cdf(gaps)
    ax1.plot(x, cdf, 'b-', linewidth=2)
    ax1.set_xlabel('Inter-burst Gap (ms)', fontsize=11)
    ax1.set_ylabel('CDF', fontsize=11)
    ax1.set_title('Gap CDF', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram (PDF approximation)
    ax2.hist(gaps, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Inter-burst Gap (ms)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Gap Distribution (Histogram)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gap vs Burst Length (if we have burst data)
    if not burst_data.empty and len(burst_data) > 1:
        # Get burst lengths for gaps (excluding the last burst)
        burst_lengths = burst_data.sort_values('BurstStart(s)')['BurstLength(ms)'].values[:-1]
        if len(burst_lengths) == len(gaps):
            ax3.scatter(burst_lengths, gaps, alpha=0.6, s=20)
            ax3.set_xlabel('Preceding Burst Length (ms)', fontsize=11)
            ax3.set_ylabel('Following Gap (ms)', fontsize=11)
            ax3.set_title('Gap vs Preceding Burst Length', fontsize=12)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Gap-Burst length\nmismatch', 
                     transform=ax3.transAxes, ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, 'Insufficient burst data', 
                 transform=ax3.transAxes, ha='center', va='center')
    
    # Plot 4: Gaps over time
    if not burst_data.empty and len(burst_data) > 1:
        df_sorted = burst_data.sort_values('BurstStart(s)')
        gap_times = df_sorted['BurstEnd(s)'].values[:-1]  # Times when gaps start
        ax4.scatter(gap_times, gaps, alpha=0.6, s=20, color='orange')
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Inter-burst Gap (ms)', fontsize=11)
        ax4.set_title('Gaps Over Time', fontsize=12)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient time data', 
                 transform=ax4.transAxes, ha='center', va='center')
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create inter-burst gap CDF from NS3 burst data')
    parser.add_argument('input_file', help='Path to the burst data text file')
    parser.add_argument('--output', '-o', help='Output plot file path', 
                        default='inter_burst_gap_cdf.png')
    parser.add_argument('--analysis-output', help='Output file for detailed analysis plots',
                        default='inter_burst_gap_analysis.png')
    parser.add_argument('--title', '-t', help='Plot title', 
                        default='Inter-burst Gap Distribution (CDF)')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use logarithmic scale for x-axis')
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
    
    # Compute inter-burst gaps
    print("\nComputing inter-burst gaps...")
    gaps = compute_inter_burst_gaps(df)
    
    if len(gaps) == 0:
        print("No inter-burst gaps found! (This can happen if bursts overlap or there's only one burst)")
        return
    
    # Create output directories if they don't exist
    for output_file in [args.output, args.analysis_output]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Plot main CDF
    plot_cdf(
        gaps=gaps,
        title=args.title,
        output_file=args.output,
        log_scale=args.log_scale
    )
    
    # Detailed analysis plots if requested
    if args.detailed:
        plot_gap_analysis(
            gaps=gaps,
            burst_data=df,
            title=f"{args.title} - Detailed Analysis",
            output_file=args.analysis_output
        )
    
    # Print detailed statistics
    print(f"\nDetailed Gap Statistics:")
    print(f"  Total bursts: {len(df)}")
    print(f"  Total gaps: {len(gaps)}")
    print(f"  Gap coverage: {len(gaps)/(len(df)-1)*100:.1f}% of possible gaps")
    if len(gaps) > 0:
        print(f"  Gap range: {gaps.min():.2f} - {gaps.max():.2f} ms")
        print(f"  Gap statistics:")
        print(f"    Mean: {np.mean(gaps):.2f} ms")
        print(f"    Median: {np.median(gaps):.2f} ms")
        print(f"    Std: {np.std(gaps):.2f} ms")
        print(f"  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"    P{p}: {np.percentile(gaps, p):.2f} ms")

if __name__ == '__main__':
    main()