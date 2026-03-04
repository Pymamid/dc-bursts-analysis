#!/usr/bin/env python3
"""
Script to generate CDF of burst Completion Times (BCTs) from burst times log file.

Input file format:
# StartTime(s) EndTime(s)
0.616232782000 0.616501831000
0.616674996000 0.616928640000
...

Usage:
python3 burst_BCTs.py /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=4DCTCPnewBurstAware/bgload=5Incast=40KB/logs/burst_times.log -o burst_BCTs.png
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_burst_times(filename):
    """
    Parse burst times file and return list of (start_time, end_time) tuples.
    
    Args:
        filename: Path to burst times log file
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    burst_times = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse start time and end time
            parts = line.split()
            if len(parts) >= 2:
                try:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    burst_times.append((start_time, end_time))
                except ValueError:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return burst_times

def calculate_bcts(burst_times):
    """
    Calculate Burst Completion Times (durations) for each burst.
    
    Args:
        burst_times: List of (start_time, end_time) tuples in seconds
        
    Returns:
        List of BCTs in seconds
    """
    if not burst_times:
        return []
    
    bcts = []
    for start_time, end_time in burst_times:
        bct = end_time - start_time
        if bct > 0:  # Only positive durations
            bcts.append(bct)
        else:
            print(f"Warning: Invalid burst duration: {bct} for burst {start_time}-{end_time}")
    
    return bcts

def plot_bct_cdf(bcts, output_filename=None):
    """
    Plot CDF of Burst Completion Times.
    
    Args:
        bcts: List of Burst Completion Times in seconds
        output_filename: Output filename (optional)
    """
    if not bcts:
        print("No BCTs to plot!")
        return
    
    # Convert to milliseconds for better readability
    bcts_ms = np.array(bcts) * 1000
    
    # Sort for CDF
    sorted_bcts = np.sort(bcts_ms)
    
    # Calculate CDF values
    y_values = np.arange(1, len(sorted_bcts) + 1) / len(sorted_bcts)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(sorted_bcts, y_values, linewidth=2, marker='o', markersize=3, alpha=0.7)
    plt.xlabel('Burst Completion Time (ms)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Burst Completion Times')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_bct = np.mean(bcts_ms)
    median_bct = np.median(bcts_ms)
    min_bct = np.min(bcts_ms)
    max_bct = np.max(bcts_ms)
    p95_bct = np.percentile(bcts_ms, 95)
    p99_bct = np.percentile(bcts_ms, 99)
    
    # Add statistics text box
    stats_text = f'Statistics:\n' \
                f'Count: {len(bcts)}\n' \
                f'Mean: {mean_bct:.2f} ms\n' \
                f'Median: {median_bct:.2f} ms\n' \
                f'Min: {min_bct:.2f} ms\n' \
                f'Max: {max_bct:.2f} ms\n' \
                f'95th %ile: {p95_bct:.2f} ms\n' \
                f'99th %ile: {p99_bct:.2f} ms'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to: {output_filename}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate CDF of burst Completion Times')
    parser.add_argument('input_file', help='Input burst times log file')
    parser.add_argument('-o', '--output', help='Output plot filename (optional)')
    
    args = parser.parse_args()
    
    try:
        # Parse burst times
        print(f"Reading burst times from: {args.input_file}")
        burst_times = parse_burst_times(args.input_file)
        
        if len(burst_times) < 1:
            print("Error: Need at least 1 burst to calculate BCTs")
            sys.exit(1)
        
        print(f"Found {len(burst_times)} bursts")
        
        # Calculate BCTs
        bcts = calculate_bcts(burst_times)
        print(f"Calculated {len(bcts)} Burst Completion Times")
        
        if len(bcts) != len(burst_times):
            print(f"Warning: {len(burst_times) - len(bcts)} bursts had invalid durations")
        
        # Plot CDF
        plot_bct_cdf(bcts, args.output)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()