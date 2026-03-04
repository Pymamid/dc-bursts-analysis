#!/usr/bin/env python3
"""
Script to plot timeseries of burst Inter-Arrival Times (IATs) from burst times log file.

Input file format:
# StartTime(s) EndTime(s)
0.616232782000 0.616501831000
0.616674996000 0.616928640000
...

Usage:
python3 burst_IAT_timeseries.py /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=4DCTCPnewBurstAware/bgload=5Incast=40KB/logs/burst_times.log -o burst_IAT_timeseries.png
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_burst_times(filename):
    """
    Parse burst times file and return list of start times.
    
    Args:
        filename: Path to burst times log file
        
    Returns:
        List of start times in seconds
    """
    start_times = []
    
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
                    start_times.append(start_time)
                except ValueError:
                    print(f"Warning: Could not parse line: {line}")
                    continue
    
    return sorted(start_times)

def calculate_iats_with_times(start_times):
    """
    Calculate Inter-Arrival Times between consecutive bursts with their corresponding times.
    
    Args:
        start_times: List of burst start times in seconds
        
    Returns:
        Tuple of (iat_times, iats) where:
        - iat_times: Time points for each IAT (using the second burst's start time)
        - iats: List of IATs in seconds
    """
    if len(start_times) < 2:
        return [], []
    
    iat_times = []
    iats = []
    for i in range(1, len(start_times)):
        iat = start_times[i] - start_times[i-1]
        iats.append(iat)
        iat_times.append(start_times[i])  # Use the arrival time of the second burst
    
    return iat_times, iats

def plot_iat_timeseries(iat_times, iats, output_filename=None):
    """
    Plot timeseries of Inter-Arrival Times.
    
    Args:
        iat_times: Time points for each IAT
        iats: List of Inter-Arrival Times in seconds
        output_filename: Output filename (optional)
    """
    if not iats:
        print("No IATs to plot!")
        return
    
    # Convert to milliseconds for better readability
    iats_ms = np.array(iats) * 1000
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: IAT vs Time
    ax1.plot(iat_times, iats_ms, linewidth=1, marker='o', markersize=2, alpha=0.7, color='blue')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Inter-Arrival Time (ms)')
    ax1.set_title('Burst Inter-Arrival Times over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IAT vs Burst Index
    burst_indices = range(1, len(iats) + 1)  # Start from 1 since first IAT is between burst 1 and 2
    ax2.plot(burst_indices, iats_ms, linewidth=1, marker='o', markersize=2, alpha=0.7, color='orange')
    ax2.set_xlabel('Burst Index')
    ax2.set_ylabel('Inter-Arrival Time (ms)')
    ax2.set_title('Burst Inter-Arrival Times vs Burst Sequence')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_iat = np.mean(iats_ms)
    median_iat = np.median(iats_ms)
    std_iat = np.std(iats_ms)
    min_iat = np.min(iats_ms)
    max_iat = np.max(iats_ms)
    
    # Add statistics text box
    stats_text = f'Statistics:\n' \
                f'Count: {len(iats)}\n' \
                f'Mean: {mean_iat:.2f} ms\n' \
                f'Median: {median_iat:.2f} ms\n' \
                f'Std Dev: {std_iat:.2f} ms\n' \
                f'Min: {min_iat:.2f} ms\n' \
                f'Max: {max_iat:.2f} ms\n' \
                f'Range: {max_iat - min_iat:.2f} ms'
    
    # Place stats on the first subplot
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Timeseries plot saved to: {output_filename}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot timeseries of burst Inter-Arrival Times')
    parser.add_argument('input_file', help='Input burst times log file')
    parser.add_argument('-o', '--output', help='Output plot filename (optional)')
    
    args = parser.parse_args()
    
    try:
        # Parse burst times
        print(f"Reading burst times from: {args.input_file}")
        start_times = parse_burst_times(args.input_file)
        
        if len(start_times) < 2:
            print("Error: Need at least 2 bursts to calculate IATs")
            sys.exit(1)
        
        print(f"Found {len(start_times)} bursts")
        
        # Calculate IATs with corresponding times
        iat_times, iats = calculate_iats_with_times(start_times)
        print(f"Calculated {len(iats)} Inter-Arrival Times")
        
        # Plot timeseries
        plot_iat_timeseries(iat_times, iats, args.output)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()