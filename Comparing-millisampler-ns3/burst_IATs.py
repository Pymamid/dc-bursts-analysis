#!/usr/bin/env python3
"""
Script to generate CDF of burst Inter-Arrival Times (IATs) from burst times log file.

Input file format:
# StartTime(s) EndTime(s)
0.616232782000 0.616501831000
0.616674996000 0.616928640000
...

Usage:
python3 burst_IATs.py /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=4DCTCPnewBurstAware/bgload=5Incast=40KB/logs/burst_times.log -o burst_IATs.png
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

def calculate_iats(start_times):
    """
    Calculate Inter-Arrival Times between consecutive bursts.
    
    Args:
        start_times: List of burst start times in seconds
        
    Returns:
        List of IATs in seconds
    """
    if len(start_times) < 2:
        return []
    
    iats = []
    for i in range(1, len(start_times)):
        iat = start_times[i] - start_times[i-1]
        iats.append(iat)
    
    return iats

def plot_iat_cdf(iats, output_filename=None):
    """
    Plot CDF of Inter-Arrival Times.
    
    Args:
        iats: List of Inter-Arrival Times in seconds
        output_filename: Output filename (optional)
    """
    if not iats:
        print("No IATs to plot!")
        return
    
    # Convert to milliseconds for better readability
    iats_ms = np.array(iats) * 1000
    
    # Sort for CDF
    sorted_iats = np.sort(iats_ms)
    
    # Calculate CDF values
    y_values = np.arange(1, len(sorted_iats) + 1) / len(sorted_iats)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(sorted_iats, y_values, linewidth=2, marker='o', markersize=3, alpha=0.7)
    plt.xlabel('Inter-Arrival Time (ms)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Burst Inter-Arrival Times')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_iat = np.mean(iats_ms)
    median_iat = np.median(iats_ms)
    min_iat = np.min(iats_ms)
    max_iat = np.max(iats_ms)
    
    # Add statistics text box
    stats_text = f'Statistics:\n' \
                f'Count: {len(iats)}\n' \
                f'Mean: {mean_iat:.2f} ms\n' \
                f'Median: {median_iat:.2f} ms\n' \
                f'Min: {min_iat:.2f} ms\n' \
                f'Max: {max_iat:.2f} ms'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to: {output_filename}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate CDF of burst Inter-Arrival Times')
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
        
        # Calculate IATs
        iats = calculate_iats(start_times)
        print(f"Calculated {len(iats)} Inter-Arrival Times")
        
        # Plot CDF
        plot_iat_cdf(iats, args.output)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

