#!/usr/bin/env python3
"""
Script to plot timeseries of ingress bytes over time from millisampler data.
Input: JSON file with ingress bytes data (like good.csv)
Output: Time series plot with x-axis from 0-2 seconds
Bandwidth: 6.25 Gbps
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

def load_data(filename):
    """Load JSON data from file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_ingress_timeseries(data, output_filename=None, max_time_seconds=2):
    """
    Plot ingress bytes timeseries.
    
    Args:
        data: Dictionary containing the measurement data
        output_filename: Output file name for the plot (optional)
        max_time_seconds: Maximum time to plot (default: 2 seconds)
    """
    
    # Extract ingress bytes (1 sample per millisecond)
    ingress_bytes = data['ingressBytes']
    
    # Extract burst data if available
    bursts = []
    if 'burst_result' in data and 'ingress' in data['burst_result']:
        burst_data = data['burst_result']['ingress']['1641906438033747']
        for burst in burst_data:
            if 'Position' in burst and 'Length' in burst:
                bursts.append({
                    'position': int(burst['Position']),
                    'length': int(burst['Length'])
                })
    
    # Time interval between samples (1ms)
    dt = 0.001  # seconds per sample
    
    # Calculate number of samples for the requested time window
    max_samples = int(max_time_seconds / dt)
    
    # Limit data to the requested time window
    if len(ingress_bytes) > max_samples:
        ingress_bytes = ingress_bytes[:max_samples]
    
    # Create time axis
    time_axis = np.arange(len(ingress_bytes)) * dt
    
    # Convert bytes to bits per second (throughput)
    # Since each sample represents bytes in a 1ms window,
    # we multiply by 1000 to get bytes/second, then by 8 for bits/second
    throughput_bps = np.array(ingress_bytes) * 1000 * 8  # bits per second
    
    # Convert to Gbps for easier reading
    throughput_gbps = throughput_bps / 1e9
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot raw ingress bytes
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, ingress_bytes, linewidth=0.8, alpha=0.8)
    
    # Mark burst regions on ingress bytes plot
    for i, burst in enumerate(bursts):
        start_time = burst['position'] * dt
        end_time = (burst['position'] + burst['length']) * dt
        
        # Only show bursts within the time window
        if start_time < max_time_seconds:
            end_time = min(end_time, max_time_seconds)
            plt.axvspan(start_time, end_time, alpha=0.05, color='red', 
                       label='Burst Region' if i == 0 else "")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Ingress Bytes per Sample')
    plt.title('Ingress Bytes over Time')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_time_seconds)
    if bursts:  # Add legend only if there are bursts
        plt.legend()
    
    # Plot throughput in Gbps
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, throughput_gbps, linewidth=0.8, alpha=0.8, color='orange')
    plt.axhline(y=6.25, color='red', linestyle='--', linewidth=2, 
                label='Bandwidth Limit (6.25 Gbps)')
    
    # Mark burst regions on throughput plot
    for i, burst in enumerate(bursts):
        start_time = burst['position'] * dt
        end_time = (burst['position'] + burst['length']) * dt
        
        # Only show bursts within the time window
        if start_time < max_time_seconds:
            end_time = min(end_time, max_time_seconds)
            plt.axvspan(start_time, end_time, alpha=0.05, color='red', 
                       label='Burst Region' if i == 0 else "")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (Gbps)')
    plt.title('Ingress Throughput over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, max_time_seconds)
    
    # Add overall statistics
    mean_throughput = np.mean(throughput_gbps)
    max_throughput = np.max(throughput_gbps)
    utilization = (mean_throughput / 6.25) * 100
    
    plt.figtext(0.02, 0.02, 
                f'Statistics:\n'
                f'Mean Throughput: {mean_throughput:.2f} Gbps\n'
                f'Peak Throughput: {max_throughput:.2f} Gbps\n'
                f'Average Utilization: {utilization:.1f}%\n'
                f'Sampling Rate: 1000 Hz (1ms per sample)\n'
                f'Duration: {len(ingress_bytes) * dt:.3f} sec\n'
                f'Bursts Detected: {len(bursts)}',
                fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save or show the plot
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot ingress bytes timeseries from millisampler data')
    parser.add_argument('input_file', help='Input JSON file (e.g., good.csv)')
    parser.add_argument('-o', '--output', help='Output plot filename (optional)')
    parser.add_argument('-t', '--time', type=float, default=2.0,
                       help='Maximum time to plot in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    try:
        # Load the data
        print(f"Loading data from: {args.input_file}")
        data = load_data(args.input_file)
        
        # Plot the timeseries
        plot_ingress_timeseries(data, args.output, args.time)
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required field in data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

