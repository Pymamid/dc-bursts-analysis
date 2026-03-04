#!/usr/bin/env python3
"""
Script to plot timeseries of ingress bytes over time from millisampler data.
Input: JSON file with ingress bytes data (like good.csv)
Output: Time series plot with x-axis from 0-2 seconds
Bandwidth: 6.25 Gbps
"""

# python3 link_util_millisampler.py ../Millisampler-analysis/good.csv -o ingress_millisampler.png 0 0.5

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

def plot_ingress_timeseries(data, output_filename=None, start_time_seconds=0, end_time_seconds=2):
    """
    Plot ingress bytes timeseries.
    
    Args:
        data: Dictionary containing the measurement data
        output_filename: Output file name for the plot (optional)
        start_time_seconds: Start time in seconds (default: 0)
        end_time_seconds: End time in seconds (default: 2)
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
    
    # Calculate sample indices for the requested time window
    start_sample = int(start_time_seconds / dt)
    end_sample = int(end_time_seconds / dt)
    
    # Limit to available data
    start_sample = max(0, min(start_sample, len(ingress_bytes) - 1))
    end_sample = min(end_sample, len(ingress_bytes))
    
    # Extract data for the time window
    windowed_ingress_bytes = ingress_bytes[start_sample:end_sample]
    
    # Create time axis starting from start_time_seconds
    time_axis = np.arange(len(windowed_ingress_bytes)) * dt + start_time_seconds
    
    # Convert bytes to bits per second (throughput)
    # Since each sample represents bytes in a 1ms window,
    # we multiply by 1000 to get bytes/second, then by 8 for bits/second
    throughput_bps = np.array(windowed_ingress_bytes) * 1000 * 8  # bits per second
    
    # Convert to Gbps for easier reading
    throughput_gbps = throughput_bps / 1e9
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot throughput in Gbps (Link Utilization)
    plt.plot(time_axis, throughput_gbps, linewidth=0.8, alpha=0.8, color='orange')
    plt.axhline(y=6.25, color='red', linestyle='--', linewidth=2, 
                label='Bandwidth Limit (6.25 Gbps)')
    
    # Mark burst regions
    for i, burst in enumerate(bursts):
        burst_start_time = burst['position'] * dt
        burst_end_time = (burst['position'] + burst['length']) * dt
        
        # Only show bursts within the time window
        if burst_end_time > start_time_seconds and burst_start_time < end_time_seconds:
            # Clip burst times to the visible window
            display_start = max(burst_start_time, start_time_seconds)
            display_end = min(burst_end_time, end_time_seconds)
            plt.axvspan(display_start, display_end, alpha=0.05, color='red', 
                       label='Burst Region' if i == 0 else "")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Link Utilization (Gbps)')
    plt.title(f'Link Utilization over Time ({start_time_seconds:.3f}s - {end_time_seconds:.3f}s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(start_time_seconds, end_time_seconds)
    
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
                f'Window: {start_time_seconds:.3f}s - {end_time_seconds:.3f}s ({len(windowed_ingress_bytes)} samples)\n'
                f'Bursts Detected: {len(bursts)}',
                fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save or show the plot
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot ingress bytes timeseries from millisampler data')
    parser.add_argument('input_file', help='Input JSON file (e.g., good.csv)')
    parser.add_argument('start_time', type=float, help='Start time in seconds')
    parser.add_argument('end_time', type=float, help='End time in seconds')
    parser.add_argument('-o', '--output', help='Output plot filename (optional)')
    
    args = parser.parse_args()
    
    # Validate time arguments
    if args.start_time >= args.end_time:
        print("Error: Start time must be less than end time.")
        sys.exit(1)
    
    if args.start_time < 0:
        print("Error: Start time must be non-negative.")
        sys.exit(1)
    
    try:
        # Load the data
        print(f"Loading data from: {args.input_file}")
        data = load_data(args.input_file)
        
        # Plot the timeseries
        plot_ingress_timeseries(data, args.output, args.start_time, args.end_time)
        
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

