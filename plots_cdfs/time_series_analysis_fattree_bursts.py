
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# python3 /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/time_series_analysis_fattree_bursts.py --canonical /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/canonical-fattree --bg_incast /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/bg-incast-fattree --burst_aware /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/burst-aware-fattree --output /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/fattree_time_series_receiver.png --title "Fattree Burst Time Series Analysis"
# Constants
LINK_SPEED_GBPS = 25.0
TIME_GRANULARITY_MS = 1.0
UTILIZATION_THRESHOLD_PCT = 0.50
FLOW_GAP_THRESHOLD_S = 0.002  # 2ms gap defines a new sub-flow
BURST_MERGE_THRESHOLD_S = 0 # 0ms gap to merge bursts (debouncing)

# Capacity per ms in bytes
BYTES_PER_MS_CAPACITY = (LINK_SPEED_GBPS * 1e9 * (TIME_GRANULARITY_MS / 1000.0)) / 8.0
BYTES_THRESHOLD = BYTES_PER_MS_CAPACITY * UTILIZATION_THRESHOLD_PCT

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze Fattree simulation logs and plot Time Series.')
    parser.add_argument('--canonical', type=str, required=True, help='Path to canonical-fattree trace directory')
    parser.add_argument('--bg_incast', type=str, required=True, help='Path to bg-incast-fattree trace directory')
    parser.add_argument('--burst_aware', type=str, required=True, help='Path to burst-aware-fattree trace directory')
    parser.add_argument('--output', type=str, default='time_series_plots.png', help='Output filename for the plots')
    parser.add_argument('--title', type=str, default='Fattree Burst Time Series Analysis', help='Title for the plots')
    return parser.parse_args()

def parse_node_map(trace_dir):
    """
    Parses node_ip_map.log to get mapping of IP -> SenderType.
    """
    node_map_path = os.path.join(trace_dir, "logs", "node_ip_map.log")
    if not os.path.exists(node_map_path):
        node_map_path = os.path.join(trace_dir, "node_ip_map.log")
        if not os.path.exists(node_map_path):
            print(f"  Warning: node_ip_map.log not found in {trace_dir}")
            return {}

    ip_map = {}
    try:
        with open(node_map_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    ip = parts[1]
                    node_type = parts[2]
                    ip_map[ip] = node_type
    except Exception as e:
        print(f"  Error reading node_ip_map.log: {e}")
    
    return ip_map

def parse_aggregator_logs(trace_dir, ip_map):
    """
    Parses aggregator_bytes_received.log to reconstruct bursts and flows over time.
    
    Returns:
        burst_subflows: List of (start_time, size) for burst flows
        bg_subflows: List of (start_time, size) for background flows
        bursts: List of burst dictionaries with time and concurrent flow counts
    """
    aggregator_log = os.path.join(trace_dir, "logs", "aggregator_bytes_received.log")
    if not os.path.exists(aggregator_log):
        print(f"  Warning: {aggregator_log} not found.")
        return [], [], []

    # Track flow sizes (sub-flows based on gaps)
    # List of (time, size)
    burst_subflows = []
    bg_subflows = []
    
    # State for sub-flow splitting: flow_id -> {'last_time': t, 'current_size': s, 'start_time': t}
    flow_state = defaultdict(lambda: {'last_time': -1.0, 'current_size': 0, 'start_time': -1.0})
    
    # Time bins: time_ms -> {'bytes': 0, 'flows': set()}
    time_bins = defaultdict(lambda: {'bytes': 0, 'flows': set()})

    try:
        with open(aggregator_log, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        time = float(parts[0])
                        sender_ip = parts[1]
                        sender_port = parts[2]
                        agg_ip = parts[3]
                        agg_port = parts[4]
                        size = int(parts[5])
                        
                        flow_id = (sender_ip, sender_port, agg_ip, agg_port)
                        sender_type = ip_map.get(sender_ip, 'Unknown')
                        
                        # Sub-flow logic
                        state = flow_state[flow_id]
                        if state['last_time'] >= 0 and (time - state['last_time'] > FLOW_GAP_THRESHOLD_S):
                            # Gap detected, finalize previous sub-flow
                            if state['current_size'] > 0:
                                if 'BurstSender' in sender_type:
                                    burst_subflows.append((state['start_time'], state['current_size']))
                                elif 'BackgroundSender' in sender_type:
                                    bg_subflows.append((state['start_time'], state['current_size']))
                            state['current_size'] = 0
                            state['start_time'] = time
                        elif state['start_time'] < 0:
                            state['start_time'] = time
                        
                        state['current_size'] += size
                        state['last_time'] = time
                        
                        # Binning
                        bin_idx = int(time * 1000) # 1ms bins
                        time_bins[bin_idx]['bytes'] += size
                        time_bins[bin_idx]['flows'].add(flow_id)
                            
                    except ValueError:
                        continue
    except Exception as e:
        print(f"  Error reading aggregator log: {e}")
        return [], [], []

    # Finalize remaining sub-flows
    for flow_id, state in flow_state.items():
        if state['current_size'] > 0:
            sender_ip = flow_id[0]
            sender_type = ip_map.get(sender_ip, 'Unknown')
            if 'BurstSender' in sender_type:
                burst_subflows.append((state['start_time'], state['current_size']))
            elif 'BackgroundSender' in sender_type:
                bg_subflows.append((state['start_time'], state['current_size']))

    # Identify and stitch bursts
    bursts = []
    sorted_bins = sorted(time_bins.keys())
    
    # Collect continuous time series data
    time_series = []
    
    if not sorted_bins:
        return burst_subflows, bg_subflows, [], []

    # Helper to get flow counts for a set of flows
    def get_flow_counts(flows):
        nb = 0
        n_bg = 0
        for flow_id in flows:
            sender_ip = flow_id[0]
            sender_type = ip_map.get(sender_ip, 'Unknown')
            if 'BurstSender' in sender_type:
                nb += 1
            elif 'BackgroundSender' in sender_type:
                n_bg += 1
        return nb, n_bg

    # Generate continuous time series
    # We iterate through sorted bins. Note: this skips empty bins (0 throughput).
    # For plotting, matplotlib handles gaps by connecting lines or we can scatter.
    for bin_idx in sorted_bins:
        t = bin_idx * (TIME_GRANULARITY_MS / 1000.0)
        b_data = time_bins[bin_idx]
        total_bytes = b_data['bytes']
        flows = b_data['flows']
        
        nb, n_bg = get_flow_counts(flows)
        
        throughput_gbps = (total_bytes * 8) / (TIME_GRANULARITY_MS / 1000.0) / 1e9
        
        time_series.append({
            'time': t,
            'throughput_gbps': throughput_gbps,
            'num_burst_flows': nb,
            'num_bg_flows': n_bg,
            'total_flows': len(flows)
        })

    current_burst_start_bin = None
    current_burst_end_bin = None
    current_burst_flows = set()
    current_burst_max_bytes = 0
    current_burst_max_connections = 0
    
    def close_burst(start_bin, end_bin, flows, max_bytes, max_connections):
        duration = (end_bin - start_bin + 1) * (TIME_GRANULARITY_MS / 1000.0)
        start_time = start_bin * (TIME_GRANULARITY_MS / 1000.0)
        
        nb, n_bg = get_flow_counts(flows)
        
        bursts.append({
            'start_time': start_time,
            'end_time': (end_bin + 1) * (TIME_GRANULARITY_MS / 1000.0),
            'duration': duration,
            'num_flows': len(flows),
            'num_burst_flows': nb,
            'num_bg_flows': n_bg,
            'max_bytes': max_bytes,
            'max_connections': max_connections
        })

    for bin_idx in sorted_bins:
        bin_data = time_bins[bin_idx]
        is_bursty = bin_data['bytes'] >= BYTES_THRESHOLD
        
        if is_bursty:
            if current_burst_start_bin is None:
                current_burst_start_bin = bin_idx
                current_burst_end_bin = bin_idx
                current_burst_flows = bin_data['flows'].copy()
                current_burst_max_bytes = bin_data['bytes']
                current_burst_max_connections = len(bin_data['flows'])
            else:
                # Check if this bin is consecutive to previous burst end
                if bin_idx == current_burst_end_bin + 1:
                    # Consecutive bursty bin: extend current burst
                    current_burst_end_bin = bin_idx
                    current_burst_flows.update(bin_data['flows'])
                    current_burst_max_bytes = max(current_burst_max_bytes, bin_data['bytes'])
                    current_burst_max_connections = max(current_burst_max_connections, len(bin_data['flows']))
                else:
                    # Gap detected: close previous burst and start new one
                    close_burst(current_burst_start_bin, current_burst_end_bin, current_burst_flows, current_burst_max_bytes, current_burst_max_connections)
                    current_burst_start_bin = bin_idx
                    current_burst_end_bin = bin_idx
                    current_burst_flows = bin_data['flows'].copy()
                    current_burst_max_bytes = bin_data['bytes']
                    current_burst_max_connections = len(bin_data['flows'])
        else:
            # Non-bursty bin: close current burst if open
            if current_burst_start_bin is not None:
                close_burst(current_burst_start_bin, current_burst_end_bin, current_burst_flows, current_burst_max_bytes, current_burst_max_connections)
                current_burst_start_bin = None

    if current_burst_start_bin is not None:
        close_burst(current_burst_start_bin, current_burst_end_bin, current_burst_flows, current_burst_max_bytes, current_burst_max_connections)

    return burst_subflows, bg_subflows, bursts, time_series

def plot_time_series(canonical_dir, bg_incast_dir, burst_aware_dir, output_file, title):
    data = {}
    
    print("Analyzing fattree simulation data...")
    
    for label, path in [('canonical', canonical_dir), ('bg-incast', bg_incast_dir), ('burst-aware', burst_aware_dir)]:
        if os.path.exists(path):
            print(f"Processing {label} from {path}...")
            ip_map = parse_node_map(path)
            burst_subflows, bg_subflows, bursts, time_series = parse_aggregator_logs(path, ip_map)
            
            data[label] = {
                'burst_subflows': burst_subflows,
                'bg_subflows': bg_subflows,
                'bursts': bursts,
                'time_series': time_series
            }
            print(f"  Found {len(bursts)} bursts, {len(time_series)} time points")
        else:
            print(f"  Directory not found: {path}")

    print("Creating Time Series plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    if title:
        fig.suptitle(title, fontsize=16)

    # 1. Throughput vs Time (Continuous)
    ax = axes[0, 0]
    for label, d in data.items():
        ts = d['time_series']
        if not ts:
            continue
        times = [t['time'] for t in ts]
        tput = [t['throughput_gbps'] for t in ts]
        ax.plot(times, tput, label=label, linewidth=1, alpha=0.8)
    
    # Add threshold line
    threshold_gbps = (BYTES_THRESHOLD * 8) / (TIME_GRANULARITY_MS / 1000.0) / 1e9
    ax.axhline(y=threshold_gbps, color='r', linestyle='--', alpha=0.5, label=f'Burst Threshold ({threshold_gbps:.1f} Gbps)')
    
    ax.set_ylabel('Throughput (Gbps)')
    ax.set_title('Link Throughput over Time (1ms granularity)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Concurrent Flows vs Time (Continuous)
    ax = axes[0, 1]
    for label, d in data.items():
        ts = d['time_series']
        if not ts:
            continue
        times = [t['time'] for t in ts]
        
        # Plot Background Flows
        bg_counts = [t['num_bg_flows'] for t in ts]
        if any(c > 0 for c in bg_counts):
            ax.plot(times, bg_counts, label=f'{label} (BG Flows)', linestyle=':', alpha=0.7)
            
        # Plot Burst Flows
        burst_counts = [t['num_burst_flows'] for t in ts]
        if any(c > 0 for c in burst_counts):
            ax.plot(times, burst_counts, label=f'{label} (Burst Flows)', linestyle='-', linewidth=1.5)

    ax.set_ylabel('Concurrent Flows')
    ax.set_title('Concurrent Active Flows over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Flow Size vs Time
    ax = axes[1, 0]
    for label, d in data.items():
        # Background Flows
        bg_flows = d['bg_subflows']
        if bg_flows:
            times = [f[0] for f in bg_flows]
            sizes = [f[1] for f in bg_flows]
            ax.scatter(times, sizes, label=f'{label} (BG Flows)', alpha=0.5, s=20)
            
        # Burst Flows
        burst_flows = d['burst_subflows']
        if burst_flows:
            times = [f[0] for f in burst_flows]
            sizes = [f[1] for f in burst_flows]
            ax.scatter(times, sizes, label=f'{label} (Burst Flows)', marker='x', s=40)

    ax.set_ylabel('Flow Size (Bytes)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Flow Sizes over Time')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4. Burst Frequency vs Time
    ax = axes[1, 1]
    for label, d in data.items():
        bursts = d['bursts']
        if len(bursts) < 2:
            continue
            
        times = [b['start_time'] for b in bursts]
        freqs = []
        plot_times = []
        
        times.sort()
        
        for i in range(1, len(times)):
            interval = times[i] - times[i-1]
            if interval > 0:
                freq = 1.0 / interval
                freqs.append(freq)
                plot_times.append(times[i])
        
        if freqs:
            ax.plot(plot_times, freqs, label=label, linewidth=1.5, marker='.', markersize=5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Burst Frequency (Hz)')
    ax.set_title('Burst Frequency over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plots saved to {output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    plot_time_series(args.canonical, args.bg_incast, args.burst_aware, args.output, args.title)
