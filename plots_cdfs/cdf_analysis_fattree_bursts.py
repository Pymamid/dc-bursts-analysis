import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# python3 cdf_analysis_fattree_bursts.py --canonical /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/canonical-fattree --bg_incast /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/bg-incast-fattree --burst_aware /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/burst-aware-fattree --output /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/attree_cdfs_DCTCP_k=8.png --title "Burst detection at Receiver similar to millisampler logic - k=8 and DCTCP and real world parameters"

# python3 /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/cdf_analysis_fattree_bursts.py --canonical /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/canonical-fattree --bg_incast /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/bg-incast-fattree --burst_aware /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/burst-aware-fattree --output /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/fattree_cdfs_receiver.png --title "Burst detection at Receiver similar to millisampler logic - k=4 and DCTCP and real world parameters"

# Constants for burst stitching
# BURST_GAP_THRESHOLD = 0.000001  # 1ns gap defines a new burst (OLD LOGIC)

# New Logic: Link Utilization
LINK_SPEED_GBPS = 25.0
TIME_GRANULARITY_MS = 1.0
UTILIZATION_THRESHOLD_PCT = 0.50
FLOW_GAP_THRESHOLD_S = 0.002  # 2ms gap defines a new sub-flow (for flow size CDF)
BURST_MERGE_THRESHOLD_S = 0 # 0ms gap to merge bursts (debouncing)

# Capacity per ms in bytes: (25 * 10^9 bits/s) * (0.001 s) / 8 bits/byte
BYTES_PER_MS_CAPACITY = (LINK_SPEED_GBPS * 1e9 * (TIME_GRANULARITY_MS / 1000.0)) / 8.0
BYTES_THRESHOLD = BYTES_PER_MS_CAPACITY * UTILIZATION_THRESHOLD_PCT

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze Fattree simulation logs and plot CDFs.')
    parser.add_argument('--canonical', type=str, required=True, help='Path to canonical-fattree trace directory')
    parser.add_argument('--bg_incast', type=str, required=True, help='Path to bg-incast-fattree trace directory')
    parser.add_argument('--burst_aware', type=str, required=True, help='Path to burst-aware-fattree trace directory')
    parser.add_argument('--output', type=str, default='cdf_plots.png', help='Output filename for the plots')
    parser.add_argument('--title', type=str, default='Fattree Burst Analysis', help='Title for the plots')
    return parser.parse_args()

def parse_node_map(trace_dir):
    """
    Parses node_ip_map.log to get mapping of IP -> SenderType.
    Returns dict: {ip_address: 'BurstSender' or 'BackgroundSender' or 'Coordinator/Receiver'}
    """
    # Try logs/node_ip_map.log first, then node_ip_map.log
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
                # Format: NodeId IPAddress Type
                if len(parts) >= 3:
                    ip = parts[1]
                    node_type = parts[2]
                    ip_map[ip] = node_type
    except Exception as e:
        print(f"  Error reading node_ip_map.log: {e}")
    
    return ip_map

def parse_aggregator_logs(trace_dir, ip_map):
    """
    Parses aggregator_bytes_received.log to reconstruct bursts at the receiver side.
    New Logic:
      - Bin packets into 1ms windows.
      - Calculate link utilization for each window.
      - If utilization > 50%, mark as bursty.
      - Stitch consecutive bursty windows into a single burst event.
    
    Returns:
        burst_flow_sizes: List of sizes for burst flows
        bg_flow_sizes: List of sizes for background flows
        bursts: List of burst dictionaries
        time_bins: Dictionary of time_ms -> {'bytes': count, 'flows': set()}
    """
    aggregator_log = os.path.join(trace_dir, "logs", "aggregator_bytes_received.log")
    if not os.path.exists(aggregator_log):
        print(f"  Warning: {aggregator_log} not found.")
        return [], [], []

    # Track flow sizes (sub-flows based on gaps)
    burst_subflows = []
    bg_subflows = []
    
    # State for sub-flow splitting: flow_id -> {'last_time': t, 'current_size': s}
    flow_state = defaultdict(lambda: {'last_time': -1.0, 'current_size': 0})
    
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
                        
                        # Sub-flow logic: Split flows if gap > threshold
                        state = flow_state[flow_id]
                        if state['last_time'] >= 0 and (time - state['last_time'] > FLOW_GAP_THRESHOLD_S):
                            # Gap detected, finalize previous sub-flow
                            if state['current_size'] > 0:
                                if 'BurstSender' in sender_type:
                                    burst_subflows.append(state['current_size'])
                                elif 'BackgroundSender' in sender_type:
                                    bg_subflows.append(state['current_size'])
                            state['current_size'] = 0
                        
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
                burst_subflows.append(state['current_size'])
            elif 'BackgroundSender' in sender_type:
                bg_subflows.append(state['current_size'])

    # Identify and stitch bursts
    bursts = []
    sorted_bins = sorted(time_bins.keys())
    
    if not sorted_bins:
        return list(burst_flows.values()), list(bg_flows.values()), []

    current_burst_start_bin = None
    current_burst_end_bin = None
    current_burst_flows = set()
    current_burst_max_bytes = 0
    current_burst_max_connections = 0
    
    # We need to iterate through time continuously to detect gaps (non-bursty windows)
    # But since we only stored bins with data, we can iterate sorted_bins and check gaps.
    # However, a bin with data might still be < threshold.
    
    # Let's iterate through the range of bins present
    # Optimization: Iterate sorted keys, but handle gaps logic
    
    # Helper to close a burst
    def close_burst(start_bin, end_bin, flows, max_bytes, max_connections):
        # Duration in seconds: (end_bin - start_bin + 1) * 1ms
        duration = (end_bin - start_bin + 1) * (TIME_GRANULARITY_MS / 1000.0)
        bursts.append({
            'start_time': start_bin * (TIME_GRANULARITY_MS / 1000.0),
            'end_time': (end_bin + 1) * (TIME_GRANULARITY_MS / 1000.0),
            'duration': duration,
            'num_flows': len(flows),
            'max_bytes': max_bytes,
            'max_connections': max_connections
        })

    for bin_idx in sorted_bins:
        bin_data = time_bins[bin_idx]
        is_bursty = bin_data['bytes'] >= BYTES_THRESHOLD
        
        if is_bursty:
            if current_burst_start_bin is None:
                # Start new burst
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
                    
                    # Start new burst
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

    # Close final burst if open
    if current_burst_start_bin is not None:
        close_burst(current_burst_start_bin, current_burst_end_bin, current_burst_flows, current_burst_max_bytes, current_burst_max_connections)

    return burst_subflows, bg_subflows, bursts, time_bins

def parse_sender_logs(trace_dir, pattern):
    """
    Parses sender logs to get flow sizes.
    Assumes logs contain flow completion info or we aggregate packet logs.
    If logs are packet-level: Time FlowID Size ...
    """
    log_pattern = os.path.join(trace_dir, "logs", pattern)
    log_files = glob.glob(log_pattern)
    flow_sizes = defaultdict(int)
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # Assuming: Time FlowID Size ...
                            flow_id = int(parts[1])
                            size = int(parts[2])
                            flow_sizes[flow_id] += size
                        except ValueError:
                            continue
        except Exception:
            pass
            
    return list(flow_sizes.values())

def compute_cdf(data):
    if not data:
        return [], []
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y

def plot_cdfs(canonical_dir, bg_incast_dir, burst_aware_dir, output_file, title):
    concurrent_flows = {'canonical': [], 'bg-incast': [], 'burst-aware': []}
    flow_sizes = {'canonical': [], 'bg-incast': [], 'burst-aware': []}
    burst_frequencies = {'canonical': [], 'bg-incast': [], 'burst-aware': []}

    print("Analyzing fattree simulation data...")

    # 1. CANONICAL-FATTREE Analysis
    if os.path.exists(canonical_dir):
        print(f"Processing canonical-fattree from {canonical_dir}...")
        
        # Get IP Map
        ip_map = parse_node_map(canonical_dir)
        
        # Parse Aggregator Logs for EVERYTHING (Bursts + Background)
        burst_flow_sizes, bg_flow_sizes, bursts, time_bins = parse_aggregator_logs(canonical_dir, ip_map)
        
        if bursts:
            concurrent_flows['canonical'] = [b['num_flows'] for b in bursts]
            
            burst_times = [b['start_time'] for b in bursts]
            if len(burst_times) > 1:
                intervals = np.diff(sorted(burst_times))
                frequencies = [1.0 / interval for interval in intervals if interval > 0]
                burst_frequencies['canonical'] = frequencies

        # Combine sizes
        total_sizes = burst_flow_sizes + bg_flow_sizes
        flow_sizes['canonical'] = total_sizes
        print(f"  Total flows analyzed: {len(total_sizes)} (BurstSender Flows: {len(burst_flow_sizes)}, BackgroundSender Flows: {len(bg_flow_sizes)})")
        print(f"  Detected High-Utilization Burst Events: {len(bursts)}")
    else:
        print(f"  Canonical directory not found: {canonical_dir}")

    # 2. BG-INCAST-FATTREE Analysis
    if os.path.exists(bg_incast_dir):
        print(f"Processing bg-incast-fattree from {bg_incast_dir}...")
        
        # Get IP Map
        ip_map = parse_node_map(bg_incast_dir)
        
        # Parse Aggregator Logs for EVERYTHING (Bursts + Background)
        burst_flow_sizes, bg_flow_sizes, bursts, time_bins = parse_aggregator_logs(bg_incast_dir, ip_map)
        
        if bursts:
            concurrent_flows['bg-incast'] = [b['num_flows'] for b in bursts]
            
            burst_times = [b['start_time'] for b in bursts]
            if len(burst_times) > 1:
                intervals = np.diff(sorted(burst_times))
                frequencies = [1.0 / interval for interval in intervals if interval > 0]
                burst_frequencies['bg-incast'] = frequencies

        # Combine sizes
        total_sizes = burst_flow_sizes + bg_flow_sizes
        flow_sizes['bg-incast'] = total_sizes
        print(f"  Total flows analyzed: {len(total_sizes)} (BurstSender Flows: {len(burst_flow_sizes)}, BackgroundSender Flows: {len(bg_flow_sizes)})")
        print(f"  Detected High-Utilization Burst Events: {len(bursts)}")

    else:
        print(f"  BG-incast directory not found: {bg_incast_dir}")

    # 3. BURST-AWARE-FATTREE Analysis
    if os.path.exists(burst_aware_dir):
        print(f"Processing burst-aware-fattree from {burst_aware_dir}...")
        
        # Get IP Map
        ip_map = parse_node_map(burst_aware_dir)
        
        # Parse Aggregator Logs for EVERYTHING (Bursts + Background)
        burst_flow_sizes, bg_flow_sizes, bursts, time_bins = parse_aggregator_logs(burst_aware_dir, ip_map)
        
        if bursts:
            concurrent_flows['burst-aware'] = [b['num_flows'] for b in bursts]
            
            burst_times = [b['start_time'] for b in bursts]
            if len(burst_times) > 1:
                intervals = np.diff(sorted(burst_times))
                frequencies = [1.0 / interval for interval in intervals if interval > 0]
                burst_frequencies['burst-aware'] = frequencies

        # Combine sizes
        total_sizes = burst_flow_sizes + bg_flow_sizes
        flow_sizes['burst-aware'] = total_sizes
        print(f"  Total flows analyzed: {len(total_sizes)} (BurstSender Flows: {len(burst_flow_sizes)}, BackgroundSender Flows: {len(bg_flow_sizes)})")
        print(f"  Detected High-Utilization Burst Events: {len(bursts)}")

    else:
        print(f"  Burst-aware directory not found: {burst_aware_dir}")

    print("Creating CDF plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=16)

    # Plot 1: Concurrent Flows per Burst
    ax = axes[0]
    for label, data in concurrent_flows.items():
        if data:
            x, y = compute_cdf(data)
            ax.plot(x, y, label=label, linewidth=2)
    ax.set_xlabel('Concurrent Flows per Burst')
    ax.set_ylabel('CDF')
    ax.set_title('Concurrent Flows Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Flow Size Distribution
    ax = axes[1]
    for label, data in flow_sizes.items():
        if data:
            x, y = compute_cdf(data)
            ax.plot(x, y, label=label, linewidth=2)
    ax.set_xlabel('Flow Size (Bytes)')
    ax.set_ylabel('CDF')
    ax.set_title('Flow Size Distribution')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Burst Frequency
    ax = axes[2]
    for label, data in burst_frequencies.items():
        if data:
            x, y = compute_cdf(data)
            ax.plot(x, y, label=label, linewidth=2)
    ax.set_xlabel('Burst Frequency (Hz)')
    ax.set_ylabel('CDF')
    ax.set_title('Burst Frequency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    plot_cdfs(args.canonical, args.bg_incast, args.burst_aware, args.output, args.title)
