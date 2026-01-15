
#!/usr/bin/env python3

# Example usage: 
# # python3 /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/cdf_analysis_fattree.py --canonical_dir /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/canonical-fattree --bg_incast_dir /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/bg-incast-fattree --burst_aware_dir /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/burst-aware-fattree --output_file /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs/fattree_cdfs_updated.png


"""
CDF Analysis Script for NS-3 Fattree Simulations

This script analyzes data from two NS-3 simulations:
1. canonical-fattree (flow-based continuous traffic)
2. bg-incast-fattree (burst + background traffic)

It generates three CDFs:
1. CDF of concurrent flows in a burst (bg-incast only)
2. CDF of flow size distribution (both)
3. CDF of burst frequency per second (bg-incast only)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate CDF plots from fattree simulation data')
    parser.add_argument('--canonical_dir', type=str, required=True,
                       help='Directory containing canonical-fattree simulation traces')
    parser.add_argument('--bg_incast_dir', type=str, required=True,
                       help='Directory containing bg-incast-fattree simulation traces')
    parser.add_argument('--burst_aware_dir', type=str, required=True,
                       help='Directory containing burst-aware-fattree simulation traces')
    parser.add_argument('--output_file', type=str,
                       default='fattree_cdfs.png',
                       help='Output filename for the CDF plot')
    return parser.parse_args()

def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# bg-incast and burst-aware fattree
def parse_flow_times_json(data):
    flows = []
    bursts = []
    for burst_id, burst_data in data.items():
        if burst_id.isdigit():
            burst_flows = []
            for ip, flow_info in burst_data.items():
                flow = {
                    'burst_id': int(burst_id),
                    'ip': ip,
                    'node_id': flow_info.get('id', 0),
                    'start_time': flow_info.get('start', 0),
                    'first_packet': flow_info.get('firstPacket', 0),
                    'end_time': flow_info.get('end', 0),
                    'flow_size': flow_info.get('flowSize', None)
                }
                flows.append(flow)
                burst_flows.append(flow)
            bursts.append({
                'burst_id': int(burst_id),
                'num_flows': len(burst_flows),
                'flows': burst_flows
            })
    return flows, bursts

# for canonical fattree
def parse_canonical_sender_logs(trace_dir):
    flows = []
    log_files = glob.glob(os.path.join(trace_dir, "logs", "canonical_bg_sender_*.log"))
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        flows.append({
                            'time': float(parts[0]),
                            'flow_id': int(parts[1]),
                            'flow_size': int(parts[2]),
                            'node_id': int(parts[3]),
                            'type': parts[4] if len(parts) > 4 else 'CANONICAL_BACKGROUND',
                            'sender': os.path.basename(log_file)
                        })
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    return flows

# for bg-incast fattree
def parse_canonical_background_sender_logs(trace_dir):
    flows = []
    log_files = glob.glob(os.path.join(trace_dir, "logs", "canonical_bg_sender_*.log"))
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        flows.append({
                            'time': float(parts[0]),
                            'flow_id': int(parts[1]),
                            'flow_size': int(parts[2]),
                            'node_id': int(parts[3]),
                            'type': parts[4] if len(parts) > 4 else 'CANONICAL_BACKGROUND',
                            'sender': os.path.basename(log_file)
                        })
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    return flows

def compute_cdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y

# dynamic_burst_sender.log exists for both burst-aware and bg-incast fattree.
def parse_dynamic_burst_sender_logs(trace_dir):
    flows = []
    log_files = glob.glob(os.path.join(trace_dir, "logs", "dynamic_burst_sender*.log"))
    if not log_files:
        log_files = [os.path.join(trace_dir, "logs", "dynamic_burst_sender.log")]
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        flows.append({
                            'time': float(parts[0]),
                            'flow_id': int(parts[1]),
                            'flow_size': int(parts[2]),
                            'node_id': int(parts[3]),
                            'type': parts[4],
                            'sender': os.path.basename(log_file)
                        })
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    return flows

# dynamic_bg_sender_*.log exists only for burst-aware fattree.
def parse_dynamic_background_sender_logs(trace_dir):
    flows = []
    log_files = glob.glob(os.path.join(trace_dir, "logs", "dynamic_bg_sender_*.log"))
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        flows.append({
                            'time': float(parts[0]),
                            'flow_id': int(parts[1]),
                            'flow_size': int(parts[2]),
                            'node_id': int(parts[3]),
                            'type': parts[4],
                            'sender': os.path.basename(log_file)
                        })
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    return flows

def plot_cdfs(canonical_dir, bg_incast_dir, burst_aware_dir, output_file):
    concurrent_flows = {'canonical': [], 'bg-incast': [], 'burst-aware': []}
    flow_sizes = {'canonical': [], 'bg-incast': [], 'burst-aware': []}
    burst_frequencies = {'canonical': [], 'bg-incast': [], 'burst-aware': []}

    print("Analyzing fattree simulation data...")

    # 1. CANONICAL-FATTREE Analysis
    if os.path.exists(canonical_dir):
        print(f"Processing canonical-fattree from {canonical_dir}...")
        flows = parse_canonical_sender_logs(canonical_dir)
        if flows:
            flow_sizes['canonical'] = [flow['flow_size'] for flow in flows]
            time_windows = defaultdict(list)
            window_size = 1.0
            for flow in flows:
                window = int(flow['time'] / window_size)
                time_windows[window].append(flow)
            concurrent_flows['canonical'] = [len(flows_in_window) for flows_in_window in time_windows.values()]
            if flows:
                total_time = max(flow['time'] for flow in flows) - min(flow['time'] for flow in flows)
                if total_time > 0:
                    flows_per_second = len(flows) / total_time
                    flows_per_minute = flows_per_second * 60
                    burst_frequencies['canonical'] = [flows_per_minute / 10] * 10
        else:
            print(f"  No canonical sender logs found in {canonical_dir}")
    else:
        print(f"  Canonical directory not found: {canonical_dir}")

    # 2. BG-INCAST-FATTREE Analysis
    if os.path.exists(bg_incast_dir):
        print(f"Processing bg-incast-fattree from {bg_incast_dir}...")
        flow_times_file = os.path.join(bg_incast_dir, "logs", "flow_times.json")
        flow_data = load_json_data(flow_times_file)
        if flow_data:
            flows, bursts = parse_flow_times_json(flow_data)
            concurrent_flows['bg-incast'] = [burst['num_flows'] for burst in bursts]
            canonical_bg_flows = parse_canonical_background_sender_logs(bg_incast_dir)
            all_flow_sizes = []
            config_file = os.path.join(bg_incast_dir, "config.json")
            config = load_json_data(config_file)
            if config and flows:
                bytes_per_burst_sender = config.get('bytesPerBurstSender', 500000)
                burst_flow_sizes = [bytes_per_burst_sender] * len(flows)
                all_flow_sizes.extend(burst_flow_sizes)
                print(f"  Found {len(flows)} burst flows from config")
            if canonical_bg_flows:
                bg_flow_sizes = [flow['flow_size'] for flow in canonical_bg_flows]
                all_flow_sizes.extend(bg_flow_sizes)
                print(f"  Found {len(canonical_bg_flows)} background flows in canonical_bg_sender logs")
            if all_flow_sizes:
                flow_sizes['bg-incast'] = all_flow_sizes
                print(f"  Total: {len(all_flow_sizes)} flows (burst + background)")
            if bursts and len(bursts) > 1:
                burst_times = []
                for burst in bursts:
                    if burst['flows']:
                        burst_times.append(burst['flows'][0]['start_time'])
                if len(burst_times) > 1:
                    intervals = np.diff(sorted(burst_times))
                    frequencies = [1.0 / interval for interval in intervals if interval > 0]
                    burst_frequencies['bg-incast'] = frequencies
        else:
            print(f"  No flow_times.json found in {bg_incast_dir}")
    else:
        print(f"  BG-incast directory not found: {bg_incast_dir}")

    # 3. BURST-AWARE-FATTREE Analysis
    if os.path.exists(burst_aware_dir):
        print(f"Processing burst-aware-fattree from {burst_aware_dir}...")
        flow_times_file = os.path.join(burst_aware_dir, "logs", "flow_times.json")
        flow_data = load_json_data(flow_times_file)
        if flow_data:
            flows, bursts = parse_flow_times_json(flow_data)
            concurrent_flows['burst-aware'] = [burst['num_flows'] for burst in bursts]

            dynamic_burst_flows = parse_dynamic_burst_sender_logs(burst_aware_dir)
            dynamic_background_flows = parse_dynamic_background_sender_logs(burst_aware_dir)

            all_flow_sizes = []
            if dynamic_burst_flows:
                all_flow_sizes.extend([flow['flow_size'] for flow in dynamic_burst_flows])
                print(f"  Found {len(dynamic_burst_flows)} burst flows in dynamic_burst_sender logs")
            if dynamic_background_flows:
                all_flow_sizes.extend([flow['flow_size'] for flow in dynamic_background_flows])
                print(f"  Found {len(dynamic_background_flows)} background flows in dynamic_bg_sender logs")
            if all_flow_sizes:
                flow_sizes['burst-aware'] = all_flow_sizes
                print(f"  Total: {len(all_flow_sizes)} flows (burst + background)")
            else:
                print(f"  No sender logs found in {burst_aware_dir}")

            if bursts and len(bursts) > 1:
                burst_times = []
                for burst in bursts:
                    if burst['flows']:
                        burst_times.append(burst['flows'][0]['start_time'])
                if len(burst_times) > 1:
                    intervals = np.diff(sorted(burst_times))
                    frequencies = [1.0 / interval for interval in intervals if interval > 0]
                    burst_frequencies['burst-aware'] = frequencies
        else:
            print(f"  No flow_times.json found in {burst_aware_dir}")
    else:
        print(f"  Burst-aware directory not found: {burst_aware_dir}")

    print("Creating CDF plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'canonical': 'blue', 'bg-incast': 'red', 'burst-aware': 'green'}

    # Plot 1: CDF of concurrent flows in a burst (bg-incast and burst-aware)
    ax1 = axes[0]
    for sim_type in ['bg-incast', 'burst-aware']:
        if concurrent_flows[sim_type]:
            x, y = compute_cdf(concurrent_flows[sim_type])
            ax1.plot(x, y, label=sim_type, color=colors[sim_type], linewidth=2)
    # Add canonical straight line at 0
    ax1.axvline(x=0, color=colors['canonical'], linestyle='--', linewidth=2, label='canonical (0)')
    ax1.set_xlabel('Number of Concurrent Flows')
    ax1.set_ylabel('CDF')
    ax1.set_title('CDF of Concurrent Flows in Burst')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: CDF of flow size distribution
    ax2 = axes[1]
    for sim_type in ['canonical', 'bg-incast', 'burst-aware']:
        if flow_sizes[sim_type]:
            sizes_kb = [size / 1024 for size in flow_sizes[sim_type]]
            x, y = compute_cdf(sizes_kb)
            ax2.plot(x, y, label=sim_type, color=colors[sim_type], linewidth=2)
    ax2.set_xlabel('Flow Size (KB)')
    ax2.set_ylabel('CDF')
    ax2.set_title('CDF of Flow Size Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: CDF of burst frequency per second (bg-incast and burst-aware)
    ax3 = axes[2]
    for sim_type in ['bg-incast', 'burst-aware']:
        if burst_frequencies[sim_type]:
            x, y = compute_cdf(burst_frequencies[sim_type])
            ax3.plot(x, y, label=sim_type, color=colors[sim_type], linewidth=2)
    # Add canonical straight line at 0
    ax3.axvline(x=0, color=colors['canonical'], linestyle='--', linewidth=2, label='canonical (0)')
    ax3.set_xlabel('Burst Frequency (per second)')
    ax3.set_ylabel('CDF')
    ax3.set_title('CDF of Burst Frequency per Second')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"CDF plots saved to {output_file}")

    print("\n=== SUMMARY STATISTICS ===")
    for sim_type in ['canonical', 'bg-incast', 'burst-aware']:
        print(f"\n{sim_type.upper()}:")
        if concurrent_flows[sim_type]:
            print(f"  Concurrent flows: mean={np.mean(concurrent_flows[sim_type]):.1f}, "
                  f"min={min(concurrent_flows[sim_type])}, max={max(concurrent_flows[sim_type])}")
        if flow_sizes[sim_type]:
            sizes_kb = [s/1024 for s in flow_sizes[sim_type]]
            print(f"  Flow sizes (KB): mean={np.mean(sizes_kb):.1f}, "
                  f"min={min(sizes_kb):.1f}, max={max(sizes_kb):.1f}")
        if burst_frequencies[sim_type]:
            print(f"  Burst frequency (per sec): mean={np.mean(burst_frequencies[sim_type]):.2f}, "
                  f"min={min(burst_frequencies[sim_type]):.2f}, max={max(burst_frequencies[sim_type]):.2f}")

if __name__ == "__main__":
    args = parse_arguments()
    plot_cdfs(args.canonical_dir, args.bg_incast_dir, args.burst_aware_dir, args.output_file)
