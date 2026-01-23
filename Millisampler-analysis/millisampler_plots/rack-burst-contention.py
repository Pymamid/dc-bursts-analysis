"""
Rack Burst Contention Analysis

For each rack, at each millisecond, count how many hosts are bursting simultaneously.
This gives us the TRUE measure of intra-rack burst contention.

Output:
- Distribution of concurrent bursting hosts per rack
- Time-weighted statistics (what fraction of time do we have N hosts bursting?)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from collections import defaultdict

def load_all_burst_data(folder_path):
    """Load ALL CSV files, grouped by rack."""
    all_files = glob.glob(os.path.join(folder_path, "burst_data_*.csv"))
    
    print(f"Found {len(all_files)} files, loading all...")
    
    # Group files by rack
    rack_to_hosts = defaultdict(dict)  # rack_id -> {host_id -> [(start, end), ...]}
    
    for i, file in enumerate(all_files):
        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i+1}/{len(all_files)} files...")
        
        filename = os.path.basename(file)
        parts = filename.replace('.csv', '').split('_')
        rack_id = None
        host_id = None
        for idx, p in enumerate(parts):
            if p == 'rackId' and idx+1 < len(parts):
                rack_id = parts[idx+1]
            if p == 'hostId' and idx+1 < len(parts):
                host_id = parts[idx+1]
        
        if not rack_id or not host_id:
            continue
        
        try:
            df = pd.read_csv(file, usecols=['Position', 'Length'])
            df = df.dropna()
            if len(df) == 0:
                continue
            
            # Store burst intervals as (start, end) tuples
            intervals = list(zip(df['Position'].astype(int).values, 
                                 (df['Position'] + df['Length']).astype(int).values))
            rack_to_hosts[rack_id][host_id] = intervals
        except Exception as e:
            pass
    
    print(f"Loaded data for {len(rack_to_hosts)} racks")
    return rack_to_hosts

def compute_contention_for_rack(host_intervals_dict):
    """
    For a single rack, compute the distribution of concurrent bursting hosts.
    
    Returns: dict mapping num_concurrent_hosts -> total_milliseconds
    """
    if not host_intervals_dict:
        return {}
    
    # Collect all events: (time, +1 for start, -1 for end)
    events = []
    for host_id, intervals in host_intervals_dict.items():
        for start, end in intervals:
            events.append((start, +1))
            events.append((end, -1))
    
    if not events:
        return {}
    
    # Sort by time, with ends (-1) before starts (+1) at same time
    events.sort(key=lambda x: (x[0], x[1]))
    
    # Sweep through events
    contention_duration = defaultdict(int)  # num_hosts -> total_ms
    
    current_count = 0
    prev_time = events[0][0]
    
    for time, delta in events:
        if time > prev_time:
            duration = time - prev_time
            contention_duration[current_count] += duration
        current_count += delta
        prev_time = time
    
    return dict(contention_duration)

def analyze_all_racks(rack_to_hosts):
    """Analyze contention for all racks."""
    
    # Global contention stats (aggregate across all racks)
    global_contention = defaultdict(int)  # num_hosts -> total_ms across all racks
    
    # Per-rack max concurrent
    rack_max_concurrent = {}
    
    print(f"\nAnalyzing contention for {len(rack_to_hosts)} racks...")
    
    for i, (rack_id, host_dict) in enumerate(rack_to_hosts.items()):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(rack_to_hosts)} racks...")
        
        contention = compute_contention_for_rack(host_dict)
        
        # Aggregate
        for num_hosts, duration in contention.items():
            if num_hosts > 0:  # Only count when at least 1 host is bursting
                global_contention[num_hosts] += duration
        
        # Track max concurrent for this rack
        if contention:
            max_concurrent = max(k for k in contention.keys() if contention[k] > 0)
            rack_max_concurrent[rack_id] = max_concurrent
    
    return global_contention, rack_max_concurrent

def plot_contention_distribution(global_contention, rack_max_concurrent, output_dir):
    """Plot the contention distribution."""
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    if not global_contention:
        print("No contention data!")
        return
    
    max_level = max(global_contention.keys())
    levels = list(range(1, max_level + 1))
    durations = [global_contention.get(l, 0) for l in levels]
    total_burst_time = sum(durations)
    
    # Convert to percentage
    percentages = [100 * d / total_burst_time if total_burst_time > 0 else 0 for d in durations]
    
    bars = ax1.bar(levels, percentages, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Number of Concurrent Bursting Hosts in Rack', fontsize=12)
    ax1.set_ylabel('% of Total Burst Time', fontsize=12)
    ax1.set_title('Intra-Rack Burst Contention Distribution', fontsize=13)
    ax1.set_xticks(levels[:20])  # Show up to 20 levels
    
    # Add value labels on bars
    for bar, pct in zip(bars[:10], percentages[:10]):
        if pct > 1:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Stats annotation
    weighted_avg = sum(l * d for l, d in global_contention.items()) / total_burst_time if total_burst_time > 0 else 0
    stats_text = (f'Total burst-ms: {total_burst_time:,}\n'
                  f'Max concurrent: {max_level}\n'
                  f'Weighted avg: {weighted_avg:.2f}')
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'rack_burst_contention.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {output_file}")
    
    # --- Plot 2: Distribution of max concurrent per rack ---
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    
    max_vals = list(rack_max_concurrent.values())
    max_bins = range(1, max(max_vals) + 2)
    
    ax3.hist(max_vals, bins=max_bins, align='left', color='coral', edgecolor='black', alpha=0.8)
    ax3.set_xlabel('Max Concurrent Bursting Hosts (per rack)', fontsize=12)
    ax3.set_ylabel('Number of Racks', fontsize=12)
    ax3.set_title('Distribution of Peak Intra-Rack Burst Contention', fontsize=13)
    
    avg_max = np.mean(max_vals)
    median_max = np.median(max_vals)
    ax3.axvline(avg_max, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_max:.1f}')
    ax3.axvline(median_max, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_max:.0f}')
    ax3.legend()
    
    output_file2 = os.path.join(output_dir, 'rack_max_contention_hist.png')
    plt.tight_layout()
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_file2}")
    
    return global_contention, max_level, weighted_avg

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder = os.path.join(script_dir, '..', 'individual_csvs')
    output_dir = script_dir
    
    print("="*60)
    print("RACK BURST CONTENTION ANALYSIS")
    print("="*60)
    print(f"\nData source: {csv_folder}")
    
    # Load all data
    rack_to_hosts = load_all_burst_data(csv_folder)
    
    # Count hosts per rack
    hosts_per_rack = [len(hosts) for hosts in rack_to_hosts.values()]
    print(f"\nRacks: {len(rack_to_hosts)}")
    print(f"Hosts per rack: min={min(hosts_per_rack)}, max={max(hosts_per_rack)}, "
          f"avg={np.mean(hosts_per_rack):.1f}, median={np.median(hosts_per_rack):.0f}")
    
    # Analyze contention
    global_contention, rack_max_concurrent = analyze_all_racks(rack_to_hosts)
    
    # Print summary
    print("\n" + "="*60)
    print("CONTENTION SUMMARY")
    print("="*60)
    
    total_burst_time = sum(global_contention.values())
    print(f"\nTotal burst-milliseconds (sum across all racks): {total_burst_time:,}")
    
    print("\nTime distribution by contention level:")
    for level in sorted(global_contention.keys())[:15]:
        duration = global_contention[level]
        pct = 100 * duration / total_burst_time if total_burst_time > 0 else 0
        print(f"  {level} concurrent hosts: {duration:>10,} ms ({pct:>5.2f}%)")
    
    if max(global_contention.keys()) > 15:
        remaining = sum(d for l, d in global_contention.items() if l > 15)
        print(f"  >15 concurrent hosts: {remaining:>10,} ms ({100*remaining/total_burst_time:.2f}%)")
    
    # Plot
    plot_contention_distribution(global_contention, rack_max_concurrent, output_dir)

if __name__ == '__main__':
    main()
