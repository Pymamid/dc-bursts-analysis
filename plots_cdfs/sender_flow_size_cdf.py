import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# Usage examples:
# python3 sender_flow_size_cdf.py --canonical /path/to/canonical-fattree --bg_incast /path/to/bg-incast-fattree --burst_aware /path/to/burst-aware-fattree --output flow_size_cdf.png --title "Flow Size CDF from Sender Logs"
# cd /home/pragna/work/DC_bursts/Analysis-scripts/plots_cdfs && python3 sender_flow_size_cdf.py --canonical /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/canonical-fattree --bg_incast /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/bg-incast-fattree --burst_aware /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/burst-aware-fattree --output sender_flow_size_DCTCP_k=8.png --title "Sender-side Flow Size CDF - k=8 DCTCP"
# Burst sender flow size (fixed)
BURST_FLOW_SIZE_BYTES = 500 * 1024  # 500KB

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze sender-side logs and plot flow size CDFs.')
    parser.add_argument('--canonical', type=str, required=True, help='Path to canonical-fattree trace directory')
    parser.add_argument('--bg_incast', type=str, required=True, help='Path to bg-incast-fattree trace directory')
    parser.add_argument('--burst_aware', type=str, required=True, help='Path to burst-aware-fattree trace directory')
    parser.add_argument('--output', type=str, default='sender_flow_size_cdf.png', help='Output filename for the plots')
    parser.add_argument('--title', type=str, default='Flow Size Distribution from Sender Logs', help='Title for the plots')
    return parser.parse_args()

def parse_background_sender_logs(trace_dir, pattern):
    """
    Parses background sender logs (canonical_bg_sender_x.log or dynamic_bg_sender_x.log).
    Format: Time FlowId FlowSize NodeId Type
    Returns list of flow sizes.
    """
    log_pattern = os.path.join(trace_dir, "logs", pattern)
    log_files = glob.glob(log_pattern)
    flow_sizes = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            # Format: Time FlowId FlowSize NodeId Type
                            flow_size = int(parts[2])
                            flow_sizes.append(flow_size)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"  Error reading {log_file}: {e}")
            
    return flow_sizes

def parse_burst_sender_logs(trace_dir):
    """
    Parses burst_coordinator.log to get the number of burst flows.
    
    The burst_coordinator.log contains the IncastScale column which indicates
    how many senders actually participate in each burst event.
    Each participating sender sends a fixed 500KB flow.
    
    Format: Time QueryId IncastScale QueryRate
    
    Returns list of flow sizes (all 500KB), num_bursts, avg_incast_scale.
    """
    burst_coordinator = os.path.join(trace_dir, "logs", "burst_coordinator.log")
    
    num_bursts = 0
    total_burst_flows = 0
    incast_scales = []
    
    # Parse burst_coordinator.log to get IncastScale per burst
    if os.path.exists(burst_coordinator):
        try:
            with open(burst_coordinator, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    # Format: Time QueryId IncastScale QueryRate
                    if len(parts) >= 3:
                        try:
                            incast_scale = int(parts[2])
                            incast_scales.append(incast_scale)
                            total_burst_flows += incast_scale
                            num_bursts += 1
                        except ValueError:
                            continue
        except Exception as e:
            print(f"  Error reading burst_coordinator.log: {e}")
    else:
        print(f"  Warning: burst_coordinator.log not found in {trace_dir}")
    
    flow_sizes = [BURST_FLOW_SIZE_BYTES] * total_burst_flows
    
    # Calculate average incast scale for reporting
    avg_incast_scale = sum(incast_scales) / len(incast_scales) if incast_scales else 0
    
    return flow_sizes, num_bursts, avg_incast_scale

def compute_cdf(data):
    if not data:
        return [], []
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y

def plot_cdfs(canonical_dir, bg_incast_dir, burst_aware_dir, output_file, title):
    # Store flow sizes by topology and type
    all_flow_sizes = {
        'canonical': {'background': [], 'burst': []},
        'bg-incast': {'background': [], 'burst': []},
        'burst-aware': {'background': [], 'burst': []}
    }

    print("Analyzing sender-side flow size data...")

    # 1. CANONICAL-FATTREE Analysis
    if os.path.exists(canonical_dir):
        print(f"\nProcessing canonical-fattree from {canonical_dir}...")
        
        # Only canonical_bg_sender_x.log files (no burst senders)
        bg_sizes = parse_background_sender_logs(canonical_dir, "canonical_bg_sender_*.log")
        all_flow_sizes['canonical']['background'] = bg_sizes
        
        print(f"  Background flows: {len(bg_sizes)}")
    else:
        print(f"  Canonical directory not found: {canonical_dir}")

    # 2. BG-INCAST-FATTREE Analysis
    if os.path.exists(bg_incast_dir):
        print(f"\nProcessing bg-incast-fattree from {bg_incast_dir}...")
        
        # Background: canonical_bg_sender_x.log
        bg_sizes = parse_background_sender_logs(bg_incast_dir, "canonical_bg_sender_*.log")
        all_flow_sizes['bg-incast']['background'] = bg_sizes
        
        # Burst: senderx_tx.log (500KB each)
        burst_sizes, num_bursts, avg_incast_scale = parse_burst_sender_logs(bg_incast_dir)
        all_flow_sizes['bg-incast']['burst'] = burst_sizes
        
        print(f"  Background flows: {len(bg_sizes)}")
        print(f"  Burst events: {num_bursts}, Avg IncastScale: {avg_incast_scale:.1f}, Total burst flows: {len(burst_sizes)}")
    else:
        print(f"  BG-incast directory not found: {bg_incast_dir}")

    # 3. BURST-AWARE-FATTREE Analysis
    if os.path.exists(burst_aware_dir):
        print(f"\nProcessing burst-aware-fattree from {burst_aware_dir}...")
        
        # Background: dynamic_bg_sender_x.log
        bg_sizes = parse_background_sender_logs(burst_aware_dir, "dynamic_bg_sender_*.log")
        all_flow_sizes['burst-aware']['background'] = bg_sizes
        
        # Burst: senderx_tx.log (500KB each)
        burst_sizes, num_bursts, avg_incast_scale = parse_burst_sender_logs(burst_aware_dir)
        all_flow_sizes['burst-aware']['burst'] = burst_sizes
        
        print(f"  Background flows: {len(bg_sizes)}")
        print(f"  Burst events: {num_bursts}, Avg IncastScale: {avg_incast_scale:.1f}, Total burst flows: {len(burst_sizes)}")
    else:
        print(f"  Burst-aware directory not found: {burst_aware_dir}")

    print("\nCreating CDF plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    colors = {'canonical': 'blue', 'bg-incast': 'orange', 'burst-aware': 'green'}
    
    # Plot 1: Background Flow Size Distribution (all topologies)
    ax = axes[0]
    for topo in ['canonical', 'bg-incast', 'burst-aware']:
        data = all_flow_sizes[topo]['background']
        if data:
            x, y = compute_cdf(data)
            ax.plot(x, y, label=f"{topo} (n={len(data)})", linewidth=2, color=colors[topo])
    ax.set_xlabel('Flow Size (Bytes)')
    ax.set_ylabel('CDF')
    ax.set_title('Background Flow Size Distribution')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Burst Flow Size Distribution (bg-incast and burst-aware only)
    ax = axes[1]
    for topo in ['bg-incast', 'burst-aware']:
        data = all_flow_sizes[topo]['burst']
        if data:
            x, y = compute_cdf(data)
            ax.plot(x, y, label=f"{topo} (n={len(data)})", linewidth=2, color=colors[topo])
    ax.set_xlabel('Flow Size (Bytes)')
    ax.set_ylabel('CDF')
    ax.set_title('Burst Flow Size Distribution (500KB fixed)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Combined Flow Size Distribution (background + burst)
    ax = axes[2]
    for topo in ['canonical', 'bg-incast', 'burst-aware']:
        bg_data = all_flow_sizes[topo]['background']
        burst_data = all_flow_sizes[topo]['burst']
        combined = bg_data + burst_data
        if combined:
            x, y = compute_cdf(combined)
            ax.plot(x, y, label=f"{topo} (n={len(combined)})", linewidth=2, color=colors[topo])
    ax.set_xlabel('Flow Size (Bytes)')
    ax.set_ylabel('CDF')
    ax.set_title('Combined Flow Size Distribution')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlots saved to {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for topo in ['canonical', 'bg-incast', 'burst-aware']:
        bg_data = all_flow_sizes[topo]['background']
        burst_data = all_flow_sizes[topo]['burst']
        combined = bg_data + burst_data
        
        print(f"\n{topo.upper()}:")
        if bg_data:
            print(f"  Background: count={len(bg_data)}, min={min(bg_data)}, max={max(bg_data)}, "
                  f"median={np.median(bg_data):.0f}, mean={np.mean(bg_data):.0f}")
        if burst_data:
            print(f"  Burst:      count={len(burst_data)}, size={BURST_FLOW_SIZE_BYTES} (500KB fixed)")
        if combined:
            print(f"  Combined:   count={len(combined)}, min={min(combined)}, max={max(combined)}, "
                  f"median={np.median(combined):.0f}, mean={np.mean(combined):.0f}")

if __name__ == "__main__":
    args = parse_arguments()
    plot_cdfs(args.canonical, args.bg_incast, args.burst_aware, args.output, args.title)
