"""
Real vs I.I.D. Replay with Inter-arrival Times

This script extends the basic I.I.D. comparison by incorporating two distributions:
1. Flow sizes (ingressBytes values) - sampled from empirical CDF
2. Inter-arrival times between burst events - sampled from empirical CDF

## PRAGNAAA --- not sure if below procedure is the one we want to disprove... read papers and find out what procedure is used in them. 
eg. in papers, flow size is different from flow sizes per ms
interarrival means interarrival time between flows, not between bursts in a time series.

Procedure:
1. Load real ingressBytes from good.csv
2. Build empirical CDFs for:
   a) Ingress bytes (flow sizes per ms)
   b) Inter-arrival times (gaps between consecutive above-threshold samples)
3. Generate synthetic time series using both CDFs:
   - Sample burst durations and heights from flow size distribution
   - Sample gaps between bursts from inter-arrival distribution
4. Detect bursts in both real and synthetic using same threshold
5. Compare burst length and height distributions (CDF)

This approach aims to preserve temporal structure that pure I.I.D. sampling loses.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Constants
LINK_RATE_BYTES_PER_MS = 1.5625e6  # 12.5 Gbps
BURST_THRESHOLD = 0.5 * LINK_RATE_BYTES_PER_MS  # 781,250 bytes/ms

def load_data_from_good_csv(filepath):
    """Load ingressBytes from good.csv"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    ingress_bytes = np.array(data['ingressBytes'])
    return ingress_bytes

def detect_bursts(samples, threshold):
    """
    Detect bursts in a time series.
    A burst is a contiguous sequence of samples >= threshold.
    
    Returns list of dicts with Position, Length, ingressMax
    """
    bursts = []
    n = len(samples)
    i = 0
    
    while i < n:
        if samples[i] >= threshold:
            start = i
            max_val = samples[i]
            
            while i < n and samples[i] >= threshold:
                max_val = max(max_val, samples[i])
                i += 1
            
            length = i - start
            bursts.append({
                'Position': start,
                'Length': length,
                'ingressMax': max_val
            })
        else:
            i += 1
    
    return bursts

def extract_temporal_features(samples, threshold):
    """
    Extract temporal features from the time series:
    1. Burst durations (length of contiguous above-threshold sequences)
    2. Burst heights (values during bursts)
    3. Inter-burst gaps (length of contiguous below-threshold sequences)
    4. Non-burst values (values during gaps)
    
    Returns dict with arrays for each feature
    """
    n = len(samples)
    
    burst_durations = []
    burst_values = []  # All individual above-threshold values
    gap_durations = []
    gap_values = []    # All individual below-threshold values
    
    i = 0
    while i < n:
        if samples[i] >= threshold:
            # We're in a burst
            start = i
            while i < n and samples[i] >= threshold:
                burst_values.append(samples[i])
                i += 1
            burst_durations.append(i - start)
        else:
            # We're in a gap
            start = i
            while i < n and samples[i] < threshold:
                gap_values.append(samples[i])
                i += 1
            gap_durations.append(i - start)
    
    return {
        'burst_durations': np.array(burst_durations),
        'burst_values': np.array(burst_values),
        'gap_durations': np.array(gap_durations),
        'gap_values': np.array(gap_values)
    }

def generate_iid_samples(real_samples, n_samples=None):
    """
    Basic I.I.D. sampling: Sample independently from empirical distribution.
    """
    if n_samples is None:
        n_samples = len(real_samples)
    
    synthetic = np.random.choice(real_samples, size=n_samples, replace=True)
    return synthetic

def generate_temporal_iid_samples(features, n_samples):
    """
    Generate synthetic samples preserving temporal structure by sampling from:
    1. Burst duration distribution
    2. Burst value distribution (for above-threshold samples)
    3. Gap duration distribution  
    4. Gap value distribution (for below-threshold samples)
    
    This alternates between bursts and gaps, sampling durations and values independently.
    """
    synthetic = []
    current_pos = 0
    
    # Start with a gap or burst based on what the real data starts with
    in_burst = False
    
    while current_pos < n_samples:
        if in_burst:
            # Sample a burst duration
            if len(features['burst_durations']) > 0:
                duration = np.random.choice(features['burst_durations'])
            else:
                duration = 1
            
            # Sample values for this burst
            if len(features['burst_values']) > 0:
                values = np.random.choice(features['burst_values'], size=duration, replace=True)
            else:
                values = np.array([BURST_THRESHOLD * 1.1] * duration)
            
            synthetic.extend(values[:min(duration, n_samples - current_pos)])
            current_pos += duration
            in_burst = False
        else:
            # Sample a gap duration
            if len(features['gap_durations']) > 0:
                duration = np.random.choice(features['gap_durations'])
            else:
                duration = 1
            
            # Sample values for this gap
            if len(features['gap_values']) > 0:
                values = np.random.choice(features['gap_values'], size=duration, replace=True)
            else:
                values = np.array([BURST_THRESHOLD * 0.1] * duration)
            
            synthetic.extend(values[:min(duration, n_samples - current_pos)])
            current_pos += duration
            in_burst = True
    
    return np.array(synthetic[:n_samples])

def plot_cdf_comparison_3way(real_values, iid_values, temporal_values, 
                              xlabel, title, output_file, log_scale=False):
    """Plot CDF comparison between real, basic I.I.D., and temporal I.I.D."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [
        (real_values, 'Real', 'blue', '-', 2.5),
        (iid_values, 'I.I.D. (basic)', 'red', '--', 2),
        (temporal_values, 'I.I.D. + Inter-arrival', 'green', '-.', 2)
    ]
    
    for values, label, color, ls, lw in datasets:
        if len(values) == 0:
            continue
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=lw, label=label)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('CDF (P(X ≤ x))', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    
    # Add statistics
    stats_lines = []
    for values, label, _, _, _ in datasets:
        if len(values) > 0:
            stats_lines.append(f'{label}: n={len(values)}, mean={np.mean(values):.1f}')
    
    stats_text = '\n'.join(stats_lines)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_file}")

def plot_feature_distributions(features, output_file):
    """Plot the extracted temporal feature distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Burst durations
    ax = axes[0, 0]
    if len(features['burst_durations']) > 0:
        sorted_vals = np.sort(features['burst_durations'])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, 'b-', linewidth=2)
        ax.set_xlabel('Burst Duration (ms)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Burst Duration Distribution (n={len(features["burst_durations"])})')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Gap durations
    ax = axes[0, 1]
    if len(features['gap_durations']) > 0:
        sorted_vals = np.sort(features['gap_durations'])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, 'r-', linewidth=2)
        ax.set_xlabel('Gap Duration (ms)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Gap (Inter-arrival) Duration Distribution (n={len(features["gap_durations"])})')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Burst values
    ax = axes[1, 0]
    if len(features['burst_values']) > 0:
        sorted_vals = np.sort(features['burst_values'])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, 'b-', linewidth=2)
        ax.axvline(BURST_THRESHOLD, color='gray', linestyle='--', label=f'Threshold={BURST_THRESHOLD:.0f}')
        ax.set_xlabel('Ingress Bytes (bytes/ms)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Burst Value Distribution (n={len(features["burst_values"])})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Gap values
    ax = axes[1, 1]
    if len(features['gap_values']) > 0:
        sorted_vals = np.sort(features['gap_values'])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, 'r-', linewidth=2)
        ax.axvline(BURST_THRESHOLD, color='gray', linestyle='--', label=f'Threshold={BURST_THRESHOLD:.0f}')
        ax.set_xlabel('Ingress Bytes (bytes/ms)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Gap Value Distribution (n={len(features["gap_values"])})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Extracted Temporal Features from Real Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    good_csv_path = os.path.join(script_dir, '..', 'good.csv')
    output_dir = script_dir
    
    print("="*70)
    print("REAL vs I.I.D. REPLAY WITH INTER-ARRIVAL TIMES")
    print("="*70)
    
    # Load real data
    print(f"\nLoading data from: {good_csv_path}")
    ingress_bytes = load_data_from_good_csv(good_csv_path)
    print(f"Loaded {len(ingress_bytes)} per-ms samples")
    
    # Extract temporal features
    print(f"\nExtracting temporal features (threshold={BURST_THRESHOLD:.0f})...")
    features = extract_temporal_features(ingress_bytes, BURST_THRESHOLD)
    
    print(f"  Burst durations: {len(features['burst_durations'])} events")
    print(f"    Mean: {np.mean(features['burst_durations']):.2f} ms")
    print(f"    Max: {np.max(features['burst_durations'])} ms")
    print(f"  Gap durations: {len(features['gap_durations'])} events")
    print(f"    Mean: {np.mean(features['gap_durations']):.2f} ms")
    print(f"    Max: {np.max(features['gap_durations'])} ms")
    print(f"  Above-threshold samples: {len(features['burst_values'])}")
    print(f"  Below-threshold samples: {len(features['gap_values'])}")
    
    # Detect real bursts
    real_bursts = detect_bursts(ingress_bytes, BURST_THRESHOLD)
    print(f"\nReal bursts detected: {len(real_bursts)}")
    
    # Plot feature distributions
    print("\nPlotting temporal feature distributions...")
    plot_feature_distributions(features, os.path.join(output_dir, 'temporal_features.png'))
    
    # Run multiple replay simulations
    n_replays = 100
    print(f"\nRunning {n_replays} replay simulations for each method...")
    
    # Basic I.I.D. replay
    basic_iid_lengths = []
    basic_iid_heights = []
    basic_iid_counts = []
    
    # Temporal I.I.D. replay
    temporal_iid_lengths = []
    temporal_iid_heights = []
    temporal_iid_counts = []
    
    for i in range(n_replays):
        # Basic I.I.D.
        basic_samples = generate_iid_samples(ingress_bytes)
        basic_bursts = detect_bursts(basic_samples, BURST_THRESHOLD)
        basic_iid_counts.append(len(basic_bursts))
        basic_iid_lengths.extend([b['Length'] for b in basic_bursts])
        basic_iid_heights.extend([b['ingressMax'] for b in basic_bursts])
        
        # Temporal I.I.D.
        temporal_samples = generate_temporal_iid_samples(features, len(ingress_bytes))
        temporal_bursts = detect_bursts(temporal_samples, BURST_THRESHOLD)
        temporal_iid_counts.append(len(temporal_bursts))
        temporal_iid_lengths.extend([b['Length'] for b in temporal_bursts])
        temporal_iid_heights.extend([b['ingressMax'] for b in temporal_bursts])
    
    # Real burst properties
    real_lengths = [b['Length'] for b in real_bursts]
    real_heights = [b['ingressMax'] for b in real_bursts]
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nReal data:")
    print(f"  Bursts: {len(real_bursts)}")
    print(f"  Avg length: {np.mean(real_lengths):.2f} ms")
    print(f"  Max length: {max(real_lengths)} ms")
    print(f"  Median length: {np.median(real_lengths):.1f} ms")
    print(f"  Avg height: {np.mean(real_heights):.0f} bytes/ms")
    
    print(f"\nBasic I.I.D. Replay ({n_replays} runs):")
    print(f"  Avg bursts per run: {np.mean(basic_iid_counts):.1f} ± {np.std(basic_iid_counts):.1f}")
    print(f"  Avg length: {np.mean(basic_iid_lengths):.2f} ms")
    print(f"  Max length: {max(basic_iid_lengths)} ms")
    print(f"  Median length: {np.median(basic_iid_lengths):.1f} ms")
    print(f"  Avg height: {np.mean(basic_iid_heights):.0f} bytes/ms")
    
    print(f"\nTemporal I.I.D. Replay ({n_replays} runs):")
    print(f"  Avg bursts per run: {np.mean(temporal_iid_counts):.1f} ± {np.std(temporal_iid_counts):.1f}")
    print(f"  Avg length: {np.mean(temporal_iid_lengths):.2f} ms")
    print(f"  Max length: {max(temporal_iid_lengths)} ms")
    print(f"  Median length: {np.median(temporal_iid_lengths):.1f} ms")
    print(f"  Avg height: {np.mean(temporal_iid_heights):.0f} bytes/ms")
    
    # Plot 1: Burst Length CDF - 3-way comparison
    print("\nPlotting burst length comparison...")
    plot_cdf_comparison_3way(
        real_lengths,
        basic_iid_lengths,
        temporal_iid_lengths,
        'Burst Length (ms)',
        'Burst Length CDF: Real vs I.I.D. Methods',
        os.path.join(output_dir, 'burst_length_with_interarrival.png'),
        log_scale=True
    )
    
    # Plot 2: Burst Height CDF - 3-way comparison
    print("Plotting burst height comparison...")
    plot_cdf_comparison_3way(
        real_heights,
        basic_iid_heights,
        temporal_iid_heights,
        'Burst Height (bytes/ms)',
        'Burst Height CDF: Real vs I.I.D. Methods',
        os.path.join(output_dir, 'burst_height_with_interarrival.png'),
        log_scale=False
    )
    
    # Plot 3: Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = [
        (real_lengths, real_heights, 'Real', 'blue', '-', 2.5),
        (basic_iid_lengths, basic_iid_heights, 'I.I.D. (basic)', 'red', '--', 2),
        (temporal_iid_lengths, temporal_iid_heights, 'I.I.D. + Inter-arrival', 'green', '-.', 2)
    ]
    
    # Length CDF
    ax1 = axes[0]
    for lengths, _, label, color, ls, lw in datasets:
        if len(lengths) == 0:
            continue
        sorted_vals = np.sort(lengths)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax1.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=lw, label=label)
    
    ax1.set_xlabel('Burst Length (ms)', fontsize=12)
    ax1.set_ylabel('CDF', fontsize=12)
    ax1.set_title('Burst Length Distribution', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Height CDF
    ax2 = axes[1]
    for _, heights, label, color, ls, lw in datasets:
        if len(heights) == 0:
            continue
        sorted_vals = np.sort(heights)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax2.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=lw, label=label)
    
    ax2.set_xlabel('Burst Height (bytes/ms)', fontsize=12)
    ax2.set_ylabel('CDF', fontsize=12)
    ax2.set_title('Burst Height Distribution', fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Real vs I.I.D. Methods: Burst Characteristics\n(Including Inter-arrival Time Distribution)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    output_combined = os.path.join(output_dir, 'burst_real_vs_iid_with_interarrival_combined.png')
    plt.savefig(output_combined, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_combined}")
    
    # Plot 4: Time series sample comparison
    print("\nPlotting sample time series comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)
    
    # Show first 500 samples for clarity
    n_show = 500
    x = np.arange(n_show)
    
    # Real
    ax = axes[0]
    ax.fill_between(x, 0, ingress_bytes[:n_show], alpha=0.7, color='blue')
    ax.axhline(BURST_THRESHOLD, color='red', linestyle='--', label='Threshold')
    ax.set_ylabel('Ingress (bytes/ms)')
    ax.set_title('Real Time Series')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Basic I.I.D. (one sample)
    basic_sample = generate_iid_samples(ingress_bytes, n_show)
    ax = axes[1]
    ax.fill_between(x, 0, basic_sample, alpha=0.7, color='red')
    ax.axhline(BURST_THRESHOLD, color='red', linestyle='--', label='Threshold')
    ax.set_ylabel('Ingress (bytes/ms)')
    ax.set_title('Basic I.I.D. Replay (one sample)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Temporal I.I.D. (one sample)
    temporal_sample = generate_temporal_iid_samples(features, n_show)
    ax = axes[2]
    ax.fill_between(x, 0, temporal_sample, alpha=0.7, color='green')
    ax.axhline(BURST_THRESHOLD, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Ingress (bytes/ms)')
    ax.set_title('I.I.D. + Inter-arrival Replay (one sample)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Time Series: Real vs Synthetic Methods', fontsize=14, y=1.01)
    plt.tight_layout()
    output_ts = os.path.join(output_dir, 'time_series_comparison.png')
    plt.savefig(output_ts, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_ts}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    main()
