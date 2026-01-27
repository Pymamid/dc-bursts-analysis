"""
Real vs I.I.D. Replay Burst Comparison

Procedure:
1. Load real ingressBytes from good.csv (2000 per-ms samples)
2. Build empirical CDF from real data
3. Generate synthetic i.i.d. samples by sampling from the CDF
4. Detect bursts in both real and synthetic using same threshold
5. Compare burst length and height distributions (CCDF)

Burst detection: A burst is a contiguous sequence of samples above the threshold.
Threshold = 0.5 * link_rate = 0.5 * 1.5625e6 = 781,250 bytes/ms
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
LINK_RATE_BYTES_PER_MS = 1.5625e6  # 12.5 Gbps
BURST_THRESHOLD = 0.5 * LINK_RATE_BYTES_PER_MS  # 781,250 bytes/ms

def load_data_from_good_csv(filepath):
    """Load ingressBytes and real burst data from good.csv"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    ingress_bytes = np.array(data['ingressBytes'])
    
    # Extract real bursts
    real_bursts = []
    if 'burst_result' in data and 'ingress' in data['burst_result']:
        for timestamp, bursts in data['burst_result']['ingress'].items():
            if isinstance(bursts, list):
                for burst in bursts:
                    if 'Position' in burst and 'Length' in burst and 'ingressMax' in burst:
                        real_bursts.append({
                            'Position': burst['Position'],
                            'Length': burst['Length'],
                            'ingressMax': burst['ingressMax'],
                            'ecnVol': burst.get('ecnVol', 0)
                        })
    
    return ingress_bytes, real_bursts

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
            # Start of a burst
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

def generate_iid_samples(real_samples, n_samples=None):
    """
    Generate i.i.d. samples by sampling from the empirical distribution.
    Each synthetic sample is drawn independently from the real distribution.
    """
    if n_samples is None:
        n_samples = len(real_samples)
    
    # Sample with replacement from real data
    synthetic = np.random.choice(real_samples, size=n_samples, replace=True)
    return synthetic

def plot_cdf_comparison(real_values, synthetic_values, xlabel, title, output_file, log_scale=False):
    """Plot CDF comparison between real and synthetic."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute CDFs
    for values, label, color, ls in [(real_values, 'Real', 'blue', '-'), 
                                      (synthetic_values, 'I.I.D. Replay', 'red', '--')]:
        if len(values) == 0:
            continue
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=2, label=label)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('CDF (P(X ≤ x))', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    
    # Add statistics
    if len(real_values) > 0 and len(synthetic_values) > 0:
        stats_text = (f'Real: n={len(real_values)} bursts, mean={np.mean(real_values):.1f}\n'
                      f'I.I.D.: n={len(synthetic_values)} bursts, mean={np.mean(synthetic_values):.1f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    good_csv_path = os.path.join(script_dir, '..', 'good.csv')
    output_dir = script_dir
    
    print("="*60)
    print("REAL vs I.I.D. REPLAY BURST COMPARISON")
    print("="*60)
    
    # Load real data
    print(f"\nLoading data from: {good_csv_path}")
    ingress_bytes, real_bursts = load_data_from_good_csv(good_csv_path)
    print(f"Loaded {len(ingress_bytes)} per-ms samples")
    print(f"Real bursts from file: {len(real_bursts)}")
    
    # Also detect bursts ourselves to verify
    detected_real_bursts = detect_bursts(ingress_bytes, BURST_THRESHOLD)
    print(f"Detected real bursts (threshold={BURST_THRESHOLD:.0f}): {len(detected_real_bursts)}")
    
    # Use detected bursts for consistency
    real_bursts = detected_real_bursts
    
    # Run multiple i.i.d. replays and aggregate
    n_replays = 100
    print(f"\nRunning {n_replays} i.i.d. replay simulations...")
    
    all_synthetic_lengths = []
    all_synthetic_heights = []
    synthetic_burst_counts = []
    
    for i in range(n_replays):
        synthetic_samples = generate_iid_samples(ingress_bytes)
        synthetic_bursts = detect_bursts(synthetic_samples, BURST_THRESHOLD)
        
        synthetic_burst_counts.append(len(synthetic_bursts))
        all_synthetic_lengths.extend([b['Length'] for b in synthetic_bursts])
        all_synthetic_heights.extend([b['ingressMax'] for b in synthetic_bursts])
    
    # Real burst properties
    real_lengths = [b['Length'] for b in real_bursts]
    real_heights = [b['ingressMax'] for b in real_bursts]
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nReal data:")
    print(f"  Bursts: {len(real_bursts)}")
    print(f"  Avg length: {np.mean(real_lengths):.2f} ms")
    print(f"  Max length: {max(real_lengths)} ms")
    print(f"  Avg height: {np.mean(real_heights):.0f} bytes/ms")
    
    print(f"\nI.I.D. Replay ({n_replays} runs):")
    print(f"  Avg bursts per run: {np.mean(synthetic_burst_counts):.1f} ± {np.std(synthetic_burst_counts):.1f}")
    print(f"  Avg length: {np.mean(all_synthetic_lengths):.2f} ms")
    print(f"  Max length: {max(all_synthetic_lengths)} ms")
    print(f"  Avg height: {np.mean(all_synthetic_heights):.0f} bytes/ms")
    
    # Plot 1: Burst Length CDF
    print("\nPlotting burst length CDF...")
    plot_cdf_comparison(
        real_lengths, 
        all_synthetic_lengths,
        'Burst Length (ms)',
        'Burst Length CDF: Real vs I.I.D. Replay',
        os.path.join(output_dir, 'burst_length_real_vs_iid.png'),
        log_scale=True
    )
    
    # Plot 2: Burst Height CDF  
    print("Plotting burst height CDF...")
    plot_cdf_comparison(
        real_heights,
        all_synthetic_heights,
        'Burst Height (bytes/ms)',
        'Burst Height CDF: Real vs I.I.D. Replay',
        os.path.join(output_dir, 'burst_height_real_vs_iid.png'),
        log_scale=False
    )
    
    # Plot 3: Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Length CDF
    ax1 = axes[0]
    for values, label, color, ls in [(real_lengths, 'Real', 'blue', '-'), 
                                      (all_synthetic_lengths, 'I.I.D. Replay', 'red', '--')]:
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax1.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=2, label=label)
    
    ax1.set_xlabel('Burst Length (ms)', fontsize=12)
    ax1.set_ylabel('CDF', fontsize=12)
    ax1.set_title('Burst Length Distribution', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Add burst count annotation
    ax1.text(0.95, 0.05, f'Real: {len(real_lengths)} bursts\nI.I.D.: {len(all_synthetic_lengths)} bursts ({n_replays} runs)',
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Height CDF
    ax2 = axes[1]
    for values, label, color, ls in [(real_heights, 'Real', 'blue', '-'), 
                                      (all_synthetic_heights, 'I.I.D. Replay', 'red', '--')]:
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax2.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=2, label=label)
    
    ax2.set_xlabel('Burst Height (bytes/ms)', fontsize=12)
    ax2.set_ylabel('CDF', fontsize=12)
    ax2.set_title('Burst Height Distribution', fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Add burst count annotation
    ax2.text(0.95, 0.05, f'Real: {len(real_heights)} bursts\nI.I.D.: {len(all_synthetic_heights)} bursts ({n_replays} runs)',
             transform=ax2.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Real vs I.I.D. Replay: Burst Characteristics', fontsize=14, y=1.02)
    plt.tight_layout()
    output_combined = os.path.join(output_dir, 'burst_real_vs_iid_combined.png')
    plt.savefig(output_combined, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_combined}")

if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    main()
