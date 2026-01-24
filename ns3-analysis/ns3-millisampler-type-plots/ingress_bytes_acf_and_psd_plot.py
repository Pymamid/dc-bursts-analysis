#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import os

#python3 ingress_bytes_acf_plot.py ../ns3-millisampler-type-output/k=8DCTCPburstawareingress.txt --output ingress_acf_ns3.png --title "Ingress Bytes ACF (DCTCP k=8 Burst-Aware)" --max-lag 500

def load_ingress_time_series(file_path):
    """Load ingress bytes time series from NS3 output file."""
    try:
        ingress_data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        timestamp = float(parts[0])  # Time(s)
                        bytes_received = int(parts[1])  # BytesReceived
                        ingress_data.append(bytes_received)
                    except (ValueError, IndexError):
                        continue
        
        if not ingress_data:
            return None
        
        return np.array(ingress_data)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def compute_psd(data, fs=1000):
    """Compute Power Spectral Density using Welch's method.
    
    Args:
        data: Time series data (ingress bytes)
        fs: Sampling frequency in Hz (1000 Hz = 1 sample per ms)
    
    Returns:
        frequencies, psd
    """
    # Use Welch's method for PSD estimation
    # nperseg controls the segment length for FFT
    nperseg = min(256, len(data) // 4)
    if nperseg < 16:
        return None, None
    
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    return frequencies, psd

def compute_acf(data, max_lag=500):
    """Compute autocorrelation function.
    
    Args:
        data: Time series data (ingress bytes)
        max_lag: Maximum lag to compute (in samples/ms)
    
    Returns:
        lags, acf values
    """
    n = len(data)
    if n < max_lag * 2:
        max_lag = n // 2
    
    if max_lag <= 0:
        return np.array([0]), np.array([1.0])
    
    # Remove mean and normalize by std
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    if data_std < 1e-10:  # Constant series
        acf = np.ones(max_lag + 1)
        acf[1:] = 0  # Only lag 0 has correlation
        return np.arange(max_lag + 1), acf
    
    data_normalized = (data - data_mean) / data_std
    
    # Compute autocorrelation using FFT for efficiency
    # Pad with zeros to avoid circular correlation
    padded_data = np.zeros(2 * n)
    padded_data[:n] = data_normalized
    
    # FFT-based correlation
    fft_data = np.fft.fft(padded_data)
    autocorr_full = np.fft.ifft(fft_data * np.conj(fft_data)).real
    
    # Normalize and extract positive lags
    autocorr_full = autocorr_full / n
    acf = autocorr_full[:max_lag + 1]
    lags = np.arange(max_lag + 1)
    
    return lags, acf

def plot_acf_analysis(time_series, title, output_file, max_lag=500):
    """Create comprehensive ACF analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    if time_series is None or len(time_series) == 0:
        fig.suptitle(f"{title} - No Data", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    # Compute ACF
    lags, acf = compute_acf(time_series, max_lag)
    
    # Plot 1: Time series (first 10 seconds)
    sample_length = min(10000, len(time_series))  # First 10 seconds
    time_axis = np.arange(sample_length) / 1000.0  # Convert ms to seconds
    ax1.plot(time_axis, time_series[:sample_length], 'b-', alpha=0.7)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Ingress Bytes', fontsize=11)
    ax1.set_title('Time Series (First 10s)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Full ACF
    ax2.plot(lags, acf, 'b-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Add confidence bounds (approximate 95% CI for white noise)
    n = len(time_series)
    ci_bound = 1.96 / np.sqrt(n)
    ax2.axhline(y=ci_bound, color='red', linestyle=':', linewidth=1, label='95% CI')
    ax2.axhline(y=-ci_bound, color='red', linestyle=':', linewidth=1)
    
    ax2.set_xlabel('Lag (ms)', fontsize=11)
    ax2.set_ylabel('Autocorrelation', fontsize=11)
    ax2.set_title(f'Autocorrelation Function', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ACF (zoomed to first 100 lags)
    zoom_lag = min(100, len(lags))
    ax3.plot(lags[:zoom_lag], acf[:zoom_lag], 'b-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.axhline(y=ci_bound, color='red', linestyle=':', linewidth=1)
    ax3.axhline(y=-ci_bound, color='red', linestyle=':', linewidth=1)
    
    # Mark significant lags
    significant_lags = np.where(np.abs(acf[:zoom_lag]) > ci_bound)[0]
    if len(significant_lags) > 1:  # Exclude lag 0
        significant_lags = significant_lags[significant_lags > 0]
        if len(significant_lags) > 0:
            ax3.scatter(significant_lags, acf[significant_lags], color='red', s=20, zorder=5)
    
    ax3.set_xlabel('Lag (ms)', fontsize=11)
    ax3.set_ylabel('Autocorrelation', fontsize=11)
    ax3.set_title('ACF (First 100 lags)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Power Spectral Density using Welch's method
    # Compute PSD using Welch's method (like millisampler version)
    freqs, psd = compute_psd(time_series, fs=1000)  # 1000 Hz = 1 sample/ms
    
    if freqs is not None and psd is not None:
        # Plot with log-log scale (skip DC component at freq=0)
        ax4.loglog(freqs[1:], psd[1:], 'g-', linewidth=2, alpha=0.8, label='PSD (Welch\'s method)')
        ax4.set_xlabel('Frequency (Hz)', fontsize=11)
        ax4.set_ylabel('Power Spectral Density (bytes²/Hz)', fontsize=11)
        ax4.set_title('Power Spectral Density', fontsize=12)
        ax4.grid(True, which='both', linestyle='--', alpha=0.3)
        ax4.legend()
        
        # Find peak frequencies
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        peak_freq = freqs[peak_idx]
        peak_power = psd[peak_idx]
        
        # Add peak frequency annotation
        if peak_freq > 0:
            ax4.annotate(f'Peak: {peak_freq:.2f} Hz', 
                        xy=(peak_freq, peak_power), xytext=(peak_freq*2, peak_power*2),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=9, color='red')
    else:
        ax4.text(0.5, 0.5, 'PSD computation failed\n(insufficient data)', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=11)
        ax4.set_xlabel('Frequency (Hz)', fontsize=11)
        ax4.set_ylabel('Power Spectral Density', fontsize=11)
        ax4.set_title('Power Spectral Density', fontsize=12)
    
    # Add overall statistics
    stats_text = f'Series length: {len(time_series):,} ms\n'
    stats_text += f'Mean: {np.mean(time_series):.1f}\n'
    stats_text += f'Std: {np.std(time_series):.1f}\n'
    
    # Find decorrelation times
    lag_half = np.argmax(acf < 0.5) if np.any(acf < 0.5) else len(acf)-1
    lag_tenth = np.argmax(acf < 0.1) if np.any(acf < 0.1) else len(acf)-1
    
    stats_text += f'ACF < 0.5 at: {lag_half} ms\n'
    stats_text += f'ACF < 0.1 at: {lag_tenth} ms'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved ACF analysis to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create autocorrelation function analysis from NS3 ingress data')
    parser.add_argument('input_file', help='Path to ingress time series file (Time(s) BytesReceived format)')
    parser.add_argument('--output', '-o', help='Output plot file path', 
                        default='ingress_bytes_acf.png')
    parser.add_argument('--title', '-t', help='Plot title', 
                        default='Ingress Bytes Autocorrelation Function')
    parser.add_argument('--max-lag', type=int, default=500,
                        help='Maximum lag to compute (in ms)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    print(f"Loading data from: {args.input_file}")
    
    # Load time series data
    print("Loading ingress time series...")
    time_series = load_ingress_time_series(args.input_file)
    
    if time_series is None:
        print("Failed to load data!")
        return
    
    print(f"Loaded time series with {len(time_series)} time points")
    print(f"Time series statistics:")
    print(f"  Length: {len(time_series)} ms ({len(time_series)/1000:.1f} seconds)")
    print(f"  Mean: {np.mean(time_series):.1f}")
    print(f"  Std: {np.std(time_series):.1f}")
    print(f"  Non-zero samples: {np.count_nonzero(time_series)} ({100*np.count_nonzero(time_series)/len(time_series):.1f}%)")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create ACF analysis
    plot_acf_analysis(
        time_series=time_series,
        title=args.title,
        output_file=args.output,
        max_lag=args.max_lag
    )
    
    # Print ACF summary
    if len(time_series) > 0:
        lags, acf = compute_acf(time_series, args.max_lag)
        
        print(f"\nAutocorrelation Analysis:")
        print(f"  ACF computed for {len(lags)} lags (0 to {args.max_lag} ms)")
        print(f"  ACF at lag 1: {acf[1]:.4f}")
        print(f"  ACF at lag 10: {acf[10] if len(acf) > 10 else 'N/A'}")
        
        # Find significant autocorrelations
        n = len(time_series)
        ci_bound = 1.96 / np.sqrt(n)
        significant_lags = np.where(np.abs(acf[1:]) > ci_bound)[0] + 1  # Exclude lag 0
        
        if len(significant_lags) > 0:
            print(f"  Significant autocorrelations at lags: {significant_lags[:10].tolist()}")
            if len(significant_lags) > 10:
                print(f"    ... and {len(significant_lags) - 10} more")
        else:
            print(f"  No significant autocorrelations found (95% confidence)")

if __name__ == '__main__':
    main()