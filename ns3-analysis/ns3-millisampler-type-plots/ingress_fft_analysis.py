#!/usr/bin/env python3

'''
# Single file FFT analysis
python3 ingress_fft_analysis.py ../ns3-millisampler-type-output/k=4newBurstAwareIncast40Bgload150_ingress.txt --output sim_data

# Compare real vs simulated ingress data
python3 ingress_fft_analysis.py ../millisampler-goodcsv-analysis/ingress_timeseries.txt --compare ../ns3-millisampler-type-output/k=4newBurstAwareIncast40Bgload150_ingress.txt --output real_vs_sim_ingress

# Custom sampling frequency
python3 ingress_fft_analysis.py file.txt --fs 1000 --output analysis
'''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import argparse

def load_ingress_timeseries(input_file):
    """Load ingress time series data"""
    # Read the data, skipping comment lines
    data = pd.read_csv(input_file, sep=r'\s+', comment='#',
                       names=['Time', 'BytesReceived'])
    return data

def compute_fft_analysis(data, fs=1000, title=""):
    """Compute and plot FFT analysis of ingress bytes"""
    
    # Extract time series
    time = data['Time'].values
    bytes_received = data['BytesReceived'].values
    
    # Ensure uniform sampling (should be 1ms = 1000 Hz)
    dt = np.mean(np.diff(time))
    actual_fs = 1.0 / dt
    print(f"Detected sampling frequency: {actual_fs:.1f} Hz")
    print(f"Using sampling frequency: {fs} Hz")
    
    # Compute FFT
    N = len(bytes_received)
    fft_vals = fft(bytes_received)
    freqs = fftfreq(N, d=1.0/fs)
    
    # Get magnitude and phase
    magnitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    
    # Power spectral density
    psd = magnitude**2 / (fs * N)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original time series
    axes[0, 0].plot(time[:min(5000, len(time))], bytes_received[:min(5000, len(bytes_received))])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Bytes Received')
    axes[0, 0].set_title(f'Ingress Time Series - {title}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # FFT Magnitude (positive frequencies only)
    pos_freqs = freqs[:N//2]
    pos_magnitude = magnitude[:N//2]
    
    axes[0, 1].semilogy(pos_freqs[1:], pos_magnitude[1:])  # Skip DC
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('FFT Magnitude')
    axes[0, 1].set_title('FFT Magnitude Spectrum')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power Spectral Density
    axes[1, 0].semilogy(pos_freqs[1:], psd[:N//2][1:])
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_title('Power Spectral Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase spectrum (for lower frequencies)
    low_freq_idx = pos_freqs < 50  # Look at frequencies below 50 Hz
    axes[1, 1].plot(pos_freqs[low_freq_idx], phase[:N//2][low_freq_idx])
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].set_title('Phase Spectrum (< 50 Hz)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return freqs, fft_vals, magnitude, phase, psd

def spectral_statistics(freqs, magnitude, psd, bytes_data):
    """Compute and print spectral statistics"""
    
    # Get positive frequencies only
    N = len(freqs)
    pos_freqs = freqs[:N//2]
    pos_magnitude = magnitude[:N//2]
    pos_psd = psd[:N//2]
    
    # Statistical measures
    print("\n=== SPECTRAL ANALYSIS STATISTICS ===")
    print(f"Total samples: {len(bytes_data)}")
    print(f"Mean bytes/sample: {np.mean(bytes_data):.2f}")
    print(f"Std bytes/sample: {np.std(bytes_data):.2f}")
    print(f"Max bytes/sample: {np.max(bytes_data)}")
    
    # DC component
    dc_power = pos_psd[0]
    total_power = np.sum(pos_psd)
    print(f"\nDC component: {pos_magnitude[0]:.2f}")
    print(f"DC power fraction: {dc_power/total_power:.4f}")
    
    # Find dominant frequencies (top 5)
    # Skip DC component for peak finding
    peak_indices = np.argsort(pos_magnitude[1:])[-5:][::-1] + 1
    
    print(f"\nTop 5 frequency components:")
    for i, peak_idx in enumerate(peak_indices):
        freq = pos_freqs[peak_idx]
        mag = pos_magnitude[peak_idx]
        power_frac = pos_psd[peak_idx] / total_power
        print(f"  {i+1}. {freq:.2f} Hz - Magnitude: {mag:.1f}, Power fraction: {power_frac:.4f}")
    
    # Bandwidth measures
    # -3dB bandwidth (half power point)
    max_power_db = 20 * np.log10(np.max(pos_magnitude[1:]))
    half_power_db = max_power_db - 3
    magnitude_db = 20 * np.log10(pos_magnitude[1:])
    
    half_power_indices = np.where(magnitude_db >= half_power_db)[0]
    if len(half_power_indices) > 0:
        bandwidth_3db = pos_freqs[half_power_indices[-1] + 1] - pos_freqs[half_power_indices[0] + 1]
        print(f"\n-3dB Bandwidth: {bandwidth_3db:.2f} Hz")
    
    # Spectral centroid (center of mass of spectrum)
    spectral_centroid = np.sum(pos_freqs[1:] * pos_magnitude[1:]) / np.sum(pos_magnitude[1:])
    print(f"Spectral centroid: {spectral_centroid:.2f} Hz")

def compare_ingress_fft(file1, file2, output_prefix="fft_comparison"):
    """Compare FFT analysis between two ingress files"""
    
    # Load both datasets
    data1 = load_ingress_timeseries(file1)
    data2 = load_ingress_timeseries(file2)
    
    label1 = file1.split('/')[-1].replace('.txt', '')
    label2 = file2.split('/')[-1].replace('.txt', '')
    
    # Ensure same length for comparison
    min_len = min(len(data1), len(data2))
    data1 = data1.iloc[:min_len]
    data2 = data2.iloc[:min_len]
    
    # Compute FFTs
    fs = 1000  # 1000 Hz sampling
    N = len(data1)
    
    freqs = fftfreq(N, d=1.0/fs)
    pos_freqs = freqs[:N//2]
    
    fft1 = fft(data1['BytesReceived'].values)
    fft2 = fft(data2['BytesReceived'].values)
    
    mag1 = np.abs(fft1)[:N//2]
    mag2 = np.abs(fft2)[:N//2]
    
    psd1 = mag1**2 / (fs * N)
    psd2 = mag2**2 / (fs * N)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series comparison (first 5 seconds)
    end_idx = min(5000, len(data1))
    axes[0, 0].plot(data1['Time'][:end_idx], data1['BytesReceived'][:end_idx], 
                    label=label1, alpha=0.7)
    axes[0, 0].plot(data2['Time'][:end_idx], data2['BytesReceived'][:end_idx], 
                    label=label2, alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Bytes Received')
    axes[0, 0].set_title('Time Series Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # FFT Magnitude comparison
    axes[0, 1].semilogy(pos_freqs[1:], mag1[1:], label=label1, alpha=0.7)
    axes[0, 1].semilogy(pos_freqs[1:], mag2[1:], label=label2, alpha=0.7)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('FFT Magnitude')
    axes[0, 1].set_title('FFT Magnitude Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PSD comparison
    axes[1, 0].semilogy(pos_freqs[1:], psd1[1:], label=label1, alpha=0.7)
    axes[1, 0].semilogy(pos_freqs[1:], psd2[1:], label=label2, alpha=0.7)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_title('PSD Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Ratio of PSDs (highlighting frequency-dependent differences)
    ratio = psd1[1:] / (psd2[1:] + 1e-12)  # Add small value to avoid division by zero
    axes[1, 1].semilogx(pos_freqs[1:], ratio)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel(f'PSD Ratio ({label1}/{label2})')
    axes[1, 1].set_title('PSD Ratio Analysis')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison statistics
    print(f"\n=== COMPARISON: {label1} vs {label2} ===")
    print(f"{label1} - Mean: {np.mean(data1['BytesReceived']):.2f}, Std: {np.std(data1['BytesReceived']):.2f}")
    print(f"{label2} - Mean: {np.mean(data2['BytesReceived']):.2f}, Std: {np.std(data2['BytesReceived']):.2f}")
    
    # Frequency domain comparison
    total_power1 = np.sum(psd1)
    total_power2 = np.sum(psd2)
    print(f"{label1} - Total power: {total_power1:.2e}")
    print(f"{label2} - Total power: {total_power2:.2e}")
    print(f"Power ratio: {total_power1/total_power2:.3f}")

def main():
    parser = argparse.ArgumentParser(description='FFT Analysis of Ingress Bytes Time Series')
    parser.add_argument('input_file', help='Input ingress time series file')
    parser.add_argument('--compare', help='Second file for FFT comparison')
    parser.add_argument('--output', default='ingress_fft', help='Output prefix for plots')
    parser.add_argument('--fs', type=float, default=1000, help='Sampling frequency (Hz)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_ingress_fft(args.input_file, args.compare, args.output)
    else:
        # Single file analysis
        data = load_ingress_timeseries(args.input_file)
        title = args.input_file.split('/')[-1]
        
        freqs, fft_vals, magnitude, phase, psd = compute_fft_analysis(data, args.fs, title)
        spectral_statistics(freqs, magnitude, psd, data['BytesReceived'].values)
        
        plt.savefig(f'{args.output}.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()