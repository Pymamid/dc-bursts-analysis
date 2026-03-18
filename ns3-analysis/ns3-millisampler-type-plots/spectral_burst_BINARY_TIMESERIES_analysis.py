#!/usr/bin/env python3

'''
# Compare real vs simulated data directly
python3 spectral_burst_analysis.py ../millisampler-goodcsv-analysis/millisampler-bursts.txt --compare ../ns3-millisampler-type-output/k=8DCTCPbgincast.txt --output real_vs_sim

# Individual analysis
python3 spectral_burst_analysis.py ../millisampler-goodcsv-analysis/millisampler-bursts.txt --analysis all --output real_data

# Focus on specific analysis
python3 spectral_burst_analysis.py file.txt --analysis interarrival --output interarrival_analysis
'''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import gaussian_kde
import argparse

def load_burst_data(input_file):
    """Load and process burst data"""
    burst_data = pd.read_csv(input_file, sep=r'\s+', comment='#',
                             names=['Length', 'BurstStart', 'IngressMax', 'MaxConnections'])
    
    # Convert to milliseconds and sort
    burst_data['BurstStart'] = burst_data['BurstStart'] * 1000  # convert to ms
    burst_data = burst_data.sort_values('BurstStart').reset_index(drop=True)
    burst_data['BurstEnd'] = burst_data['BurstStart'] + burst_data['Length']
    
    return burst_data

def create_binary_timeseries(burst_data, max_time=None):
    """Create binary time series (0=no burst, 1=burst)"""
    if max_time is None:
        max_time = int(burst_data['BurstEnd'].max()) + 1
    
    time_series = np.zeros(max_time, dtype=int)
    for _, row in burst_data.iterrows():
        start = int(row['BurstStart'])
        end = int(row['BurstEnd'])
        time_series[start:end] = 1
    
    return time_series

def power_spectral_density(time_series, fs=1000, title=""):
    """Compute and plot power spectral density"""
    freqs, psd = signal.periodogram(time_series, fs=fs)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs[1:], psd[1:])  # Skip DC component
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Power Spectral Density - {title}')
    plt.grid(True, alpha=0.3)
    return freqs, psd

def autocorrelation_analysis(time_series, max_lags=1000, title=""):
    """Compute and plot autocorrelation function"""
    # Compute autocorrelation
    autocorr = np.correlate(time_series - np.mean(time_series), 
                           time_series - np.mean(time_series), mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Plot
    lags = np.arange(min(len(autocorr), max_lags))
    plt.figure(figsize=(10, 6))
    plt.plot(lags, autocorr[:len(lags)])
    plt.xlabel('Lag (ms)')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Function - {title}')
    plt.grid(True, alpha=0.3)
    
    return lags, autocorr[:len(lags)]

def burst_interarrival_analysis(burst_data, title=""):
    """Analyze burst interarrival times"""
    starts = burst_data['BurstStart'].values
    interarrivals = np.diff(starts)
    
    # Remove very small interarrivals (overlapping bursts)
    interarrivals = interarrivals[interarrivals > 0]
    
    # Plot distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(interarrivals, bins=50, density=True, alpha=0.7)
    plt.xlabel('Interarrival Time (ms)')
    plt.ylabel('Density')
    plt.title(f'Interarrival Distribution - {title}')
    plt.yscale('log')
    
    # Plot log-log to check for power law
    plt.subplot(1, 3, 2)
    counts, bins = np.histogram(interarrivals, bins=50)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    valid_idx = counts > 0
    plt.loglog(bin_centers[valid_idx], counts[valid_idx], 'o-')
    plt.xlabel('Interarrival Time (ms)')
    plt.ylabel('Count')
    plt.title('Log-Log Distribution')
    
    # Spectral analysis of interarrival times
    if len(interarrivals) > 10:
        plt.subplot(1, 3, 3)
        # Interpolate to regular time series for FFT
        time_uniform = np.linspace(0, len(interarrivals)-1, len(interarrivals))
        freqs = fftfreq(len(interarrivals))
        fft_vals = np.abs(fft(interarrivals))
        
        # Plot positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        plt.semilogy(pos_freqs[1:], pos_fft[1:])
        plt.xlabel('Normalized Frequency')
        plt.ylabel('FFT Magnitude')
        plt.title('Interarrival Spectrum')
    
    plt.tight_layout()
    return interarrivals

def burst_duration_analysis(burst_data, title=""):
    """Analyze burst durations"""
    durations = burst_data['Length'].values
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(durations, bins=30, density=True, alpha=0.7)
    plt.xlabel('Burst Duration (ms)')
    plt.ylabel('Density')
    plt.title(f'Duration Distribution - {title}')
    plt.yscale('log')
    
    # Log-log plot
    plt.subplot(1, 3, 2)
    counts, bins = np.histogram(durations, bins=30)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    valid_idx = counts > 0
    plt.loglog(bin_centers[valid_idx], counts[valid_idx], 'o-')
    plt.xlabel('Burst Duration (ms)')
    plt.ylabel('Count')
    plt.title('Log-Log Distribution')
    
    # Spectral analysis of duration sequence
    if len(durations) > 10:
        plt.subplot(1, 3, 3)
        freqs = fftfreq(len(durations))
        fft_vals = np.abs(fft(durations))
        
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        plt.semilogy(pos_freqs[1:], pos_fft[1:])
        plt.xlabel('Normalized Frequency')
        plt.ylabel('FFT Magnitude')
        plt.title('Duration Sequence Spectrum')
    
    plt.tight_layout()
    return durations

def compare_spectral_features(real_file, sim_file, output_prefix="comparison"):
    """Compare spectral features between real and simulated data"""
    
    # Load data
    real_data = load_burst_data(real_file)
    sim_data = load_burst_data(sim_file)
    
    # Create time series (use same duration for fair comparison)
    max_time = max(int(real_data['BurstEnd'].max()), int(sim_data['BurstEnd'].max())) + 1
    real_ts = create_binary_timeseries(real_data, max_time)
    sim_ts = create_binary_timeseries(sim_data, max_time)
    
    # 1. Power Spectral Density Comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    freqs_real, psd_real = signal.periodogram(real_ts, fs=1000)
    freqs_sim, psd_sim = signal.periodogram(sim_ts, fs=1000)
    
    plt.semilogy(freqs_real[1:], psd_real[1:], label='Real Data', alpha=0.7)
    plt.semilogy(freqs_sim[1:], psd_sim[1:], label='Simulated Data', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('PSD Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Autocorrelation Comparison
    plt.subplot(1, 3, 2)
    lags_real, autocorr_real = np.arange(500), np.correlate(real_ts - np.mean(real_ts), 
                                                            real_ts - np.mean(real_ts), mode='full')
    autocorr_real = autocorr_real[autocorr_real.size // 2:autocorr_real.size // 2 + 500]
    autocorr_real = autocorr_real / autocorr_real[0]
    
    lags_sim, autocorr_sim = np.arange(500), np.correlate(sim_ts - np.mean(sim_ts), 
                                                          sim_ts - np.mean(sim_ts), mode='full')
    autocorr_sim = autocorr_sim[autocorr_sim.size // 2:autocorr_sim.size // 2 + 500]
    autocorr_sim = autocorr_sim / autocorr_sim[0]
    
    plt.plot(lags_real, autocorr_real, label='Real Data', alpha=0.7)
    plt.plot(lags_sim, autocorr_sim, label='Simulated Data', alpha=0.7)
    plt.xlabel('Lag (ms)')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Burst Rate Comparison
    plt.subplot(1, 3, 3)
    real_burst_rate = np.mean(real_ts)
    sim_burst_rate = np.mean(sim_ts)
    
    # Moving average burst rate
    window = 1000  # 1 second windows
    real_rate_ma = np.convolve(real_ts, np.ones(window)/window, mode='valid')
    sim_rate_ma = np.convolve(sim_ts, np.ones(window)/window, mode='valid')
    
    plt.plot(real_rate_ma[:5000], label=f'Real (avg={real_burst_rate:.3f})', alpha=0.7)
    plt.plot(sim_rate_ma[:5000], label=f'Sim (avg={sim_burst_rate:.3f})', alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Burst Rate (1s window)')
    plt.title('Temporal Burst Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_spectral_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Real Data:")
    print(f"  Total bursts: {len(real_data)}")
    print(f"  Burst rate: {real_burst_rate:.4f}")
    print(f"  Avg duration: {real_data['Length'].mean():.2f}ms")
    print(f"  Avg interarrival: {np.diff(real_data['BurstStart']).mean():.2f}ms")
    
    print(f"\nSimulated Data:")
    print(f"  Total bursts: {len(sim_data)}")
    print(f"  Burst rate: {sim_burst_rate:.4f}")
    print(f"  Avg duration: {sim_data['Length'].mean():.2f}ms")
    print(f"  Avg interarrival: {np.diff(sim_data['BurstStart']).mean():.2f}ms")

def main():
    parser = argparse.ArgumentParser(description='Spectral Analysis of Burst Data')
    parser.add_argument('input_file', help='Input burst data file')
    parser.add_argument('--compare', help='Second file for comparison')
    parser.add_argument('--output', default='spectral_analysis', help='Output prefix')
    parser.add_argument('--analysis', choices=['psd', 'autocorr', 'interarrival', 'duration', 'all'], 
                       default='all', help='Type of analysis')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_spectral_features(args.input_file, args.compare, args.output)
    else:
        # Single file analysis
        burst_data = load_burst_data(args.input_file)
        time_series = create_binary_timeseries(burst_data)
        title = args.input_file.split('/')[-1]
        
        if args.analysis in ['psd', 'all']:
            power_spectral_density(time_series, title=title)
            plt.savefig(f'{args.output}_psd.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        if args.analysis in ['autocorr', 'all']:
            autocorrelation_analysis(time_series, title=title)
            plt.savefig(f'{args.output}_autocorr.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        if args.analysis in ['interarrival', 'all']:
            burst_interarrival_analysis(burst_data, title=title)
            plt.savefig(f'{args.output}_interarrival.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        if args.analysis in ['duration', 'all']:
            burst_duration_analysis(burst_data, title=title)
            plt.savefig(f'{args.output}_duration.png', dpi=300, bbox_inches='tight')
            plt.show()

if __name__ == "__main__":
    main()