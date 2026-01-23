import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import os

def load_ingress_bytes(file_path):
    """Load ingress bytes from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df['ingressBytes'].values
    except Exception as e:
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

def main():
    # Path to the ingress_bytes folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ingress_folder = os.path.join(script_dir, '..', 'ingress_bytes')
    output_dir = script_dir
    
    print(f"Loading data from: {ingress_folder}")
    
    # Get all CSV files
    all_files = glob.glob(os.path.join(ingress_folder, "ingress_bytes_*.csv"))
    print(f"Found {len(all_files)} files")
    
    # Collect PSDs from all hosts
    all_psds = []
    common_freqs = None
    
    for i, file in enumerate(all_files):
        if i % 500 == 0:
            print(f"Processing file {i}/{len(all_files)}...")
        
        data = load_ingress_bytes(file)
        if data is None or len(data) < 64:
            continue
        
        freqs, psd = compute_psd(data, fs=1000)  # 1000 Hz = 1 sample/ms
        if freqs is None:
            continue
        
        # Interpolate to common frequency grid if needed
        if common_freqs is None:
            common_freqs = freqs
            all_psds.append(psd)
        else:
            # Interpolate PSD to common frequency grid
            psd_interp = np.interp(common_freqs, freqs, psd)
            all_psds.append(psd_interp)
    
    print(f"Successfully processed {len(all_psds)} hosts")
    
    if len(all_psds) == 0:
        print("No valid PSDs computed!")
        return
    
    # Convert to numpy array and compute average
    all_psds = np.array(all_psds)
    avg_psd = np.mean(all_psds, axis=0)
    std_psd = np.std(all_psds, axis=0)
    
    # Also compute median and percentiles for robustness
    median_psd = np.median(all_psds, axis=0)
    p25_psd = np.percentile(all_psds, 25, axis=0)
    p75_psd = np.percentile(all_psds, 75, axis=0)
    
    # Plot averaged PSD
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot with log-log scale
    ax.loglog(common_freqs[1:], avg_psd[1:], 'b-', linewidth=2, label='Mean PSD')
    ax.loglog(common_freqs[1:], median_psd[1:], 'g--', linewidth=2, label='Median PSD')
    ax.fill_between(common_freqs[1:], p25_psd[1:], p75_psd[1:], 
                    alpha=0.3, color='blue', label='25th-75th percentile')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (bytes²/Hz)', fontsize=12)
    ax.set_title(f'Averaged Power Spectral Density of Ingress Bytes\n({len(all_psds)} hosts)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add statistics
    stats_text = (f'Hosts: {len(all_psds):,}\n'
                  f'Sampling: 1 kHz (1 ms)')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ingress_bytes_psd.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

if __name__ == '__main__':
    main()
