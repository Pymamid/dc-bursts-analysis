import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_ingress_bytes(file_path):
    """Load ingress bytes from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df['ingressBytes'].values
    except Exception as e:
        return None

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
    
    # Normalize data (subtract mean, divide by std)
    data_normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
    
    # Compute autocorrelation using numpy correlate
    acf_full = np.correlate(data_normalized, data_normalized, mode='full')
    acf_full = acf_full / n  # Normalize by length
    
    # Take only positive lags (including 0)
    acf = acf_full[n-1:n-1+max_lag+1]
    lags = np.arange(max_lag + 1)
    
    return lags, acf

def main():
    # Path to the ingress_bytes folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ingress_folder = os.path.join(script_dir, '..', 'ingress_bytes')
    output_dir = script_dir
    
    print(f"Loading data from: {ingress_folder}")
    
    # Get all CSV files
    all_files = glob.glob(os.path.join(ingress_folder, "ingress_bytes_*.csv"))
    print(f"Found {len(all_files)} files")
    
    # Parameters
    max_lag = 500  # Maximum lag in ms
    
    # Collect ACFs from all hosts
    all_acfs = []
    
    for i, file in enumerate(all_files):
        if i % 500 == 0:
            print(f"Processing file {i}/{len(all_files)}...")
        
        data = load_ingress_bytes(file)
        if data is None or len(data) < max_lag * 2:
            continue
        
        lags, acf = compute_acf(data, max_lag=max_lag)
        all_acfs.append(acf)
    
    print(f"Successfully processed {len(all_acfs)} hosts")
    
    if len(all_acfs) == 0:
        print("No valid ACFs computed!")
        return
    
    # Convert to numpy array and compute average
    all_acfs = np.array(all_acfs)
    avg_acf = np.mean(all_acfs, axis=0)
    std_acf = np.std(all_acfs, axis=0)
    
    # Compute percentiles for robustness
    median_acf = np.median(all_acfs, axis=0)
    p25_acf = np.percentile(all_acfs, 25, axis=0)
    p75_acf = np.percentile(all_acfs, 75, axis=0)
    
    lags = np.arange(len(avg_acf))
    
    # Plot averaged ACF
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(lags, avg_acf, 'b-', linewidth=2, label='Mean ACF')
    ax.fill_between(lags, p25_acf, p75_acf, 
                    alpha=0.3, color='blue', label='25th-75th percentile')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Add confidence bounds (approximate 95% CI for white noise)
    n_avg = np.mean([len(load_ingress_bytes(f)) for f in all_files[:100] if load_ingress_bytes(f) is not None])
    ci_bound = 1.96 / np.sqrt(n_avg)
    ax.axhline(y=ci_bound, color='red', linestyle=':', linewidth=1, label=f'95% CI (white noise)')
    ax.axhline(y=-ci_bound, color='red', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Lag (ms)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title(f'Averaged Autocorrelation Function of Ingress Bytes\n({len(all_acfs)} hosts)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, max_lag)
    
    # Add statistics
    # Find first lag where ACF drops below 0.5 and 0.1
    lag_half = np.argmax(avg_acf < 0.5) if np.any(avg_acf < 0.5) else max_lag
    lag_tenth = np.argmax(avg_acf < 0.1) if np.any(avg_acf < 0.1) else max_lag
    
    stats_text = (f'Hosts: {len(all_acfs):,}\n'
                  f'ACF < 0.5 at lag: {lag_half} ms\n'
                  f'ACF < 0.1 at lag: {lag_tenth} ms')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ingress_bytes_acf.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_file}")

if __name__ == '__main__':
    main()
