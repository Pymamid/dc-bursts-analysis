import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Directory containing the ingress bytes CSV files
ingress_dir = "/home/pragna/work/DC_bursts/Analysis-scripts/Millisampler-analysis/ingress_bytes"

# Collect start timestamps from all files
start_timestamps = []
file_info = []  # To store (filename, start_timestamp)

# Read all CSV files in the directory
csv_files = glob.glob(os.path.join(ingress_dir, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

for filepath in csv_files:
    try:
        df = pd.read_csv(filepath)
        if 'start_timestamp_ms' in df.columns:
            # Get the first start timestamp (they seem to be the same within a file)
            ts = df['start_timestamp_ms'].iloc[0]
            start_timestamps.append(ts)
            file_info.append((os.path.basename(filepath), ts))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

print(f"Collected {len(start_timestamps)} start timestamps")

# Convert to numpy array for analysis
timestamps = np.array(start_timestamps)

# Calculate statistics
min_ts = np.min(timestamps)
max_ts = np.max(timestamps)
mean_ts = np.mean(timestamps)
std_ts = np.std(timestamps)
range_ts = max_ts - min_ts

print(f"\n=== Start Timestamp Statistics ===")
print(f"Min timestamp: {min_ts}")
print(f"Max timestamp: {max_ts}")
print(f"Mean timestamp: {mean_ts}")
print(f"Std deviation: {std_ts}")
print(f"Range (max - min): {range_ts} ms")
print(f"Range in seconds: {range_ts / 1000} s")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram of start timestamps
ax1 = axes[0, 0]
ax1.hist(timestamps, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Start Timestamp (ms)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Start Timestamps')
ax1.ticklabel_format(style='plain', axis='x')

# 2. Histogram of relative timestamps (normalized to min)
ax2 = axes[0, 1]
relative_ts = timestamps - min_ts
ax2.hist(relative_ts, bins=50, edgecolor='black', alpha=0.7, color='green')
ax2.set_xlabel('Relative Start Time from First (ms)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Relative Start Times')

# 3. CDF of start timestamps
ax3 = axes[1, 0]
sorted_ts = np.sort(relative_ts)
cdf = np.arange(1, len(sorted_ts) + 1) / len(sorted_ts)
ax3.plot(sorted_ts, cdf, linewidth=2, color='red')
ax3.set_xlabel('Relative Start Time from First (ms)')
ax3.set_ylabel('CDF')
ax3.set_title('CDF of Start Timestamps')
ax3.grid(True, alpha=0.3)

# 4. Scatter plot - index vs timestamp to see ordering
ax4 = axes[1, 1]
sorted_indices = np.argsort(timestamps)
ax4.scatter(range(len(timestamps)), relative_ts[sorted_indices], alpha=0.5, s=5, color='purple')
ax4.set_xlabel('File Index (sorted by timestamp)')
ax4.set_ylabel('Relative Start Time (ms)')
ax4.set_title('Start Timestamps Across Files (Sorted)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/pragna/work/DC_bursts/Analysis-scripts/Millisampler-analysis/millisampler_plots/start_timestamp_variation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to: millisampler_plots/start_timestamp_variation.png")

# Additional analysis: Check if timestamps are unique or shared across hosts
unique_timestamps = np.unique(timestamps)
print(f"\n=== Additional Analysis ===")
print(f"Number of unique start timestamps: {len(unique_timestamps)}")
print(f"Number of files: {len(timestamps)}")

# Find files with same timestamps
if len(unique_timestamps) < len(timestamps):
    print("\nSome files share the same start timestamp.")
    ts_counts = pd.Series(timestamps).value_counts()
    print(f"Top 10 most common timestamps:")
    print(ts_counts.head(10))
