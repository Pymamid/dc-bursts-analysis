import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the burst data
print("Loading burst data...")
df = pd.read_csv('burst_data.csv')

print(f"Total bursts: {len(df)}")
print(f"Data columns: {df.columns.tolist()}")
print(f"Sample data:")
print(df.head())

# Calculate actual burst start and end times (both in milliseconds)
print("\nCalculating burst timings...")
df['burst_start_ms'] = df['StartTimestamp'] + df['Position']
df['burst_end_ms'] = df['burst_start_ms'] + df['Length']

print(f"Burst start time range: {df['burst_start_ms'].min():.0f} - {df['burst_start_ms'].max():.0f} ms")
print(f"Burst end time range: {df['burst_end_ms'].min():.0f} - {df['burst_end_ms'].max():.0f} ms")
print(f"Total time span: {df['burst_end_ms'].max() - df['burst_start_ms'].min():.0f} ms")

# Analysis 1: Count simultaneous bursts using event-based approach
print("\nAnalyzing simultaneous bursts per millisecond...")

def count_overlapping_bursts_efficient():
    """
    Use an efficient event-based approach to count overlapping bursts
    """
    # Create events for burst starts and ends
    events = []
    
    # Add start events (+1) and end events (-1)
    for _, row in df.iterrows():
        start_ms = int(np.floor(row['burst_start_ms']))
        end_ms = int(np.ceil(row['burst_end_ms']))
        events.append((start_ms, 1))  # Burst starts
        events.append((end_ms, -1))   # Burst ends
    
    # Sort events by time
    events.sort()
    
    # Process events to count overlaps at each millisecond
    simultaneous_counts = []
    current_count = 0
    prev_time = None
    
    for time_ms, delta in events:
        if prev_time is not None and time_ms != prev_time and current_count > 0:
            # Record the count for the previous time period
            simultaneous_counts.append(current_count)
        
        current_count += delta
        prev_time = time_ms
    
    return np.array(simultaneous_counts)

# More efficient approach: sample at millisecond boundaries where bursts actually exist
def count_simultaneous_at_sample_points():
    """
    Sample at key time points to get distribution of simultaneous bursts
    """
    # Get unique millisecond timestamps where bursts start or end
    sample_times = set()
    
    for _, row in df.iterrows():
        start_ms = int(np.floor(row['burst_start_ms']))
        end_ms = int(np.ceil(row['burst_end_ms']))
        sample_times.update(range(start_ms, end_ms + 1))
    
    sample_times = sorted(sample_times)
    print(f"Sampling at {len(sample_times)} unique millisecond points...")
    
    simultaneous_counts = []
    
    for i, ms in enumerate(sample_times):
        # Count bursts active at this millisecond
        count = ((df['burst_start_ms'] <= ms) & (df['burst_end_ms'] > ms)).sum()
        if count > 0:
            simultaneous_counts.append(count)
        
        # Progress indicator
        if i % 50000 == 0:
            print(f"Processed {i}/{len(sample_times)} time points...")
    
    return np.array(simultaneous_counts)

# Use the more efficient sampling approach
simultaneous_counts = count_simultaneous_at_sample_points()

print(f"Non-zero simultaneous burst counts: {len(simultaneous_counts)}")
print(f"Simultaneous burst counts range: {simultaneous_counts.min()} - {simultaneous_counts.max()}")

# Analysis 2: Burst durations
print("\nAnalyzing burst durations...")
durations = df['Length']
print(f"Duration range: {durations.min()} - {durations.max()} ms")

# Create CDFs
def create_cdf(data):
    """Create cumulative distribution function data"""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y

# CDF 1: Number of simultaneous bursts
simultaneous_data, simultaneous_cdf = create_cdf(simultaneous_counts)

# CDF 2: Burst durations
duration_data, duration_cdf = create_cdf(durations.values)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: CDF of simultaneous bursts
ax1.plot(simultaneous_data, simultaneous_cdf, 'b-', linewidth=2, alpha=0.8)
ax1.set_xlabel('Number of Simultaneous Bursts')
ax1.set_ylabel('Cumulative Probability')
ax1.set_title('CDF: Number of Bursts Occurring Simultaneously\nat a Given Millisecond')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

# Plot 2: CDF of burst durations (log scale)
ax2.plot(duration_data, duration_cdf, 'r-', linewidth=2, alpha=0.8)
ax2.set_xlabel('Burst Duration (ms)')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('CDF: Duration of Bursts (Log Scale)')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0.9)

plt.tight_layout()
plt.savefig('burst_cdfs_final.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\n=== SIMULTANEOUS BURSTS STATISTICS ===")
print(f"Mean simultaneous bursts: {simultaneous_counts.mean():.2f}")
print(f"Median simultaneous bursts: {np.median(simultaneous_counts):.2f}")
print(f"95th percentile: {np.percentile(simultaneous_counts, 95):.2f}")
print(f"99th percentile: {np.percentile(simultaneous_counts, 99):.2f}")
print(f"Max simultaneous bursts: {simultaneous_counts.max()}")

print(f"\n=== BURST DURATION STATISTICS ===")
print(f"Mean duration: {durations.mean():.2f} ms")
print(f"Median duration: {durations.median():.2f} ms")
print(f"95th percentile: {np.percentile(durations, 95):.2f} ms")
print(f"99th percentile: {np.percentile(durations, 99):.2f} ms")
print(f"Max duration: {durations.max():.2f} ms")

# Distribution breakdown
print(f"\n=== SIMULTANEOUS BURSTS DISTRIBUTION ===")
simul_dist = Counter(simultaneous_counts)
for count, freq in sorted(simul_dist.items())[:10]:  # Show first 10
    percentage = (freq / len(simultaneous_counts)) * 100
    print(f"{count} simultaneous bursts: {freq} milliseconds ({percentage:.1f}%)")

print(f"\n=== BURST DURATION DISTRIBUTION ===")
duration_dist = Counter(durations.values)
for duration, freq in sorted(duration_dist.items())[:10]:  # Show first 10
    percentage = (freq / len(durations)) * 100
    print(f"{duration:.1f} ms duration: {freq} bursts ({percentage:.1f}%)")

print(f"\nPlots saved as 'burst_cdfs_final.png'")
