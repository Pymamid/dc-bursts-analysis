#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 transition_matrix.py <burst_file.txt>")
    sys.exit(1)

input_file = sys.argv[1]

# Read the burst data file
# Format: # BurstLength(ms) BurstStart(s) IngressMax(Bytes) MaxConnections
burst_data = pd.read_csv(input_file, sep=r'\s+', comment='#',
                         names=['Length', 'BurstStart', 'IngressMax', 'MaxConnections'])

# burst_start is in seconds - multiply by 1000 to get milliseconds.
burst_data['BurstStart'] = burst_data['BurstStart'] * 1000  # convert to ms
# sort by burst start time to get the correct order of bursts
burst_data = burst_data.sort_values('BurstStart').reset_index(drop=True)

# Filter bursts to only include those starting between 1.0s and 3.0s (1000ms to 3000ms)
burst_data = burst_data[(burst_data['BurstStart'] >= 1000) & (burst_data['BurstStart'] <= 3000)].reset_index(drop=True)

if len(burst_data) == 0:
    print("No bursts found in the 1-3s time window!")
    sys.exit(1)

burst_data['BurstEnd'] = burst_data['BurstStart'] + burst_data['Length']

# Create a time series array to mark burst periods for 1-3s window (2000ms total)
# Time series covers 1000ms to 3000ms, so index 0 = 1000ms, index 1999 = 2999ms
time_start_ms = 1000
time_end_ms = 3000
total_duration = time_end_ms - time_start_ms  # 2000ms
time_series = np.zeros(total_duration, dtype=int)

for _, row in burst_data.iterrows():
    start = int(row['BurstStart']) - time_start_ms  # Adjust for 1s offset
    end = int(row['BurstEnd']) - time_start_ms      # Adjust for 1s offset
    
    # Ensure bounds are within the time window
    start = max(0, start)
    end = min(total_duration, end)
    
    if start < end:  # Only mark if valid range
        time_series[start:end] = 1  # Mark burst periods with 1

# compute transitions

x_prev = time_series[:-1]
x_next = time_series[1:]

count_00 = np.sum((x_prev == 0) & (x_next == 0))
count_01 = np.sum((x_prev == 0) & (x_next == 1))
count_10 = np.sum((x_prev == 1) & (x_next == 0))
count_11 = np.sum((x_prev == 1) & (x_next == 1))

count_0 = np.sum(x_prev == 0)
count_1 = np.sum(x_prev == 1)

p_0_0 = count_00/count_0
p_1_0 = count_01/count_0
p_0_1 = count_10/count_1
p_1_1 = count_11/count_1

print("printing transition probabilities:")
print(p_0_0)
print(p_1_0)
print(p_0_1)
print(p_1_1)

print("Printing counts:")
print(count_00)
print(count_01)
print(count_10)
print(count_11)

print("Printing total counts:")
print(count_0)
print(count_1)

print("Printing r:")
print(p_1_1/p_1_0)