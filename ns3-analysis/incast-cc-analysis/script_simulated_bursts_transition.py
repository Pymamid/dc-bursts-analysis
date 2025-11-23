import os
import pandas as pd
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

burst_times_file_path = os.path.join(cur_dir, '../../../../chris_ns3/ns-3-dev-git/scratch/traces/trace_directory/logs/burst_times.log')
agg_bytes_file_path = os.path.join(cur_dir, '../../../../chris_ns3/ns-3-dev-git/scratch/traces/trace_directory/logs/aggregator_bytes_received.log')
# format is start_time end_time in seconds. ignore first line

burst_df = pd.DataFrame(columns=["StartTime", "EndTime"])
with open(burst_times_file_path, 'r') as file:
    lines = file.readlines()[1:]  # skip first line
    for line in lines:
        start_time, end_time = map(float, line.strip().split(' '))
        burst_df = pd.concat([burst_df, pd.DataFrame({"StartTime": [start_time], "EndTime": [end_time]})], ignore_index=True)

burst_df['StartTimeMs'] = (burst_df['StartTime'] * 1000).round()
burst_df['EndTimeMs'] = (burst_df['EndTime'] * 1000).round()

# take min_start_time and max_end_time from agg_bytes_file_path
# read data from file - timestamp(s), sender ip, sender port, aggregator ip, aggregator port, bytes received
# just need to read first and last timestamp
with open(agg_bytes_file_path, 'r') as file:
    lines = file.readlines()[1:]  # skip first line
    first_line = lines[0]
    last_line = lines[-1]
    min_timestamp_s = float(first_line.strip().split(' ')[0]) * 1000
    max_timestamp_s = float(last_line.strip().split(' ')[0]) * 1000

min_start_time = min_timestamp_s
max_end_time = max_timestamp_s
length = int(max_end_time - min_start_time)


print(length)

time_series = np.zeros(length)

# Create a time series array to mark burst periods
for _, row in burst_df.iterrows():
    start = int(row['StartTimeMs'] - min_start_time)
    end = int(row['EndTimeMs'] - min_start_time)
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

