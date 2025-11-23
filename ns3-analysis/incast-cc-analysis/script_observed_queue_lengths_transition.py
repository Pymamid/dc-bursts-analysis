import os
import pandas as pd
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

agg_bytes_file_path = os.path.join(cur_dir, '../../../../chris_ns3/ns-3-dev-git/scratch/traces/trace_directory/logs/aggregator_bytes_received.log')

linkbandwidth = 12500 #Mbps

# read data from file - timestamp(s), sender ip, sender port, aggregator ip, aggregator port, bytes received
# use read_lines and skip first line

data = pd.DataFrame(columns=["TimestampMs", "BytesReceived"])
with open(agg_bytes_file_path, 'r') as file:
    lines = file.readlines()[1:]  # skip first line
    for line in lines:
        parts = line.strip().split(' ')
        timestamp_s = float(parts[0])
        bytes_received = int(parts[5])
        timestamp_ms = int(timestamp_s * 1000)
        data = pd.concat([data, pd.DataFrame({"TimestampMs": [timestamp_ms], "BytesReceived": [bytes_received]})], ignore_index=True)

print(data.head())

min_time = data['TimestampMs'].min()
max_time = data['TimestampMs'].max()
length = int(max_time - min_time) + 1
time_series = np.zeros(length)
data['TimeIndex'] = (data['TimestampMs'] - min_time).astype(int)
for _, row in data.iterrows():
    time_series[row['TimeIndex']] += row['BytesReceived']

# output time_series for verification to a file
output_verification_path = os.path.join(cur_dir, 'aggregated_link_utilization_bytes_per_ms.csv')
pd.DataFrame({'TimeMs': np.arange(len(time_series)), 'BytesReceivedPerMs': time_series}).to_csv(output_verification_path, index=False)



# Convert bytes received to Mbps
time_series_mbps = (time_series / (1024 * 1024)) * 8 * 1000 # bytes to megabits, per millisecond to per second

# save time_series_mbps to csv for plotting
output_path = os.path.join(cur_dir, 'aggregated_link_utilization_mbps.csv')
pd.DataFrame({'TimeMs': np.arange(len(time_series_mbps)), 'UtilizationMbps': time_series_mbps}).to_csv(output_path, index=False)


# bursts are defined as periods where the link utilization is above 50% of link bandwidth
threshold = 0.5 * linkbandwidth
bursts = time_series_mbps > threshold

# compute transitions
x_prev = bursts[:-1]
x_next = bursts[1:]

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


