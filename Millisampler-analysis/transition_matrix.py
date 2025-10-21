import pandas as pd
import numpy as np

burst_data = pd.read_csv('burst_data.csv')

burst_data['Starttime'] = burst_data['StartTimestamp'].round()

min_start_time = burst_data['Starttime'].min()
max_end_time = (burst_data['Starttime']).max() + 1667 # Length is in milliseconds, add 1667ms to the max end time, because thats the length of a run.

length = max_end_time - min_start_time
print(length)

time_series = np.zeros(int(length))

burst_data['BurstStart'] = burst_data['Starttime'] + burst_data['Position'] - min_start_time
burst_data['BurstEnd'] = burst_data['BurstStart'] + burst_data['Length']
# Create a time series array to mark burst periods
for _,row in burst_data.iterrows():
    start = int(row['BurstStart'])
    end = int(row['BurstEnd'])
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

print(p_0_0)
print(p_1_0)
print(p_0_1)
print(p_1_1)