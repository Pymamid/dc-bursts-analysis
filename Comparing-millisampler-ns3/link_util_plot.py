# take input aggregated csv file as argument, and output link util plot with burst threshold marked. plots according to window specified. start and end timestamp in seconds. if not specified, plot full range.

# link bandwidth is 25Gbps

# input aggregated csv is in ms intervals, with columns Time(s),BytesReceived

# python3 link_util_plot.py dctcpk=8ingress_aggregated.csv dctcpk=8linkutil.png 0 10


import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def load_aggregated_data(filepath):
    timestamps = []
    bytes_received = []
    
    with open(filepath, 'r') as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            parts = line.split(',')
            if len(parts) != 2:
                continue  # skip malformed lines

            try:
                timestamp = float(parts[0])  # Time (s)
                bytes_recv = int(parts[1])   # Bytes received
                timestamps.append(timestamp)
                bytes_received.append(bytes_recv)
            except ValueError:
                continue  # skip lines with invalid data
    
    return np.array(timestamps), np.array(bytes_received)

def plot_link_utilization(timestamps, bytes_received, output_path):
    # Convert bytes to bits and calculate utilization
    bits_received = bytes_received * 8
    link_bandwidth_bps = 25 * 1e6  # 25 Gbps = 25 * 10^9 bps = 25 * 1e6 bpms (since we are working with ms intervals)
    utilization = bits_received / link_bandwidth_bps
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, utilization, label='Link Utilization', color='blue')
    # plot horizontal line for burst threshold at 0.5 utilization
    plt.axhline(y=0.5, color='red', linestyle='--', label='Burst Threshold (0.5)')
    plt.xlabel('Time (s)')
    plt.ylabel('Utilization')
    plt.title('Link Utilization Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def filter_time_range(timestamps, bytes_received, start_time=None, end_time=None):
    if start_time is None and end_time is None:
        return timestamps, bytes_received

    mask = np.ones_like(timestamps, dtype=bool)
    if start_time is not None:
        mask = mask & (timestamps >= start_time)
    if end_time is not None:
        mask = mask & (timestamps <= end_time)

    return timestamps[mask], bytes_received[mask]

def main():
    if len(sys.argv) not in (3, 5):
        print("Usage: python link_util_plot.py <aggregated_csv_file> <output_plot_path> [<start_time_s> <end_time_s>]")
        print("Example (full range): python link_util_plot.py input.csv output.png")
        print("Example (0 to 2s): python link_util_plot.py input.csv output.png 0 2")
        print("Example (4 to 9s): python link_util_plot.py input.csv output.png 4 9")
        sys.exit(1)
    
    agg_csv_file = sys.argv[1]
    output_plot_path = sys.argv[2]

    start_time = None
    end_time = None
    if len(sys.argv) == 5:
        try:
            start_time = float(sys.argv[3])
            end_time = float(sys.argv[4])
        except ValueError:
            print("Error: start_time and end_time must be numbers (seconds).")
            sys.exit(1)

        if start_time > end_time:
            print("Error: start_time must be less than or equal to end_time.")
            sys.exit(1)

    if not os.path.isfile(agg_csv_file):
        print(f"Error: File {agg_csv_file} does not exist.")
        sys.exit(1)

    print(f"Loading aggregated data from {agg_csv_file}...")
    timestamps, bytes_received = load_aggregated_data(agg_csv_file)
    
    print(f"Data loaded. Number of data points: {len(timestamps)}")

    timestamps, bytes_received = filter_time_range(
        timestamps,
        bytes_received,
        start_time=start_time,
        end_time=end_time,
    )

    if len(timestamps) == 0:
        if start_time is None and end_time is None:
            print("Error: No valid data points found in the input file.")
        else:
            print(f"Error: No data points found in time range [{start_time}, {end_time}] seconds.")
        sys.exit(1)

    if start_time is not None and end_time is not None:
        print(f"Filtered data points in [{start_time}, {end_time}] seconds: {len(timestamps)}")
    
    print(f"Generating link utilization plot and saving to {output_plot_path}...")
    plot_link_utilization(timestamps, bytes_received, output_plot_path)
    print("Plot saved successfully.")


if __name__ == "__main__":
    main()