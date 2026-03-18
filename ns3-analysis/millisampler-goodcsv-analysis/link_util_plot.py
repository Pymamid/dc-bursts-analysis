
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def load_ingress_timeseries_data(filepath):
    timestamps = []
    bytes_received = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip empty lines and comment headers

            parts = line.split()
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
    link_bandwidth_bps = 12.5 * 1e6  # 25 Gbps = 25 * 10^9 bps = 25 * 1e6 bpms (since we are working with ms intervals)
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
        print("Usage: python link_util_plot.py <ingress_timeseries_file> <output_plot_path> [<start_time_s> <end_time_s>]")
        print("Example (full range): python link_util_plot.py ingress_timeseries.txt output.png")
        print("Example (0 to 2s): python link_util_plot.py ingress_timeseries.txt output.png 0 2")
        print("Example (4 to 9s): python link_util_plot.py ingress_timeseries.txt output.png 4 9")
        sys.exit(1)
    
    timeseries_file = sys.argv[1]
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

    if not os.path.isfile(timeseries_file):
        print(f"Error: File {timeseries_file} does not exist.")
        sys.exit(1)

    print(f"Loading ingress timeseries data from {timeseries_file}...")
    timestamps, bytes_received = load_ingress_timeseries_data(timeseries_file)
    
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