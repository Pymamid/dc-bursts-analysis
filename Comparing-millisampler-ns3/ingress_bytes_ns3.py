# aggregator bytes received format is - # Time (s) , sender IP , sender port , aggregator IP , aggregator port , bytes received , tcp flags

# it prints out timestamps of when each packet is received.
# we want to aggregate in millisec intervals and plot the ingress bytes over time.

# take file (with path) as input, and output a plot.

# python3 ingress_bytes_ns3.py /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8DCTCPafter-parameter-alignment/bg-incast-fattree/logs/aggregator_bytes_received.log dctcpk=8ingress.png

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
def load_data(filepath):
    timestamps = []
    bytes_received = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip malformed lines

            parts = line.replace(',', ' ').split()
            if len(parts) < 7:
                continue  # skip malformed lines

            try:
                timestamp = float(parts[0])  # Time (s)
                bytes_recv = int(parts[5])   # bytes received
                timestamps.append(timestamp)
                bytes_received.append(bytes_recv)
            except ValueError:
                continue  # skip lines with invalid data
    
    return np.array(timestamps), np.array(bytes_received)

def aggregate_bytes(timestamps, bytes_received, interval_ms=1):
    # Convert interval to seconds
    interval_s = interval_ms / 1000.0
    
    # Determine the range of timestamps
    start_time = np.min(timestamps)
    end_time = np.max(timestamps)
    
    # Create bins for aggregation
    bins = np.arange(start_time, end_time + interval_s, interval_s)
    
    # Aggregate bytes received in each bin
    aggregated_bytes, _ = np.histogram(timestamps, bins=bins, weights=bytes_received)
    
    # Get the midpoints of the bins for plotting
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    return bin_midpoints, aggregated_bytes

def plot_ingress_bytes(timestamps, bytes_received, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, bytes_received, label='Ingress Bytes', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Bytes Received')
    plt.title('Ingress Bytes Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python ingress_bytes_ns3.py <input_csv_path> <output_plot_path>")
        print("Example: python ingress_bytes_ns3.py data.csv output_plot.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {input_path}...")
    timestamps, bytes_received = load_data(input_path)
    if len(timestamps) == 0:
        print("Error: No valid rows found. Expected format:")
        print("<time> <sender_ip> <sender_port> <aggregator_ip> <aggregator_port> <bytes_received> <tcp_flags>")
        sys.exit(1)
    print(f"Loaded {len(timestamps)} data points")
    
    # Aggregate bytes in 1ms intervals
    print("Aggregating bytes in 1ms intervals...")
    agg_timestamps, agg_bytes = aggregate_bytes(timestamps, bytes_received, interval_ms=1)
    
    # save aggregated data to a new file for reference
    agg_data_path = output_path.replace('.png', '_aggregated.csv')
    with open(agg_data_path, 'w') as f:
        f.write("Time(s),BytesReceived\n")
        for t, b in zip(agg_timestamps, agg_bytes):
            f.write(f"{t},{b}\n")
    print(f"Aggregated data saved to {agg_data_path}")

    # Plot and save
    print(f"Generating plot and saving to {output_path}...")
    plot_ingress_bytes(agg_timestamps, agg_bytes, output_path)
    print("Done!")

if __name__ == "__main__":
    main()