import sys
import os
import argparse

# python3 process_throughput.py /home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8after-parameter-alignment/bg-incast-fattree/logs --output-dir /home/pragna/work/DC_bursts/Analysis-scripts/ns3-analysis/ns3-millisampler-type-output

def main():
    parser = argparse.ArgumentParser(description="Process throughput logs.")
    parser.add_argument("log_dir", help="Directory containing the logs")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files (default: same as log directory)")
    parser.add_argument("--link-rate", type=float, default=25000000000, help="Link rate in bps (default: 25Gbps)")
    
    
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir if args.output_dir else log_dir
    link_rate_bps = args.link_rate
    
    # Threshold for burst detection in bytes per ms
    # Rate = bytes * 8 * 1000 bits/s
    # Condition: Rate >= 0.5 * link_rate
    # bytes * 8000 >= 0.5 * link_rate
    # bytes >= (0.5 * link_rate) / 8000
    burst_bytes_threshold = (0.5 * link_rate_bps) / 8000.0
    
    print(f"Link rate: {link_rate_bps/1e9} Gbps")
    print(f"Burst threshold: {burst_bytes_threshold:.2f} bytes/ms")
    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(log_dir, "aggregator_bytes_received.log")

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        sys.exit(1)

    # Dictionary to store bytes and connections per millisecond for each receiver
    # receiver_ip -> { time_ms_index -> {'bytes': count, 'conns': set(sender_id)} }
    # sender_id is "ip:port"
    receiver_data = {}
    global_max_ms = 0

    print(f"Reading {input_file}...")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                # Expected format:
                # Time (s) , sender IP , sender port , aggregator IP , aggregator port , bytes received , tcp flags
                # 0.526572 10.3.1.5 49153 10.0.0.1 9 1067 0x4e1
                
                if len(parts) < 6:
                    continue

                try:
                    time_s = float(parts[0])
                    sender_ip = parts[1]
                    sender_port = parts[2]
                    receiver_ip = parts[3]
                    bytes_received = int(parts[5])
                    
                    # Convert to millisecond index
                    ms_idx = int(time_s * 1000)
                    
                    if ms_idx > global_max_ms:
                        global_max_ms = ms_idx

                    if receiver_ip not in receiver_data:
                        receiver_data[receiver_ip] = {}

                    if ms_idx not in receiver_data[receiver_ip]:
                        receiver_data[receiver_ip][ms_idx] = {'bytes': 0, 'conns': set()}

                    entry = receiver_data[receiver_ip][ms_idx]
                    entry['bytes'] += bytes_received
                    entry['conns'].add(f"{sender_ip}:{sender_port}")

                except ValueError:
                    # Skip lines with parse errors
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Process each receiver
    for receiver_ip, time_map in receiver_data.items():
        # Output 1: Throughput per ms
        throughput_filename = f"receiver_{receiver_ip}_ingress.txt"
        throughput_path = os.path.join(output_dir, throughput_filename)
        
        # Output 2: Burst statistics
        burst_filename = f"receiver_{receiver_ip}_bursts.txt"
        burst_path = os.path.join(output_dir, burst_filename)
        
        print(f"Processing receiver {receiver_ip}...")
        
        try:
            with open(throughput_path, 'w') as t_out, open(burst_path, 'w') as b_out:
                # Write headers
                t_out.write("# Time(s) BytesReceived\n")
                b_out.write("# BurstLength(ms) BurstStart(s) IngressMax(Bytes) MaxConnections\n")
                
                # Burst tracking state
                in_burst = False
                burst_start_ms = 0
                burst_max_bytes = 0
                burst_max_conns = 0
                burst_length_ms = 0

                # Using global_max_ms to be consistent with the simulation duration
                for ms in range(global_max_ms + 1):
                    entry = time_map.get(ms, {'bytes': 0, 'conns': set()})
                    bytes_val = entry['bytes']
                    conns_count = len(entry['conns'])
                    time_sec = ms / 1000.0
                    
                    # Write throughput
                    t_out.write(f"{time_sec:.3f} {bytes_val}\n")
                    
                    # Check burst condition
                    is_bursty = bytes_val >= burst_bytes_threshold
                    
                    if is_bursty:
                        if not in_burst:
                            # Start new burst
                            in_burst = True
                            burst_start_ms = ms
                            burst_max_bytes = bytes_val
                            burst_max_conns = conns_count
                            burst_length_ms = 1
                        else:
                            # Continue burst
                            burst_length_ms += 1
                            if bytes_val > burst_max_bytes:
                                burst_max_bytes = bytes_val
                            if conns_count > burst_max_conns:
                                burst_max_conns = conns_count
                    else:
                        if in_burst:
                            # End of burst, write stats
                            burst_start_sec = burst_start_ms / 1000.0
                            b_out.write(f"{burst_length_ms} {burst_start_sec:.3f} {burst_max_bytes} {burst_max_conns}\n")
                            in_burst = False
                
                # If ended in a burst, write it out
                if in_burst:
                    burst_start_sec = burst_start_ms / 1000.0
                    b_out.write(f"{burst_length_ms} {burst_start_sec:.3f} {burst_max_bytes} {burst_max_conns}\n")

            print(f"Generated {throughput_filename} and {burst_filename}")
                    
        except Exception as e:
            print(f"Error writing output for {receiver_ip}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
