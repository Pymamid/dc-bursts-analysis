#!/usr/bin/env python3

import json
import sys
import os

def process_burst_data(input_file, output_file):
    """
    Process good.csv to extract burst information and create millisampler-bursts.txt
    """
    try:
        # Read the JSON data from good.csv
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract bursts from the data
        bursts = []
        if 'burst_result' in data and 'ingress' in data['burst_result']:
            burst_data = data['burst_result']['ingress']['1641906438033747']
            for burst in burst_data:
                bursts.append(burst)
        
        if not bursts:
            print("No burst data found in the file!")
            return
        
        # dt = 0.001 seconds (sampling frequency is 1000 Hz as mentioned in the file)
        dt = 0.001
        
        # Write output file
        with open(output_file, 'w') as f:
            # Write header
            f.write("# BurstLength(ms) BurstStart(s) IngressMax(Bytes) MaxConnections\n")
            
            # Process each burst
            for i, burst in enumerate(bursts):
                try:
                    burst_length_ms = burst['Length']  # Already in milliseconds
                    burst_start_s = burst['Position'] * dt  # Convert to seconds
                    ingress_max_bytes = burst['ingressMax']
                    max_connections = burst['maxConnections']
                    
                    # Write burst data
                    f.write(f"{burst_length_ms} {burst_start_s:.3f} {ingress_max_bytes} {max_connections}\n")
                except KeyError as e:
                    print(f"Error processing burst {i}: Missing key {e}")
                    print(f"Burst keys: {list(burst.keys())}")
                    continue
        
        print(f"Successfully processed {len(bursts)} bursts from {input_file}")
        print(f"Output written to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    # Define input and output files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "good.csv")
    output_file = os.path.join(script_dir, "millisampler-bursts.txt")
    
    process_burst_data(input_file, output_file)

if __name__ == "__main__":
    main()