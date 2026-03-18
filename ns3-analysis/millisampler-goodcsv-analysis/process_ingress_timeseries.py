#!/usr/bin/env python3

import json
import sys
import os

def process_ingress_timeseries(input_file, output_file):
    """
    Process good.csv to extract ingressBytes array and create time series output
    """
    try:
        # Read the JSON data from good.csv
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract ingressBytes array
        if 'ingressBytes' not in data:
            print("Error: No 'ingressBytes' array found in the file!")
            return
        
        ingress_bytes = data['ingressBytes']
        
        if not ingress_bytes:
            print("Error: ingressBytes array is empty!")
            return
        
        # dt = 0.001 seconds (sampling frequency is 1000 Hz)
        dt = 0.001
        
        # Write output file
        with open(output_file, 'w') as f:
            # Write header
            f.write("# Time(s) BytesReceived\n")
            
            # Process each time sample
            for i, bytes_received in enumerate(ingress_bytes):
                time_s = i * dt
                f.write(f"{time_s:.3f} {bytes_received}\n")
        
        print(f"Successfully processed {len(ingress_bytes)} time samples from {input_file}")
        print(f"Time series output written to {output_file}")
        print(f"Duration: {(len(ingress_bytes)-1) * dt:.3f} seconds")
        
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
    output_file = os.path.join(script_dir, "ingress_timeseries.txt")
    
    process_ingress_timeseries(input_file, output_file)

if __name__ == "__main__":
    main()