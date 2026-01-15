#python3 analyze_fct.py "/home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/SavedTraces/k=8/bg-incast-fattree/logs"

import pandas as pd
import glob
import os
import re
import json
import matplotlib.pyplot as plt
import sys

def parse_node_ip_map(filepath):
    """Parses node_ip_map.log to create a mapping from IP to NodeId."""
    ip_to_node = {}
    node_to_ip = {}
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return {}, {}
        
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                ip = parts[1]
                ip_to_node[ip] = node_id
                node_to_ip[node_id] = ip
    return ip_to_node, node_to_ip

def parse_sender_logs(logs_dir, ip_to_node, node_to_ip):
    """
    Parses canonical_bg_sender_*.log and dynamic_burst_sender.log
    Returns a DataFrame with columns: [FlowId, NodeId, SenderIP, StartTime, ExpectedSize, Type]
    """
    flows = []
    
    # 1. Parse Canonical Background Sender Logs
    canonical_files = glob.glob(os.path.join(logs_dir, "canonical_bg_sender_*.log"))
    for filepath in canonical_files:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                # Format: Time FlowId FlowSize NodeId Type
                if len(parts) >= 5:
                    time = float(parts[0])
                    flow_id = int(parts[1])
                    size = int(parts[2])
                    node_id = int(parts[3])
                    flow_type = parts[4]
                    
                    flows.append({
                        'StartTime': time,
                        'FlowId': flow_id,
                        'ExpectedSize': size,
                        'NodeId': node_id,
                        'SenderIP': node_to_ip.get(node_id),
                        'Type': 'Background'
                    })

    # 2. Parse Dynamic Burst Sender Log (Incast)
    burst_log = os.path.join(logs_dir, "dynamic_burst_sender.log")
    if os.path.exists(burst_log):
        with open(burst_log, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                # Format: Time FlowId FlowSize NodeId Type
                # Note: The log header says "Time (s) Flow ID Flow Size (bytes) Node ID Type"
                if len(parts) >= 5:
                    time = float(parts[0])
                    flow_id = int(parts[1])
                    size = int(parts[2])
                    node_id = int(parts[3])
                    flow_type = parts[4]
                    
                    flows.append({
                        'StartTime': time,
                        'FlowId': flow_id,
                        'ExpectedSize': size,
                        'NodeId': node_id,
                        'SenderIP': node_to_ip.get(node_id),
                        'Type': 'Incast'
                    })
                    
    df = pd.DataFrame(flows)
    if not df.empty:
        df = df.sort_values('StartTime').reset_index(drop=True)
    return df

def parse_aggregator_log(filepath):
    """
    Parses aggregator_bytes_received.log to reconstruct received flows.
    Returns a dictionary of flows keyed by (SenderIP, SenderPort)
    Value: List of flow dicts [{FirstPacketTime, LastPacketTime, TotalBytes, FinReceived}]
    """
    received_flows = {} # (IP, Port) -> [flow_stats_1, flow_stats_2, ...]
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {}

    GAP_THRESHOLD = 0.5 # Seconds to consider a new flow on the same port

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            # Format: Time, sender IP, sender port, agg IP, agg port, bytes, flags
            if len(parts) >= 7:
                time = float(parts[0])
                src_ip = parts[1]
                src_port = int(parts[2])
                bytes_rx = int(parts[5])
                flags_hex = parts[6]
                flags = int(flags_hex, 16)
                
                key = (src_ip, src_port)
                
                if key not in received_flows:
                    received_flows[key] = []
                    
                start_new = False
                if not received_flows[key]:
                    start_new = True
                else:
                    last_flow = received_flows[key][-1]
                    if time - last_flow['LastPacketTime'] > GAP_THRESHOLD:
                         start_new = True
                
                if start_new:
                    received_flows[key].append({
                        'FirstPacketTime': time,
                        'LastPacketTime': time, # Initialize
                        'TotalBytes': 0,
                        'FinReceived': False
                    })
                
                current_flow = received_flows[key][-1]
                current_flow['LastPacketTime'] = time
                current_flow['TotalBytes'] += bytes_rx
                
                # Check for FIN flag (0x1) - valid for TCP
                if flags & 0x1:
                    current_flow['FinReceived'] = True

    return received_flows

def match_flows(sent_df, received_flows):
    """
    Matches sent flows (intent) with received flows (reality) based on IP and Time.
    """
    results = []
    
    # Organize received flows by IP for faster lookup
    # Flatten structure: list of all flows for an IP, sorted by time
    flows_by_ip = {}
    for (ip, port), flow_list in received_flows.items():
        if ip not in flows_by_ip:
            flows_by_ip[ip] = []
        
        for f in flow_list:
            flow_copy = f.copy()
            flow_copy['Port'] = port
            flows_by_ip[ip].append(flow_copy)
        
    # Sort received flows by start time for each IP
    for ip in flows_by_ip:
        flows_by_ip[ip].sort(key=lambda x: x['FirstPacketTime'])
        
    # Track used matches to avoid double counting?
    # For now, simplistic matching
    
    for idx, row in sent_df.iterrows():
        sender_ip = row['SenderIP']
        start_time = row['StartTime']
        
        if sender_ip not in flows_by_ip:
            continue
            
        candidates = flows_by_ip[sender_ip]
        
        # Heuristic: Find the first flow on this IP that started at or after the sender's start time
        # Allow for a tiny bit of clock skew/jitter, but generally rx_start > tx_start
        # Also check that it hasn't been matched yet? (Optional, skipping for simplicity)
        
        best_match = None
        min_diff = float('inf')
        
        for cand in candidates:
            # The packet must be received AFTER it was sent
            if cand['FirstPacketTime'] >= start_time:
                diff = cand['FirstPacketTime'] - start_time
                if diff < min_diff and diff < 1.0: # Assuming delay is within 1 second
                    # Also check byte count if needed, but let's trust timing first
                    best_match = cand
                    min_diff = diff
            elif start_time - cand['FirstPacketTime'] < 0.001: 
                # Handling barely-negative diffs due to potential logging order precision issues?
                # Unlikely in NS3 deterministic mode, but good safety
                diff = abs(start_time - cand['FirstPacketTime'])
                if diff < min_diff:
                    best_match = cand
                    min_diff = diff

        if best_match:
            fct = best_match['LastPacketTime'] - start_time
            results.append({
                'FlowId': row['FlowId'],
                'Type': row['Type'],
                'Size': row['ExpectedSize'],
                'StartTime': start_time,
                'EndTime': best_match['LastPacketTime'],
                'FCT': fct,
                'RxBytes': best_match['TotalBytes'],
                'FinReceived': best_match['FinReceived']
            })
            
    return pd.DataFrame(results)

def main():
    logs_dir = "logs" # Default relative path

    if len(sys.argv) > 1:
        logs_dir = sys.argv[1]
    
    # If not found, try the specific path from user context
    if not os.path.exists(logs_dir):
        possible_path = "/home/pragna/work/chris_ns3/ns-3-dev-git/scratch/traces/bg-incast-fattree/logs"
        if os.path.exists(possible_path):
            logs_dir = possible_path
    
    print(f"Reading logs from: {logs_dir}")
    
    ip_map_file = os.path.join(logs_dir, "node_ip_map.log")
    agg_file = os.path.join(logs_dir, "aggregator_bytes_received.log")
    
    print("Parsing IP Map...")
    ip_to_node, node_to_ip = parse_node_ip_map(ip_map_file)
    
    print("Parsing Sender Logs...")
    sent_df = parse_sender_logs(logs_dir, ip_to_node, node_to_ip)
    print(f"Found {len(sent_df)} generated flows.")
    
    print("Parsing Aggregator Log...")
    rx_flows = parse_aggregator_log(agg_file)
    print(f"Found {len(rx_flows)} unique received TCP flows (5-tuples).")
    
    print("Matching Flows...")
    fct_df = match_flows(sent_df, rx_flows)
    print(f"Matched {len(fct_df)} flows.")
    
    if not fct_df.empty:
        output_csv = "fct_analysis.csv"
        fct_df.to_csv(output_csv, index=False)
        print(f"Saved analysis to {output_csv}")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Plot Background FCT
        bg_flows = fct_df[fct_df['Type'] == 'Background']
        if not bg_flows.empty:
            plt.scatter(bg_flows['StartTime'], bg_flows['FCT'], label='Background', alpha=0.6, marker='o')
            
        # Plot Incast FCT
        incast_flows = fct_df[fct_df['Type'] == 'Incast']
        if not incast_flows.empty:
            plt.scatter(incast_flows['StartTime'], incast_flows['FCT'], label='Incast', alpha=0.6, marker='x')

        plt.xlabel('Start Time (s)')
        plt.ylabel('Flow Completion Time (s)')
        plt.title('Flow Completion Times over Simulation')
        plt.legend()
        plt.grid(True)
        plt.savefig("fct_plot.png")
        print("Saved plot to fct_plot.png")
        
        # CDF Plot
        plt.figure(figsize=(10, 6))
        if not bg_flows.empty:
            sorted_fct = np.sort(bg_flows['FCT'])
            p = 1. * np.arange(len(sorted_fct)) / (len(sorted_fct) - 1)
            plt.plot(sorted_fct, p, label='Background')
            
        if not incast_flows.empty:
            sorted_fct = np.sort(incast_flows['FCT'])
            p = 1. * np.arange(len(sorted_fct)) / (len(sorted_fct) - 1)
            plt.plot(sorted_fct, p, label='Incast')
            
        plt.xlabel('Flow Completion Time (s)')
        plt.ylabel('CDF')
        plt.title('CDF of Flow Completion Times')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.savefig("fct_cdf.png")
        print("Saved plot to fct_cdf.png")

    else:
        print("No matches found. Check logs.")

if __name__ == "__main__":
    import numpy as np # Import here to avoid dependency if script fails earlier
    main()
