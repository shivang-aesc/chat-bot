#!/usr/bin/env python3
"""
External ping script to keep the FAQ Bot server alive on Render.
This can be used as a backup method or run from another service.
"""

import requests
import time
import os
import sys
from datetime import datetime

def ping_server(server_url: str):
    """Ping the server health endpoint"""
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"[{datetime.now()}] ✅ Server pinged successfully")
            return True
        else:
            print(f"[{datetime.now()}] ❌ Server ping failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error pinging server: {str(e)}")
        return False

def main():
    # Get server URL from environment or command line
    server_url = os.getenv("SERVER_URL")
    if not server_url:
        if len(sys.argv) > 1:
            server_url = sys.argv[1]
        else:
            print("Usage: python ping_server.py <server_url>")
            print("Or set SERVER_URL environment variable")
            sys.exit(1)
    
    # Get ping interval from environment (default: 5 minutes)
    ping_interval = int(os.getenv("PING_INTERVAL", "300"))
    
    print(f"Starting ping service for: {server_url}")
    print(f"Ping interval: {ping_interval} seconds")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            ping_server(server_url)
            time.sleep(ping_interval)
    except KeyboardInterrupt:
        print("\nPing service stopped")

if __name__ == "__main__":
    main() 