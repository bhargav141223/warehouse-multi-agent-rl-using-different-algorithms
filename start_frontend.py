#!/usr/bin/env python3
"""Script to start the frontend development server"""
import subprocess
import sys
import os

def start_frontend():
    """Start the Next.js frontend development server"""
    print("Starting Multi-Agent Warehouse Navigation Frontend...")
    print("=" * 60)
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    # Check if node_modules exists
    node_modules = os.path.join(frontend_dir, "node_modules")
    if not os.path.exists(node_modules):
        print("Installing npm dependencies...")
        # Try npm first, then yarn
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["yarn", "install"], cwd=frontend_dir, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Error: Neither npm nor yarn found. Please install Node.js.")
                sys.exit(1)
    
    # Start the development server
    print("Starting Next.js development server on http://localhost:3000")
    print("=" * 60)
    
    try:
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(["yarn", "dev"], cwd=frontend_dir, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Failed to start development server.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    start_frontend()
