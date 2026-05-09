#!/usr/bin/env python3
"""Script to start the backend server"""
import subprocess
import sys
import os

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting Multi-Agent Warehouse Navigation Backend...")
    print("=" * 60)
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    
    # Check if virtual environment exists, create if not
    venv_path = os.path.join(backend_dir, "venv")
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    
    # Determine the correct python/pip paths based on OS
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    # Install requirements
    print("Installing dependencies...")
    req_file = os.path.join(backend_dir, "requirements.txt")
    subprocess.run([pip_path, "install", "-r", req_file], check=True)
    
    # Start the server
    print("Starting FastAPI server on http://localhost:8000")
    print("API documentation available at: http://localhost:8000/docs")
    print("=" * 60)
    
    try:
        subprocess.run([python_path, "-m", "uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], 
                      cwd=backend_dir, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    start_backend()
