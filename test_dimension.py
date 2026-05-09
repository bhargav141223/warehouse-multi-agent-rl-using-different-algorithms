#!/usr/bin/env python3
"""Test script to check observation dimensions"""
import sys
import os

# Add backend to path
backend_path = 'C:/Users/bharg/OneDrive/Desktop/final_rl/backend'
sys.path.insert(0, backend_path)

# Change to backend directory
os.chdir(backend_path)

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")
print(f"Files in dir: {os.listdir('.')[:10]}")

from env import SimpleEnvironment

# Create environment
env = SimpleEnvironment()
obs, _ = env.reset()

print(f"Environment: {env.__class__.__name__}")
print(f"Number of agents: {env.num_agents}")
print(f"Grid size: {env.grid_size}")
print(f"Observation shape: {obs[0].shape}")
print(f"Observation size (flattened): {obs[0].flatten().shape[0]}")
print(f"Total obs for critic (all agents): {obs[0].flatten().shape[0] * env.num_agents}")
