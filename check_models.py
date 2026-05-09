#!/usr/bin/env python3
"""Check saved model dimensions"""
import sys
sys.path.insert(0, 'C:/Users/bharg/OneDrive/Desktop/final_rl/backend')
import os
os.chdir('C:/Users/bharg/OneDrive/Desktop/final_rl/backend')

import torch

models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]

print(f"Found {len(model_files)} model files:")
for f in model_files:
    path = os.path.join(models_dir, f)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Check critic state dict
        if 'critic_state_dict' in checkpoint:
            critic_sd = checkpoint['critic_state_dict']
            # Check first layer weight
            if 'fc1.weight' in critic_sd:
                shape = critic_sd['fc1.weight'].shape
                print(f"  {f}: critic fc1.weight = {shape}")
            else:
                print(f"  {f}: no fc1.weight in critic")
        else:
            print(f"  {f}: no critic_state_dict")
    except Exception as e:
        print(f"  {f}: Error - {e}")
