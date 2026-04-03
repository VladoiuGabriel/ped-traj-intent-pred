import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from dataset import PedestrianDataset, collate_fn
from torch.utils.data import DataLoader
from model import PedTrajModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nusc = NuScenes(
    version='v1.0-mini',
    dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes',
    verbose=False
)

dataset = PedestrianDataset(nusc, r'C:\Users\Gabi\pedtraj\data\nuscenes')
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = PedTrajModel(device=device)
model.bridge = model.bridge.to(device)
model.clip   = model.clip.to(device)
model.flow   = model.flow.to(device)
model.flow.eval()
model.bridge.eval()

batch = next(iter(loader))
obs    = batch['obs'].to(device)
images = batch['images']

print("=== PROMPT INSPECT ===")
for b in range(obs.shape[0]):
    coords = ", ".join(
        [f"({obs[b, i, 0]:.2f}, {obs[b, i, 1]:.2f})" for i in range(obs.shape[1])]
    )
    prompt = (f"A pedestrian was observed at positions: {coords}. "
              f"Predict the future trajectory.")
    print(f"Prompt {b}: {prompt}\n")

print("=== QWEN OUTPUT INSPECT ===")
with torch.no_grad():
    text_features = model.encode_obs_trajectory(obs)

print(f"text_features shape: {text_features.shape}")
print(f"text_features mean:  {text_features.mean().item():.4f}")
print(f"text_features std:   {text_features.std().item():.4f}")
print(f"text_features min:   {text_features.min().item():.4f}")
print(f"text_features max:   {text_features.max().item():.4f}")


diff = (text_features[0] - text_features[1]).abs().mean().item()
print(f"\nmean absolute diff between sample 0 and 1: {diff:.4f}")


print("\n=== CLIP OUTPUT INSPECT ===")
with torch.no_grad():
    visual_tokens = model.encode_image(images)

print(f"visual_tokens shape: {visual_tokens.shape}")
print(f"visual_tokens mean:  {visual_tokens.mean().item():.4f}")
print(f"visual_tokens std:   {visual_tokens.std().item():.4f}")

diff_vis = (visual_tokens[0] - visual_tokens[1]).abs().mean().item()
print(f"\nMean absolute diff between image 0 and 1: {diff_vis:.4f}")

print("\n=== CONTEXT INSPECT ===")
with torch.no_grad():
    context = model.get_context(images, obs)
print(f"context shape: {context.shape}")
print(f"context mean:  {context.mean().item():.4f}")
print(f"context std:   {context.std().item():.4f}")