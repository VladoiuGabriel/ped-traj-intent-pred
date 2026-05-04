import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from nuscenes.nuscenes import NuScenes
from dataset import PedestrianDataset, collate_fn
from torch.utils.data import DataLoader
from model import PedTrajModel

device = 'cuda'

nusc = NuScenes(version='v1.0-mini', dataroot='/home/student02/data/nuscenes', verbose=False)
dataset = PedestrianDataset(nusc, '/home/student02/data/nuscenes')
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model = PedTrajModel(device=device)
model.projector = model.projector.to(device)
model.flow = model.flow.to(device)
model.eval()

batch = next(iter(loader))
obs = batch['obs'].to(device)
images = batch['images']

with torch.no_grad():
    planning_tokens = model.get_planning_token(images, obs)

print(f"planning token shape: {planning_tokens.shape}")
print(f"planning token std:   {planning_tokens.std().item():.4f}")

for i in range(4):
    for j in range(i+1, 4):
        diff = (planning_tokens[i] - planning_tokens[j]).abs().mean().item()
        print(f"diff sample {i} vs {j}: {diff:.4f}")

model.projector = model.projector.to(device)

with torch.no_grad():
    planning_tokens = model.get_planning_token(images, obs)
    context = model.projector(planning_tokens)

print(f"\nplanning token std:  {planning_tokens.std().item():.4f}")
print(f"planning token mean: {planning_tokens.mean().item():.4f}")

print(f"\ncontext shape: {context.shape}")
print(f"context std:   {context.std().item():.4f}")
print(f"context mean:  {context.mean().item():.4f}")

for i in range(4):
    for j in range(i+1, 4):
        diff = (context[i] - context[j]).abs().mean().item()
        print(f"diff context {i} vs {j}: {diff:.4f}")