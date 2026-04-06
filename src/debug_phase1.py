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
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = PedTrajModel(device=device)
model.projector = model.projector.to(device)
model.flow = model.flow.to(device)
model.eval()

batch = next(iter(loader))
obs = batch['obs'].to(device)
pred_gt = batch['pred_gt'].to(device)
images = batch['images']

print("checking planning token...")
with torch.no_grad():
    planning_token = model.get_planning_token(images, obs)

print(f"planning token shape: {planning_token.shape}")
print(f"planning token mean:  {planning_token.mean().item():.4f}")
print(f"planning token std:   {planning_token.std().item():.4f}")
print(f"planning token min:   {planning_token.min().item():.4f}")
print(f"planning token max:   {planning_token.max().item():.4f}")
print(f"has nan: {torch.isnan(planning_token).any().item()}")
print(f"has inf: {torch.isinf(planning_token).any().item()}")

print("\nchecking projector output...")
context = model.projector(planning_token)
print(f"context shape: {context.shape}")
print(f"context mean:  {context.mean().item():.4f}")
print(f"context has nan: {torch.isnan(context).any().item()}")
print(f"context has inf: {torch.isinf(context).any().item()}")

print("\nchecking flow matching loss...")
loss = model.flow_matching_loss(context, pred_gt)
print(f"loss: {loss.item()}")