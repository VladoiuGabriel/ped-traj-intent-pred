import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from nuscenes.nuscenes import NuScenes
from dataset import PedestrianDataset, collate_fn
from torch.utils.data import DataLoader
from model import PedTrajModel

device = 'cuda'

print('1. loading nuScenes...', flush=True)
nusc = NuScenes(version='v1.0-mini', dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes', verbose=False)

print('2. building dataset...', flush=True)
dataset = PedestrianDataset(nusc, r'C:\Users\Gabi\pedtraj\data\nuscenes')
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

print('3. loading model...', flush=True)
model = PedTrajModel(device=device)
model.dit = model.dit.to(device)
model.bridge = model.bridge.to(device)
model.clip = model.clip.to(device)

print('4. getting first batch...', flush=True)
batch = next(iter(loader))
print('5. got batch!', flush=True)

obs = batch['obs'].to(device)
pred_gt = batch['pred_gt'].to(device)
images = batch['images']

print('6. running forward pass...', flush=True)
loss = model(images, obs, pred_gt)
print(f'7. loss: {loss.item():.4f}', flush=True)
print('SUCCESS', flush=True)