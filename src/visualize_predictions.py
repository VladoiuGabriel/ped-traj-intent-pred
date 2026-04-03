import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from dataset import PedestrianDataset, collate_fn
from torch.utils.data import DataLoader
from model import PedTrajModel


def visualize_predictions(checkpoint_path, n_examples=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes',
        verbose=False
    )

    dataset = PedestrianDataset(nusc, r'C:\Users\Gabi\pedtraj\data\nuscenes')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    print("Loading model...", flush=True)
    model = PedTrajModel(device=device)
    model.bridge = model.bridge.to(device)
    model.clip   = model.clip.to(device)
    model.flow   = model.flow.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.bridge.load_state_dict(ckpt['bridge'])
    model.flow.load_state_dict(ckpt['flow'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val loss: {ckpt['val_loss']:.4f})", flush=True)

    model.flow.eval()
    model.bridge.eval()

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for idx, batch in enumerate(loader):
        if idx >= n_examples:
            break

        obs     = batch['obs'].to(device)
        pred_gt = batch['pred_gt'].to(device)
        images  = batch['images']

        with torch.no_grad():
            predictions = model.predict(images, obs, n_samples=6)

        obs_np   = obs[0].cpu().numpy()
        gt_np    = pred_gt[0].cpu().numpy()
        preds_np = predictions.cpu().numpy()

        ax = axes[idx]

        for k in range(6):
            full_pred = np.vstack([obs_np[-1:], preds_np[k]])
            ax.plot(full_pred[:, 0], full_pred[:, 1],
                    color='lightcoral', linewidth=1.5, alpha=0.6, zorder=2)

        ax.plot(obs_np[:, 0], obs_np[:, 1],
                'b-o', linewidth=2.5, markersize=6,
                label='Observation', zorder=3)

        full_gt = np.vstack([obs_np[-1:], gt_np])
        ax.plot(full_gt[:, 0], full_gt[:, 1],
                'g--o', linewidth=2.5, markersize=6,
                label='Ground truth', zorder=4)

        ax.plot(0, 0, 'ko', markersize=8, zorder=5)

        gt_exp = torch.tensor(gt_np).unsqueeze(0).expand(6, -1, -1)
        ade_per = torch.norm(predictions.cpu() - gt_exp, dim=-1).mean(dim=-1)
        min_ade = ade_per.min().item()

        ax.plot([], [], color='lightcoral', linewidth=1.5,
                alpha=0.6, label='Predictions (x6)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'Example {idx+1} | minADE: {min_ade:.3f}m')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Pedestrian Trajectory Predictions\n'
                 'Blue=observation, Green=ground truth, Red=predictions',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()
    print("Saved: predictions.png")


if __name__ == '__main__':
    ckpt_path = r'C:\Users\Gabi\pedtraj\checkpoints\best_model.pt'
    visualize_predictions(ckpt_path, n_examples=4)