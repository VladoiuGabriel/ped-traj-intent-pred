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
import argparse


def polar_to_cartesian_np(coords):
    r     = coords[..., 0:1]
    theta = coords[..., 1:2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.concatenate([x, y], axis=-1)


def load_model(ckpt_path, device, use_lora=False):
    model = PedTrajModel(device=device, use_lora=use_lora)
    model.projector   = model.projector.to(device)
    model.flow        = model.flow.to(device)
    model.plan_norm   = model.plan_norm.to(device)
    model.obs_encoder = model.obs_encoder.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.projector.load_state_dict(ckpt['projector'])
    model.flow.load_state_dict(ckpt['flow'])
    model.plan_norm.load_state_dict(ckpt['plan_norm'])
    model.obs_encoder.load_state_dict(ckpt['obs_encoder'])

    if use_lora and 'vlm_lora' in ckpt:
        model.vlm.load_state_dict(ckpt['vlm_lora'])

    model.projector.eval()
    model.flow.eval()
    model.vlm.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} "
          f"| ADE: {ckpt['ade']:.3f}m | FDE: {ckpt['fde']:.3f}m", flush=True)
    return model


def visualize(ckpt_path, dataroot, use_lora, n_examples, label, device):
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    dataset = PedestrianDataset(nusc, dataroot)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = load_model(ckpt_path, device, use_lora=use_lora)

    cols = 4
    rows = (n_examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    collected = 0
    for batch in loader:
        if collected >= n_examples:
            break

        obs     = batch['obs'].to(device)
        pred_gt = batch['pred_gt'].to(device)
        images  = batch['images']

        with torch.no_grad():
            preds = model.predict(images, obs, n_samples=6)

        obs_raw   = obs[0].cpu().numpy()
        gt_raw    = pred_gt[0].cpu().numpy()
        preds_raw = preds.cpu().numpy()

        obs_cart   = polar_to_cartesian_np(obs_raw)
        gt_cart    = polar_to_cartesian_np(gt_raw)


        gt_exp  = torch.tensor(gt_cart).unsqueeze(0).expand(6, -1, -1)
        ade_per = torch.norm(torch.tensor(preds_raw) - gt_exp, dim=-1).mean(dim=-1)
        min_ade = ade_per.min().item()

        ax = axes[collected]

        colors = plt.cm.Reds(np.linspace(0.4, 0.9, 6))
        for k in range(6):
            x = preds_raw[k, :, 0]
            y = preds_raw[k, :, 1]
            ax.plot(x, y, color=colors[k], linewidth=1.5, alpha=0.8)
            ax.scatter(x, y, color=colors[k], s=15, alpha=0.8, zorder=3)

        ax.plot(obs_cart[:, 0], obs_cart[:, 1],
                'b-o', linewidth=2, markersize=4, label='obs')
        ax.plot(gt_cart[:, 0], gt_cart[:, 1],
                'g--o', linewidth=2, markersize=4, label='gt')
        ax.plot(obs_cart[-1, 0], obs_cart[-1, 1], 'ko', markersize=6)

        ax.set_title(f'ADE: {min_ade:.2f}m', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

        if collected == 0:
            ax.legend(fontsize=6)

        collected += 1

    for i in range(collected, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'{label}\nBlue=observation, Green=ground truth, Red=predictions', fontsize=12)
    plt.tight_layout()
    out_path = f'viz_{label.replace(" ", "_")}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {out_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--dataroot',   default='/home/student02/data/nuscenes')
    parser.add_argument('--use_lora',   action='store_true')
    parser.add_argument('--n_examples', type=int, default=8)
    parser.add_argument('--label',      default='model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualize(
        ckpt_path=args.ckpt,
        dataroot=args.dataroot,
        use_lora=args.use_lora,
        n_examples=args.n_examples,
        label=args.label,
        device=device
    )