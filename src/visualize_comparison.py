import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nuscenes.nuscenes import NuScenes
from dataset import PedestrianDataset, collate_fn
from torch.utils.data import DataLoader
from model import PedTrajModel


def load_model(ckpt_path, device, use_lora=False):
    model = PedTrajModel(
        device=device,
        use_lora=use_lora
    )
    model.projector = model.projector.to(device)
    model.flow      = model.flow.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.projector.load_state_dict(ckpt['projector'])
    model.flow.load_state_dict(ckpt['flow'])

    if use_lora and 'vlm_lora' in ckpt:
        model.vlm.load_state_dict(ckpt['vlm_lora'])

    model.projector.eval()
    model.flow.eval()
    model.vlm.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} "
          f"| ADE: {ckpt['ade']:.3f}m | FDE: {ckpt['fde']:.3f}m", flush=True)
    return model


def visualize_comparison(
    phase1_ckpt,
    phase2_ckpt,
    dataroot,
    n_examples=8,
    device='cuda'
):
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=dataroot,
        verbose=False
    )
    dataset = PedestrianDataset(nusc, dataroot)
    loader  = DataLoader(
        dataset, batch_size=1,
        shuffle=True, collate_fn=collate_fn
    )

    print("loading phase 1 model...", flush=True)
    model1 = load_model(phase1_ckpt, device, use_lora=False)

    print("loading phase 2 model...", flush=True)
    model2 = load_model(phase2_ckpt, device, use_lora=True)

    fig = plt.figure(figsize=(n_examples * 4, 10))
    gs  = gridspec.GridSpec(2, n_examples, hspace=0.4, wspace=0.3)

    collected = 0
    for batch in loader:
        if collected >= n_examples:
            break

        obs     = batch['obs'].to(device)
        pred_gt = batch['pred_gt'].to(device)
        images  = batch['images']

        with torch.no_grad():
            preds1 = model1.predict(images, obs, n_samples=6)
            preds2 = model2.predict(images, obs, n_samples=6)

        obs_raw    = obs[0].cpu().numpy()
        gt_raw     = pred_gt[0].cpu().numpy()
        preds1_raw = preds1.cpu().numpy()
        preds2_raw = preds2.cpu().numpy()

        offset = obs_raw[-1].copy()

        obs_norm    = obs_raw    - offset
        gt_norm     = gt_raw     - offset
        preds1_norm = preds1_raw - offset
        preds2_norm = preds2_raw - offset

        def compute_min_ade(preds, gt):
            gt_exp = torch.tensor(gt).unsqueeze(0).expand(6, -1, -1)
            ade = torch.norm(torch.tensor(preds) - gt_exp, dim=-1).mean(dim=-1)
            return ade.min().item()

        ade1 = compute_min_ade(preds1_raw, gt_raw)
        ade2 = compute_min_ade(preds2_raw, gt_raw)

        for row, (preds_norm, phase, ade) in enumerate([
            (preds1_norm, "Phase 1", ade1),
            (preds2_norm, "Phase 2", ade2)
        ]):
            ax = fig.add_subplot(gs[row, collected])

            for k in range(6):
                ax.plot(preds_norm[k, :, 0], preds_norm[k, :, 1],
                        color='lightcoral', linewidth=1.2, alpha=0.6)

            ax.plot(obs_norm[:, 0], obs_norm[:, 1],
                    'b-o', linewidth=2, markersize=4, label='obs')

            ax.plot(gt_norm[:, 0], gt_norm[:, 1],
                    'g--o', linewidth=2, markersize=4, label='gt')

            ax.plot(0, 0, 'ko', markersize=6)

            ax.set_title(f'{phase}\nADE: {ade:.2f}m', fontsize=8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            if collected == 0:
                ax.legend(fontsize=6)

        collected += 1

    plt.suptitle(
        'Phase 1 (frozen VLM) vs Phase 2 (LoRA)\n'
        'Blue=observation, Green=ground truth, Red=predictions',
        fontsize=12
    )
    out_path = 'comparison_phase1_vs_phase2.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {out_path}", flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1_ckpt', default='/home/student02/data/checkpoints/best_model.pt')
    parser.add_argument('--phase2_ckpt', default='/home/student02/data/checkpoints/phase2/best_model.pt')
    parser.add_argument('--dataroot',    default='/home/student02/data/nuscenes')
    parser.add_argument('--n_examples',  type=int, default=8)
    args = parser.parse_args()

    visualize_comparison(
        phase1_ckpt=args.phase1_ckpt,
        phase2_ckpt=args.phase2_ckpt,
        dataroot=args.dataroot,
        n_examples=args.n_examples
    )