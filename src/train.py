import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, random_split
from nuscenes.nuscenes import NuScenes

from dataset import PedestrianDataset, collate_fn
from model import PedTrajModel

import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',         default='/home/student02/data/nuscenes')
    parser.add_argument('--save_dir',         default='/home/student02/data/checkpoints')
    parser.add_argument('--version',          default='v1.0-mini')
    parser.add_argument('--vlm_name',         default='Qwen/Qwen2-VL-3B-Instruct')
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--epochs',           type=int,   default=50)
    parser.add_argument('--lr',               type=float, default=3e-4)
    parser.add_argument('--log_every',        type=int,   default=10)
    parser.add_argument('--val_split',        type=float, default=0.2)
    parser.add_argument('--sigma',            type=float, default=0.1)
    parser.add_argument('--waypoint_dropout', type=float, default=0.15)
    args = parser.parse_args()
    return vars(args)


def compute_ade(pred, gt):
    gt_exp = gt.unsqueeze(0).expand_as(pred)
    ade_per_sample = torch.norm(pred - gt_exp, dim=-1).mean(dim=-1)
    return ade_per_sample.min().item()


def compute_fde(pred, gt):
    fde_per_sample = torch.norm(pred[:, -1, :] - gt[-1, :], dim=-1)
    return fde_per_sample.min().item()


def train():
    CONFIG = get_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}", flush=True)

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    print("loading nuScenes...", flush=True)
    nusc = NuScenes(
        version=CONFIG['version'],
        dataroot=CONFIG['dataroot'],
        verbose=False
    )

    print("building dataset...", flush=True)
    full_dataset = PedestrianDataset(nusc, CONFIG['dataroot'])

    val_size   = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"train: {train_size} | val: {val_size}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, collate_fn=collate_fn
    )

    print("loading model...", flush=True)
    model = PedTrajModel(
        device=device,
        vlm_name=CONFIG['vlm_name'],
        sigma=CONFIG['sigma'],
        waypoint_dropout=CONFIG['waypoint_dropout']
    )
    model.projector = model.projector.to(device)
    model.flow      = model.flow.to(device)

    trainable_params = (
        list(model.projector.parameters()) +
        list(model.flow.parameters())
    )
    print(f"trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs']
    )

    best_val_loss = float('inf')

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.projector.train()
        model.flow.train()
        model.vlm.eval()

        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            obs     = batch['obs'].to(device)
            pred_gt = batch['pred_gt'].to(device)
            images  = batch['images']

            optimizer.zero_grad()
            loss = model(images, obs, pred_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % CONFIG['log_every'] == 0:
                print(f"  epoch {epoch} | batch {batch_idx}/{len(train_loader)} "
                      f"| loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.projector.eval()
        model.flow.eval()

        val_loss = 0.0
        val_ade  = 0.0
        val_fde  = 0.0
        n_val    = 0

        with torch.no_grad():
            for batch in val_loader:
                obs     = batch['obs'].to(device)
                pred_gt = batch['pred_gt'].to(device)
                images  = batch['images']

                loss = model(images, obs, pred_gt)
                val_loss += loss.item()

                preds = model.predict(images[:1], obs[:1], n_samples=6)
                val_ade += compute_ade(preds, pred_gt[0])
                val_fde += compute_fde(preds, pred_gt[0])
                n_val   += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_ade      = val_ade  / n_val
        avg_fde      = val_fde  / n_val

        print(f"\nepoch {epoch}/{CONFIG['epochs']} | "
              f"train: {avg_train_loss:.4f} | "
              f"val: {avg_val_loss:.4f} | "
              f"minADE: {avg_ade:.3f}m | "
              f"minFDE: {avg_fde:.3f}m\n", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')
            torch.save({
                'epoch':      epoch,
                'projector':  model.projector.state_dict(),
                'flow':       model.flow.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   avg_val_loss,
                'ade':        avg_ade,
                'fde':        avg_fde,
            }, ckpt_path)
            print(f"saved best model at {ckpt_path}", flush=True)

    print(f"\ntraining complete, best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()