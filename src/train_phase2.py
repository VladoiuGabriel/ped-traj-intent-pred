import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from nuscenes.nuscenes import NuScenes
import wandb

from dataset import PedestrianDataset, collate_fn
from model import PedTrajModel

import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',         default='/home/student02/data/nuscenes')
    parser.add_argument('--save_dir', default='/home/student02/data/checkpoints/phase2_v2')
    parser.add_argument('--phase1_ckpt',      default='/home/student02/data/checkpoints/phase1_v2/best_model.pt')
    parser.add_argument('--version',          default='v1.0-mini')
    parser.add_argument('--vlm_name',         default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--batch_size',       type=int,   default=4)
    parser.add_argument('--epochs',           type=int,   default=50)
    parser.add_argument('--lr_flow',          type=float, default=3e-4)
    parser.add_argument('--lr_lora',          type=float, default=3e-5)
    parser.add_argument('--lora_rank',        type=int,   default=16)
    parser.add_argument('--lora_alpha',       type=int,   default=16)
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

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    is_main = local_rank == 0

    if is_main:
        wandb.init(
            project="ped-traj-pred",
            name="phase2_ddp",
            config=CONFIG
        )

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    if is_main:
        print("loading nuScenes...", flush=True)
    nusc = NuScenes(
        version=CONFIG['version'],
        dataroot=CONFIG['dataroot'],
        verbose=False
    )

    if is_main:
        print("building dataset...", flush=True)
    full_dataset = PedestrianDataset(nusc, CONFIG['dataroot'])

    val_size   = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    if is_main:
        print(f"train: {train_size} | val: {val_size}", flush=True)

    train_sampler = DistributedSampler(train_dataset)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    if is_main:
        print("loading model with lora...", flush=True)
    model = PedTrajModel(
        device=device,
        vlm_name=CONFIG['vlm_name'],
        sigma=CONFIG['sigma'],
        waypoint_dropout=CONFIG['waypoint_dropout'],
        use_lora=True,
        lora_rank=CONFIG['lora_rank'],
        lora_alpha=CONFIG['lora_alpha']
    )
    model.projector = model.projector.to(device)
    model.flow      = model.flow.to(device)

    if is_main:
        print("loading phase 1 checkpoint...", flush=True)
    ckpt = torch.load(CONFIG['phase1_ckpt'], map_location=device)
    model.projector.load_state_dict(ckpt['projector'])
    model.flow.load_state_dict(ckpt['flow'])
    if is_main:
        print(f"loaded phase 1 checkpoint from epoch {ckpt['epoch']}", flush=True)

    model.projector = DDP(model.projector, device_ids=[local_rank])
    model.flow      = DDP(model.flow,      device_ids=[local_rank])

    lora_params = [p for p in model.vlm.parameters() if p.requires_grad]
    flow_params  = (
        list(model.projector.parameters()) +
        list(model.flow.parameters())
    )

    if is_main:
        print(f"lora params: {sum(p.numel() for p in lora_params)/1e6:.2f}M", flush=True)
        print(f"flow params: {sum(p.numel() for p in flow_params)/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW([
        {'params': flow_params, 'lr': CONFIG['lr_flow']},
        {'params': lora_params, 'lr': CONFIG['lr_lora']}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs']
    )

    best_val_loss = float('inf')

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_sampler.set_epoch(epoch)
        model.projector.train()
        model.flow.train()
        model.vlm.train()

        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            obs     = batch['obs'].to(device)
            pred_gt = batch['pred_gt'].to(device)
            images  = batch['images']

            optimizer.zero_grad()
            loss = model(images, obs, pred_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                flow_params + lora_params, max_norm=1.0
            )
            optimizer.step()

            train_loss += loss.item()

            if is_main and batch_idx % CONFIG['log_every'] == 0:
                print(f"  epoch {epoch} | batch {batch_idx}/{len(train_loader)} "
                      f"| loss: {loss.item():.4f}", flush=True)
                wandb.log({'train_loss_step': loss.item()})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.projector.eval()
        model.flow.eval()
        model.vlm.eval()

        val_loss = 0.0
        val_ade  = 0.0
        val_fde  = 0.0
        n_val    = 0

        if is_main:
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

            wandb.log({
                'epoch':      epoch,
                'train_loss': avg_train_loss,
                'val_loss':   avg_val_loss,
                'minADE':     avg_ade,
                'minFDE':     avg_fde
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')
                torch.save({
                    'epoch':      epoch,
                    'projector':  model.projector.module.state_dict(),
                    'flow':       model.flow.module.state_dict(),
                    'vlm_lora':   model.vlm.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'val_loss':   avg_val_loss,
                    'ade':        avg_ade,
                    'fde':        avg_fde,
                }, ckpt_path)
                print(f"saved best model at {ckpt_path}", flush=True)

        dist.barrier()

    if is_main:
        wandb.finish()
        print(f"\ntraining complete, best val loss: {best_val_loss:.4f}")

    dist.destroy_process_group()


if __name__ == '__main__':
    train()