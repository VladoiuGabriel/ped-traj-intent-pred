import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from dataloader import get_pedestrian_trajectories
import os


def cartesian_to_polar(coords):
    r = np.linalg.norm(coords, axis=-1, keepdims=True)
    theta = np.arctan2(coords[..., 1:2], coords[..., 0:1])
    return np.concatenate([r, theta], axis=-1)


class PedestrianDataset(Dataset):
    """
    PyTorch dataset for pedestrian trajectory prediction
    each sample contains:
      - image:   CAM_FRONT image at the last observed frame
      - obs:     (4, 2) observed trajectory in polar coords (r, theta)
      - pred_gt: (6, 2) ground truth future trajectory in polar coords (r, theta)
    """

    def __init__(self, nusc, dataroot, only_moving=True, min_displacement=0.5):
        self.nusc = nusc
        self.dataroot = dataroot

        raw_trajs = get_pedestrian_trajectories(nusc)

        self.samples = []
        for traj in raw_trajs:
            if only_moving:
                all_pos = np.vstack([traj['obs'], traj['pred']])
                displacement = np.max(
                    np.linalg.norm(all_pos - all_pos[0], axis=1)
                )
                if displacement < min_displacement:
                    continue

            img_path = self._get_cam_front_path(traj['instance_token'], traj['obs'])
            if img_path is None:
                continue

            origin = traj['obs'][-1].copy()
            obs_norm  = traj['obs']  - origin
            pred_norm = traj['pred'] - origin

            obs_polar  = cartesian_to_polar(obs_norm)
            pred_polar = cartesian_to_polar(pred_norm)

            self.samples.append({
                'img_path': img_path,
                'obs':      obs_polar.astype(np.float32),
                'pred_gt':  pred_polar.astype(np.float32),
                'origin':   origin.astype(np.float32)
            })

        print(f"Dataset built: {len(self.samples)} samples")

    def _get_cam_front_path(self, instance_token, obs_positions):
        try:
            instance = self.nusc.get('instance', instance_token)
            ann_token = instance['first_annotation_token']

            n_obs = len(obs_positions)
            count = 0
            last_ann = None

            while ann_token:
                ann = self.nusc.get('sample_annotation', ann_token)
                if count == n_obs - 1:
                    last_ann = ann
                    break
                ann_token = ann['next']
                count += 1

            if last_ann is None:
                return None

            sample = self.nusc.get('sample', last_ann['sample_token'])

            cam_token = sample['data']['CAM_FRONT']
            cam_data  = self.nusc.get('sample_data', cam_token)
            img_path  = os.path.join(self.dataroot, cam_data['filename'])

            return img_path if os.path.exists(img_path) else None

        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['img_path']).convert('RGB')

        return {
            'image':   image,
            'obs':     torch.tensor(sample['obs']),
            'pred_gt': torch.tensor(sample['pred_gt']),
            'origin':  torch.tensor(sample['origin'])
        }


def collate_fn(batch):
    return {
        'images':  [b['image']  for b in batch],
        'obs':     torch.stack([b['obs']     for b in batch]),
        'pred_gt': torch.stack([b['pred_gt'] for b in batch]),
        'origin':  torch.stack([b['origin']  for b in batch])
    }


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes',
        verbose=False
    )

    dataset = PedestrianDataset(
        nusc,
        dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes'
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))
    print(f"\nFirst batch:")
    print(f"  images:  {len(batch['images'])} PIL Images")
    print(f"  obs:     {batch['obs'].shape}")
    print(f"  pred_gt: {batch['pred_gt'].shape}")
    print(f"  origin:  {batch['origin'].shape}")