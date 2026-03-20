import numpy as np
from nuscenes.nuscenes import NuScenes


def get_pedestrian_trajectories(nusc, min_obs=4, min_pred=6):
    """
    extract pedestrian trajectories from nuScenes

    args:
        nusc: NuScenes object
        min_obs: minimum observation frames-4
        min_pred: minimum prediction frames-6

    returns:
        list of dicts with obs and pred trajectories
    """
    trajectories = []
    min_len = min_obs + min_pred

    for instance in nusc.instance:
        category = nusc.get('category', instance['category_token'])
        if not category['name'].startswith('human.pedestrian'):
            continue

        positions = []
        ann_token = instance['first_annotation_token']

        while ann_token:
            ann = nusc.get('sample_annotation', ann_token)
            x, y, _ = ann['translation']
            positions.append([x, y])
            ann_token = ann['next']

        if len(positions) < min_len:
            continue

        positions = np.array(positions)

        for start in range(len(positions) - min_len + 1):
            obs = positions[start : start + min_obs]
            pred = positions[start + min_obs : start + min_obs + min_pred]
            trajectories.append({
                'obs': obs,
                'pred': pred,
                'instance_token': instance['token']
            })

    return trajectories


if __name__ == '__main__':
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes',
        verbose=False
    )

    trajs = get_pedestrian_trajectories(nusc)

    print(f"Extracted trajectories: {len(trajs)}")
    print(f"\nFirst trajectory example:")
    print(f"  obs shape:  {trajs[0]['obs'].shape}")
    print(f"  pred shape: {trajs[0]['pred'].shape}")
    print(f"  obs (x,y):\n{trajs[0]['obs']}")
    print(f"  pred (x,y):\n{trajs[0]['pred']}")

    moving = 0
    stationary = 0

    for traj in trajs:
        all_pos = np.vstack([traj['obs'], traj['pred']])
        displacement = np.max(np.linalg.norm(all_pos - all_pos[0], axis=1))
        if displacement > 0.5:
            moving += 1
        else:
            stationary += 1

    print(f"\nTotal trajectories: {len(trajs)}")
    print(f"Moving (>0.5m):     {moving}")
    print(f"Stationary:         {stationary}")