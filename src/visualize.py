import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from dataloader import get_pedestrian_trajectories

def plot_trajectories(trajs, n=20, only_moving=True, min_displacement=0.5):
    """
    Visualize pedestrian trajectories in BEV (Bird's Eye View).
    One trajectory per unique pedestrian instance.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    seen_instances = set()
    plotted = 0

    for traj in trajs:
        if plotted >= n:
            break

      
        if traj['instance_token'] in seen_instances:
            continue

        all_pos = np.vstack([traj['obs'], traj['pred']])
        displacement = np.max(np.linalg.norm(all_pos - all_pos[0], axis=1))

        if only_moving and displacement < min_displacement:
            continue

        seen_instances.add(traj['instance_token'])

        obs  = traj['obs']
        pred = traj['pred']

        origin = obs[0].copy()
        obs_norm  = obs  - origin
        pred_norm = pred - origin

        ax.plot(obs_norm[:, 0], obs_norm[:, 1],
                'b-o', linewidth=2, markersize=4, alpha=0.7)

        ax.plot(pred_norm[:, 0], pred_norm[:, 1],
                'g--o', linewidth=2, markersize=4, alpha=0.7)

        ax.plot(0, 0, 'ko', markersize=6)

        plotted += 1

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Pedestrian trajectories BEV — {plotted} unique pedestrians\n'
                 f'Blue = observation (2s), Green = ground truth (3s)')
    ax.legend(['Observation', 'Ground truth', 'Start'], loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('trajectories_bev.png', dpi=150)
    plt.show()
    print("Saved: trajectories_bev.png")


if __name__ == '__main__':
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes',
        verbose=False
    )

    trajs = get_pedestrian_trajectories(nusc)
    plot_trajectories(trajs, n=20, only_moving=True)