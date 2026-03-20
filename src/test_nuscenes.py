from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.0-mini',
    dataroot=r'C:\Users\Gabi\pedtraj\data\nuscenes',
    verbose=True
)

print(f"\n--- stats ---")
print(f"scenes: {len(nusc.scene)}")
print(f"samples: {len(nusc.sample)}")
print(f"total n of instances: {len(nusc.instance)}")

pedestrians = [
    inst for inst in nusc.instance
    if nusc.get('category', inst['category_token'])['name'].startswith('human.pedestrian')
]
print(f"pedestrian instances: {len(pedestrians)}")