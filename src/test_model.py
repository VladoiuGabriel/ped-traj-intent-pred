import torch
from PIL import Image
import numpy as np
from model import PedTrajModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


print("\nloading model...")
model = PedTrajModel(device=device)
model.dit = model.dit.to(device)
model.bridge = model.bridge.to(device)
model.clip = model.clip.to(device)


dummy_image = Image.fromarray(
    np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
)
obs = torch.randn(2, 4, 2).to(device)      
pred_gt = torch.randn(2, 6, 2).to(device)  


print("\ntesting training forward pass...")
loss = model(dummy_image, obs, pred_gt)
print(f"loss: {loss.item():.4f}")


print("\ntesting inference...")
obs_single = torch.randn(1, 4, 2).to(device)
predictions = model.predict(dummy_image, obs_single, n_samples=6)
print(f"predictions shape: {predictions.shape}") 

print(f"\nVRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("\n Ok")