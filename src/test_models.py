import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#clip
print("\nLoading CLIP-ViT-B/16...")
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
clip_model = clip_model.to(device)
clip_model.eval()

#freeze clip
for param in clip_model.parameters():
    param.requires_grad = False

print(f"CLIP loaded — parameters: {sum(p.numel() for p in clip_model.parameters()) / 1e6:.1f}M (frozen)")
print(f"VRAM after CLIP: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

#qwen
print("\nLoading Qwen2.5-1.5B...")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.float16, 
    device_map="auto"
)

print(f"Qwen2.5 loaded — parameters: {sum(p.numel() for p in qwen_model.parameters()) / 1e6:.1f}M")
print(f"VRAM after Qwen2.5: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

#fw pass test
print("\nTesting forward pass...")

#dummy image
from PIL import Image
import numpy as np
dummy_img = Image.fromarray(np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8))
inputs = clip_processor(images=dummy_img, return_tensors="pt").to(device)

with torch.no_grad():
    clip_output = clip_model(**inputs)
    patch_tokens = clip_output.last_hidden_state 

print(f"CLIP output shape: {patch_tokens.shape}")

#dummy text
tokens = qwen_tokenizer("A pedestrian is walking.", return_tensors="pt").to(device)
with torch.no_grad():
    qwen_output = qwen_model(**tokens, output_hidden_states=True)
    last_hidden = qwen_output.hidden_states[-1] 

print(f"Qwen2.5 hidden state shape: {last_hidden.shape}")
print(f"\nFinal VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("\n Ok")