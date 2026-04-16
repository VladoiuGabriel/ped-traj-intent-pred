import sys
import os
import torch
import argparse

def inspect(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"epoch:     {ckpt['epoch']}")
    print(f"val_loss:  {ckpt['val_loss']:.4f}")
    print(f"minADE:    {ckpt['ade']:.3f}m")
    print(f"minFDE:    {ckpt['fde']:.3f}m")
    print(f"keys:      {[k for k in ckpt.keys()]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    args = parser.parse_args()
    inspect(args.ckpt)