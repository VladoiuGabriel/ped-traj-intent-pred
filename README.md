# Generative Pedestrian Trajectory Prediction with Vision-Language Context

Predict where a pedestrian will walk next from a few observed positions and the front-camera image, using flow matching and a vision-language model for scene context.

<p align="center">
  <img src="docs/architecture.png" width="900" alt="Architecture overview">
</p>

## Context

This repository contains the code developed to obtain the results reported in the paper *Generative Pedestrian Trajectory Prediction with Vision-Language Context for Autonomous Vehicles*, accepted at IEEE ICCP 2026 (Cluj-Napoca).

📄 [Paper on ResearchGate](https://www.researchgate.net/publication/416978796) · IEEE Xplore link to be added once available

## Overview

The same past motion can lead to many plausible futures, and the outcome depends on the scene (crosswalk, parked cars, other people). This project treats prediction as a **generative** problem:

- **Flow matching** learns a velocity field that transports Gaussian noise to a 12-waypoint future trajectory, so the model samples several diverse futures instead of one average path.
- **FlowDiT**, a small Diffusion-Transformer-style network, denoises the trajectory: self-attention over the waypoint tokens, cross-attention to a context vector, and adaLN conditioning on the flow time.
- **Vision-language context.** Qwen2.5-VL-3B reads the front-camera image and produces a single **[PLAN] token**. Its last-layer hidden state is projected to a 256-d scene vector and fused with an encoding of the observed motion.
- **Three-stage training:** (1) motion only, (2) frozen VLM with the projector, (3) LoRA fine-tuning of the VLM (r=16 on Q, K, V, O, with a 10x lower learning rate).
- **Cartesian vs. polar** trajectory representations are both supported. Cartesian works better with a Gaussian prior.

## Results

The numbers below are the ones reported in the paper. They are evaluated on nuScenes pedestrians: 4 observed steps → 12 predicted steps at 2 Hz. Metrics in metres; minADE/minFDE over K samples drawn from one pool of 10.

| Model | minADE₁ | minADE₁₀ | minFDE₁ | minFDE₁₀ |
|---|---|---|---|---|
| Cartesian, motion only | 0.844 | 0.402 | 1.719 | 0.710 |
| Polar, motion only | 0.885 | 0.406 | 1.842 | 0.743 |
| **Cartesian + VLM** | **0.822** | **0.395** | **1.702** | **0.707** |
| Polar + VLM | 0.877 | 0.401 | 1.822 | 0.741 |

Adding the vision-language context lowers minADE by 2.6% (K=1), 4.0% (K=5) and 1.7% (K=10).

## Repository structure

```
src/
  dataloader.py   # extracts pedestrian tracks from nuScenes
  dataset.py      # builds (image, observed, future) samples
  model.py        # planning-token projector, FlowDiT, full model
  train.py        # training
  visualize*.py   # qualitative plots
```

## Setup

```bash
git clone https://github.com/VladoiuGabriel/ped-traj-intent-pred.git
cd ped-traj-intent-pred
pip install torch transformers qwen-vl-utils nuscenes-devkit peft matplotlib
```

Download [nuScenes](https://www.nuscenes.org/nuscenes) (full trainval for the full experiments, `v1.0-mini` for a quick test). The Qwen2.5-VL-3B weights are downloaded automatically from Hugging Face on first run.

## Training

```bash
python src/train.py \
  --dataroot /path/to/nuscenes \
  --save_dir /path/to/checkpoints \
  --version v1.0-trainval
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{vladoiu2026generative,
  title     = {Generative Pedestrian Trajectory Prediction with Vision-Language Context for Autonomous Vehicles},
  author    = {Vladoiu, Cristian-Gabriel and Di Bella, Leandro and Jurca, Mihnea-Bogdan and Munteanu, Adrian},
  booktitle = {IEEE International Conference on Intelligent Computer Communication and Processing (ICCP)},
  year      = {2026},
  address   = {Cluj-Napoca, Romania}
}
```

## Acknowledgements

Built on [nuScenes](https://www.nuscenes.org/) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).
