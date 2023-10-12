<div align="center">
  <img src="./assets/logo.png" width="30%">
</div>
<h1 align="center">💃DreamWaltz: Make a Scene with Complex 3D Animatable Avatars</h1>
<p align="center">

### [Project Page](https://dreamwaltz3d.github.io/) | [Arxiv](https://arxiv.org/abs/2305.12529) | [Video]()

This repository contains the official implementation of NeurIPS 2023 paper:
> DreamWaltz: Make a Scene with Complex 3D Animatable Avatars
<br>[Yukun Huang](https://github.com/hyk1996/)<sup>1,2</sup>, [Jianan Wang](https://github.com/wendyjnwang/)<sup>1</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>1</sup>, [He Cao](https://github.com/CiaoHe/)<sup>1</sup>, [Xianbiao Qi](http://scholar.google.com/citations?user=odjSydQAAAAJ&hl=zh-CN/)<sup>1</sup>, Yukai Shi<sup>1</sup>, Zheng-Jun Zha<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>1</sup>International Digital Economy Academy &nbsp; <sup>2</sup>University of Science and Technology of China


## Introduction
DreamWaltz is a learning framework for text-driven 3D animatable avatar creation using pretrained 2D diffusion model [ControlNet](https://github.com/lllyasviel/ControlNet) and human parametric model [SMPL](https://github.com/vchoutas/smplx). The core idea is to optimize a deformable NeRF representation from skeleton-conditioned diffusion supervisions, which ensures 3D consistency and generalization to arbitrary poses.

<p align="middle">
<img src="assets/teaser.gif" width="80%">
<br>
<em>Figure 1. DreamWaltz can generate animatable avatars (a) and construct complex scenes (b)(c)(d).</em>
</p>

## Installation

This code is heavily based on the excellent [latent-nerf](https://github.com/eladrich/latent-nerf) and [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) projects. Please install dependencies:
```
pip install -r requirements.txt
```
The cuda extension for instant-ngp is built at runtime as in [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion#build-extension-optional).

## Prepare SMPL Weights

We use smpl and vposer models for avatar creation and animation learning, please follow the instructions in [smplx](https://github.com/vchoutas/smplx#downloading-the-model) and [human_body_prior](https://github.com/nghorbani/human_body_prior) to download the model weights, and build a directory with the following structure:
```
smpl_models
├── smpl
│   ├── SMPL_FEMALE.pkl
│   └── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
└── vposer
    └── v2.0
        ├── snapshots
        ├── V02_05.yaml
        └── V02_05.log
```
Then, update the model paths `SMPL_ROOT` and `VPOSER_ROOT` in `configs/paths.py`.

## Prepare Motion Sequences
You might need to prepare SMPL-format human motion sequences to animate the generated avatars. Our code provide a data api for [AIST++](https://google.github.io/aistplusplus_dataset), which is a high-quality dance video database with SMPL annotations. Please download the SMPL annotations from [this website](https://google.github.io/aistplusplus_dataset/download.html), build a directory with the following structure:
```
aist
├── gWA_sFM_cAll_d26_mWA5_ch13.pkl
├── gWA_sFM_cAll_d27_mWA0_ch15.pkl
├── gWA_sFM_cAll_d27_mWA2_ch17.pkl
└── ...
```
and update the data path `AIST_ROOT` in `configs/paths.py`.

## Getting Started

DreamWaltz mainly consists of two training stages: (I) _Canonical Avatar Creation_ and (II) _Animatable Avatar Learning_.

The following commands are also provided in `run.sh`.

#### 1. SMPL-Guided NeRF Initialization

To pretrain NeRF using mask images rendered from canonical-posed SMPL mesh:
```bash
python train.py \
  --log.exp_name "pretrained" \
  --log.pretrain_only True \
  --prompt.scene canonical-A \
  --prompt.smpl_prompt depth \
  --optim.iters 10000
```
The obtained pretrained ckpt is available to different text prompts.

#### 2. Canonical Avatar Creation

To learn a NeRF-based canonical avatar representation using ControlNet-based SDS:

```bash
text="a wooden robot"
avatar_name="wooden_robot"
pretrained_ckpt="./outputs/pretrained/checkpoints/step_010000.pth"
# the pretrained ckpt is available to different text prompts

python train.py \
  --guide.text "${text}" \
  --log.exp_name "canonical/${avatar_name}" \
  --optim.ckpt "${pretrained_ckpt}" \
  --optim.iters 30000 \
  --prompt.scene canonical-A
```

#### 3. Animatable Avatar Learning

To learn a NeRF-based animatable avatar representation using ControlNet-based SDS:

```bash
text="a wooden robot"
avatar_name="wooden_robot"
canonical_ckpt="./outputs/canonical/${avatar_name}/checkpoints/step_030000.pth"

python train.py \
  --animation True \
  --guide.text "${text}" \
  --log.exp_name "animatable/${avatar_name}" \
  --optim.ckpt "${canonical_ckpt}" \
  --optim.iters 50000 \
  --prompt.scene random \
  --render.cuda_ray False
```

#### 4. Make a Dancing Video

To make a dancing video based on the well-trained animatable avatar representation and the target motion sequences:

```bash
scene="gWA_sFM_cAll_d27_mWA2_ch17,180-280"
# "gWA_sFM_cAll_d27_mWA2_ch17" is the filename of motion sequences in AIST++
# "180-280" is the range of video frame indices: [180, 280]

avatar_name="wooden_robot"
animatable_ckpt="./outputs/animatable/${avatar_name}/checkpoints/step_050000.pth"

python train.py \
    --animation True \
    --log.eval_only True \
    --log.exp_name "videos/${avatar_name}" \
    --optim.ckpt "${animatable_ckpt}" \
    --prompt.scene "${scene}" \
    --render.cuda_ray False \
    --render.eval_fix_camera True
```
The resulting video can be found in `PROJECT_ROOT/outputs/videos/${avatar_name}/results/128x128/`.

## Results

### Canonical Avatars

<p align="middle">
<image src="assets/canonical_half.gif" width="80%">
<br>
<em>Figure 2. DreamWaltz can create canonical avatars from textual descriptions.</em>
</p>

### Animatable Avatars

<p align="middle">
<image src="assets/animation_sp.gif" width="80%">
<br>
<em>Figure 3. DreamWaltz can animate canonical avatars given motion sequences.</em>
</p>
  
### Complex Scenes

<p align="middle">
<image src="assets/animation_obj.gif" width="80%">
<br>
<em>Figure 4. DreamWaltz can make complex 3D scenes with avatar-object interactions.</em>
</p>

<p align="middle">
<image src="assets/animation_scene.gif" width="80%">
<br>
<em>Figure 5. DreamWaltz can make complex 3D scenes with avatar-scene interactions.</em>
</p>

<p align="middle">
<image src="assets/animation_mp.gif" width="80%">
<br>
<em>Figure 6. DreamWaltz can make complex 3D scenes with avatar-avatar interactions.</em>
</p>

## Reference
If you find this repository useful for your work, please consider citing it as follows:
```bibtex
@article{huang2023dreamwaltz,
 title={DreamWaltz: Make a Scene with Complex 3D Animatable Avatars},
 author={Yukun Huang and Jianan Wang and Ailing Zeng and He Cao and Xianbiao Qi and Yukai Shi and Zheng-Jun Zha and Lei Zhang},
 year={2023},
 eprint={2305.12529},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}

@article{huang2023dreamtime,
 title={DreamTime: An Improved Optimization Strategy for Text-to-3D Content Creation},
 author={Yukun Huang and Jianan Wang and Yukai Shi and Xianbiao Qi and Zheng-Jun Zha and Lei Zhang},
 year={2023},
 eprint={2306.12422},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}
