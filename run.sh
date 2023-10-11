#!/bin/bash

# ---------- SMPL-Guided NeRF Pretraining ---------- #
python train.py \
  --log.exp_name "pretrained" \
  --log.pretrain_only True \
  --prompt.scene canonical-A \
  --prompt.smpl_prompt depth \
  --optim.iters 10000


# ---------- Text-to-Avatar Generation ---------- #
text="a wooden robot"
avatar_name="wooden_robot"

pretrained_ckpt="./outputs/pretrained/checkpoints/step_010000.pth"
canonical_ckpt="./outputs/canonical/${avatar_name}/checkpoints/step_030000.pth"
animatable_ckpt="./outputs/animatable/${avatar_name}/checkpoints/step_050000.pth"

# Canonical Avatar Creation
python train.py \
  --guide.text "${text}" \
  --log.exp_name "canonical/${avatar_name}" \
  --optim.ckpt "${pretrained_ckpt}" \
  --optim.iters 30000 \
  --prompt.scene canonical-A

# Animatable Avatar Learning
python train.py \
  --animation True \
  --guide.text "${text}" \
  --log.exp_name "animatable/${avatar_name}" \
  --optim.ckpt "${canonical_ckpt}" \
  --optim.iters 50000 \
  --prompt.scene random \
  --render.cuda_ray False


# ---------- Make a Dancing Video ---------- #
scene="gWA_sFM_cAll_d27_mWA2_ch17,180-280"
# "gWA_sFM_cAll_d27_mWA2_ch17" is the filename of motion sequences in AIST++
# "180-280" is the range of video frame indices: [180, 280]

python train.py \
    --animation True \
    --log.eval_only True \
    --log.exp_name "videos/${avatar_name}" \
    --optim.ckpt "${animatable_ckpt}" \
    --prompt.scene "${scene}" \
    --render.cuda_ray False \
    --render.eval_fix_camera True
