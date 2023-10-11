import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger

from configs import RenderConfig
from .network import NeRFNetwork

from .deformation.skeletal_motion import SkeletalMotion


class DreamWaltz(NeRFNetwork):
    def __init__(self, cfg: RenderConfig, smpl, smpl_params_cnl, **cfg_extra):
        super().__init__(cfg, **cfg_extra)
        assert cfg.cuda_ray is False
        self.cfg = cfg
        self.cfg_extra = cfg_extra
        # Module
        self.skeletal_motion = SkeletalMotion(cfg, smpl, smpl_params_cnl)
        # Dynamic Vars
        self.smpl_data_obs = None
        if not cfg.joint_train:
            self.set_requires_grad(self, False)

    @staticmethod
    def set_requires_grad(self, flag):
        self.encoder.requires_grad_(flag)
        self.sigma_net.requires_grad_(flag)
        self.sigma_scale.requires_grad_(flag)
        if self.bg_radius > 0:
            self.encoder_bg.requires_grad_(flag)
            self.bg_net.requires_grad_(flag)

    def common_forward(self, x, **kwargs):
        """
        Inputs:
            - x: [N, 3], in [-bound, bound]
            - smpl_data_obs: dict, {}
        """
        x, extra_outputs = self.skeletal_motion(x, self.smpl_data_obs)
        return super().common_forward(x, **kwargs, **extra_outputs)

    def common_forward_vanilla(self, x, **kwargs):
        return super().common_forward(x, **kwargs)

    def get_params(self, lr):
        params = [
            {'params': self.skeletal_motion.parameters(), 'lr': lr},
        ]
        if self.cfg.joint_train:
            # Get Current NeRF Params
            params += [
                {'params': self.encoder.parameters(), 'lr': lr * 10},
                {'params': self.sigma_net.parameters(), 'lr': lr},
                {'params': self.sigma_scale, 'lr': lr},
            ]
            if self.bg_radius > 0:
                params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
                params.append({'params': self.bg_net.parameters(), 'lr': lr})
        return params
