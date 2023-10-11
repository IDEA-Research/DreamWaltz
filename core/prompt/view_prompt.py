import os
import random
from configs.train_config import RenderConfig
import numpy as np
import torch


class ViewPrompt:
    def __init__(self, cfg: RenderConfig, mode='sjc') -> None:
        # Augmentation Mode: Latent-NeRF
        # Augmentation Mode: SJC
        if mode in ('sjc', 'dreamfusion-v3', 'mix'):
            self.views = ['front', 'side', 'backside', 'side', 'overhead', 'bottom']
        # Augmentation Mode: DreamFusion
        elif mode in ('latent-nerf', 'stable-dreamfusion', 'dreamfusion', 'dreamfusion-v2'):
            self.views = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
        else:
            raise NotImplementedError
        self.mode = mode
        self.optional_patterns = (
            "{0}, {1} view",
            "{1} view of {0}",
        )
        # Angle Range
        angle_front = np.deg2rad(cfg.angle_front)
        angle_overhead = np.deg2rad(cfg.angle_overhead)
        self.azimuth_range = [
            angle_front/2,
            np.pi - angle_front/2,
            np.pi + angle_front/2,
            2*np.pi - angle_front/2,
        ]
        self.pitch_range = [
            angle_overhead,
            np.pi - angle_overhead,
        ]

    @property
    def pattern(self):
        if self.mode in ('stable-dreamfusion', 'latent-nerf', 'dreamfusion', 'dreamfusion-v3'):
            return self.optional_patterns[0]
        elif self.mode in ('sjc', 'dreamfusion-v2'):
            return self.optional_patterns[1]
        elif self.mode == 'mix':
            return random.choice(self.optional_patterns)
        else:
            raise NotImplementedError

    def get_view_direction(self, thetas, phis):
        #                   phis [B,];          thetas: [B,]
        # front = 0         [0, front)
        # side (left) = 1   [front, 180)
        # back = 2          [180, 180+front)
        # side (right) = 3  [180+front, 360)
        # top = 4                               [0, overhead]
        # bottom = 5                            [180-overhead, 180]

        # init
        pitch = self.pitch_range
        azimuth = self.azimuth_range

        res = torch.zeros(thetas.shape[0], dtype=torch.long)

        # first determine by phis
        res[phis >= azimuth[3] or phis < azimuth[0]] = 0
        res[azimuth[0] <= phis < azimuth[1]] = 1
        res[azimuth[1] <= phis < azimuth[2]] = 2
        res[azimuth[2] <= phis < azimuth[3]] = 3

        # override by thetas
        res[thetas <= pitch[0]] = 4  # overhead
        if 'bottom' in self.views:
            res[thetas >= pitch[1]] = 5  # bottom

        return res

    def get_a_batch(self, batch_size, device):
        
        pitch = self.pitch_range
        azimuth = self.azimuth_range

        dirs, phis, thetas = [], [], []

        assert batch_size <= len(self.views)

        for i in range(len(self.views)):

            dirs.append(i)

            if i == 0:
                phi = (2 * torch.rand(1, device=device) - 1) * azimuth[0]
                phi[phi < 0] += 2 * np.pi
                theta = torch.rand(1, device=device) * (pitch[1] - pitch[0]) + pitch[0]
            elif i == 1:
                phi = torch.rand(1, device=device) * (azimuth[1] - azimuth[0]) + azimuth[0]
                theta = torch.rand(1, device=device) * (pitch[1] - pitch[0]) + pitch[0]
            elif i == 2:
                phi = torch.rand(1, device=device) * (azimuth[2] - azimuth[1]) + azimuth[1]
                theta = torch.rand(1, device=device) * (pitch[1] - pitch[0]) + pitch[0]
            elif i == 3:
                phi = torch.rand(1, device=device) * (azimuth[3] - azimuth[2]) + azimuth[2]
                theta = torch.rand(1, device=device) * (pitch[1] - pitch[0]) + pitch[0]
            elif i == 4:
                phi = torch.rand(1, device=device) * azimuth[3]
                theta = torch.rand(1, device=device) * pitch[0]
            elif i == 5:
                phi = torch.rand(1, device=device) * azimuth[3]
                theta = torch.rand(1, device=device) * pitch[0] + pitch[1]
            else:
                assert 0
            
            phis.append(phi)
            thetas.append(theta)

        dirs = torch.tensor(dirs, dtype=torch.long, device=device)
        phis = torch.cat(phis)
        thetas = torch.cat(thetas)

        indices = torch.randperm(len(self.views))[:batch_size]

        return phis[indices], thetas[indices], dirs[indices]
