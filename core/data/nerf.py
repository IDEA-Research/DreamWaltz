import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.nerf.utils.render_utils import get_rays
from configs import RenderConfig
from core.utils.pose import rand_pose, circle_pose, custom_pose


class NeRFDataset:
    def __init__(self, cfg: RenderConfig, device, image_prompt=None, view_prompt=None, type='train', H=64, W=64, eval_size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test

        self.training = self.type in ['train', 'all']

        self.H = H
        self.W = W
        self.cx = self.H / 2
        self.cy = self.W / 2

        self.radius_range = cfg.radius_range
        self.fovy_range = cfg.fovy_range
        self.eval_size = eval_size

        self.fix_camera = cfg.eval_fix_camera
        self.fix_animation = cfg.eval_fix_animation
        self.camera_track = cfg.eval_camera_track

        self.view_prompt = view_prompt
        self.image_prompt = image_prompt

        if hasattr(self.image_prompt, 'num_frame'):
            self.num_frame = self.image_prompt.num_frame
            if not self.training and not self.fix_animation and self.num_frame > 0:
                self.eval_size = min(eval_size, self.num_frame)
        else:
            self.num_frame = None

    def collate(self, index):

        B = len(index)  # always 1
        fixed_viewpoint = False
        if self.training:
            # random pose on the fly
            poses, dirs = rand_pose(B, self.device, radius_range=self.radius_range, theta_range=self.cfg.theta_range, phi_range=self.cfg.phi_range,
                                    jitter=self.cfg.jitter_pose, vertical_jitter=self.cfg.vertical_jitter, view_prompt=self.view_prompt,
                                    batched=self.cfg.batched_view, uniform_sphere_rate=self.cfg.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])

            # smpl prompt
            cond_images = self.image_prompt(intrinsics, poses)
        else:
            if not self.fix_animation and not self.fix_camera and self.eval_size > 150:
                percent = index[0] % 150 / 150
            else:
                percent = index[0] / self.eval_size

            # circle pose
            radius = self.radius_range[1] * self.cfg.eval_radius_rate
            phi, theta = custom_pose(percent, eval_theta=self.cfg.eval_theta, fix_camera=self.fix_camera, camera_track=self.camera_track)
            poses, dirs = circle_pose(self.device, radius=radius, theta=theta, phi=phi, view_prompt=self.view_prompt)

            # fixed focal
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])

            # smpl prompt
            if self.num_frame is not None:
                assert len(index) == 1
                if self.fix_animation:
                    frame_idx = 0
                else:
                    if self.eval_size < self.num_frame:
                        frame_idx = index[0]
                        # frame_idx = int((index[0] / self.eval_size) * self.num_frame)
                    else:
                        frame_idx = index[0]
                cond_images = self.image_prompt(intrinsics, poses, frame_idx=frame_idx)
            else:
                cond_images = self.image_prompt(intrinsics, poses)

        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'fixed_viewpoint': fixed_viewpoint,
        }
        if isinstance(cond_images, dict):
            data.update(cond_images)
        else:
            data['cond_images'] = cond_images

        return data

    def dataloader(self, batch_size=1):
        loader = DataLoader(list(range(self.eval_size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader


if __name__ == '__main__':
    for one_batch in NeRFDataset(RenderConfig(), torch.device('cpu')).dataloader():
        print(one_batch)
        break
