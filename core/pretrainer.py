import os.path as osp
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List

import imageio
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from torchvision.utils import save_image

from core.data import NeRFDataset, DataLoaderIter
from core.trainer import Trainer


class PreTrainer(Trainer):

    def train(self):
        logger.info('Starting training ^_^')
        self.nerf.train()

        pbar = tqdm(total=self.max_step, initial=self.train_step, bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        train_loader = DataLoaderIter(self.dataloaders['train'])

        while self.train_step < self.max_step:
            # Keep going over dataloader until finished the required number of iterations
            data = train_loader.next()

            # Update Density Grid
            if self.nerf.cuda_ray and self.train_step % self.cfg.render.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    self.nerf.update_extra_state()

            self.train_step += 1
            if self.train_step % 100 == 0:
                pbar.update(100)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                pred_rgbs, loss, mask = self.train_forward(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.train_step % self.cfg.log.save_interval == 0:
                self.save_checkpoint(full=False)

            if self.train_step % self.cfg.log.snapshot_interval == 0 or self.train_step in (1, 100, 200, 300, 400):
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                self.nerf.train()

            if self.train_step % self.cfg.log.snapshot_interval == 0 or self.train_step in (1, 100, 200, 300, 400):
                self.snapshot_rgb(pred_rgbs)
                self.snapshot(image=data['cond_images'], filename=self.cond_type)
                save_image(mask, osp.join(str(self.train_renders_path / 'condition'), f'mask_{self.train_step:06d}.jpg'))
                logger.info(f'train_step={self.train_step:06d}')

        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True, save_as_image=True)
        # self.evaluate(self.dataloaders['val_large'], self.final_renders_path_wo_bg, save_as_video=True, save_as_image=True, disable_background=True)
        if self.dataloaders['val_large_64x64'] is not None:
            self.evaluate(self.dataloaders['val_large_64x64'], self.final_renders_path_64x64, save_as_video=True, save_as_image=False)

        logger.info('\tDone!')

    def train_forward(self, data: Dict[str, Any]):
        # Render images
        nerf_outputs = self.render_for_train(data)
        pred_rgb, pred_ws = nerf_outputs['image'], nerf_outputs['weights_sum']
 
        mask = np.array(data['cond_images'][0])  # (512, 512, 3)
        mask = np.max(mask, axis=2) > 1e-6  # (512, 512)
        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

        mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode='nearest')

        loss = self.losses['mse_loss'](pred_ws, mask.to(pred_ws.device))  # * 10.0

        return pred_rgb, loss, mask
