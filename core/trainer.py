import os.path as osp
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List

import imageio
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict

from core import utils
from configs.train_config import TrainConfig
from core.nerf.renderer import NeRFRenderer
from core.data import NeRFDataset, DataLoaderIter
from core.prompt.smpl_prompt import SMPLPrompt, AnimatedSMPLPrompt
from core.prompt.view_prompt import ViewPrompt
from core.diffusion.stable_diffusion import ScoreDistillationSampling
from core.diffusion.controlnet import ControllableScoreDistillationSampling
from core.utils import make_path, tensor2numpy


def visualize(imgs, image_path):
    from einops import rearrange
    from torchvision import utils as vutils
    from imageio import imwrite

    imgs = rearrange(imgs, "N H W C -> N C H W", C=3)
    imgs = torch.from_numpy(imgs)
    pane = vutils.make_grid(imgs, padding=2, nrow=4)
    pane = rearrange(pane, "C H W -> H W C", C=3).numpy()
    imwrite(image_path, pane)


class BasicTrainer:
    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def snapshot(self, image: Union[List[Image.Image], Image.Image], filename: Union[List[str], str], dirname='condition'):
        if isinstance(image, list) and isinstance(filename, List):
            for _image, _filename in zip(image, filename):
                self.snapshot(_image, _filename, dirname)
        else:
            save_path = self.train_renders_path / dirname / f'{filename}_{self.train_step:05d}.jpg'
            save_path.parent.mkdir(exist_ok=True)
            image.save(save_path)

    def snapshot_rgb(self, pred_rgbs: torch.Tensor, filename='rgb', save_raw=False):
        # Save Raw
        if save_raw:
            save_path = self.train_renders_path / 'rgb_raw' / f'{filename}_{self.train_step:05d}.pt'
            save_path.parent.mkdir(exist_ok=True)
            torch.save(pred_rgbs[0].detach().cpu(), save_path)
        # Save Viz
        pred_rgb_vis = self.diffusion.decode_latents(pred_rgbs) if self.nerf.latent_mode else pred_rgbs  # [B, 4, H_, W_] -> [B, H, W, 3]
        pred_rgb_vis = pred_rgb_vis.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        save_path = self.train_renders_path / 'rgb' / f'{filename}_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)
        Image.fromarray(tensor2numpy(pred_rgb_vis[0])).save(save_path)

    def snapshot_grad(self, gradients: torch.Tensor, filename='grad', save_raw=False, use_norm=True, pseudo_color=False):
        # Gradients
        if gradients is not None:
            # Save Raw
            if save_raw:
                save_path = self.train_renders_path / 'grad_raw' / f'{filename}_{self.train_step:05d}.pt'
                save_path.parent.mkdir(exist_ok=True)
                torch.save(gradients[0].detach().cpu(), save_path)
            # Save Viz
            save_path = self.train_renders_path / 'grad' / f'{filename}_{self.train_step:05d}.jpg'
            save_path.parent.mkdir(exist_ok=True)
            if pseudo_color:
                gradients = torch.abs(gradients[0]).mean(dim=0, keepdim=False)
                if use_norm:
                    gradients /= torch.std(gradients)
                    # gradients /= torch.norm(gradients.flatten(), p=2)
                gradients = gradients.detach().cpu().numpy()
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                # plt.gca().imshow(gradients, cmap=plt.get_cmap('jet'))
                plt.gca().imshow(gradients)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
            else:
                if use_norm:
                    gradients /= torch.std(gradients)
                pred_grad_vis = self.diffusion.decode_latents(gradients).permute(0, 2, 3, 1).contiguous()
                Image.fromarray(tensor2numpy(pred_grad_vis[0])).save(save_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.nerf.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return

        missing_keys, unexpected_keys = self.nerf.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        if self.cfg.render.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.nerf.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.nerf.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] # + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except Exception as e:
                logger.warning("Failed to load optimizer.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("loaded scaler.")
            except Exception as e:
                logger.warning("Failed to load scaler.")

    def save_checkpoint(self, full=False):

        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if self.nerf.cuda_ray:
            state['mean_count'] = self.nerf.mean_count
            state['mean_density'] = self.nerf.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()

        state['model'] = self.nerf.state_dict()

        file_path = f"{name}.pth"

        self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)


class Trainer(BasicTrainer):
    def __init__(self, cfg: TrainConfig, device=None):
        self.cfg = cfg
        self.train_step = 0
        self.max_step = self.cfg.optim.iters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'snapshots' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'snapshots' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results' / '128x128')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_prompt = ViewPrompt(self.cfg.render, mode=self.cfg.prompt.view_prompt)

        self.use_controlnet = self.cfg.guide.use_controlnet
        self.cond_type = self.cfg.prompt.smpl_prompt.split(',')
        if self.cfg.animation:
            self.smpl_prompt = AnimatedSMPLPrompt(self.cfg.prompt, cond_type=self.cond_type, num_person=self.cfg.prompt.num_person, scene=self.cfg.prompt.scene)
        else:
            self.smpl_prompt = SMPLPrompt(self.cfg.prompt, cond_type=self.cond_type, num_person=self.cfg.prompt.num_person, scene=self.cfg.prompt.scene)

        # Build NeRF
        self.nerf = self.init_nerf()

        self.dataloaders = self.init_dataloaders()

        self.diffusion = self.init_diffusion()
        self.text_z_uc, self.text_z, self.text_z_with_dirs = self.init_text_embeddings()
        self.losses = self.init_losses()
        self.optimizer, self.scaler, self.scheduler = self.init_optimizer()

        # Export generated images
        if not cfg.log.eval_only:
            logger.info('Export generated samples...')
            # self.export_samples(make_path(self.exp_path / 'snapshots' / 'diffusion'))

        self.export_conditions(make_path(self.exp_path / 'snapshots' / 'smpl'))

        # Load checkpoints
        self.past_checkpoints = []

        # Resume
        if self.cfg.optim.resume:
            print(f'Load model weights and optimizer checkpoints from {self.cfg.optim.ckpt}...')
            self.load_checkpoint(model_only=False)

        if self.cfg.optim.ckpt is not None:
            print(f'Load model weights from {self.cfg.optim.ckpt}...')
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        if self.cfg.optim.ckpt_extra is not None:
            print(f'Load extra model weights from {self.cfg.optim.ckpt_extra}...')
            self.nerf.load_extra_checkpoint(self.cfg.optim.ckpt_extra, device=self.device)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    @torch.no_grad()
    def export_samples(self, save_path, num_samples=2):
        # sd without augmentations
        text_embeds = torch.cat((
            self.text_z_uc.expand(num_samples, -1, -1),
            self.text_z.expand(num_samples, -1, -1),
        ))
        images = self.diffusion.batched_prompt_to_img(text_embeds, guidance_scale=7.5)  # np.array: [N, H, W, 3]
        visualize(images, osp.join(save_path, 'export_uncond_7.5.jpg'))
        images = self.diffusion.batched_prompt_to_img(text_embeds)  # np.array: [N, H, W, 3]
        visualize(images, osp.join(save_path, 'export_uncond.jpg'))

        # sd with augmentations
        if self.cfg.prompt.append_direction:
            for view, text_z in zip(self.view_prompt.views, self.text_z_with_dirs):
                text_embeds = torch.cat((
                    self.text_z_uc.expand(num_samples, -1, -1),
                    text_z.expand(num_samples, -1, -1),
                ))
                images = self.diffusion.batched_prompt_to_img(text_embeds, guidance_scale=7.5)  # [N, H, W, 3]
                visualize(images, osp.join(save_path, f'export_{view}_7.5.jpg'))
                images = self.diffusion.batched_prompt_to_img(text_embeds)  # [N, H, W, 3]
                visualize(images, osp.join(save_path, f'export_{view}.jpg'))

    @torch.no_grad()
    def export_conditions(self, save_path, cond_image_only=True):
        # controlnet with conditions
        save_dir_smpl = str(save_path)
        cond_inputs = []
        for cond_name in self.cond_type:
            # export videos
            cond_image = self.smpl_prompt.write_video(
                save_dir=save_dir_smpl,
                save_video=f'{cond_name}.mp4',
                save_image=f'{cond_name}.png',
                cond_type=cond_name,
            )
            cond_inputs.append(cond_image)
        # export images
        if not cond_image_only:
            images = self.diffusion.batch_prompt_control_to_img(
                prompts=self.cfg.guide.text,
                cond_inputs=cond_inputs,
                guidance_scale=12.5,
                controlnet_conditioning_scale=[1.0 / len(cond_inputs) for _ in range(len(cond_inputs))],
            )  # np.array: [N, H, W, 3]
            visualize(images, osp.join(save_dir_smpl, 'control.jpg'))

    def init_nerf(self) -> NeRFRenderer:
        if self.cfg.animation:
            smpl = self.smpl_prompt.hs.smpl.model
            smpl_params_cnl = self.smpl_prompt.hs_canonical.smpl_params
            smpl_params_cnl['vertices'] = self.smpl_prompt.hs_canonical.vertices
            smpl_params_cnl['joints'] = self.smpl_prompt.hs_canonical.joints[:, :self.smpl_prompt.hs_canonical.num_joints+1, :]
            if self.smpl_prompt.num_person == 1:
                from core.nerf.dreamwaltz import DreamWaltz as NeRFNetwork
                model = NeRFNetwork(self.cfg.render, smpl=smpl, smpl_params_cnl=smpl_params_cnl)
            else:
                raise NotImplementedError
        else:
            from core.nerf.network import NeRFNetwork
            model = NeRFNetwork(self.cfg.render)

        logger.info(f'Loaded {self.cfg.render.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)

        if self.cfg.render.density_blob == 'smpl':
            model.density_smpl = self.smpl_prompt.hs.export_density

        return model.to(self.device)

    def init_diffusion(self):
        if self.use_controlnet:
            condition_type = self.cond_type
            diffusion_model = ControllableScoreDistillationSampling(
                device=self.device,
                condition_type=condition_type,
                model_name=self.cfg.guide.diffusion_name,
                weight_mode=self.cfg.guide.loss_type,
                fp16=self.cfg.guide.diffusion_fp16,
                guidance_scale=self.cfg.guide.guidance_scale,
                concept_name=self.cfg.guide.concept_name,
                latent_mode=self.nerf.latent_mode,
                guide_cfg=self.cfg.guide,
            )
        else:
            diffusion_model = ScoreDistillationSampling(
                device=self.device,
                model_name=self.cfg.guide.diffusion_name,
                weight_mode=self.cfg.guide.loss_type,
                fp16=self.cfg.guide.diffusion_fp16,
                guidance_scale=self.cfg.guide.guidance_scale,
                concept_name=self.cfg.guide.concept_name,
                latent_mode=self.nerf.latent_mode,
                guide_cfg=self.cfg.guide,
            )
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def init_optimizer(self) -> Tuple[Optimizer, Any]:
        # Optimizer
        if self.cfg.optim.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        elif self.cfg.optim.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-2)
        elif self.cfg.optim.optimizer == 'adan':
            from .solver.adan import Adan
            optimizer = Adan(self.nerf.get_params(5 * self.cfg.optim.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        elif self.cfg.optim.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        elif self.cfg.optim.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.nerf.get_params(self.cfg.optim.lr))
        else:
            raise NotImplementedError
        # Grad Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
        # LR Scheduler
        from core.solver import make_scheduler
        lr_policy = self.cfg.optim.lr_policy
        if lr_policy in ('none', 'constant'):
            scheduler = None
        elif lr_policy == 'step':
            scheduler = make_scheduler(optimizer, lr_policy, step_size=int(self.max_step * 0.7))
        elif lr_policy == 'multistep':
            scheduler = make_scheduler(optimizer, lr_policy, step_size=int(self.max_step * 0.7))
        elif lr_policy == 'warmup':
            scheduler = make_scheduler(optimizer, lr_policy, milestones=[int(self.max_step * 0.7),], warmup_iter=1000)
        elif lr_policy == 'lambda':
            def lr_lambda(i: int):
                # i: 0, 1, ..., 10000
                idx = int((1 - i / lr_lambda.max_step) * 1000)  # 1000 -> 0
                if idx == 1000:
                    return 1.0
                else:
                    return 1 - lr_lambda.alphas_cumprod[idx].item()
            lr_lambda.max_step = self.max_step
            lr_lambda.alphas_cumprod = self.diffusion.alphas_cumprod  # [0.9991 -> 0.0047]
            scheduler = make_scheduler(optimizer, lr_policy, lr_lambda=lr_lambda)
        return optimizer, scaler, scheduler

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_h, train_w = self.cfg.render.train_h, self.cfg.render.train_w
        eval_h, eval_w = self.cfg.render.eval_h, self.cfg.render.eval_w
        _NeRFDataset = partial(NeRFDataset, self.cfg.render, device=self.device, view_prompt=self.view_prompt, image_prompt=self.smpl_prompt)
        # Build training set
        train_dataloader = _NeRFDataset(type='train', H=train_h, W=train_w).dataloader(batch_size=self.cfg.optim.batch_size)
        # Build test set
        val_loader = _NeRFDataset(type='val', H=eval_h, W=eval_w, eval_size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = _NeRFDataset(type='val', H=eval_h, W=eval_w, eval_size=self.cfg.log.full_eval_size).dataloader()
        return {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}

    def init_losses(self) -> Dict[str, Callable]:
        losses = {}
        from core.loss.sparsity_loss import SparsityLoss
        losses['sparsity_loss'] = SparsityLoss(self.cfg.optim)
        if self.cfg.optim.lambda_shape > 0 and self.cfg.guide.shape_path is not None:
            from core.loss.shape_loss import ShapeLoss
            losses['shape_loss'] = ShapeLoss(self.cfg.guide)
        losses['mse_loss'] = torch.nn.MSELoss()
        return losses

    def init_text_embeddings(self) -> Tuple:
        ref_text = self.cfg.guide.text
        text_z_uc, text_z = self.diffusion.get_text_embeds([ref_text], concat=False)  # [1, 77, 768], [1, 77, 768]
        text_z_with_dirs = []
        if self.cfg.prompt.append_direction:
            for v in self.view_prompt.views:
                text = self.view_prompt.pattern.format(ref_text, v)
                _, text_zd = self.diffusion.get_text_embeds([text], concat=False)
                text_z_with_dirs.append(text_zd)
            text_z_with_dirs = torch.cat(text_z_with_dirs)# [N, 77, 768]
        return text_z_uc, text_z, text_z_with_dirs

    def train(self):
        logger.info('Starting training ^_^')
        self.nerf.train()

        pbar = tqdm(total=self.max_step, initial=self.train_step, bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        train_loader = DataLoaderIter(self.dataloaders['train'])

        while self.train_step < self.max_step:
            # Keep going over dataloader until finished the required number of iterations
            data = train_loader.next()

            # Set Current Body Pose
            if self.cfg.animation:
                self.nerf.smpl_data_obs = data['smpl_data']

            # Update Density Grid
            if self.nerf.cuda_ray and self.train_step % self.cfg.render.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    self.nerf.update_extra_state()

            self.train_step += 1
            if self.train_step % 5 == 0:
                pbar.update(5)

            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                pred_rgbs, loss, sd_outputs = self.train_forward(data)

            gradients = sd_outputs['gradients']
            noise_residual = sd_outputs['noise_residual']
            t = sd_outputs['t']

            self.optimizer.zero_grad()

            latent_grads = self.scaler.scale(gradients) if self.cfg.optim.grad_scale else gradients
            if isinstance(loss, float):
                sd_outputs['latents'].backward(gradient=latent_grads, retain_graph=False)
            else:
                sd_outputs['latents'].backward(gradient=latent_grads, retain_graph=True)
                self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.train_step % self.cfg.log.save_interval == 0:
                self.save_checkpoint(full=False)

            if self.train_step % self.cfg.log.evaluate_interval == 0 or self.train_step in (250,):  #  or self.train_step in (1, 100, 200, 300, 400)
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                self.nerf.train()

            if self.train_step % self.cfg.log.snapshot_interval == 0 or self.train_step in (1, 100, 200, 300, 400):
                self.snapshot_rgb(pred_rgbs)
                self.snapshot_grad(noise_residual)
                self.snapshot(image=data['cond_images'], filename=self.cond_type)
                logger.info(f'train_step={self.train_step:06d}, t={t.item():04d}')

        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True, save_as_image=True)

        logger.info('\tDone!')

    def train_forward(self, data: Dict[str, Any]):
        # Render images
        nerf_outputs = self.render_for_train(data)
        pred_rgb, pred_ws = nerf_outputs['image'], nerf_outputs['weights_sum']
        B = pred_rgb.size(0)
 
        # Text embeddings
        if self.cfg.prompt.append_direction:
            text_embeds = torch.cat((
                self.text_z_uc.expand(B, -1, -1),
                self.text_z_with_dirs[data['dir']],
            ))
        else:
            text_embeds = torch.cat((
                self.text_z_uc.expand(B, -1, -1),
                self.text_z.expand(B, -1, -1),
            ))

        # Guidance loss
        if self.use_controlnet:
            sd_outputs = self.diffusion.estimate(
                text_embeddings=text_embeds,
                inputs=pred_rgb,
                cond_inputs=data['cond_images'],
                controlnet_conditioning_scale=self.cfg.guide.controlnet_scale,
                train_step=self.train_step,
                max_iteration=self.max_step,
                backward=False,
            )
        else:
            sd_outputs = self.diffusion.estimate(
                text_embeddings=text_embeds,
                inputs=pred_rgb,
                train_step=self.train_step,
                max_iteration=self.max_step,
                backward=False,
            )

        sd_outputs['gradients'] *= self.cfg.optim.lambda_guidance

        loss = self.losses['sparsity_loss'](pred_ws, self.train_step, self.max_step)

        return pred_rgb, loss, sd_outputs

    def render_for_train(self, data: Dict[str, Any]):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        outputs = self.nerf.render(rays_o, rays_d, staged=False, perturb=True, bg_color=None, ambient_ratio=ambient_ratio, shading=shading)

        outputs['image'] = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs['weights_sum'] = outputs['weights_sum'].reshape(B, 1, H, W)

        with torch.no_grad():
            pred_depth = outputs['depth'].reshape(B, H, W).unsqueeze(-1).permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
            # outputs['depth'] = (pred_depth - torch.min(pred_depth)) / (torch.max(pred_depth) - torch.min(pred_depth))
            outputs['depth'] = pred_depth / torch.max(pred_depth)
            
        return outputs

    def evaluate(self, dataloader: DataLoader, save_path: Path, disable_background: bool = False, save_as_video: bool = False, save_as_image: bool = True,
                 output_types=('image', 'depth', 'normal', 'weights_sum')):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        self.nerf.eval()
        save_path.mkdir(exist_ok=True)
        if save_as_image:
            for output_type in output_types:
                (save_path / output_type).mkdir(exist_ok=True)

        if save_as_video:
            all_preds = defaultdict(list)

        for i, data in enumerate(dataloader):

            # Set Current Body Pose
            if self.cfg.animation:
                self.nerf.smpl_data_obs = data['smpl_data']
                cond_images = data['cond_images'][0]
                (save_path / 'smpl').mkdir(exist_ok=True)
                cond_images.save(save_path / 'smpl' / f'{i:04d}.png')
                # self.smpl_prompt.hs.export_mesh_to_file(save_path / 'smpl' / f'{i:04d}.obj')

            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                outputs = self.eval_render(data, disable_background=disable_background)

            for output_type in output_types:
                pred = tensor2numpy(outputs[output_type][0])

                if save_as_video:
                    all_preds[output_type].append(pred)

                if save_as_image:
                    if output_type != 'image':
                        Image.fromarray(pred).save(save_path / output_type / f"{self.train_step:06d}_{i:01d}.png")
                    elif not self.cfg.log.skip_rgb:
                        Image.fromarray(pred).save(save_path / "image" / f"{self.train_step:06d}_{i:01d}.jpg")

        if save_as_video:
            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.train_step}_{name}.mp4", video, fps=25, quality=8, macro_block_size=1)

            for output_type, preds in all_preds.items():
                if not (output_type == 'image' and self.cfg.log.skip_rgb):
                    dump_vid(np.stack(preds, axis=0), output_type)

        logger.info('Done!')

    @torch.no_grad()
    def eval_render(self, data, disable_background=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        bg_color = 'nerf' if self.nerf.bg_mode == 'nerf' else 'white'

        outputs = self.nerf.render(rays_o, rays_d, staged=False, perturb=False, bg_color=bg_color,
                                   light_d=light_d, ambient_ratio=ambient_ratio, shading=shading,
                                   disable_background=disable_background)

        # RGB
        if self.nerf.latent_mode:
            pred_latent = outputs['image'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent.device)
            else:
                pred_rgb = self.diffusion.decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)

        # Weights_sum
        pred_weights_sum = outputs['weights_sum'].reshape(B, H, W).unsqueeze(-1).repeat(1, 1, 1, 3)
        pred_weights_sum = (pred_weights_sum - torch.min(pred_weights_sum)) / torch.max(pred_weights_sum)

        # Depth
        pred_depth = outputs['depth'].reshape(B, H, W).unsqueeze(-1).repeat(1, 1, 1, 3)
        # pred_depth = (pred_depth - torch.min(pred_depth)) / (torch.max(pred_depth) - torch.min(pred_depth))
        pred_depth = pred_depth / torch.max(pred_depth)

        # Normal
        shading = 'normal'
        outputs_normals = self.nerf.render(rays_o, rays_d, staged=False, perturb=False, bg_color=None,
                                           light_d=light_d, ambient_ratio=ambient_ratio, shading=shading,
                                           disable_background=True)
        pred_normals = outputs_normals['image'][:, :, :3].reshape(B, H, W, 3).contiguous()

        return {
            'image': pred_rgb,
            'depth': pred_depth,
            'normal': pred_normals,
            'weights_sum': pred_weights_sum,
        }

    def full_eval(self):
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=self.cfg.render.eval_save_video,
                       disable_background=self.cfg.render.eval_disable_background)
