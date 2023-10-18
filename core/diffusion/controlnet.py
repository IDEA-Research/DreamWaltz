from transformers import logging
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector, MidasDetector, CannyDetector
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from .time_prior import TimePrioritizedScheduler

logging.set_verbosity_error()


MODEL_CARDS = {
    'pose': "lllyasviel/sd-controlnet-openpose",
    'depth': "lllyasviel/sd-controlnet-depth",
    'canny': "lllyasviel/sd-controlnet-canny",
    'seg': "lllyasviel/sd-controlnet-seg",
    'normal': "lllyasviel/sd-controlnet-normal",
    # 'pose': "fusing/stable-diffusion-v1-5-controlnet-openpose",
    # 'depth': "fusing/stable-diffusion-v1-5-controlnet-depth",
    # 'canny': "fusing/stable-diffusion-v1-5-controlnet-canny",
    # 'seg': "fusing/stable-diffusion-v1-5-controlnet-seg",
    # 'normal': "fusing/stable-diffusion-v1-5-controlnet-normal",
}

class ControlNet(nn.Module):
    def __init__(self, device, condition_type: List[str], latent_mode=True, concept_name=None, guidance_scale=100.0, fp16=True):
        super().__init__()

        self.device = device
        self.condition_type = condition_type
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.guidance_scale = guidance_scale

        assert concept_name is None

        logger.info(f'loading stable diffusion with ControlNet-{condition_type}...')
        control_models = []
        cond_processors = []
        for cond_type in condition_type:
            controlnet = ControlNetModel.from_pretrained(MODEL_CARDS[cond_type], torch_dtype=torch.float16 if fp16 else torch.float32)
            control_models.append(controlnet)
            # cond_processors, not used for now
            # if cond_type == 'pose':
            #     cond_processors.append(OpenposeDetector.from_pretrained('lllyasviel/ControlNet'))
            # elif cond_type == 'depth':
            #     cond_processors.append(MidasDetector.from_pretrained('lllyasviel/ControlNet'))
            # elif cond_type == 'canny':
            #     cond_processors.append(CannyDetector.from_pretrained('lllyasviel/ControlNet'))
            # else:
            #     raise NotImplementedError(f'Not support {cond_type} yet')

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=control_models, 
            safety_checker=None, 
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).to(self.device)

        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.controlnet = self.pipe.controlnet
        self.cond_processors = cond_processors

        logger.info(f'\t successfully loaded ControlNet-{condition_type}!')

    def get_text_embeds(self, prompt, concat=True):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length',
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        if concat:
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            return text_embeddings
        else:
            return uncond_embeddings, text_embeddings

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=None, latents=None):

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def encode_images(self, images):
        images = 2 * images - 1  # [B, 3, H, W]
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def decode_latents(self, latents, to_uint8=False, to_numpy=False):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        with torch.no_grad():
            latents = 1 / self.vae.config.scaling_factor * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        if to_uint8:
            imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bf16
        if to_numpy:
            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        return imgs

    def calc_condition(self, image_path):
        results = []
        image = load_image(image_path)
        for processor in self.cond_processors:
            # process image with each condition processor
            cond = processor(image)
            # store the result
            results.append(cond)
        return results

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=Image.Resampling.LANCZOS))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=None, latents=None):

        # Prompts -> text embeds: [unconditioned embedding, text embedding]
        if isinstance(prompts, torch.Tensor):
            text_embeds = prompts
        else:
            if isinstance(prompts, str):
                prompts = [prompts]
            text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]

        # Text embeds -> img latents: [1, 4, 64, 64]
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

        # Img latents -> images
        images = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()  # [1, 512, 512, 3]
        images = (images * 255).round().astype('uint8')

        return images

    def batched_prompt_to_img(self, prompts, batch_size=4, **kwargs):
        if isinstance(prompts, torch.Tensor):
            assert prompts.size(0) % 2 == 0  # [z_uncond, z_cond]
            num_samples = prompts.size(0) // 2
            uncond_embeds_list = torch.split(prompts[:num_samples], batch_size)
            cond_embeds_list = torch.split(prompts[num_samples:], batch_size)
        else:
            raise NotImplementedError
        images_list = []
        for uncond_embeds, cond_embeds in zip(uncond_embeds_list, cond_embeds_list):
            text_embeds = torch.cat((uncond_embeds, cond_embeds))
            images = self.prompt_to_img(text_embeds, **kwargs)
            images_list.append(images)
        return np.concatenate(images_list)

    def batch_prompt_control_to_img(self, prompts, cond_inputs, height=512, width=512, num_inference_steps=50,
                                    guidance_scale=None, controlnet_conditioning_scale=1.0, dtype=torch.float):  
        """
        Generates images from prompts and control inputs using the controlnet model.
        
        Args:
        - prompts (str or List[str]): The prompts to generate images from.
        - cond_inputs (torch.Tensor or List[PIL.Image.Image]): The control inputs to condition the generation on.
        - height (int): The height of the generated images.
        - width (int): The width of the generated images.
        - num_inference_steps (int): The number of inference steps to use.
        - guidance_scale (float or None): The guidance scale to use.
        - controlnet_conditioning_scale (float): The conditioning scale to use for the controlnet model.

        Returns:
        - images (np.ndarray): The generated images.
        """
        # assert prompts is str or a list of str
        assert isinstance(prompts, str) or isinstance(prompts, List(str))
        batch_size = 1 if isinstance(prompts, str) else len(prompts)

        # prepare cond_inputs
        if isinstance(self.controlnet, ControlNetModel):
            cond_inputs = self.prepare_image(
                cond_inputs, height, width, batch_size=batch_size, 
                num_images_per_prompt=1, device=self.device, dtype=dtype
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            cond_inputs_temp = []
            for cond_input in cond_inputs:
                cond_input = self.prepare_image(
                    cond_input, height, width, batch_size=batch_size,
                    num_images_per_prompt=1, device=self.device, dtype=dtype
                )
                cond_inputs_temp.append(cond_input)
            cond_inputs = cond_inputs_temp

        # inference
        image = self.pipe(
            prompts, num_inference_steps=num_inference_steps, image=cond_inputs,
            guidance_scale=guidance_scale, controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type='np.array'
        ).images  # [1, 512, 512, 3]

        # Img to Numpy
        return (image * 255).round().astype('uint8')


class ControllableScoreDistillationSampling(ControlNet):
    def __init__(self, device, model_name='v1.5', weight_mode='sjc', guide_cfg=None, **kwargs):

        super().__init__(device, **kwargs)

        self.cfg = guide_cfg

        self.tp_scheduler = TimePrioritizedScheduler(guide_cfg, scheduler=self.scheduler, device=device, num_train_timesteps=self.num_train_timesteps)
        self.add_noise = self.tp_scheduler.add_noise
        self.get_timestep = self.tp_scheduler.get_timestep
        self.alphas = self.tp_scheduler.alphas
        self.betas = self.tp_scheduler.betas
        self.alphas_cumprod = self.tp_scheduler.alphas_cumprod

        self.weight_mode = weight_mode
        self.guidance_adjust = guide_cfg.guidance_adjust

    def get_guidance_scale(self, train_step, max_iteration):
        if self.guidance_adjust == 'constant':
            guidance_scale = self.guidance_scale
        elif self.guidance_adjust == 'uniform':
            guidance_scale = np.random.uniform(7.5, self.guidance_scale)
        elif self.guidance_adjust == 'linear':
            guidance_delta = (self.guidance_scale - 7.5) / (max_iteration - 1)
            guidance_scale = self.guidance_scale - (train_step - 1) * guidance_delta
        elif self.guidance_adjust == 'linear_rev':
            guidance_delta = (self.guidance_scale - 7.5) / (max_iteration - 1)
            guidance_scale = 7.5 + (train_step - 1) * guidance_delta
        else:
            raise NotImplementedError
        return guidance_scale

    def calc_gradients(self, noise_residual, noise_pred, t):
        # Weight
        if self.weight_mode != 'sjc-v2':
            if self.weight_mode in ('dreamfusion', 'stable-dreamfusion'):
                w = (1 - self.alphas_cumprod[t])
            elif self.weight_mode == 'latent-nerf':
                w = (1 - self.alphas_cumprod[t]) * torch.sqrt(self.alphas_cumprod[t])
            elif self.weight_mode == 'sjc':
                w = torch.ones_like(self.alphas_cumprod[t])
            else:
                raise NotImplementedError
            gradients = w.reshape(-1, 1, 1, 1) * noise_residual
        else:
            gradients = noise_pred
        # Reg
        if self.cfg.grad_clip:
            gradients = gradients.clamp(-1, 1)
        if self.cfg.grad_norm:
            gradients = torch.nn.functional.normalize(noise_residual, p=2, dim=(1, 2, 3))
        return gradients

    def estimate(self, text_embeddings, inputs, train_step, max_iteration, cond_inputs=None, controlnet_conditioning_scale=1.0, cross_attention_kwargs=None, backward=True):
        """
            text_embeddings: [2N, 77, 768]
            inputs: [N, 4, 64, 64]
            cond_inputs: conditional preprocessed images
        """
        batch_size = inputs.size(0)
        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)
        # controlnet_conditioning_scale = [item / len(controlnet_conditioning_scale) for item in controlnet_conditioning_scale]

        if cond_inputs is None:
            # TODO: write logic processing inputs with different processors
            # Here we can add logic to handle different types of inputs
            # For example, if inputs are images, we can preprocess them using a specific function
            # If inputs are text, we can tokenize them and pass them through an embedding layer
            # For now, we will raise a NotImplementedError to remind ourselves to implement this logic
            raise NotImplementedError('TODO: write logic processing inputs with different processors')

        # prepare images
        if isinstance(self.controlnet, ControlNetModel):
            cond_inputs = self.prepare_image(
                cond_inputs, 512, 512, batch_size=batch_size, 
                num_images_per_prompt=1, device=self.device, dtype=inputs.dtype
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            cond_inputs_temp = []
            for cond_input in cond_inputs:
                cond_input = self.prepare_image(
                    cond_input, 512, 512, batch_size=batch_size,
                    num_images_per_prompt=1, device=self.device, dtype=inputs.dtype
                )
                cond_inputs_temp.append(cond_input)
            cond_inputs = cond_inputs_temp

        # Adaptive guidance scale
        guidance_scale = self.get_guidance_scale(train_step, max_iteration)
        do_classifier_free_guidance = guidance_scale > 1.0

        # Adaptive timestep
        t = self.get_timestep(batch_size, train_step, max_iteration)

        # Interp to 512x512 to be fed into vae.
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_images(pred_rgb_512)
        else:
            latents = inputs

        # Encode image into latents with vae, requires grad!
        # Predict the noise residual with unet, no grad!
        with torch.no_grad():
            # 1. Add Noise
            noise = torch.randn_like(latents)
            latents_noisy = self.add_noise(latents, noise, t)

            # 2. Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_inputs,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 4. Noise Residual
            noise_residual = noise_pred - noise
            gradients = self.calc_gradients(noise_residual, noise_pred, t)

            # 5. Denoise (Optional)
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            alpha_comp_t = self.alphas_cumprod[t]
            latents_denoise = (latents_noisy - noise_pred * beta_t / torch.sqrt(1 - alpha_comp_t)) / torch.sqrt(alpha_t)

        # Manually backward, since we omitted an item in grad and cannot simply autodiff.
        if backward:
            latents.backward(gradient=gradients, retain_graph=True)

        return {
            'latents': latents,
            'latents_denoise': latents_denoise,
            'gradients': gradients,
            'noise_residual': noise_residual,
            't': t,
        }
