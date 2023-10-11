from huggingface_hub import hf_hub_download
from transformers import logging
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from .time_prior import TimePrioritizedScheduler

logging.set_verbosity_error()


MODEL_CARDS = {
    'v1.4': "CompVis/stable-diffusion-v1-4",
    'v1.5': "runwayml/stable-diffusion-v1-5",
    'v2.0b': "stabilityai/stable-diffusion-2-base",
    'v2.0': "stabilityai/stable-diffusion-2",
    'v2.1b': "stabilityai/stable-diffusion-2-1-base",
    'v2.1': "stabilityai/stable-diffusion-2-1",
}


class StableDiffusion(nn.Module):
    def __init__(self, device, model_name='v1.4', concept_name=None, fp16=False, latent_mode=True, guidance_scale=100.0):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            # logger.warning('try to load hugging face access token from the default place,'
            #                ' make sure you have run `huggingface-cli login`.')
            self.token = True

        if model_name in MODEL_CARDS.keys():
            model_name = MODEL_CARDS[model_name]

        self.device = device
        self.model_name = model_name
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.guidance_scale = guidance_scale

        logger.info(f'loading stable diffusion with {model_name}...')

        local_files_only = False
        if not fp16:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, local_files_only=local_files_only)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, local_files_only=local_files_only, torch_dtype=torch.float16, revision="fp16")

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.vae = pipe.vae.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.scheduler = pipe.scheduler

        if concept_name is not None:
            self.load_concept(concept_name)

        logger.info(f'\t successfully loaded stable diffusion!')

    def load_concept(self, concept_name):
        repo_id_embeds = f"sd-concepts-library/{concept_name}"
        learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
        with open(token_path, 'r') as file:
            placeholder_token_string = file.read()

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}."
                f" Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

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
            imgs = imgs.cpu().permute(0, 2, 3, 1).float().numpy()
        return imgs

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


class ScoreDistillationSampling(StableDiffusion):
    def __init__(self, device, model_name='v1.4', weight_mode='sjc', concept_name=None, fp16=False, latent_mode=True, guidance_scale=100, guide_cfg=None):
        super().__init__(device, model_name, concept_name, fp16, latent_mode, guidance_scale)
        self.cfg = guide_cfg

        self.tp_scheduler = TimePrioritizedScheduler(guide_cfg, scheduler=self.scheduler, device=device, num_train_timesteps=self.num_train_timesteps)
        self.add_noise = self.tp_scheduler.add_noise
        self.get_timestep = self.tp_scheduler.get_timestep
        self.alphas = self.tp_scheduler.alphas
        self.betas = self.tp_scheduler.betas
        self.alphas_cumprod = self.tp_scheduler.alphas_cumprod

        self.weight_mode = weight_mode
        self.guidance_adjust = guide_cfg.guidance_adjust

    def prepare_time_schedule(self):
        p2_cumsum = (self.p2_lambda / torch.sum(self.p2_lambda)).flip(dims=(0,)).cumsum(dim=0)
        return p2_cumsum.detach().cpu().numpy()

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

    def estimate(self, text_embeddings, inputs, train_step, max_iteration, backward=True):
        """
            text_embeddings: [2N, 77, 768]
            inputs: [N, 4, 64, 64]
            inputs: latents or images
        """
        batch_size = inputs.size(0)

        # Adaptive guidance scale
        guidance_scale = self.get_guidance_scale(train_step, max_iteration)

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

            # 2. Predict Noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_input = torch.cat([t] * 2)

            noise_pred = self.unet(latent_model_input, t_input, encoder_hidden_states=text_embeddings).sample

            # 3. Perform Guidance (high scale from paper!)
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


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='a photo of stable diffusion model')
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    with torch.no_grad():
        sd = StableDiffusion(torch.device('cuda'))
        images = []
        for _ in range(4):
            _images = sd.prompt_to_img([opt.prompt for _ in range(4)], opt.H, opt.W, opt.steps)
            images.append(_images)
        images = np.concatenate(images)
