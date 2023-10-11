import torch
import bisect
import matplotlib.pyplot as plt
from configs.train_config import GuideConfig


class TimePrioritizedScheduler:
    def __init__(self, cfg: GuideConfig, scheduler, device, num_train_timesteps=1000) -> None:
        # Hyper Params
        self.device = device
        self.time_sampling = cfg.time_sampling
        self.min_step = int(num_train_timesteps * cfg.min_timestep)
        self.max_step = int(num_train_timesteps * cfg.max_timestep)
        self.num_timesteps = num_train_timesteps
        self.time_prior = cfg.time_prior

        # Scheduler
        self.scheduler = scheduler

        # Coefficients
        self.betas = self.scheduler.betas.to(self.device)
        self.alphas = self.scheduler.alphas.to(self.device)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

        # Step Mode
        if self.time_sampling.startswith('stage'):
            self.num_stage = int(self.time_sampling[6:])
            self.time_sampling = 'stage'
            timesteps_per_stage = (self.max_step - self.min_step) // self.num_stage
            self.intervals_per_stage = []
            for i in range(self.num_stage, 0, -1):
                self.intervals_per_stage.append(
                    (self.min_step + timesteps_per_stage*(i-1), self.min_step + timesteps_per_stage*i),
                )
            print(self.intervals_per_stage)

        # Annealed Mode
        if self.time_sampling == 'annealed':
            # Time Weights
            self.weights_dict = {
                'uniform': torch.ones_like(self.alphas_cumprod),
            }
            for k, v in self.weights_dict.items():
                v[:self.min_step] = 0.0
                v[self.max_step+1:] = 0.0
                self.weights_dict[k] = v / torch.sum(v)
            # Convert Weights to Iteration-Time Mapping
            weights_flip = self.weights_dict[self.time_prior].flip(dims=(0,))
            self.weights_cumsum = (weights_flip).cumsum(dim=0).detach().cpu().numpy()

    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_timestep(self, batch_size, train_step, max_iteration):
        if self.time_sampling == 'uniform':
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, [batch_size,], dtype=torch.long, device=self.device)
        elif self.time_sampling == 'constant':
            mid_step = (self.min_step + self.max_step) // 2
            t = torch.randint(mid_step, mid_step + 1, [batch_size,], dtype=torch.long, device=self.device)
        elif self.time_sampling == 'linear':
            timestep_delta = (self.max_step - self.min_step) / (max_iteration - 1)
            timestep = int(self.max_step - (train_step - 1) * timestep_delta)
            t = torch.ones([batch_size,], dtype=torch.long, device=self.device) * timestep
        elif self.time_sampling == 'stage':
            iters_per_stage = max_iteration // self.num_stage
            i_stage = min(train_step // iters_per_stage, self.num_stage - 1)
            min_step, max_step = self.intervals_per_stage[i_stage]
            t = torch.randint(min_step, max_step + 1, [batch_size,], dtype=torch.long, device=self.device)
        elif self.time_sampling == 'annealed':
            timestep = self.get_annealed_t_from_weights_cumsum(train_step, max_iteration)
            t = torch.ones([batch_size,], dtype=torch.long, device=self.device) * timestep
        else:
            raise NotImplementedError
        return t

    def get_annealed_t_from_weights_cumsum(self, train_step, max_iteration):
        # Current Training Stage
        current_state = train_step / max_iteration
        # Bi-Search
        t = self.num_timesteps - bisect.bisect_left(self.weights_cumsum, current_state)  # [0, ..., 1000]
        t = max(t, self.min_step)
        t = min(t, self.max_step)
        return t

    def draw_curves(self, save_dir):

        def _draw_curve(_x, _y, label):
            if isinstance(_x, torch.Tensor):
                _y = _y.detach().cpu().numpy()
            if isinstance(_y, torch.Tensor):
                _y = _y.detach().cpu().numpy()
            plt.plot(_x, _y, label=label)

        # Figure: Weights
        if self.time_sampling == 'annealed':
            time_steps = list(range(1, 1000+1))
            plt.figure()
            for k, v in self.weights_dict.items():
                _draw_curve(time_steps, v, k)
            plt.legend()
            plt.savefig(save_dir / 'tp_weights.png')
            plt.clf()

        # Figure: I2T
        train_steps = list(range(1, 1000+1))
        i_each_step = [i / len(train_steps) for i in train_steps]
        t_each_step = [self.get_timestep(1, i, len(train_steps)).item() for i in train_steps]
        plt.figure()
        _draw_curve(i_each_step, t_each_step, None)
        plt.savefig(save_dir / 'tp_i2t.png')
        plt.clf()
