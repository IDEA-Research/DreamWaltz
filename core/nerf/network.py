import torch
from torch import nn
import torch.nn.functional as F

from configs import RenderConfig
from .encoder import get_encoder
from .utils.nerf_utils import trunc_exp, MLP, NeRFType
from .utils.render_utils import safe_normalize
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self, cfg: RenderConfig, num_layers=3, hidden_dim=64, num_layers_bg=2, hidden_dim_bg=64, density_activation='exp'):

        super().__init__(cfg, latent_mode=cfg.nerf_type == NeRFType.latent)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        additional_dim_size = 1 if self.latent_mode else 0

        # Foreground Network
        self.encoder, self.in_dim = get_encoder(cfg.backbone, input_dim=3, desired_resolution=2048 * self.bound, interpolation='smoothstep')
        self.sigma_net = MLP(self.in_dim, 4 + additional_dim_size, hidden_dim, num_layers, bias=True)
        self.sigma_scale = torch.nn.Parameter(torch.tensor(0.0))

        if density_activation == 'exp':
            self.density_activation = trunc_exp
        elif density_activation == 'softplus':
            self.density_activation = F.softplus
        elif density_activation == 'scaling':
            def density_activation(x, density_shift=-1.0):
                x = x * torch.exp(self.sigma_scale)
                return F.softplus(x + density_shift)
            self.density_activation = density_activation
        else:
            assert 0
        self.density_blob_type = cfg.density_blob

        # Background Network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)
            self.bg_net = MLP(self.in_dim_bg, 3 + additional_dim_size, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None
            if self.latent_mode:
                self.bg_colors = {
                    'white': torch.tensor([[2.1750,  1.4431, -0.0317, -1.1624]]).cuda(),
                    'black': torch.tensor([[-0.9952, -2.6023,  1.1155,  1.2966]]).cuda(),
                    'gray': torch.tensor([[0.9053, -0.7003,  0.5424,  0.1057]]).cuda(),
                }
            else:
                self.bg_colors = {
                    'white': torch.tensor([[1.0, 1.0, 1.0]]).cuda(),
                    'black': torch.tensor([[0.0, 0.0, 0.0]]).cuda(),
                    'gray': torch.tensor([[0.5, 0.5, 0.5]]).cuda(),
                }

    # add a density blob to the scene center
    def density_blob(self, x, blob_density=10, blob_radius=0.5):
        # x: [B, N, 3]
        if self.density_blob_type == 'none':
            return 0.0
        d = (x ** 2).sum(-1)
        if self.density_blob_type == 'gaussian':
            g = 5 * torch.exp(-d / (2 * 0.2 ** 2))
        elif self.density_blob_type == 'sqrt':
            g = blob_density * (1 - torch.sqrt(d) / blob_radius)
        else:
            assert 0, self.density_blob_type
        return g

    def common_forward(self, x, mask=None, **kwargs):
        # sigma
        enc = self.encoder(x, bound=self.bound)
        h = self.sigma_net(enc)

        if self.density_blob_type == 'smpl':
            sigma = self.density_activation(h[..., 0] + self.density_smpl(x))
        else:
            sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = h[..., 1:]
        # albedo
        if not self.latent_mode:
            albedo = torch.sigmoid(h[..., 1:])
        # mask
        if mask is not None:
            sigma *= mask

        # return
        return sigma, albedo

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo = self.common_forward(x)
        
        if shading == 'albedo':
            color = albedo
        else:
            # normal
            normal = self.normal(x)

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ -l).clamp(min=0)  # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)

            if self.latent_mode:
                # pad color with a single dimension of zeros
                color = torch.cat([color, torch.zeros((color.shape[0], 1), device=color.device)], axis=1)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        sigma, albedo = self.common_forward(x)
        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def normal(self, x, normalize=True):
        normal = self.finite_difference_normal(x)
        if normalize:
            normal = safe_normalize(normal)
        normal[torch.isnan(normal)] = 0
        return normal

    def background(self, d):
        if isinstance(d, str):
            rgbs = self.bg_colors[d]
        else:
            h = self.encoder_bg(d)  # [N, C]
            rgbs = self.bg_net(h)
            if not self.latent_mode:
                rgbs = torch.sigmoid(rgbs)
        return rgbs

    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.sigma_scale, 'lr': lr},
        ]

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return normal
