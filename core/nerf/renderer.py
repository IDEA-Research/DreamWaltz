import math
import torch
import torch.nn as nn
from random import choice

from loguru import logger

from configs import RenderConfig
from .utils.render_utils import sample_pdf, custom_meshgrid, safe_normalize, near_far_from_bound


class NeRFRenderer(nn.Module):
    def __init__(self, cfg: RenderConfig, latent_mode: bool=True):
        super().__init__()

        self.opt = cfg
        self.bound = cfg.bound
        self.cascade = 1 + math.ceil(math.log2(cfg.bound))
        self.grid_size = cfg.grid_size
        self.cuda_ray = cfg.cuda_ray
        self.min_near = cfg.min_near
        self.density_thresh = cfg.density_thresh
        self.bg_mode = cfg.bg_mode
        self.bg_radius = cfg.bg_radius if self.bg_mode == 'nerf' else 0.0
        self.latent_mode = latent_mode
        self.img_dims = 3+1 if self.latent_mode else 3
        if self.cuda_ray:
            logger.info('Loading CUDA ray marching module (compiling might take a while)...')
            from .raymarching import rgb as raymarchingrgb
            from .raymarching import latent as raymarchinglatent
            logger.info('\tDone.')
            self.raymarching = raymarchinglatent if self.latent_mode else raymarchingrgb

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-cfg.bound, -cfg.bound, -cfg.bound, cfg.bound, cfg.bound, cfg.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def build_extra_state(self, grid_size):
        # density grid
        density_grid = torch.zeros([self.cascade, grid_size ** 3]) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0
    
    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=None):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return

        if S is None:
            S = self.grid_size

        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = self.raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign 
                        tmp_grid[cas, indices] = sigmas

        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = self.raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def mix_background(self, image, weights_sum, bg_color, rays_d):
        if bg_color is not None:
            bg_mode = bg_color
        else:
            bg_mode = self.bg_mode

        if bg_mode in ('gaussian', 'normal'):
            bg_image = torch.randn_like(image)
        elif bg_mode == 'zero':
            bg_image = 0.0
        elif bg_mode in ('rand', 'random'):
            bg_image = torch.randn((1, self.img_dims)).to(image.device)
        elif bg_mode == 'nerf':
            bg_image = self.background(rays_d).to(image.device)  # [N, 3]
        else:
            bg_image = self.background(bg_mode).to(image.device)  # [N, 3]

        return image + (1 - weights_sum).unsqueeze(-1) * bg_image

    def run(self, rays_o, rays_d, num_steps=64, upsample_steps=64, light_d=None, ambient_ratio=1.0, shading='albedo',
             bg_color=None, perturb=False, disable_background=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        # nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # nears.unsqueeze_(-1)
        # fars.unsqueeze_(-1)
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        # query density and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        if shading == 'albedo':
            rgbs = density_outputs['albedo']
        else:
            _, rgbs = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, self.img_dims) # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]

        # calculate depth 
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if not disable_background:
            image = self.mix_background(image, weights_sum, bg_color, rays_d)

        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)
        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        # results['sigmas'] = sigmas
        # results['rgbs'] = rgbs
        # results['alphas'] = alphas

        return results

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False,
                 max_steps=1024, T_thresh=1e-4, disable_background=False):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        results = {}
        xyzs, sigmas = None, None
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, ts, rays = self.raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                self.cascade, self.grid_size, nears, fars, perturb, dt_gamma, max_steps)

            sigmas, rgbs = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = self.raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh)

            # weights normalization
            results['weights'] = weights

        else:
            # allocate outputs 
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, self.img_dims, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0

            while step < max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = self.raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound,
                     self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                self.raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

        # mix background color
        if not disable_background:
            image = self.mix_background(image, weights_sum, bg_color, rays_d)

        image = image.reshape(*prefix, self.img_dims)
        depth = depth.reshape(*prefix)
        weights_sum = weights_sum.reshape(*prefix)
        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        # results['sigmas'] = sigmas
        # results['rgbs'] = rgbs
        return results

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, self.img_dims), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch

            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results

    # @torch.no_grad()
    # def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):

    #     if resolution is None:
    #         resolution = self.grid_size

    #     if self.cuda_ray:
    #         density_thresh = min(self.mean_density, self.density_thresh) \
    #             if np.greater(self.mean_density, 0) else self.density_thresh
    #     else:
    #         density_thresh = self.density_thresh
        
    #     # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
    #     if self.opt.density_activation == 'softplus':
    #         density_thresh = density_thresh * 25
        
    #     sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    #     # query
    #     X = torch.linspace(-1, 1, resolution).split(S)
    #     Y = torch.linspace(-1, 1, resolution).split(S)
    #     Z = torch.linspace(-1, 1, resolution).split(S)

    #     for xi, xs in enumerate(X):
    #         for yi, ys in enumerate(Y):
    #             for zi, zs in enumerate(Z):
    #                 xx, yy, zz = custom_meshgrid(xs, ys, zs)
    #                 pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
    #                 val = self.density(pts.to(self.aabb_train.device))
    #                 sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

    #     print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

    #     vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
    #     vertices = vertices / (resolution - 1.0) * 2 - 1

    #     # clean
    #     vertices = vertices.astype(np.float32)
    #     triangles = triangles.astype(np.int32)
    #     vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)
        
    #     # decimation
    #     if decimate_target > 0 and triangles.shape[0] > decimate_target:
    #         vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

    #     v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
    #     f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

    #     # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    #     # mesh.export(os.path.join(path, f'mesh.ply'))

    #     def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
    #             fp.write(f'map_Kd {name}albedo.png \n')

    #     _export(v, f)
