import math
import torch
import torch.nn as nn
from configs.train_config import RenderConfig


###############################################################################
## Network Components - compute motion base
###############################################################################
SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}


class MotionBasisComputer(nn.Module):
    r"""Compute motion bases between the target pose and canonical pose."""

    def __init__(self, total_bones=24):
        super(MotionBasisComputer, self).__init__()
        self.total_bones = total_bones

    def _construct_G(self, R_mtx, T):
        r''' Tile ration matrix and translation vector to build a 4x4 matrix.

        Args:
            R_mtx: Tensor (B, TOTAL_BONES, 3, 3)
            T:     Tensor (B, TOTAL_BONES, 3)

        Returns:
            G:     Tensor (B, TOTAL_BONES, 4, 4)
        '''
        batch_size, total_bones = R_mtx.shape[:2]
        assert total_bones == self.total_bones

        G = torch.zeros(size=(batch_size, total_bones, 4, 4),
                        dtype=R_mtx.dtype, device=R_mtx.device)
        G[:, :, :3, :3] = R_mtx
        G[:, :, :3, 3] = T
        G[:, :, 3, 3] = 1.0
    
        return G

    def forward(self, dst_Rs, dst_Ts, cnl_gtfms):
        r"""
        Args:
            dst_Rs:    Tensor (B, TOTAL_BONES, 3, 3)
            dst_Ts:    Tensor (B, TOTAL_BONES, 3)
            cnl_gtfms: Tensor (B, TOTAL_BONES, 4, 4)
                
        Returns:
            scale_Rs: Tensor (B, TOTAL_BONES, 3, 3)
            Ts:       Tensor (B, TOTAL_BONES, 3)
        """
        dst_gtfms = torch.zeros_like(cnl_gtfms)

        local_Gs = self._construct_G(dst_Rs, dst_Ts)    
        dst_gtfms[:, 0, :, :] = local_Gs[:, 0, :, :]

        for i in range(1, self.total_bones):
            dst_gtfms[:, i, :, :] = torch.matmul(
                                        dst_gtfms[:, SMPL_PARENT[i], 
                                                  :, :].clone(),
                                        local_Gs[:, i, :, :])

        dst_gtfms = dst_gtfms.view(-1, 4, 4)
        inv_dst_gtfms = torch.inverse(dst_gtfms)
        
        cnl_gtfms = cnl_gtfms.view(-1, 4, 4)
        f_mtx = torch.matmul(cnl_gtfms, inv_dst_gtfms)
        f_mtx = f_mtx.view(-1, self.total_bones, 4, 4)

        scale_Rs = f_mtx[:, :, :3, :3]
        Ts = f_mtx[:, :, :3, 3]

        return scale_Rs, Ts


###############################################################################
## Init Functions
###############################################################################
def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]


def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


###############################################################################
## Networks
###############################################################################
class SkeletalMotionMLP(nn.Module):
    def __init__(self, input_size, output_size=24, mlp_width=64, mlp_depth=5):
        super(SkeletalMotionMLP, self).__init__()

        block_mlps = [nn.Linear(input_size, mlp_width), nn.ReLU()]

        for i in range(1, mlp_depth):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, output_size)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, x):
        for i in range(len(self.block_mlps)):
            x = self.block_mlps[i](x)
        x = torch.softmax(x, dim=-1)
        # x = torch.sigmoid(x)
        return x


class SkeletalSDF(nn.Module):
    def __init__(self, input_size, output_size=1, mlp_width=64, mlp_depth=5):
        super(SkeletalSDF, self).__init__()

        block_mlps = [nn.Linear(input_size, mlp_width), nn.ReLU()]

        for i in range(1, mlp_depth):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, output_size)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, x, a=0.001):
        x = self.forward_mlp(x)
        # density = torch.sigmoid(- x / a) / a
        density = torch.sigmoid(- x / a)
        return density

    def forward_mlp(self, x):
        for i in range(len(self.block_mlps)):
            x = self.block_mlps[i](x)
        return x


class NonRigidMLP(nn.Module):
    def __init__(self, cfg: RenderConfig, input_size, condition_size=0):
        super(NonRigidMLP, self).__init__()

        self.mode = cfg.non_rigid_motion

        self.block_mlps, self.layers_to_cat_inputs = self.build_mlp(input_size, condition_size)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def build_mlp(self, input_size, condition_size=0, mlp_width=128, mlp_depth=6, skips=[4]):

        block_mlps = [nn.Linear(input_size + condition_size, mlp_width), nn.ReLU()]

        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            if i in skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width + input_size, mlp_width), nn.ReLU()]
            else:
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 3)]

        return nn.ModuleList(block_mlps), layers_to_cat_inputs

    def forward(self, inputs, conditions=None):
        if self.mode in ('none', 'None'):
            offsets = 0.0
        else:
            offsets = self.forward_mlp(inputs, conditions)
        return offsets

    def forward_mlp(self, inputs, conditions=None):
        if conditions is not None:
            _inputs = torch.cat([inputs, conditions], dim=-1)
        else:
            _inputs = inputs

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                _inputs = torch.cat([_inputs, inputs], dim=-1)
            _inputs = self.block_mlps[i](_inputs)

        return _inputs


###############################################################################
## misc functions
###############################################################################
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
