import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import igl
from smplx.lbs import blend_shapes, batch_rodrigues, vertices2joints, batch_rigid_transform
from configs.train_config import RenderConfig
from core.nerf.encoder import get_encoder
from .network_utils import SkeletalSDF, NonRigidMLP


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x.float(), y.t().float(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


def to_RT(G):
    R = G[..., :3, :3]
    T = G[..., :3, 3]
    return R, T


def point_transform(pts, G=None, R=None, T=None):
    """
    Input:
        - pts: tensor, [..., 3]
        - G: tensor, [..., 4, 4]
        - R: tensor, [..., 3, 3]
        - T: tensor, [..., 3]
    """
    if G is not None:
        R, T = to_RT(G)
        # pts = to_homogeneous(pts).unsqueeze(-1)  # [..., 4, 1]
        # pts = torch.matmul(G, pts)
        # pts = pts[..., :3, 0]
    # else:
    pts = pts.unsqueeze(-1)  # [..., 3, 1]
    return torch.matmul(R, pts)[..., :, 0] + T


class MotionBasisComputer(object):
    r"""Compute motion bases between the target pose and canonical pose."""

    def __init__(self, smpl):
        super(MotionBasisComputer, self).__init__()
        self.num_joints = smpl.NUM_JOINTS + 1
        self.parents = smpl.parents  # [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        self.v_template = smpl.v_template
        self.faces = smpl.faces
        self.shapedirs = smpl.shapedirs
        self.J_regressor = smpl.J_regressor
        self.lbs_weights = smpl.lbs_weights
        # default params
        self.betas = smpl.betas
        self.global_orient = smpl.global_orient

    def get_standard_joints(self, betas):
        # Add shape contribution
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        # Get the joints: NxJx3 tensor
        return vertices2joints(self.J_regressor, v_shaped)

    def smpl_params_to_T(self, body_pose, betas=None, global_orient=None, transl=None, use_vertex=False, **kwargs):
        r""" Convert smpl params to rigid transform matrix.

        Args:
            - body_pose:      Tensor, [Batch_Size, (Total_Joints - 1) x 3]
            - betas:          Tensor, [Batch_Size, 10]
            - global_orient:  Tensor, [Batch_Size, 3]
            - transl:         Tensor,  [Batch_Size, 3]

        Returns:
            - posed_joints : torch.tensor BxNx3
                The locations of the joints after applying the pose rotations
            - rel_transforms : torch.tensor BxNx4x4
                The relative (with respect to the root joint) rigid transformations for all the joints
        """
        batch_size = body_pose.shape[0]

        if betas is None:
            betas = self.betas
        if global_orient is None:
            global_orient = self.global_orient

        # convert rotation vectors to rotation matrix
        full_pose = torch.cat([global_orient, body_pose], dim=1)  # [N, J*3]
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        # joint transform
        joints = self.get_standard_joints(betas)
        posed_joints, T_joint = batch_rigid_transform(rot_mats, joints, self.parents)

        if not use_vertex:
            # translation
            if transl is not None:
                transl = transl.unsqueeze(1)  # [B, 1, 3]
                posed_joints += transl
                T_joint[:, :, :3, 3] += transl
            # return
            return posed_joints, T_joint
        else:
            # vertex transform
            W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
            T_vertex = torch.matmul(W, T_joint.view(batch_size, self.num_joints, 16)).view(batch_size, -1, 4, 4)
            # vertex transform
            # homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
            # v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
            # v_homo = torch.matmul(T_vertex, torch.unsqueeze(v_posed_homo, dim=-1))
            # verts = v_homo[:, :, :3, 0]
            posed_verts = None
            # translation
            if transl is not None:
                transl = transl.unsqueeze(1)  # [B, 1, 3]
                posed_joints += transl
                T_joint[:, :, :3, 3] += transl
                T_vertex[:, :, :3, 3] += transl
            # return
            return posed_joints, T_joint, posed_verts, T_vertex

    def observation_to_standard_to_canonical(self, Gs_obs, Gs_cnl, ret_RT=False):
        r"""
        Args:
            Gs_obs:    Tensor (B, NUM_JOINTS/NUM_VERTICES, 4, 4)
            Gs_cnl:    Tensor (B, NUM_JOINTS/NUM_VERTICES, 4, 4)

        Returns:
            Rs:       Tensor (B, NUM_JOINTS/NUM_VERTICES, 3, 3)
            Ts:       Tensor (B, NUM_JOINTS/NUM_VERTICES, 3)
        """
        batch_size = Gs_obs.shape[0]

        Gs_obs = Gs_obs.view(-1, 4, 4)
        inv_Gs_obs = torch.inverse(Gs_obs)
        Gs_cnl = Gs_cnl.view(-1, 4, 4)

        Gs_final = torch.matmul(Gs_cnl, inv_Gs_obs).view(batch_size, -1, 4, 4)

        if ret_RT:
            Rs = Gs_final[:, :, :3, :3]
            Ts = Gs_final[:, :, :3, 3]
            return Rs, Ts
        else:
            return Gs_final


class SkeletalMotion(nn.Module):
    def __init__(self, cfg: RenderConfig, smpl, smpl_data_cnl) -> None:
        super().__init__()
        self.mode = cfg.skeletal_motion
        # Motion Basis
        self.motion_basis = MotionBasisComputer(smpl)
        self.num_joints = self.motion_basis.num_joints
        self.faces = self.motion_basis.faces.astype(dtype=np.int64)
        # Joint Encode
        self.skeletal_motion_encode = cfg.skeletal_motion_encode
        if self.skeletal_motion_encode == 'freq':
            self.freq_encoder, pos_embed_size = get_encoder('frequency', input_dim=6, multires=4)
            # self.freq_encoder, pos_embed_size = get_encoder('frequency', input_dim=self.num_joints*3, multires=4)
        elif self.skeletal_motion_encode == 'euclidean':
            pos_embed_size = self.num_joints
        elif self.skeletal_motion_encode == 'offset':
            pos_embed_size = self.num_joints * 3
        elif self.skeletal_motion_encode == 'vertex_offset':
            pos_embed_size = self.motion_basis.v_template.shape[0] * 3
        else:
            assert 0, self.skeletal_motion_encode
        # Build Network
        self.sk_sdf = SkeletalSDF(input_size=pos_embed_size, output_size=1)
        self.sk_thres = cfg.skeletal_motion_thres
        # self.sk_thres = nn.Parameter(torch.tensor(cfg.skeletal_motion_thres), requires_grad=False)
        # Non-Rigid Motion
        self.nr_motion_mlp = NonRigidMLP(cfg, input_size=pos_embed_size, condition_size=0)
        # Set Canonical Config
        self.set_canonical_config(smpl_data_cnl)

    def set_canonical_config(self, smpl_data_cnl):
        vertices_cnl = torch.from_numpy(smpl_data_cnl['vertices'][0])
        _, Gs_cnl_joint, _, Gs_cnl_vertex = \
            self.motion_basis.smpl_params_to_T(**smpl_data_cnl, use_vertex=True)  # BxNx3
        self.register_buffer('Gs_cnl_joint', Gs_cnl_joint)
        self.register_buffer('Gs_cnl_vertex', Gs_cnl_vertex)
        self.vertices_cnl = vertices_cnl

    def encode_offsets(self, pts, joints):
        """
        Inputs:
            - pts: tensor, [N, 3]
            - joints: tensor, [J, 3]
        Outputs:
            - embed: tensor, [N, D]
        """
        if self.skeletal_motion_encode == 'euclidean':
            return euclidean_dist(pts, joints)  # [N, J]
        elif self.skeletal_motion_encode == 'freq':
            offsets = pts.unsqueeze(1) - joints.unsqueeze(0)  # [N, J, 3]
            offsets = offsets.reshape(pts.size(0), -1)  # [N, J*3]
            return self.freq_encoder(offsets)
        elif self.skeletal_motion_encode in ('offset', 'vertex_offset'):
            offsets = pts.unsqueeze(1) - joints.unsqueeze(0)  # [N, J, 3]
            return offsets.reshape(pts.size(0), -1)  # [N, J*3]

    def forward(self, pts, smpl_data_obs):
        mode = self.mode
        extra_ouptuts = {}
        if mode == 'closest_joint':
            pts_new = self.forward_closest_joint(pts, smpl_data_obs)
        elif mode == 'closest_vertex':
            pts_new, extra_ouptuts = self.forward_closest_vertex(pts, smpl_data_obs)
        elif mode == 'ours':
            pts_new, extra_ouptuts = self.forward_ours(pts, smpl_data_obs)
        elif mode == 'ours_wo_weight':
            pts_new, extra_ouptuts = self.forward_ours_wo_weighting(pts, smpl_data_obs)
        elif mode in ('none', 'None'):
            pts_new = pts
        else:
            assert 0
        return pts_new.float(), extra_ouptuts

    def forward_closest_vertex(self, pts, smpl_data_obs):
        with torch.no_grad():
            _, _, _, Gs_obs_vertex = self.motion_basis.smpl_params_to_T(**smpl_data_obs, use_vertex=True)
            Gs_obs_vertex = Gs_obs_vertex.to(pts.device)
            Gs_vertex = self.motion_basis.observation_to_standard_to_canonical(Gs_obs_vertex, self.Gs_cnl_vertex, ret_RT=False)[0]  # [V, 4, 4]
            # Calculate Distance
            vertices_obs = smpl_data_obs['vertices'][0]  # [V, 3]
            dists, face_id, _ = igl.point_mesh_squared_distance(pts.cpu().numpy(), vertices_obs, self.faces)
            vertices_idx = self.faces[face_id][:, np.random.randint(0, 3)]
            # To Torch
            dists = torch.from_numpy(dists).to(pts.device).half()
            return point_transform(pts, Gs_vertex[vertices_idx, :, :]), {'dist': dists}

    def forward_closest_joint(self, pts, smpl_data_obs):
        with torch.no_grad():
            _, Gs_obs_joint = self.motion_basis.smpl_params_to_T(**smpl_data_obs, use_vertex=False)
            Gs_obs_joint = Gs_obs_joint.to(pts.device)
            Gs_joint = self.motion_basis.observation_to_standard_to_canonical(Gs_obs_joint, self.Gs_cnl_joint, ret_RT=False)[0]  # [J, 4, 4]

            joints_obs = torch.from_numpy(smpl_data_obs['joints'][0]).to(pts.device)  # [J, 3]
            joints_idx = torch.argmin(euclidean_dist(pts, joints_obs), dim=1)   # [N]

            pts_new = point_transform(pts, Gs_joint[joints_idx, :, :])
            return pts_new

    @staticmethod
    def find_nearest_points(pts: torch.Tensor, verts: torch.Tensor):
        """
          - pts: torch.Tensor, [N, 3]
          - verts: torch.Tensor, [V, 3]
        """
        diffs = ((pts[..., :, None, :] - verts[..., None, :, :]) ** 2).sum(-1)  # [N, 1, 3] - [1, V, 3] -> [N, V, 3] -> [N x V]
        nearest_vi = diffs.argmin(-1)
        return nearest_vi, diffs

    def forward_ours(self, pts, smpl_data_obs):
        """
        Vars:
            - dists:            (N,) float
            - face_id:          (N,) int
            - vertices_idx:     (N,) int
        """
        device = pts.device
        with torch.no_grad():
            # Motion Basis
            _, _, _, Gs_obs_vertex = self.motion_basis.smpl_params_to_T(**smpl_data_obs, use_vertex=True)
            Gs_obs_vertex = Gs_obs_vertex.to(pts.device)
            Gs_vertex = self.motion_basis.observation_to_standard_to_canonical(Gs_obs_vertex, self.Gs_cnl_vertex, ret_RT=False)[0]  # [V, 4, 4]
            # Signed Distance
            vertices_obs = smpl_data_obs['vertices'][0]  # [V, 3]
            dists, face_id, _ = igl.point_mesh_squared_distance(pts.cpu().numpy(), vertices_obs, self.faces)
            # dists, face_id, _ = igl.signed_distance(pts.cpu().numpy(), vertices_obs, self.faces, return_normals=False)
            # Skeletal Transform
            vertices_idx = self.faces[face_id][:, np.random.randint(0, 3)]
            pts_xyz_cnl = point_transform(pts, Gs_vertex[vertices_idx, :, :])
            # Position Embedding
            vertices_xyz_cnl = torch.from_numpy(vertices_obs[vertices_idx]).to(device)
            pos_embed = self.freq_encoder(torch.cat((pts_xyz_cnl, vertices_xyz_cnl), dim=-1))
            # To Torch
            dists = torch.from_numpy(dists).to(pts.device).half()

        # Mask
        thres_init = 0.0
        # thres ~ (thres_init, +inf)
        thres = self.sk_sdf.forward_mlp(pos_embed)[:, 0]
        thres = torch.relu(thres) + thres_init
        # thres = torch.clamp(thres, 0.0, 0.5)
        mask = torch.sigmoid(- (dists - thres) / self.sk_thres)

        # Non-Rigid Motion
        offsets = self.nr_motion_mlp(pos_embed)  # [N, 3]
        pts_xyz_cnl += offsets

        # Return
        extra_ouptuts = {
            'mask': mask,
            'dist': dists,
        }
        return pts_xyz_cnl, extra_ouptuts

    def forward_ours_wo_weighting(self, pts, smpl_data_obs):
        with torch.no_grad():
            # Motion Basis
            _, _, _, Gs_obs_vertex = self.motion_basis.smpl_params_to_T(**smpl_data_obs, use_vertex=True)
            Gs_obs_vertex = Gs_obs_vertex.to(pts.device)
            Gs_vertex = self.motion_basis.observation_to_standard_to_canonical(Gs_obs_vertex, self.Gs_cnl_vertex, ret_RT=False)[0]  # [V, 4, 4]
            # Signed Distance
            vertices_obs = smpl_data_obs['vertices'][0]  # [V, 3]
            dists, face_id, _ = igl.point_mesh_squared_distance(pts.cpu().numpy(), vertices_obs, self.faces)
            # Skeletal Transform
            vertices_idx = self.faces[face_id][:, np.random.randint(0, 3)]
            pts_xyz_cnl = point_transform(pts, Gs_vertex[vertices_idx, :, :])

        # Return
        extra_ouptuts = {
            'mask': None,
            'dist': dists,
        }
        return pts_xyz_cnl, extra_ouptuts
