import random
import numpy as np
import torch
import torch.nn.functional as F

from core.nerf.utils.render_utils import safe_normalize


def angle2center(radius, thetas, phis):
    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]
    return centers


def uniform_sphere_sampling(radius, size, device):
    unit_centers = F.normalize(
        torch.stack([
            (torch.rand(size, device=device) - 0.5) * 2.0,
            torch.rand(size, device=device),
            (torch.rand(size, device=device) - 0.5) * 2.0,
        ], dim=-1), p=2, dim=1)
    thetas = torch.acos(unit_centers[:,1])
    phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
    phis[phis < 0] += 2 * np.pi
    centers = unit_centers * radius.unsqueeze(-1)
    return thetas, phis, centers


def rand_pose(size, device, radius_range=(1, 1.5), theta_range=(0, 150), phi_range=(0, 360), view_prompt=None, jitter=False, vertical_jitter=None, batched=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)

    dirs = None

    if random.random() < uniform_sphere_rate:
        assert batched is False
        thetas, phis, centers = uniform_sphere_sampling(radius, size, device)
    else:
        if not batched:
            thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        else:
            phis, thetas, dirs = view_prompt.get_a_batch(size, device=device)
        centers = angle2center(radius, thetas, phis)

    if view_prompt is not None and dirs is None:
        dirs = view_prompt.get_view_direction(thetas, phis)

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    lookat_vector = safe_normalize(targets - centers)

    # vertical jitters
    if vertical_jitter is not None:
        vertical_offsets = np.random.uniform(*vertical_jitter)  # -0.5, +0.5
        lookat_vector[..., 1] += vertical_offsets

    # other vector
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(lookat_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, lookat_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, lookat_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses, dirs


def circle_pose(device, radius=1.25, theta=80, phi=0, return_dirs=False, view_prompt=None):
    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)
    centers = angle2center(radius, thetas, phis)

    lookat_vector = - safe_normalize(centers)
    right_vector = safe_normalize(torch.cross(lookat_vector, torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0), dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, lookat_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, lookat_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = view_prompt.get_view_direction(thetas, phis)
    else:
        dirs = None

    return poses, dirs


def custom_pose(percent, eval_theta, fix_camera=False, camera_track='circle'):
    def _linear_func1(x):
        # -1 -> +1 -> -1
        if x < 0.5:
            return x * 4 - 1
        else:
            return 1 - (x - 0.5) * 4

    def _linear_func2(x):
        # 0 -> +1 -> 0 -> -1 -> 0
        if x < 0.25:
            return x * 4
        elif 0.25 <= x <= 0.75:
            return - (x - 0.25) * 4 + 1
        else:
            return (x - 0.75) * 4 - 1
    
    # Sample phi and theta

    if fix_camera or camera_track == 'fixed':
        phi = 0
        theta = eval_theta
    elif camera_track == 'circle':
        phi = percent * 360
        theta = eval_theta
    elif camera_track == 'shake_linear':
        delta_phi = 45
        delta_theta = 20
        phi = _linear_func1(percent) * delta_phi
        theta = eval_theta + _linear_func2(percent) * delta_theta
    elif camera_track == 'shake':
        delta_phi = 45
        delta_theta = 20
        phi = np.sin(percent * 2 * np.pi + np.pi / 2) * delta_phi
        theta = eval_theta + np.sin(percent * 2 * np.pi) * delta_theta
    elif camera_track == 'wave':
        delta_theta = 30
        phi = percent * 360
        theta = eval_theta + np.sin(percent * 2 * np.pi) * delta_theta
    else:
        assert 0, camera_track
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    return phi, theta


def index2pose(index, H=512, W=512, radius=3.0, theta=90, phi=0, eval_size=100, camera_track='circle', device='cuda'):
    # phi = (index / eval_size) * 360
    phi, theta = custom_pose(index / eval_size, eval_theta=theta, fix_camera=False, camera_track=camera_track)
    poses, _ = circle_pose(device=device, radius=radius, theta=theta, phi=phi)
    # fixed focal
    cx = H / 2
    cy = W / 2
    fov = (40 + 70) / 2
    focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
    intrinsics = np.array([focal, focal, cx, cy])

    return intrinsics, poses.cpu().numpy()


def visualize_pose(poses, size=0.1):
    # poses: [B, 4, 4]
    import trimesh
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def NeRF_data_to_standard(intrinsics, cam2world, H=None, W=None):
    # Tensor to Numpy
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(cam2world, torch.Tensor):
        cam2world = cam2world.cpu().numpy()
    # intrinsics
    intrinsics = intrinsics_Vec2Mat(intrinsics, H=H, W=W)
    # Extrinsic
    if cam2world.ndim == 3:
        assert cam2world.shape[0] == 1
        cam2world = cam2world[0]
    extrinsic = SE3_inverse(cam2world)
    return intrinsics, extrinsic


def intrinsics_Vec2Mat(intrinsics, H=None, W=None):
    """
    Input:
        intrinsics: np.array, shape = (4,)
    Return:
        intrinsics: np.array, shape = (3, 3)
    """
    assert intrinsics.ndim == 1 and intrinsics.shape[-1] == 4
    if H is None or W is None:
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ])
    else:
        Hc, Wc = intrinsics[2] * 2, intrinsics[3] * 2
        K = np.array([
            [intrinsics[0] * H / Hc, 0.0, H / 2],
            [0.0, intrinsics[1] * W / Wc, W / 2],
            [0.0, 0.0, 1.0],
        ])
    return K


def SE3_Mat2RT(extrinsic):
    """
    Input:
        extrinsic: np.array, shape = (N, 4, 4) or (4, 4)
    Return:
        R: np.array, shape = (N, 3, 3) or (3, 3)
        T: np.array, shape = (N, 3, 1) or (3, 1)
    """
    if extrinsic.ndim == 2:
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3][:, np.newaxis]
    elif extrinsic.ndim == 3:
        R = extrinsic[:, :3, :3]
        T = extrinsic[:, :3, 3][:, np.newaxis]
    return R, T


def SE3_RT2Mat(R, T):
    """
    Input:
        R: np.array, shape = (N, 3, 3) or (3, 3)
        T: np.array, shape = (N, 3, 1) or (3, 1)
    Return:
        extrinsic: np.array, shape = (N, 4, 4) or (4, 4)
    """
    if R.ndim == 2 and T.ndim == 2:
        extrinsic = np.zeros((4, 4))
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T[:, 0]
    elif R.ndim == 3 and T.ndim == 3:
        extrinsic = np.zeros((R.shape[0], 4, 4))
        extrinsic[:, :3, :3] = R
        extrinsic[:, :3, 3] = T[:, :, 0]
    return extrinsic


def SE3_inverse(mat):
    """
    Input:
        mat: np.array, shape = (4, 4)
    Return:
        mat: np.array, shape = (4, 4)
    """
    R, T = SE3_Mat2RT(mat)
    R = np.linalg.inv(R)
    T = np.dot(R, - T)
    return SE3_RT2Mat(R, T)
