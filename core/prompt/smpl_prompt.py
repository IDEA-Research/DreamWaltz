import os
import os.path as osp
import cv2
import math
import imageio
import torch
import torch.utils.dlpack
import numpy as np
from PIL import Image
import pickle
import os.path as osp
import open3d as o3d
from typing import Iterable

import smplx

from configs.paths import SMPL_ROOT, VPOSER_ROOT
from configs.train_config import PromptConfig
from core.utils.point3d import *
from core.utils.pose import index2pose, SE3_Mat2RT, NeRF_data_to_standard
from core.data.aist import AIST


def build_human_body_prior(model_path=VPOSER_ROOT):
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
    vp, vp_cfg = load_model(model_path, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
    if torch.cuda.is_available():
        vp = vp.to('cuda')
    return vp


def smpl_to_openpose(model_type='smplx', openpose_format='coco18', use_hands=True, use_face=True, use_face_contour=False):
    # https://github.com/vchoutas/smplify-x/blob/3e11ff1daed20c88cd00239abf5b9fc7ba856bb6/smplifyx/utils.py#L96
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
    elif openpose_format == 'coco18':
        # coco18_names = [
        #     'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        #     'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
        #     'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
        # ]
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)


class MySMPL(object):
    def __init__(self, batch_size, model_type, model_path=SMPL_ROOT, gender='neutral') -> None:
        # Build SMPL Model
        assert model_type in ('smpl', 'smplh', 'smplx')
        smpl_cfgs = {
            'model_path': model_path,
            'model_type': model_type,
            'gender': gender,
            'batch_size': batch_size,
            'num_betas': 10,
            'ext': 'npz',
            'use_face_contour': False,
        }
        self.model_type = model_type
        self.model = smplx.create(**smpl_cfgs)
        self.batch_size = batch_size
        # Build Pose Prior
        self.vp = build_human_body_prior()

    def sample_body_pose(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        body_pose = self.vp.sample_poses(num_poses=batch_size)['pose_body']  # tensor with shape of (N, 21, 3)
        if self.model_type == 'smpl':
            body_pose_hands = torch.zeros((batch_size, 2, 3)).to(body_pose.device)
            body_pose = torch.cat((body_pose, body_pose_hands), dim=1)
        body_pose = body_pose.contiguous().view(batch_size, -1).cpu()  # body_pose shape = (N, 63)
        return body_pose

    def normalize(self, vertices, joints, keypoints, scale=0.5, transl_mode='pelvis'):  # scale=0.5
        """
        Input & Return:
            vertices: np.array, [N, 10475, 3]
            joints: np.array, [N, 127, 3]
            keypoints: np.array, [N, 18, 3]
        """
        assert vertices.ndim == 3 and vertices.shape[-1] == 3
        assert joints.ndim == 3 and joints.shape[-1] == 3
        assert keypoints.ndim == 3 and keypoints.shape[-1] == 3
        # Translation
        if transl_mode == 'pelvis':
            center = joints[:, [0,], :]
        else:
            selected_indices = {
                'body': np.asarray([1, 8, 11]),
                'hip': np.asarray([8, 11]),
                'all': np.asarray([i for i in range(18)]),
            }
            center = keypoints[:, selected_indices[transl_mode], :].mean(axis=(0, 1), keepdims=True)  # shape = [1, 1, 3]
        vertices -= center
        joints -= center
        keypoints -= center
        # Scale
        if scale is not None:
            scale /= np.max(np.linalg.norm(keypoints, ord=2, axis=-1))
            vertices *= scale
            joints *= scale
            keypoints *= scale
        return vertices, joints, keypoints

    def __call__(self, body_pose=None, random_pose=True, **kwargs):
        """
        Input:
        Return:
            vertices: np.array, shape = (N, V, 3)
            joints: np.array, shape = (N, J, 3)
            keypoints: np.array, shape = (N, 18, 3)
        """
        # SMPL param sampling
        batch_size = self.batch_size
        if body_pose is None and random_pose:
            body_pose = self.sample_body_pose(batch_size=batch_size)
        # SMPL model inference
        output = self.model(body_pose=body_pose, return_verts=True, **kwargs)
        vertices = output.vertices.detach().cpu().numpy()  # Vertices shape = (N, 10475, 3)
        joints = output.joints.detach().cpu().numpy()      # Joints shape = (N, 127, 3)
        # Select keypoints
        keypoints = joints[..., smpl_to_openpose(self.model_type, openpose_format='coco18'), :]
        # Normalize
        # vertices, joints, keypoints = self.normalize(vertices, joints, keypoints)
        return vertices, joints, keypoints


class _HumanScene(object):
    face_indices = (0, 14, 15, 16, 17)
    body_indices = [i for i in range(18) if i not in (0, 14, 15, 16, 17)]
    face_keypoints = ('nose', 'r_eye', 'l_eye', 'r_ear', 'l_ear')

    def __init__(self, model_type, offset_y=0.25) -> None:
        self.model_type = model_type
        self.num_joints = 23 if model_type == 'smpl' else 21
        self.offset_y = offset_y

    def add_offset_to_smpl_params(self, smpl_params):
        if 'transl' in smpl_params:
            transl_offset = torch.zeros_like(smpl_params['transl'])
            transl_offset[..., 1] += self.offset_y
            smpl_params['transl'] += transl_offset  # [B, 3]
        else:
            transl_offset = torch.zeros_like(smpl_params['body_pose'])[..., :3]
            transl_offset[..., 1] += self.offset_y
            smpl_params['transl'] = transl_offset
        return smpl_params

    def build_scene(self):
        meshs = []
        ray_casting_scene = o3d.t.geometry.RaycastingScene()
        for each_vertices in self.vertices:
            mesh = o3d.geometry.TriangleMesh(
                vertices = o3d.utility.Vector3dVector(each_vertices),
                triangles = o3d.utility.Vector3iVector(self.triangles),
            )
            mesh.compute_vertex_normals()
            meshs.append(mesh)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            ray_casting_scene.add_triangles(mesh_t)
        return meshs, ray_casting_scene

    def export_depth_map(self, intrinsics, extrinsic, width=512, height=512, inverse=True, normalize=True):
        """
        Input:
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        """
        # Rays are 6D vectors with origin and ray direction.
        # Here we use a helper function to create rays for a pinhole camera.
        rays = self.ray_casting_scene.create_rays_pinhole(intrinsics, extrinsic, width_px=width, height_px=height)

        # Compute the ray intersections.
        ans = self.ray_casting_scene.cast_rays(rays)
        depth = ans['t_hit'].numpy()

        # Inverse and Normalize
        if inverse:
            depth = 1.0 / depth
        if normalize:
            depth -= np.min(depth)
            depth /= np.max(depth)
        image = np.asarray(depth * 255.0, np.uint8)
        image = np.stack([image, image, image], axis=2)
        return Image.fromarray(image)

    def export_pose_map(self, intrinsics, extrinsic, width=512, height=512, occlusion_culling=True):
        """
        Input:
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        Variable:
            self.keypoints: np.array, [N, K, 3]
        """
        # Init
        N, K, _ = self.keypoints.shape
        R, T = SE3_Mat2RT(extrinsic)
        # Transform
        kp_world = self.keypoints.reshape(-1, 3)
        kp_camera = transform_keypoints_to_novelview(kp_world, None, None, R, T)
        kp_image = project_camera3d_to_2d(kp_camera, intrinsics)  # [N*18, 2]
        kp_image = kp_image.reshape(N, K, 2)  # [N, 18, 2]
        # Occlusion
        if occlusion_culling:
            CAM_world = np.dot(np.linalg.inv(R), - T)
            occluded = self.detect_occlusion(CAM_world)  # [N, 18]
            kp_image[occluded, :] = None
        # Draw
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        image = self.draw_bodypose(canvas, kp_image)
        return image

    def export_mesh_map(self, intrinsics, extrinsic, width=512, height=512, focal=512.0, device=None):
        import pytorch3d
        import pytorch3d.renderer
        from scipy.spatial.transform import Rotation

        ''' Render the mesh under camera coordinates
        vertices: (N_v, 3), vertices of mesh
        faces: (N_f, 3), faces of mesh
        translation: (3, ), translations of mesh or camera
        focal: float, focal length of camera
        height: int, height of image
        width: int, width of image
        device: "cpu"/"cuda:0", device of torch
        :return: the rgba rendered image
        '''

        if device is None:
            device = torch.device('cuda')

        vertices = torch.from_numpy(self.vertices[0]).to(device)
        faces = torch.from_numpy(self.triangles.astype(np.int64)).to(device)
        # translation = self.smpl_params['transl'].to(device)

        # print(vertices.shape)     torch.Size([6890, 3])
        # print(faces.shape)        torch.Size([13776, 3])
        # print(translation.shape)  torch.Size([1, 3])

        # add the translation
        # vertices = vertices + translation

        # upside down the mesh
        # rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        # rot = torch.from_numpy(rot).to(device)

        # vertices = torch.matmul(rot, vertices.T).T

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
        textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
        mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces], textures=textures)

        focal = intrinsics[0][0].item()

        R = torch.from_numpy(extrinsic[np.newaxis, :3, :3])
        T = torch.from_numpy(extrinsic[np.newaxis, :3, 3])
        R = R.transpose(1,2)
        R[:,:,0:2] = -R[:,:,0:2] # y = -y, z = -z

        # print(R.shape)  # [4, 4]
        # print(T.shape)  # [4, 4]

        if not hasattr(self, 'mesh_renderer'):
            # Define the settings for rasterization and shading.
            raster_settings = pytorch3d.renderer.RasterizationSettings(
                # image_size=(height, width),   # (H, W)
                image_size=height,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Define the material
            materials = pytorch3d.renderer.Materials(
                ambient_color=((1, 1, 1),),
                diffuse_color=((1, 1, 1),),
                specular_color=((1, 1, 1),),
                shininess=64,
                device=device
            )

            # Place a directional light in front of the object.
            lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 2, 3),))
            # lights = pytorch3d.renderer.AmbientLights(ambient_color=((1.0, 1.0, 1.0),), device=device)

            # Create a phong renderer by composing a rasterizer and a shader.
            renderer = pytorch3d.renderer.MeshRenderer(
                rasterizer=pytorch3d.renderer.MeshRasterizer(
                    raster_settings=raster_settings
                ),
                shader=pytorch3d.renderer.SoftPhongShader(
                    device=device,
                    lights=lights,
                    materials=materials
                )
            )

            self.mesh_renderer = renderer

        # Initialize a camera.
        # R: Rotation matrix of shape (N, 3, 3)
        # T: Translation matrix of shape (N, 3)
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=(
                (2 * focal / min(height, width), 2 * focal / min(height, width)),
            ),
            R=R,
            T=T,
            image_size=((height, width),),
            device=device,
        )

        # Do rendering
        color_batch = self.mesh_renderer(mesh, cameras=cameras)  # [1, 512, 512, 4]

        # To Image
        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        input_img = np.zeros_like(color[:, :, :3])
        alpha = 1.0
        image_vis = alpha * color[:, :, :3] * valid_mask + (1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        image_vis = image_vis.astype(np.uint8)

        image = Image.fromarray(image_vis, mode='RGB')
        return image

    def export_distance(self, query_points: torch.Tensor, signed=True):
        """
        Input:
            query_points: torch.Tensor, [..., 3]
        Return:
            distances: torch.Tensor, [...]
        """
        if isinstance(query_points, torch.Tensor):
            device = query_points.device
            query_points = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query_points.detach().cpu()))
        if signed:
            distances = self.ray_casting_scene.compute_signed_distance(query_points)
        else:
            distances = self.ray_casting_scene.compute_distance(query_points)
        distances = torch.utils.dlpack.from_dlpack(distances.to_dlpack()).to(device)
        return distances

    def export_density(self, query_points: torch.Tensor, a=0.001):

        def inv_softplus(bias):
            """Inverse softplus function.
            Args:
                bias (float or tensor): the value to be softplus-inverted.
            """
            is_tensor = True
            if not isinstance(bias, torch.Tensor):
                is_tensor = False
                bias = torch.tensor(bias)
            out = bias.expm1().clamp_min(1e-6).log()
            if not is_tensor and out.numel() == 1:
                return out.item()
            return out

        distances = self.export_distance(query_points)
        # print(torch.min(distances), torch.mean(distances), torch.max(distances))
        density = torch.sigmoid(- distances / a) / a  # [0, 1000]
        # t = torch.sigmoid(- distances / a) / a  # [0, 1000]
        # density = torch.clamp(inv_softplus(t), min=0.0, max=1/a)

        return density

    def detect_occlusion(self, CAM_world, thres=0.02):
        """
        Input:
            CAM_world: np.array, [3, 1], camera position in world coordinates
        Return:
            occluded: np.array, bool, [N, 18]
        """
        KP_world = self.keypoints                                              # [N, 18, 3]
        CAM_world = np.broadcast_to(CAM_world.T, (*(KP_world.shape[:-1]), 3))  # [N, 18, 3]
        t_far = np.linalg.norm(KP_world - CAM_world, ord=2, axis=2)            # [N, 18]

        O_xyz = CAM_world
        D_xyz = KP_world - CAM_world
        D_xyz /= np.linalg.norm(D_xyz, ord=2, axis=2, keepdims=True)  # [N, 18, 3]
        rays = np.concatenate((O_xyz, D_xyz), axis=2)   # [N, 18, 6]

        outputs = self.ray_casting_scene.cast_rays(np.asarray(rays, dtype=np.float32))
        t_hit = outputs['t_hit'].numpy()                # [N, 18]
        geometry_ids = outputs['geometry_ids'].numpy()  # [2, 18], int

        occluded = (t_far - t_hit) > thres      # [N, 18], bool

        # Face Occlusion
        occluded_face = occluded[:, self.face_indices]  # [N, 5], bool

        # Body Occlusion
        occluded_body = occluded[:, self.body_indices]
        self_geometry_ids = np.array([[i for i in range(len(self.meshs))]]).T  # [N, 1]
        self_occluded_body = geometry_ids[:, self.body_indices] == self_geometry_ids  # [N, 13], bool
        occluded_body = occluded_body & (~ self_occluded_body)  # [N, 13], bool

        # Return
        occluded = np.zeros_like(occluded, dtype=np.bool_)
        occluded[:, self.face_indices] = occluded_face
        occluded[:, self.body_indices] = occluded_body
        return occluded

    @staticmethod
    def draw_bodypose(canvas, keypoints_2d):
        """
        canvas = np.zeros_like(input_image), np.array, [H x W x 3]
        keypoints_2d: np.array, [N, 18, 2], N is the number of people
        """
        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        assert keypoints_2d.shape[1] == 18 and keypoints_2d.ndim in (2, 3)
        if keypoints_2d.ndim == 2:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
        N = keypoints_2d.shape[0]
        for p in range(N):
            # draw points
            for i in range(18):
                x, y = keypoints_2d[p, i]
                if is_nan(x) or is_nan(y):
                    continue
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
            # draw lines
            for i in range(17):
                indices = np.array(limbSeq[i]) - 1
                cur_canvas = canvas.copy()
                X = keypoints_2d[p, indices, 1]
                Y = keypoints_2d[p, indices, 0]
                if is_nan(Y[0]) or is_nan(Y[1]) or is_nan(X[0]) or is_nan(X[1]):
                    continue
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return Image.fromarray(canvas)

    def export_geometry(self, plot_joints=True):
        # Export Geometry for Visualization
        geometry = self.meshs.copy()
        if plot_joints:
            for joints in self.joints:
                joints_pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(joints))
                joints_pcl.paint_uniform_color([1.0, 0.0, 0.0])
                geometry.append(joints_pcl)
        return geometry
        # import open3d.web_visualizer
        # open3d.web_visualizer.draw(geometry)

    def export_mesh_to_file(self, filename):
        o3d.io.write_triangle_mesh(str(filename), self.meshs[0], write_triangle_uvs=False)


class RandomScene(_HumanScene):
    def __init__(self, num_person=1, model_type='smpl', spacing=0.8, **kwargs) -> None:
        super().__init__(model_type=model_type, **kwargs)
        # Pre-defined Scene
        transl_pattern = {
            1: None,
            2: torch.tensor([[-spacing, 0.0, 0.0], [+spacing, 0.0, 0.0]]),
            3: torch.tensor([[0.0, 0.0, +spacing], [-spacing, 0.0, 0.0], [+spacing, 0.0, 0.0]]),
            4: torch.tensor([[+spacing, 0.0, +spacing], [+spacing, 0.0, -spacing],
                             [-spacing, 0.0, +spacing], [-spacing, 0.0, -spacing]]),
            5: torch.tensor([[+spacing, 0.0, +spacing], [+spacing, 0.0, -spacing], [0.0, 0.0, 0.0],
                             [-spacing, 0.0, +spacing], [-spacing, 0.0, -spacing]]),
        }
        self.num_person = num_person
        # Set SMPL Params
        smpl_params = {
            'transl': transl_pattern[num_person],
        }
        smpl_params = self.add_offset_to_smpl_params(smpl_params)
        self.smpl_params = smpl_params
        # SMPL Inference
        smpl = MySMPL(batch_size=num_person, model_type=model_type)
        self.vertices, self.joints, self.keypoints = smpl(**smpl_params)
        self.triangles = smpl.model.faces
        # Build Scene
        self.meshs, self.ray_casting_scene = self.build_scene()


class CanonicalScene(_HumanScene):
    def __init__(self, scene, model_type='smpl', **kwargs) -> None:
        super().__init__(model_type=model_type, **kwargs)
        # Load data from 3DPW dataset
        self.num_person = 1
        smpl_params = {
            'betas': torch.zeros((1, 10)),
            'body_pose': torch.zeros((1, self.num_joints*3)),
            'global_orient': torch.zeros((1, 3)),
        }
        if scene == 'canonical-T':
            body_pose = smpl_params['body_pose'].reshape(1, self.num_joints, 3)
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +1.0])
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -1.0])
            smpl_params['body_pose'] = body_pose.reshape(1, 63)
        elif scene == 'canonical-A':
            body_pose = smpl_params['body_pose'].reshape(1, self.num_joints, 3)
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi/4])
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +np.pi/4])
            smpl_params['body_pose'] = body_pose.reshape(1, -1)
        # Set SMPL Params
        smpl_params = self.add_offset_to_smpl_params(smpl_params)
        self.smpl_params = smpl_params
        # SMPL Model
        smpl = MySMPL(batch_size=self.num_person, model_type=model_type)
        self.smpl = smpl
        self.triangles = smpl.model.faces
        self.vertices, self.joints, self.keypoints = smpl(**smpl_params)
        # Build Scene
        self.meshs, self.ray_casting_scene = self.build_scene()

    def set_frame_index(self, frame_idx):
        pass


class AnimatedScene(_HumanScene):
    def __init__(self, scene, num_person=1, model_type='smpl', pop_transl=True, **kwargs) -> None:
        super().__init__(model_type=model_type, **kwargs)
        # Load data from dataset
        self.scene = scene
        if scene not in ('random', 'rand'):
            self.smpl_seqs, self.num_person, self.num_frame = self.load_smpl_sequences(scene, num_person=num_person, model_type=model_type)
        else:
            self.smpl_seqs, self.num_person, self.num_frame = {}, 1, 0
        # SMPL Model
        self.smpl = MySMPL(batch_size=self.num_person, model_type=model_type)
        self.smpl_params = None
        self.triangles = self.smpl.model.faces
        # Initialization
        self.vertices, self.joints, self.keypoints = None, None, None
        self.meshs, self.ray_casting_scene = None, None
        # Others
        self.pop_transl = pop_transl

    def load_smpl_sequences(self, scene, num_person, model_type):
        # SceneName Format: "dance", "dance,200-275", etc.
        scene_args = scene.split(',')
        scene = scene_args[0]
        if len(scene_args) > 1:
            frame_interval = scene_args[1]
            frame_interval = tuple(map(int, frame_interval.split('-'))) if '-' in frame_interval else eval(frame_interval)
        else:
            frame_interval = None
        # Load Data
        smpl_seqs = AIST().get_smpl_params(scene, model_type=model_type)
        # Preprocess SMPL Seqs
        # identity_idx = [i for i in range(num_person)]
        identity_idx = None
        smpl_seqs = self.preprocess_smpl_sequences(smpl_seqs, identity_idx=identity_idx, frame_interval=frame_interval)
        num_person, num_frame, _ = smpl_seqs['body_pose'].shape
        # Return
        return smpl_seqs, num_person, num_frame

    def preprocess_smpl_sequences(self, smpl_seqs, identity_idx, frame_interval, to_tensor=True):

        global_orient = smpl_seqs['global_orient']
        body_pose = smpl_seqs['body_pose']
        betas = smpl_seqs['betas']
        transl = smpl_seqs['transl']

        if identity_idx is not None:
            if isinstance(identity_idx, Iterable):
                global_orient = global_orient[identity_idx, ...]
                body_pose = body_pose[identity_idx, ...]
                betas = betas[identity_idx, ...]
                transl = transl[identity_idx, ...]
            else:
                global_orient = global_orient[identity_idx][np.newaxis, ...]
                body_pose = body_pose[identity_idx][np.newaxis, ...]
                betas = betas[identity_idx][np.newaxis, ...]
                transl = transl[identity_idx][np.newaxis, ...]

        if frame_interval is not None:
            if isinstance(frame_interval, Iterable):
                global_orient = global_orient[:, range(*frame_interval), :]
                body_pose = body_pose[:, range(*frame_interval), :]
                transl = transl[:, range(*frame_interval), :]
            else:
                fid = frame_interval
                global_orient = global_orient[:, fid:fid+1, :]
                body_pose = body_pose[:, fid:fid+1, :]
                transl = transl[:, fid:fid+1, :]

        smpl_seqs = {
            'global_orient': global_orient,  # (2, F, 3)
            'body_pose': body_pose,          # (2, F, 63/69)
            'betas': betas,                  # (2, F, 10)
            'transl': transl,                # (2, F, 3)
        }

        if to_tensor:
            for k in smpl_seqs.keys():
                smpl_seqs[k] = torch.tensor(smpl_seqs[k], dtype=torch.float)

        return smpl_seqs

    def set_frame_index(self, frame_idx):
        # Select Frame
        smpl_params = {}
        if self.scene in ('random', 'rand'):
            smpl_params['body_pose'] = self.smpl.sample_body_pose(self.smpl.batch_size)
        else:
            if frame_idx is None:
                frame_idx = np.random.randint(0, self.num_frame)
            frame_idx %= self.num_frame
            for k, v in self.smpl_seqs.items():
                if v.ndim == 3:
                    smpl_params[k] = v[:, frame_idx, :]
                elif v.ndim == 2:
                    smpl_params[k] = v
        # Adjust Translate Vector to Center
        if 'transl' in smpl_params.keys():
            if self.num_person == 1:
                if self.pop_transl:
                    smpl_params.pop('transl')
                else:
                    raise NotImplementedError
                    # smpl_params['transl'] -= torch.mean(self.smpl_seqs['transl'], dim=(0, 1), keepdim=True)[:, 0, :]
            else:
                transl = smpl_params['transl']  # [N, 3]
                smpl_params['transl'] -= torch.mean(transl, dim=0, keepdim=True)  # [N, 3] - [1, 3]
        # Set SMPL Params
        smpl_params = self.add_offset_to_smpl_params(smpl_params)
        self.smpl_params = smpl_params
        # SMPL Inference
        self.vertices, self.joints, self.keypoints = self.smpl(**smpl_params)
        # Build Scene
        self.meshs, self.ray_casting_scene = self.build_scene()


# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
class SMPLPrompt:
    def __init__(self, cfg: PromptConfig, cond_type, scene='canonical-A', num_person=1, height=512, width=512):
        # Init
        self.cond_type = cond_type
        self.height, self.width = height, width
        # Build Scene
        if scene == 'random':
            self.hs = RandomScene(num_person=num_person, offset_y=cfg.smpl_offset_y)
        elif scene.startswith('canonical'):
            self.hs = CanonicalScene(scene=scene, offset_y=cfg.smpl_offset_y)
        else:
            assert 0, scene
        self.num_person = self.hs.num_person

    def __call__(self, intrinsics, cam2world, cond_type=None, frame_idx=None):
        """
        Input:
            intrinsics: shape = [4, ]
            cam2world: shape = [N, 4, 4]
            cond_type: List[str]
        Return:
            cond_images: list of [PIL.Image]
        """
        if cond_type is None:
            cond_type = self.cond_type
        if isinstance(cond_type, str):
            cond_type = [cond_type,]
        intrinsics, extrinsic = NeRF_data_to_standard(intrinsics, cam2world, H=self.height, W=self.width)
        cond_images = []
        for _cond in cond_type:
            if _cond == 'pose':
                cond_image = self.hs.export_pose_map(intrinsics, extrinsic, width=self.width, height=self.height)
            elif _cond == 'depth':
                cond_image = self.hs.export_depth_map(intrinsics, extrinsic, width=self.width, height=self.height)
            elif _cond == 'mesh':
                cond_image = self.hs.export_mesh_map(intrinsics, extrinsic, width=self.width, height=self.height)
            else:
                assert 0, _cond
            cond_images.append(cond_image)
        return cond_images

    def write_video(self, save_dir='./', save_image='output.png', save_video='output.mp4', cond_type=None):
        import os
        import os.path as osp
        os.makedirs(save_dir, exist_ok=True)
        images = []
        for i in range(100):
            intrinsics, cam2world = index2pose(i, H=self.height, W=self.width)
            image = self(intrinsics, cam2world, cond_type=cond_type)[0]
            if i == 0:
                image.save(osp.join(save_dir, save_image))
            images.append(np.array(image))
        imageio.mimsave(osp.join(save_dir, save_video), np.array(images), fps=25, quality=8, macro_block_size=1)
        return image


class AnimatedSMPLPrompt:
    def __init__(self, cfg: PromptConfig, cond_type, scene='dance', num_person=1, height=512, width=512):
        # Init
        self.cond_type = cond_type
        self.height, self.width = height, width
        # Animated Scene Initialization
        if 'canonical' in scene:
            self.hs = CanonicalScene(scene=scene, offset_y=cfg.smpl_offset_y)
            self.num_person = 1
            self.num_frame = 1
        else:
            self.hs = AnimatedScene(scene=scene, num_person=num_person, offset_y=cfg.smpl_offset_y, pop_transl=cfg.pop_transl)
            self.num_person = self.hs.num_person
            self.num_frame = self.hs.num_frame
        # Canonical Scene Initialization
        self.hs_canonical = CanonicalScene(scene='canonical-A', offset_y=cfg.smpl_offset_y)
        # Others
        self.set_frame_index = self.hs.set_frame_index

    def __call__(self, intrinsics, cam2world, cond_type=None, frame_idx=None):
        """
        Input:
            intrinsics: shape = [4, ]
            cam2world: shape = [N, 4, 4]
            cond_type: List[str]
        Return:
            cond_images: list of [PIL.Image]
        """
        # Sampling
        self.hs.set_frame_index(frame_idx)
        smpl_data = self.hs.smpl_params
        smpl_data['vertices'] = self.hs.vertices
        smpl_data['joints'] = self.hs.joints[:, :self.hs.num_joints+1, :]
        # Condition Images
        if cond_type is None:
            cond_type = self.cond_type
        if isinstance(cond_type, str):
            cond_type = [cond_type,]
        intrinsics, extrinsic = NeRF_data_to_standard(intrinsics, cam2world, H=self.height, W=self.width)
        cond_images = []
        for _cond in cond_type:
            if _cond == 'pose':
                cond_image = self.hs.export_pose_map(intrinsics, extrinsic, width=self.width, height=self.height)
            elif _cond == 'depth':
                cond_image = self.hs.export_depth_map(intrinsics, extrinsic, width=self.width, height=self.height)
            elif _cond == 'mesh':
                cond_image = self.hs.export_mesh_map(intrinsics, extrinsic, width=self.width, height=self.height)
            else:
                assert 0, _cond
            cond_images.append(cond_image)
        return {
            'cond_images': cond_images,
            'smpl_data': smpl_data,
        }

    def write_video(self, save_dir='./', save_image='output.png', save_video='output.mp4', save_sequence=None, cond_type=None,
                    num_frames=None, camera_track='fixed', radius=3.0):
        os.makedirs(save_dir, exist_ok=True)
        if save_sequence is not None:
            os.makedirs(save_sequence, exist_ok=True)
        images = []
        if num_frames is None:
            num_frames = self.num_frame
        if num_frames <= 0:
            num_frames = 100
        for i in range(num_frames):
            intrinsics, cam2world = index2pose(i, H=self.height, W=self.width, radius=radius, camera_track=camera_track)
            image = self(intrinsics, cam2world, cond_type=cond_type, frame_idx=i)['cond_images'][0]
            if i == 0:
                image.save(osp.join(save_dir, save_image))
            if save_sequence is not None:
                image.save(osp.join(save_sequence, f'{i:04d}.png'))
            images.append(np.array(image))
        imageio.mimsave(osp.join(save_dir, save_video), np.array(images), fps=25, quality=8, macro_block_size=1)
        return image
