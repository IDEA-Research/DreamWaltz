import torch
import numpy as np
from typing import Any
import numpy as np
import torch
import os
import os.path as osp
import pickle as pkl
from glob import glob
from typing import Iterable

from configs.paths import AIST_ROOT


class AIST(object):
    def __init__(self, root=AIST_ROOT) -> None:
        self.root = root
        filenames = os.listdir(root)
        filepaths = [osp.join(root, fn) for fn in filenames]
        dat = {}
        for fn, fp in zip(filenames, filepaths):
            fn = osp.splitext(fn)[0]
            dat[fn] = fp
        self.dat = dat

    def get_smpl_params(self, filename, model_type='smpl'):

        fps = 60
        stand_fps = 25
        fps_step = np.ceil(fps / stand_fps)

        data_gt = pkl.load(open(self.dat[filename], 'rb'))

        pose_params = data_gt['smpl_poses'][np.newaxis, ...]  # [1, F, 24*3]
        transl = data_gt['smpl_trans'][np.newaxis, ...]  # [1, F, 3]
        betas = np.zeros((1, 10))
        global_orient = pose_params[:, :, :3]
        body_pose = pose_params[:, :, 3:]

        slected_frames = [i for i in range(pose_params.shape[1]) if i % fps_step == 0]
        global_orient = global_orient[:, slected_frames, :]
        body_pose = body_pose[:, slected_frames, :]
        transl = transl[:, slected_frames, :]

        if model_type in ('smplx', 'smplh'):
            # Layout of 23 joints of SMPL
            # https://www.researchgate.net/figure/Layout-of-23-joints-in-the-SMPL-models_fig2_351179264
            body_pose = body_pose[:, :, :-2*3]

        smpl_params = {
            'global_orient': global_orient,  # (2, F, 3)
            'body_pose': body_pose,          # (2, F, 63/69)
            'betas': betas,                  # (2, 10)
            'transl': transl,                # (2, F, 3)
        }

        return smpl_params
