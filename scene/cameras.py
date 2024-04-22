#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from PIL import Image
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch

class Camera(nn.Module):
    def __init__(self, cam_info, resolution, uid,
                 gt_alpha_mask=None,trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.colmap_id = cam_info.colmap_id
        self.R = cam_info.R
        self.T = cam_info.T
        self.FoVx = cam_info.FovX
        self.FoVy = cam_info.FovY
        self.image_name = cam_info.image_name
        self.image_path = cam_info.image_path
        self.cam_intr = torch.tensor(cam_info.cam_intr.params * cam_info.downsample,dtype=torch.float32,device=data_device)
        self.image_type = cam_info.image_type
        self.uid = uid
        self.resolution = resolution
        

        try:
            self.data_device = torch.device(data_device)
            # self.cam_intr.device = self.data_device
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if self.image_type == "all":
            self.image = PILtoTorch(cam_info.image, resolution).clamp(0.0, 1.0).to(self.data_device)
            # self.image = cam_info.image


            # self.original_image = self.image.clamp(0.0, 1.0).to(self.data_device)
            # self.image_width = self.original_image.shape[2]
            # self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.image *= gt_alpha_mask.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, 
                                                     zfar=self.zfar, 
                                                     fovX=self.FoVx, 
                                                     fovY=self.FoVy, 
                                                     params = self.cam_intr,
                                                     w=self.resolution[0],
                                                     h=self.resolution[1]).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    @property
    def get_image(self):
        if self.image_type == "all":
            return self.image
        if self.image_type == "iterable":
            return PILtoTorch(Image.open(self.image_path), self.resolution).clamp(0.0, 1.0).to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

