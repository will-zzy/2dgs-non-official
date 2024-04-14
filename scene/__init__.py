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

import os
import random
import json
import numpy as np
import glob
import cv2
from tqdm import tqdm
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.colmap_utils import read_cameras_binary, read_cameras_text, read_images_binary, read_images_text

def params_from_models(camera_model:np.ndarray):
    ## Radial-Tangential distortion model
    fx = camera_model[0]
    fy = camera_model[1]
    cx = camera_model[2]
    cy = camera_model[3]
    K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]])
    try:
        distortion = np.array([
            camera_model[4],
            camera_model[5],
            camera_model[6],
            camera_model[7]
        ])
    except:
        distortion = np.array([0,0,0,0])
    return K,distortion
class Scene:

    gaussians : GaussianModel

    def __init__(self, config, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.config = config
        self.root_dir = config.root_dir
        self.loaded_iter = None
        self.gaussians = gaussians
        self.img_downsample = self.config.dataset.downsample
        self.img_dir = os.path.join(self.root_dir,f"images_undistorted_{self.img_downsample}")
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.root_dir, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        self.train_cameras = {}
        self.test_cameras = {}
        # self.read_meta()
        
        self.load_images() # 得到img_path
        self.scene_info = self.load_cam()  
         

        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.root_dir, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.root_dir, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, self.config.dataset)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, self.config.dataset)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.root_dir,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent,init_rots=True)

    def load_cam(self):
        if os.path.exists(os.path.join(self.config.root_dir, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.config.root_dir, 
                                                          self.img_dir,
                                                          self.config.dataset)
        elif os.path.exists(os.path.join(self.config.root_dir, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.config.root_dir, self.config.white_background, self.config.eval)
        else:
            assert False, "Could not recognize scene type!"
        return scene_info
    
    def load_images(self):
        if not os.path.exists(self.img_dir):    
            self.undist_and_downsample()
        pass    
    def undist_and_downsample(self):
        PREFIX = f"images_undistorted_{self.img_downsample}"
        output_dir = os.path.join(self.root_dir,PREFIX)
        input_dir = os.path.join(self.root_dir,"images")
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)
            
            models, img_names = self.read_meta() # 读取内参和图像名字
            
            for img_idx in tqdm(range(0,len(img_names))):
                img_path = os.path.join(input_dir,img_names[img_idx])
                output_path = os.path.join(output_dir,img_names[img_idx])
                distorted = cv2.imread(img_path)
                img_wh = int(distorted.shape[1]*self.img_downsample), int(distorted.shape[0]*self.img_downsample)   # W,H,3

                
                if self.config.dataset.undistortion and models[img_idx+1].model in ["OPENCV"]:
                    K,distortion = params_from_models(models[img_idx+1].params)
                    undistorted = cv2.undistort(distorted,
                                                K,
                                                distortion)
                else:
                    undistorted = distorted
                    
                undistorted = cv2.resize(undistorted,(img_wh[0],img_wh[1]))
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                cv2.imwrite(output_path,undistorted)

                    
                    
                
                
                
                
                
                
        else:
            return 
        
        
    def read_meta(self): # 返回畸变参数和图像位置
        
        try:
            img_data = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
            cam_data = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        except:
            img_data = read_images_text(os.path.join(self.root_dir, 'sparse/0/images.txt'))
            cam_data = read_cameras_text(os.path.join(self.root_dir, 'sparse/0/cameras.txt'))   
        img_names = [img_data[k].name for k in img_data]
        
        
        return cam_data,img_names
            
            
            
        
    def load_pcd(self):
        
        pass



    def save(self, iteration):
        point_cloud_path = os.path.join(self.root_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]