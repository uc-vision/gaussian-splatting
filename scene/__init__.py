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

import numpy as np
import torch
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams

from scene.cameras import Camera

from camera_geometry.scan import FrameSet
import camera_geometry
from camera_geometry.scan.views import load_frames, CameraImage
import open3d as o3d
import cv2

def from_colmap_transform(m):
    R = np.transpose(m[:3,:3])
    T = m[:3, 3]
    return R, T


def load_cloud(scan:FrameSet) -> BasicPointCloud:
  assert 'sparse' in scan.models, "No sparse model found in scene.json"
  pcd_file = scan.find_file(scan.models.sparse.filename)
  pcd = o3d.io.read_point_cloud(str(pcd_file))

  points = np.asarray(pcd.points)
  return BasicPointCloud(points=points, 
                         colors=np.asarray(pcd.colors), 
                         normals=np.zeros_like(points))



def camera_extent(frames):
    cam_centers = [frame.camera.location for frame in frames]
    cam_centers = np.stack(cam_centers)

    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)

    return diagonal * 1.1




def load_cameras(scan:FrameSet,  scale=1.0, device="cuda:0"):

  def to_camera(i, frame:CameraImage):
      camera:camera_geometry.Camera = frame.camera
      R, T = from_colmap_transform(camera.camera_t_parent)

      bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

      return Camera(colmap_id=i,
                    uid=i,
                    image_name=frame.rgb_file,
                    R=R, T=T,
                    FoVx=camera.fov[0], 
                    FoVy=camera.fov[1],
                    image=torch.from_numpy(bgr).permute(2, 0, 1),
                    data_device=device,
                    gt_alpha_mask=None
                    )
      

  frames = load_frames(
      scan.with_image_scale(scale), alpha=0.0, centered=True)
  
  cameras = [cam_frame for frame in frames
            for cam_frame in frame]
  
  return [to_camera(i, cam_frame) for i, cam_frame in enumerate(cameras)], camera_extent(cameras) * 0.1



class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        if not os.path.exists(args.source_path):
            raise Exception("Scan path {} does not exist".format(args.source_path))

        scan = FrameSet.load(args.source_path)

        image_scale = args.resolution if args.resolution > 0 else 1.0

        self.train_cameras, self.cameras_extent = load_cameras(scan, scale=image_scale)
        self.test_cameras = []


        np.random.shuffle(self.train_cameras)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            pcd = load_cloud(scan)
            self.gaussians.create_from_pcd(pcd, spatial_lr_scale=self.cameras_extent * 0.1)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras