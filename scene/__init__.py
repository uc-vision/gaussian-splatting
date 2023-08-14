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

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from utils.camera_utils import camera_to_JSON
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams

from scene.cameras import Camera

from camera_geometry.scan import FrameSet
import camera_geometry
from camera_geometry.scan.views import load_frames_with, undistort_cameras, CameraImage, Undistortion
from camera_geometry.transforms import translate_44
from tqdm import tqdm

from scan_tools.crop_points import visibility_depths

import open3d as o3d
import cv2

def from_colmap_transform(m):
    R = np.transpose(m[:3,:3])
    T = m[:3, 3]
    return R, T


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec


def add_bg_points(pcd, n_points=10000, radius=100.0):
    dist = np.random.rand(n_points, 1) * radius

    points = sample_spherical(n_points) * dist
    colors = np.ones_like(points)

    aug = o3d.geometry.PointCloud()
    aug.points=o3d.utility.Vector3dVector(np.concatenate([pcd.points, points]))
    aug.colors=o3d.utility.Vector3dVector(np.concatenate([pcd.colors, colors]))

    return aug


def load_cloud(scan:FrameSet) -> BasicPointCloud:
  assert 'sparse' in scan.models, "No sparse model found in scene.json"
  pcd_file = scan.find_file(scan.models.sparse.filename)

  return o3d.io.read_point_cloud(str(pcd_file))


def to_basic_cloud(pcd):
  points = np.asarray(pcd.points)
  colors = np.asarray(pcd.colors)

  return BasicPointCloud(points=points, 
                         colors=colors,
                         normals=np.zeros_like(points))






def camera_extents(scan:FrameSet):
    cam_centers = np.stack([camera.location for camera in scan.expand_cameras()])
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)

    distances = np.linalg.norm(cam_centers - avg_cam_center, axis=0, keepdims=True)
    diagonal = np.max(distances)

    return avg_cam_center.reshape(3), diagonal * 1.1




def load_cameras(scan:FrameSet, undistortions:Dict[str, Undistortion], device="cuda:0"):

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
      

  frames = load_frames_with(scan, undistortions)
  cameras = [cam_frame for frame in frames
            for cam_frame in frame]
  
  return [to_camera(i, cam_frame) for i, cam_frame in enumerate(cameras)]

def write_images(model_path, cameras):
    with ThreadPoolExecutor(8) as pool:

      def write_image(camera):
          image_path = Path(model_path) / camera.image_name
          image_path.parent.mkdir(parents=True, exist_ok=True)

          bgr = camera.image.permute(1, 2, 0).cpu().numpy()
          cv2.imwrite(str(image_path), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

      tqdm(pool.map(write_image, cameras), total=len(cameras))
           
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
            if load_iteration is None:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        if not os.path.exists(args.source_path):
            raise Exception("Scan path '{}' does not exist".format(args.source_path))


        scan = FrameSet.load(args.source_path).with_image_scale(args.resolution)
        centre, self.cameras_extent = camera_extents(scan)

        undistortions = undistort_cameras(scan.cameras, alpha=0, centered=True)
        cameras = {k:dist.undistorted for k, dist in undistortions.items()}

        scan = scan.transform(translate_44(-centre[0], -centre[1], -centre[2])).with_cameras(cameras)
        pcd = load_cloud(scan).translate(-centre)

        print("Loading images...")
        self.train_cameras = load_cameras(scan, undistortions)
        self.test_cameras = []

        np.random.shuffle(self.train_cameras)

        if self.loaded_iter is not None:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
                                                           
        else:
            o3d.io.write_point_cloud(os.path.join(self.model_path, "input.ply"), pcd)          
            json_cameras = [camera_to_JSON(id, cam) for id, cam in enumerate(self.train_cameras)]
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cameras, file)

            print(f"Writing images...")
            write_images(self.model_path, self.train_cameras)

            scan_file = os.path.join(self.model_path, "scene.json")
            scan = scan.with_model("sparse", "input.ply")

            print(f"Writing scene.json...")
            scan.save(scan_file)

            if self.gaussians is not None:
              # pcd = add_bg_points(pcd, n_points=len(pcd.points) // 2, radius=2000.0)
              # _, min_depths = visibility_depths(scan.expand_cameras(), np.asarray(pcd.points))
              # base_scale = self.cameras_extent / 1000.0

              self.gaussians.create_from_pcd(pcd, spatial_lr_scale=self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
