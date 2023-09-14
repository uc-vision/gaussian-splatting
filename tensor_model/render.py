import argparse
from pathlib import Path
from natsort import natsorted
import numpy as np
import open3d as o3d
import torch
from .fov_camera import FOVCamera, load_camera_json
from .loading import from_pcd

from camera_geometry.scan import FrameSet, Camera

from .renderer import render, to_camera


def find_cloud(p:Path):
  clouds = [d / "point_cloud.ply" for d in p.iterdir() if d.is_dir()]
  clouds = natsorted(clouds)

  if len(clouds) == 0:
    raise FileNotFoundError(f"No point clouds found in {str(p)}")

  return clouds[0]


def camera_to_fov(camera:Camera) -> FOVCamera:
   assert camera.has_distortion == False, "Simple FOV camera does not have distortion"
  
   return FOVCamera(
      focal_length = camera.focal_length[0],
      position = camera.location,
      rotation = camera.rotation,
      image_size = camera.image_size,
      image_name = ""
    )
   




def main():
  torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
  np.set_printoptions(precision=4, suppress=True, linewidth=200)

  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=Path)
  
  args = parser.parse_args()

  cloud_filename = find_cloud(args.input / 'point_cloud')
  print("Loading", cloud_filename)

  pcd = o3d.t.io.read_point_cloud(str(cloud_filename))
  scene = FrameSet.load_file(args.input / 'scene.json')

  gaussians = from_pcd(pcd)

  cameras = scene.expand_cameras()
  cameras2 = load_camera_json(args.input / 'cameras.json')

  for camera1, (k, camera2) in zip(cameras, cameras2.items()):
    fov = camera_to_fov(camera1)

    # view, proj, pos = [torch.from_numpy(t).to(device=device, dtype=torch.float32) 
    #         for t in (camera.camera_t_parent, np.linalg.inv(camera.projection), camera.location)]


    spcam = to_camera(0, camera1)

    print("view\n", spcam.world_view_transform,  fov.camera_t_world.transpose(1, 0))


    print("view\n", spcam.projection_matrix, "\n", fov.ndc_t_camera)
        
    # print("???", camera1.parent_t_camera, camera1.location)

    # print("view\n", spcam.world_view_transform, "\n", fov.camera_t_world.transpose(1, 0))


    exit(0)
    # render(camera, gaussians, bg_color=torch.Tensor(0, 0, 0))

  

if __name__ == '__main__':
  main()
