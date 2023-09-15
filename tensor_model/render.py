import argparse
from pathlib import Path
from natsort import natsorted
import numpy as np
import open3d as o3d
import torch
from .fov_camera import FOVCamera, load_camera_json
from .loading import from_pcd

from camera_geometry.scan import FrameSet, Camera

from .renderer import  render


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
  parser.add_argument('--device', default='cuda')
  
  args = parser.parse_args()

  cloud_filename = find_cloud(args.input / 'point_cloud')
  print("Loading", cloud_filename)

  pcd = o3d.t.io.read_point_cloud(str(cloud_filename))
  # scene = FrameSet.load_file(args.input / 'scene.json')

  device = torch.device(args.device)

  gaussians = from_pcd(pcd).to(device)
  cameras = load_camera_json(args.input / 'cameras.json')

  for k, camera in cameras.items():
    render(camera, gaussians, bg_color=torch.Tensor(0, 0, 0))

  

if __name__ == '__main__':
  main()
