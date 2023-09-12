import argparse
from pathlib import Path
from natsort import natsorted
import open3d as o3d
import torch

from tensor_model.loading import from_pcd
from camera_geometry.scan import FrameSet

from .renderer import render


def find_cloud(p:Path):
  clouds = [d / "point_cloud.ply" for d in p.iterdir() if d.is_dir()]
  clouds = natsorted(clouds)

  if len(clouds) == 0:
    raise FileNotFoundError(f"No point clouds found in {str(p)}")

  return clouds[0]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=Path)
  
  args = parser.parse_args()


  cloud_filename = find_cloud(args.input / 'point_cloud')
  print("Loading", cloud_filename)

  pcd = o3d.t.io.read_point_cloud(str(cloud_filename))
  scene = FrameSet.load_file(args.input / 'scene.json')

  gaussians = from_pcd(pcd)

  cameras = scene.expand_cameras()
  for camera in cameras:
    render(camera, gaussians, bg_color=torch.Tensor(0, 0, 0))

  

if __name__ == '__main__':
  main()
