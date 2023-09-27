import argparse
from pathlib import Path
from typing import List
from beartype import beartype
import numpy as np


from tqdm import tqdm
from scan_tools.crop_points import visibility

import torch
from splatting.camera.fov import FOVCamera

from splatting.gaussians.loading import from_pcd, read_gaussians, to_pcd, write_gaussians
from splatting.gaussians.workspace import load_workspace


def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=torch.float32, device=points.device)], axis=-1)

def _transform_points(transform, points):
  assert points.shape[
      -1] == 3, 'transform_points: expected 3d points of ...x3, got:' + str(
          points.shape)

  homog = make_homog(points).reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ homog
  return transformed[..., 0].reshape(-1, 4)

def project_points(transform, xyz):
  homog = _transform_points(transform, xyz)
  depth = homog[..., 2:3]
  xy = homog[..., 0:2] 
  return (xy / depth), depth


@beartype
def visibility(cameras:List[FOVCamera], points:torch.Tensor, near=0.1, far=torch.inf):
  counts = torch.zeros(len(points), dtype=torch.int32, device=points.device)

  projections = np.array([camera.projection for camera in cameras])
  torch_projections = torch.from_numpy(projections).to(dtype=torch.float32, device=points.device)

  for camera, proj in tqdm(zip(cameras, torch_projections)):
  
    proj, depth = project_points(proj, points)
    width, height = camera.image_size

    valid = ((proj[:, 0] >= 0) & (proj[:, 0] < width) & 
             (proj[:, 1] >= 0) & (proj[:, 1] < height)
             & (depth[:, 0] > near) & (depth[:, 0] < far)
             )
    counts[valid] += 1
  return counts


def crop_model(model, cameras:List[FOVCamera], args):
  counts = visibility(cameras, model.positions, near = args.near, far = args.far)

  min_views = max(1, len(cameras) * args.min_percent / 100)
  cropped = model[counts >= min_views]

  print(f"Cropped model from {model.batch_shape} to {cropped.batch_shape} points, with at least {min_views} views")
  return cropped


def main():

  parser = argparse.ArgumentParser(description="Extract a point cloud from a gaussian splatting model")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  parser.add_argument("--scan", type=str,  help="Input scan file")
  
  parser.add_argument("--far", default=20.0, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.2, type=float, help="Min depth to determine the visible ROI")

  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')

  parser.add_argument("--statistical_outliers", type=float, default=None)
  parser.add_argument("--radius_outliers", type=float, default=None)
  parser.add_argument("--knn", type=int, default=100)

  
  parser.add_argument("--write", action="store_true", help="Write the cropped model to a file")
  parser.add_argument("--show", action="store_true")

  args = parser.parse_args()

  workspace = load_workspace(args.model_path)

  with torch.inference_mode():
    if args.model_path.is_dir():
      workspace = load_workspace(args.model_path)
      model = workspace.load_model(workspace.latest_iteration())
    else:
      model = read_gaussians(args.model_path)

    model = model.to(args.device)
    model = crop_model(model, workspace.cameras, args)

    if any([args.radius_outliers, args.statistical_outliers]):
      pcd = to_pcd(model.cpu())

      if args.statistical_outliers is not None:
        pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=args.knn, std_ratio=args.statistical_outliers)
      
      if args.radius_outliers is not None:
        pcd, _ = pcd.remove_radius_outliers(nb_points=args.knn, search_radius=args.radius_outliers)
      
      num_removed = model.batch_shape[0] - pcd.point['positions'].shape[0]

      model = from_pcd(pcd)
      print(f"Removed {num_removed} outliers")

    if args.write:

      output = args.model_path / "point_cloud" / "cropped" / "point_cloud.ply"
      output.parent.mkdir(parents=True, exist_ok=True)
      
      write_gaussians(output, model)
      print(f"Wrote {model} to {output}")

    if args.show:
      from splatting.viewer.scene_widget import show_workspace
      show_workspace(workspace, model)



if __name__ == "__main__":
  main()  






