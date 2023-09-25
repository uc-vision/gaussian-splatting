import argparse
from pathlib import Path
from scan_tools.scan_roi import scan_roi

from camera_geometry import FrameSet
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scan_tools.crop_points import visibility
import numpy as np

import open3d as o3d
import torch


def load_cropped(args, op:OptimizationParams):

  model = GaussianModel(3)
  model.load_ply(args.cloud)

  model.training_setup(op, 0)
  
  pos = model.get_xyz
  scan = FrameSet.load(args.scan)

  counts = visibility(scan.expand_cameras(), pos.cpu().numpy(), (args.min_depth, args.max_depth))

  # roi = scan_roi(scan.expand_cameras(), pos.cpu().numpy(), max_depth = args.max_depth, 
  #                min_views = max(1, scan.total_views * args.proportion / 100), margin_percent=20, trim_percentile=2, aligned=True)
  
  # min_bound, max_bound = [torch.from_numpy(b).to(dtype=torch.float32, device="cuda") 
  #                         for b in (roi.min_bound, roi.max_bound)]

  # include = (pos > min_bound) & (pos < max_bound)
  min_views = max(1, scan.total_views * args.proportion / 100)
  sizes = model.get_scaling.max(dim=1).values.cpu().numpy()

  model.prune_points((counts < min_views) | (sizes > args.max_size))
  print(model.get_scaling.max())

  print(f"Cropped model from {pos.shape[0]} to {model.get_xyz.shape[0]} points, with {min_views} views")

  return model


def main():

  parser = argparse.ArgumentParser(description="Extract a point cloud from a gaussian splatting model")
  parser.add_argument("cloud", type=Path, help="Path to the gaussian splatting point cloud")
  parser.add_argument("--scan", type=str,  help="Input scan file")
  
  parser.add_argument("--max_depth", default=20.0, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--min_depth", default=0.2, type=float, help="Min depth to determine the visible ROI")
  parser.add_argument("--proportion", type=float, default=0, help="Minimum proportion of views to be included")
  parser.add_argument("--max_size", type=float, default=np.inf, help="Cull points of large size")

  parser.add_argument("--output", type=Path,  help="Write cropped ply file")

  parser.add_argument("--show", action="store_true")

  op = OptimizationParams(parser)
  args = parser.parse_args()

  if args.scan is None:
    args.scan = str(args.cloud.parent.parent.parent / "scene.json")

  if args.output is None:
    args.output = args.cloud.parent.parent / "iteration_cropped" / "point_cloud.ply"

  op = op.extract(args)

  with torch.inference_mode():
    model = load_cropped(args, op)

    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      model.save_ply(str(args.output))
      print(f"Wrote {args.output}")

    if args.show:
      points = model.get_xyz.cpu().numpy()
      colors = model._features_dc.squeeze().cpu().numpy()

      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(points)
      pcd.colors = o3d.utility.Vector3dVector(colors)

      o3d.visualization.draw_geometries([pcd])

    


if __name__ == "__main__":
  main()  






