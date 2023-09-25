from splatting.gaussians.loading import read_gaussians, to_rgb, write_gaussians
import argparse
from pathlib import Path
import open3d as o3d

import taichi as ti

def display_outliers(cloud, ind):
    inlier_cloud = cloud.select_by_mask(ind)
    outlier_cloud = cloud.select_by_mask(ind, invert=True)

    outlier_cloud.paint_uniform_color([1., 0., 0.])
    o3d.visualization.draw([inlier_cloud, outlier_cloud])


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=Path)
  parser.add_argument('--write', type=Path)
  parser.add_argument('--show', action='store_true')
  parser.add_argument('--densify', default=1, type=int)
  args = parser.parse_args()

  ti.init(ti.gpu)

  gaussians = read_gaussians(args.input)
  pcd = to_rgb(gaussians, densify=args.densify)

  if args.output is None and not args.show:
    raise ValueError("Must specify --output or --show")

  if args.show:
    o3d.visualization.draw([pcd])
  
  if args.write:
    o3d.t.io.write_point_cloud(str(args.write), pcd)