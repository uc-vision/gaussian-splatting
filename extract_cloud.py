import argparse
import math
from pathlib import Path
from typing import Tuple
import open3d as o3d
from scan_tools.scan_roi import scan_roi

import numpy as np
from camera_geometry import FrameSet
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel

import torch
import taichi as ti
from taichi.math import vec3

@ti.func
def quat_to_mat(q:ti.math.vec4) -> ti.math.mat3:
  w, x, y, z = q
  x2, y2, z2 = x*x, y*y, z*z

  return ti.math.mat3(
    1 - 2*y2 - 2*z2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
    2*x*y + 2*w*z, 1 - 2*x2 - 2*z2, 2*y*z - 2*w*x,
    2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x2 - 2*y2
  )

@ti.data_oriented
class AABox:
  lower:ti.math.vec3
  upper:ti.math.vec3

  @ti.func
  def extents(self) -> ti.math.vec3:
    return self.upper - self.lower

@ti.func 
def splat_bounds(q:ti.math.vec4, scale:ti.math.vec3) -> AABox:
  axes = scale * quat_to_mat(q)

  lower, upper = ti.math.vec3(np.inf), ti.math.vec3(-np.inf)
  for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
    corner = axes @ ti.math.vec3(i, j, k)
    lower = ti.min(lower, corner)
    upper = ti.max(upper, corner)

@ti.func 
def to_covariance(q:ti.math.vec4, scale:ti.math.vec3) -> ti.math.mat3:
  axes = scale * quat_to_mat(q)
  return axes @ axes.T

@ti.func
def gaussian(x:ti.math.vec3, mean:ti.math.vec3, inv_cov:ti.math.mat3) -> ti.f32:
  x = x - mean
  return ti.exp(-0.5 * x @ inv_cov @ x)


def block_bitmask(size, chunk, levels=3):
    
    root_size = [math.ceil(x/(chunk ** (levels - 1))) for x in size]
    snode:ti.SNode = ti.root.bitmasked(ti.ijk, root_size)

    for i in range(levels - 1):
        snode = snode.bitmasked(ti.ijk, (chunk,chunk,chunk))
    
    return snode


class DensityGrid:

  def init(self, bounds:AABox, cell_size:float, chunk_size=8):
    self.density = ti.field(ti.f16)
    self.color = ti.Vector.field(3, ti.f16)

    self.bounds = bounds
    self.grid_size = ti.math.ceil(bounds.extents / cell_size)
    self.cells = block_bitmask(self.grid_size, chunk_size)

    self.cells.place(self.density)
    self.cells.place(self.color)


def render_grid(xyz:torch.Tensor, scale:torch.Tensor, q:torch.Tensor, cell_size:float):
  bounds = AABox(
    lower = vec3(*torch.min(xyz, dim=1).values),
    upper = vec3(*torch.max(xyz, dim=1).values))
  
  grid = DensityGrid(bounds, cell_size)



def load_cropped(args, op:OptimizationParams):

  model = GaussianModel(3)
  model.load_ply(args.cloud)

  model.training_setup(op)
  
  pos = model.get_xyz
  scan = FrameSet.load(args.scan)
  
  roi = scan_roi(scan.expand_cameras(), pos.cpu().numpy(), max_depth = args.max_depth, 
                 min_views = scan.total_views * args.proportion / 100, margin_percent=20, trim_percentile=2, aligned=True)
  
  min_bound, max_bound = [torch.from_numpy(b).to(dtype=torch.float32, device="cuda") 
                          for b in (roi.min_bound, roi.max_bound)]

  include = (pos > min_bound) & (pos < max_bound)
  model.prune_points(~torch.all(include, dim=1))
  
  return model


def main():

  parser = argparse.ArgumentParser(description="Extract a point cloud from a gaussian splatting model")
  parser.add_argument("--cloud", type=Path, help="Path to the gaussian splatting point cloud")
  parser.add_argument("--scan", type=str,  help="Input scan file")
  

  parser.add_argument("--max_depth", default=20.0, help="Max depth to determine the visible ROI")
  parser.add_argument("--proportion", type=int, default=30, help="Minimum proportion of views to be included")

  parser.add_argument("--write_crop", type=str,  help="Write cropped ply file")
  parser.add_argument("--cell_cm", default=1.0, help="Grid size to extract the ROI at (default 1.0)")

  op = OptimizationParams(parser)
  args = parser.parse_args()

  op = op.extract(args)

  with torch.inference_mode():
    model = load_cropped(args, op)

    if args.write_crop is not None:
      model.save_ply(args.write_crop)

  
  # ti.init(arch=ti.cuda)
  # render_grid(model.get_xyz, model.get_scaling, model.get_rotation, args.grid_size)

  # cloud.point["colors"] = color
  # o3d.visualization.draw([cloud, roi])


if __name__ == "__main__":
  main()  






