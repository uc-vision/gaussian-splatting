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
from geometry_grid.taichi_geometry.conversion import struct_size

import torch
import taichi as ti
from taichi.math import vec3, vec4

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
def splat_bounds(p:vec3, q:vec4, scale:vec3) -> AABox:
  axes = scale * quat_to_mat(q)

  lower, upper = ti.math.vec3(np.inf), ti.math.vec3(-np.inf)
  for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
    corner = axes @ ti.math.vec3(i, j, k)
    lower = ti.min(lower, corner)
    upper = ti.max(upper, corner)

  return AABox(lower + p, upper + p)

@ti.func 
def to_covariance(q:ti.math.vec4, scale:ti.math.vec3) -> ti.math.mat3:
  axes = scale * quat_to_mat(q)
  return axes @ axes.T

@ti.func
def gaussian(x:ti.math.vec3, mean:ti.math.vec3, inv_cov:ti.math.mat3) -> ti.f32:
  x = x - mean
  return ti.exp(-0.5 * ti.math.dot(x, inv_cov @ x))


def block_bitmask(size, chunk, levels=3):
    
    root_size = [math.ceil(x/(chunk ** (levels - 1))) for x in size]
    snode:ti.SNode = ti.root.bitmasked(ti.ijk, root_size)

    for i in range(levels - 1):
        snode = snode.bitmasked(ti.ijk, (chunk,chunk,chunk))
    
    return snode


@ti.data_oriented
class Splat:
  p: vec3
  q: vec4
  scale: vec3
  color: vec3
  opacity: ti.f32

  @ti.func
  def from_vec(self, v:ti.template()):
    self.p = v[0:3]
    self.q = v[3:7]
    self.scale = v[7:10]
    self.color = v[10:13]
    self.opacity = v[13]
  
spat_vec=ti.types.vector(dtype=ti.f32, n=struct_size(Splat))




class DensityGrid:

  def init(self, bounds:AABox, cell_size:float, chunk_size=8):
    self.rgb_density = ti.Vector.field(4, ti.f16)

    self.bounds = bounds
    self.grid_size = ti.math.ceil(bounds.extents / cell_size)
    self.cells = block_bitmask(self.grid_size, chunk_size)

    self.cell_size = cell_size

    self.cells.place(self.rgb_density)

  @ti.kernel
  def splat(self, splats:ti.types.ndarray(dtype=spat_vec), threshold:ti.f32):
    
    for i in ti.static(range(splats.shape[0])):
      splat = Splat()
      splat.from_vec(splats[i])

      splat.p -= self.bounds.lower
      splat.scale /= self.cell_size

      r = splat_bounds(splat.p, splat.q, 3 * splat.scale)
      
      lower = ti.max(ti.cast(ti.floor(r.lower), ti.i32), 0)
      upper = ti.min(ti.cast(ti.ceil(r.upper), ti.i32), self.grid_size - 1)

      inv_cov = ti.math.inverse(to_covariance(splat.q, splat.scale))

      for i in ti.grouped(ti.ndrange(lower, upper)):
        density = gaussian(i, splat.p, inv_cov) * splat.opacity

        if density > threshold:
          self.rgb_density[i] += vec4(splat.color, density)

        
    


def render_grid(model:GaussianModel, cell_size:float):
  xyz = model.get_xyz

  bounds = AABox(
    lower = vec3(*torch.min(xyz, dim=1).values),
    upper = vec3(*torch.max(xyz, dim=1).values))
  
  grid = DensityGrid(bounds, cell_size)
  splats = torch.concatenate([xyz, 
                              model.get_rotation, 
                              model._features_dc, 
                              model.get_opacity], dim=1)
  
  grid.splat(splats, 0.01)



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

  
  ti.init(arch=ti.cuda)


  render_grid(model, args.grid_size)

  # cloud.point["colors"] = color
  # o3d.visualization.draw([cloud, roi])


if __name__ == "__main__":
  main()  






