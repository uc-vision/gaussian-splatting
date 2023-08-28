import argparse
from pathlib import Path
import open3d as o3d

import numpy as np
import torch

from .gaussians import Gaussians
from natsort import natsorted

def to_pcd(gaussians:Gaussians) -> o3d.t.geometry.PointCloud:
  pcd = o3d.t.geometry.PointCloud()
  pcd.point['positions'] = gaussians.positions.numpy()
  pcd.point['opacity'] = gaussians.opacity.numpy()

  for i in range(3):
    pcd.point[f'f_dc_{i}'] = gaussians.sh_dc[:, i:i+1].numpy()

  for i in range(gaussians.sh_rest.shape[-1]):
    pcd.point[f'f_rest_{i}'] = gaussians.sh_rest[:, i:i+1].numpy()

  for i in range(3):
    pcd.point[f'scale_{i}'] = gaussians.scaling[:, i:i+1].numpy()

  for i in range(4):
    pcd.point[f'rot_{i}'] = gaussians.rotation[:, i:i+1].numpy()

  return pcd


def from_pcd(pcd:o3d.t.geometry.PointCloud) -> Gaussians:
  def get_keys(ks):
    values = [pcd.point[k].numpy() for k in ks]
    return torch.from_numpy(np.concatenate(values, axis=-1))

  positions = pcd.point['positions'].numpy()

  attrs = sorted(dir(pcd.point))
  sh_attrs = [k for k in attrs if k.startswith('f_rest_') or k.startswith('f_dc_')]
  
  n_sh = len(sh_attrs) // 3
  sh_degree = int(np.sqrt(n_sh))

  assert sh_degree * sh_degree == n_sh, "SH feature count must be square"

  return Gaussians(
    positions = torch.from_numpy(positions),
    sh_dc = get_keys([f'f_dc_{k}' for k in range(3)]),
    sh_rest = get_keys([f'f_rest_{k}' for k in range(3 * (sh_degree * sh_degree - 1))]),
    scaling = get_keys([f'scale_{k}' for k in range(3)]),
    rotation = get_keys([f'rot_{k}' for k in range(4)]),
    opacity = get_keys(['opacity'])
  )


def write_gaussians(filename:str, gaussians:Gaussians):
  pcd = to_pcd(gaussians)
  o3d.t.io.write_point_cloud(filename, pcd)

def read_gaussians(filename:str) -> Gaussians:
  pcd:o3d.t.geometry.PointCloud = o3d.t.io.read_point_cloud(filename)
  if not 'positions' in pcd.point:
    raise ValueError(f"Could not load point cloud from {filename}")

  return from_pcd(pcd)
  






