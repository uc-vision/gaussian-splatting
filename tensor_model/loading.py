import argparse
import fileinput
from pathlib import Path
import open3d as o3d

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import Gaussians, inverse_sigmoid
from natsort import natsorted

def to_pcd(gaussians:Gaussians) -> o3d.t.geometry.PointCloud:
  pcd = o3d.t.geometry.PointCloud()
  pcd.point['positions'] = gaussians.positions.numpy()
  pcd.point['opacity'] = inverse_sigmoid(gaussians.opacity).numpy()

  sh_dc, sh_rest = gaussians.split_sh()
  sh_dc = sh_dc.view(-1, 3)
  sh_rest = sh_rest.view(sh_rest.shape[0], -1)

  for i in range(3):
    pcd.point[f'f_dc_{i}'] = sh_dc[:, i:i+1].numpy()

  for i in range(sh_rest.shape[-1]):
    pcd.point[f'f_rest_{i}'] = sh_rest[:, i:i+1].numpy()

  for i in range(3):
    pcd.point[f'scale_{i}'] = torch.log(gaussians.scaling[:, i:i+1]).numpy()

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
  deg = int(np.sqrt(n_sh))

  assert deg * deg == n_sh, "SH feature count must be square"
  log_scaling = get_keys([f'scale_{k}' for k in range(3)])

  sh_dc = get_keys([f'f_dc_{k}' for k in range(3)]).view(positions.shape[0], 1, 3)
  sh_rest = get_keys([f'f_rest_{k}' for k in range(3 * (deg * deg - 1))]).view(positions.shape[0], -1, 3)
  rotation = get_keys([f'rot_{k}' for k in range(4)])
  opacity_logit = get_keys(['opacity'])

  return Gaussians(
    positions = torch.from_numpy(positions),
    rotation = F.normalize(rotation, -1),
    opacity = torch.sigmoid(opacity_logit),
    sh_features = torch.cat([sh_dc, sh_rest], dim=1),
    scaling = torch.exp(log_scaling)
  )


def write_gaussians(filename:str, gaussians:Gaussians):
  pcd = to_pcd(gaussians)
  o3d.t.io.write_point_cloud(filename, pcd)

def read_gaussians(filename:str) -> Gaussians:
  pcd:o3d.t.geometry.PointCloud = o3d.t.io.read_point_cloud(filename)
  if not 'positions' in pcd.point:
    raise ValueError(f"Could not load point cloud from {filename}")

  return from_pcd(pcd)
  






