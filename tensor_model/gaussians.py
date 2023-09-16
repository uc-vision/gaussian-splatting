
from dataclasses import dataclass, replace
import math
from tensorclass import TensorClass
import torch
from torch import functional as F

from tensor_model.geometry import quat_to_mat_kernel
from utils.sh_utils import RGB2SH

from .typecheck import NVec3, Vec1, Vec3, Vec4, typechecked, Float32, Tensor
from simple_knn._C import distCUDA2

def sh_features(deg):
  return (deg + 1) ** 2 

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def check_sh_degree(sh_features):
  n_sh = sh_features.shape[1]
  n = int(math.sqrt(n_sh))

  assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
  return n
  
@dataclass
class Gaussians(TensorClass):

  positions : Float32[Tensor, '3']
  sh_features : Float32[Tensor, 'N 3']
  scaling   : Float32[Tensor, '3']
  rotation  : Float32[Tensor, '4']
  opacity   : Float32[Tensor, '1']

  def __post_init__(self, broadcast, convert_types):
    super().__post_init__(broadcast, convert_types)
    check_sh_degree(self.sh_features)

  def split_sh(self):
    return self.sh_features[:, :1], self.sh_features[:, 1:]


  @property
  def sh_degree(self):
    return check_sh_degree(self.sh_features)



  @property
  def rotation_matrix(self):
    return quat_to_mat_kernel(self.rotation)

  @typechecked
  @staticmethod
  def from_points(points:NVec3, colors:NVec3, min_scale:float = 0.0001, sh_degree=0):

    dist2 = torch.clamp_min(distCUDA2(points), min_scale)
    scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
    
    rotations = torch.randn((points.shape[0], 4), device=points.device)
    opacity =  torch.full((points.shape[0], 1), fill_value=0.1,
                           dtype=torch.float, device=points.device)
    
    sh_features = torch.zeros((points.shape[0], num_features, 3), device=points.device)
    sh_features[:, 0] =  RGB2SH(colors)

    num_features = sh_features(sh_degree)
    return Gaussians(
      positions = points,
      sh_features = sh_features,
      scaling = scales,
      rotation = F.normalize(rotations),
      opacity_logit = opacity)



@dataclass
class GaussianParameters(TensorClass):

  positions : Float32[Tensor, '3']

  sh_dc     : Float32[Tensor, '3 1']
  sh_rest   : Float32[Tensor, 'N 3']

  log_scaling   : Float32[Tensor, '3']
  rotation      : Float32[Tensor, '4']
  opacity_logit : Float32[Tensor, '1']

  @property
  def sh_features(self):
    return torch.cat([self.sh_dc, self.sh_rest], dim=1)
  
  @property
  def unit_rotation(self):
    return F.normalize(self.rotation) 
  
  @property
  def rotation_matrix(self):
    return quat_to_mat_kernel(self.rotation)
  
  @property
  def scaling(self):
    return torch.exp(self.log_scaling)
  
  @property
  def opacity(self):
    return torch.sigmoid(self.opacity_logit)
  
  @property
  def sh_degree(self):
    n_sh = 1 + (self.sh_rest.shape[1]) 
    n = int(math.sqrt(n_sh))

    assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({self.sh_rest.shape})"
    return (n - 1)

  @property
  def with_inc_degree(self) -> 'Gaussians':
    assert self.sh_degree < 3, "SH degree cannot exceed 3"

    num_features = sh_features(self.sh_degree + 1)
    sh_rest = torch.zeros(dtype=self.sh_rest.dtype, device=self.sh_rest.device, size=(self.sh_rest.shape[0], num_features - 1, 3))
    sh_rest[:, :self.sh_rest.shape[1]] = self.sh_rest

    return replace(self, sh_rest=sh_rest)



      

