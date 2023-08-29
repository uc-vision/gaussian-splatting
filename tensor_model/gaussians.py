
from dataclasses import dataclass
import math
from tensorclass import TensorClass
import torch
from torch import functional as F

from tensor_model.geometry import quat_to_mat_kernel
from utils.sh_utils import RGB2SH

from .typecheck import NVec3, Vec1, Vec3, Vec4, typechecked, VecN
from simple_knn._C import distCUDA2



def inverse_sigmoid(x):
    return math.log(x/(1-x))

@dataclass
class Gaussians(TensorClass):

  positions : Vec3
  sh_dc : Vec3
  sh_rest : VecN

  log_scaling : Vec3
  rotation : Vec4
  opacity_logit: Vec1

  @property
  def sh_features(self):
    return torch.cat([self.sh_dc, self.sh_rest], dim=-1)
  
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

  @typechecked
  @staticmethod
  def from_points(points:NVec3, colors:NVec3, min_scale:float = 0.0001, sh_degree=1):

    dist2 = torch.clamp_min(distCUDA2(points), min_scale)
    scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
    
    rotations = torch.randn((points.shape[0], 4), device=points.device)
    opacity =  torch.full((points.shape[0], 1), fill_value=inverse_sigmoid(0.1),
                           dtype=torch.float, device=points.device)

    return Gaussians(
      positions = points,
      sh_dc = RGB2SH(colors),
      sh_rest = torch.zeros((points.shape[0], 3 * (sh_degree + 1) ** 2), device=points.device),
      log_scaling = torch.log(scales),
      rotation = F.normalize(rotations),
      opacity_logit = opacity)
      

