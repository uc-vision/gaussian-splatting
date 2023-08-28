
from dataclasses import dataclass
from tensorclass import TensorClass

from .typecheck import Vec1, Vec3, Vec4, typechecked, VecN, Float32



@dataclass
class Gaussians(TensorClass):

  positions : Vec3
  sh_dc : Vec3
  sh_rest : VecN

  scaling : Vec3
  rotation : Vec4
  opacity : Vec1

  
