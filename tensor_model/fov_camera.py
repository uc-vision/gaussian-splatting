from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
import json


@dataclass
class FOVCamera:
   
  camera_t_world: np.ndarray
  fovH : float
  image_size : Tuple[int, int]


  @property   
  def projection_ndc(self, near, far):
    fx = 0.5 * math.tan(self.fovH / 2)  
    fy = fx * self.aspect 

    z_sign = 1.0
    z = z_sign * far / (far - near)
    w = -z_sign * far * near / (far - near)

    return np.array(
      [[fx, 0,  0,  0],
        [0, fy, 0,  0],
        [0, 0,  z,  w],
        [0, 0,  1,  0]]
    )
  
  @property
  def aspect(self):
    width, height = self.image_size
    return width / height
  
  @property
  def location(self):
    return self.camera_t_world[:, 3]
  
  @property
  def intrinsic(self):
    
    width, height = self.image_size
    fx = width * (0.5 * math.tan(self.fovH / 2))
    fy = fx * self.aspect

    return np.array(
      [[fx, 0,  width / 2],
        [0, fy, height / 2],
        [0, 0,  1]]
    )
         
   

def load_json_cameras(filename):
  with open(filename) as f:
    data = json.load(f)
