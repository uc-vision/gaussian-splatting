from dataclasses import dataclass, replace
import math

from camera_geometry import Camera

from pathlib import Path
from typing import Tuple

import numpy as np
import json


@dataclass
class FOVCamera:
   
  position: np.ndarray
  rotation: np.ndarray
  focal_length : float
  image_size : Tuple[int, int]

  image_name: str
  near:float  = 0.1
  far :float  = 200.0

  @property
  def ndc_t_camera(self):

    z_sign = 1.0

    f = self.focal_length
    w, h = self.image_size

    z_scale = z_sign * self.far / (self.far - self.near)
    z_off =  -(self.far * self.near) / (self.far - self.near)

    return np.array(
      [[2 * f / w, 0,  0,  0],
        [0, 2 * f / h, 0,  0],
        [0, 0,  z_scale,  z_off],
        [0, 0,  z_sign,  0]]
    )
  
  @property
  def aspect(self):
    width, height = self.image_size
    return width / height
  
  @property 
  def resized(self, scale_factor) -> 'FOVCamera':
    width = round(width * scale_factor)
    height = round(height * scale_factor)

    width, height = self.image_size
    return replace(self,
      image_size=(width, height),
      focal_length=self.focal_length * scale_factor
    )
    

  @property
  def world_t_camera(self):
    return join_rt(self.rotation, self.position)
  
  @property
  def camera_t_world(self):
    return np.linalg.inv(self.world_t_camera)
  
  @property
  def ndc_t_world(self):
    return self.ndc_t_camera @ self.camera_t_world

  @property
  def fov(self):
    return tuple(math.atan2(x, self.focal_length * 2) * 2 for x in self.image_size)
  
  @property
  def intrinsic(self):
    
    width, height = self.image_size
    f = self.focal_length

    return np.array(
      [[f, 0,  width / 2],
        [0, f, height / 2],
        [0, 0,  1]]
    )
  
    
  @property
  def image_t_camera(self):
    m44 = np.eye(4)
    m44[:3, :3] = self.intrinsic
    
    return m44
  
  @property
  def image_t_world(self):
    return self.image_t_camera @ self.camera_t_world

  @property
  def projection(self):
    return self.image_t_world
  
def join_rt(R, T):
  Rt = np.zeros((4, 4))
  Rt[:3, :3] = R
  Rt[:3, 3] = T
  Rt[3, 3] = 1.0

  return Rt
         

def split_rt(Rt):
  R = Rt[:3, :3]
  T = Rt[:3, 3]
  return R, T



def camera_to_fov(camera:Camera) -> FOVCamera:
  assert camera.has_distortion == False, "Simple FOV camera does not have distortion"
  R, T = split_rt(camera.camera_t_parent)

  return FOVCamera(
    position = T,
    rotation = R,
    focal_length = camera.focal_length[0],
    image_size = camera.image_size
  )


def from_json(camera_info) -> Tuple[FOVCamera, Path]:
  pos = np.array(camera_info['position'])
  rotation = np.array(camera_info['rotation']).reshape(3, 3)

  return FOVCamera(
    position=pos,
    rotation=rotation,
    image_size=(camera_info['width'], camera_info['height']),
    focal_length=camera_info['fx'],
    image_name=camera_info['img_name']
  )



def load_camera_json(filename:Path):
  cameras = sorted(json.loads(filename.read_text()), key=lambda x: x['id'])

  return {camera_info['id']: from_json(camera_info) for camera_info in cameras}
  
