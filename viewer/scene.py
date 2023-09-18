from typing import Tuple
import trimesh
import pyrender

import numpy as np
from scipy.spatial.transform import Rotation as R

from tensor_model.fov_camera import FOVCamera, join_rt


def normalize(v):
  return v / np.linalg.norm(v)


def look_at(eye, target, up=np.array([0., 0., 1.])):
  forward = normalize(target - eye)
  left = normalize(np.cross(up, forward))
  true_up = np.cross(forward, left)
  return np.stack([left, true_up, forward])



def look_at_pose(eye, target, up=np.array([0., 0., 1.])):
  pose = np.eye(4)
  pose[:3, :3] = look_at(eye, target, up)
  pose[:3, 3] = eye
  return pose

def make_sphere(pos, color, radius):
  sphere = trimesh.creation.icosphere(radius=radius)
  sphere.visual.vertex_colors = color
  sphere_mesh = pyrender.Mesh.from_trimesh(sphere)
  node = pyrender.Node(mesh=sphere_mesh, translation=pos)
  return node

def fov_to_focal(fov, image_size):
  return image_size / (2 * np.tan(fov / 2))

flip_yz = np.array([
  [1, 0, 0],
  [0, -1, 0],
  [0, 0, -1]
])

class Scene:
  def __init__(self):

    self.scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
    # self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    # self.scene.add(self.light, pose=look_at_pose(np.array([2, 2, 2]), np.array([0, 0, 0])))

    for i in range(10):
      node = make_sphere(np.random.randn(3) , color=np.random.rand(3), radius=0.1)
      self.scene.add_node(node)

    self.camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
    self.cam_node = self.scene.add(self.camera, pose=np.eye(4))

   
  def add_node(self, node):
    self.scene.add_node(node)

  def add(self, mesh, pose=None):
    self.scene.add(mesh, pose=pose)

   
  def set_fov_camera(self, camera:FOVCamera):
    self.camera.yfov = camera.fov[1]

    rotation = camera.rotation  @ flip_yz 
    self.cam_node.matrix = join_rt(rotation, camera.position)



  def get_fov_camera(self, image_size) -> FOVCamera:
    focal_length = fov_to_focal(self.camera.yfov, image_size[1])
    return FOVCamera(
      position = self.cam_pos,
      rotation = self.cam_rotation @ flip_yz,
      image_size = image_size,
      focal_length = focal_length,
      image_name = "scene_camera"
    )

  def look_at(self, pos:np.array, target:np.ndarray, up:np.ndarray=np.array([0, 0, 1])):
    self.cam_node.matrix = look_at_pose(pos, target, up)


  @property 
  def cam_rotation(self):
    return self.cam_node.matrix[:3, :3]
  
  @property 
  def cam_pos(self):
    return self.cam_node.matrix[:3, 3]
  
  @cam_pos.setter
  def cam_pos(self, value):
    m = self.cam_node.matrix
    m[:3, 3] = value
    self.cam_node.matrix = m
  

  def move_camera(self, delta:np.ndarray):
    self.cam_pos += self.cam_rotation @ delta


  def rotate_camera(self, ypr):
    m = self.cam_node.matrix
    m[:3, :3] =  m[:3, :3] @ R.from_euler('yxz', ypr).as_matrix() 
    self.cam_node.matrix = m

