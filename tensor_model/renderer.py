

from dataclasses import dataclass
from typing import Tuple
import torch
import math

from scene import from_colmap_transform
from scene import Camera as SplatCamera

from .gaussians import Gaussians

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_geometry import Camera

import numpy as np



def to_camera(i, camera:Camera):
    R, T = from_colmap_transform(camera.camera_t_parent)


    return SplatCamera(colmap_id=i,
                  uid=i,
                  image_name="foo",
                  R=R, T=T,
                  FoVx=camera.fov[0], 
                  FoVy=camera.fov[1],
                  image=torch.zeros(*camera.image_size, 3, dtype=torch.float32),
                  data_device='cuda',
                  gt_alpha_mask=None
                  )

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
         
   
def camera_to_fov(camera:Camera) -> FOVCamera:
   assert camera.has_distortion == False, "Simple FOV camera does not have distortion"
  
   return FOVCamera(
      fov = camera.fov,
      camera_t_world = camera.camera_t_parent,
      image_size = camera.image_size
    )
   
   
   

def render(camera:Camera, model : Gaussians, bg_color : torch.Tensor):
  
    device = model.device

    fov = camera_to_fov(camera)


    # view, proj, pos = [torch.from_numpy(t).to(device=device, dtype=torch.float32) 
    #         for t in (camera.camera_t_parent, np.linalg.inv(camera.projection), camera.location)]


    spcam = to_camera(0, camera)
    print("view\n", spcam.world_view_transform, "\n", fov.camera_t_world.transpose(0, 1))

    # print("proj\n", spcam.full_proj_transform, "\n", proj)
    # print(spcam.camera_center, pos)

    
    
    print(camera.intrinsic, camera.image_size)

    return

    # print(spcam.projection_matrix, "\n", camera.intrinsic)
  
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fovW * 0.5),
        tanfovy=math.tan(fovH * 0.5),
        bg=bg_color.to(device),
        scale_modifier=1.0,
        viewmatrix=view.transpose(0, 1),
        projmatrix=proj,
        sh_degree=model.sh_degree,
        campos=pos,
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = torch.zeros_like(model.positions, device=device)

    if torch.is_grad_enabled():
      means2D.requires_grad_(True).retain_grad()

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means2D = means2D,
        means3D = model.positions,
        shs = model.sh_features,
        colors_precomp = None,
        opacities = model.opacity,
        scales = model.scaling,
        rotations = model.rotation)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}
