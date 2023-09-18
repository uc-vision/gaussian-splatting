from dataclasses import asdict, dataclass
import torch
import math

from .fov_camera import FOVCamera, split_rt
from .gaussians import Gaussians

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import numpy as np
 

@dataclass
class RenderOutputs:
  image : torch.Tensor
  radii : torch.Tensor
   
  
def render_gaussians(camera:FOVCamera, model : Gaussians, bg_color : torch.Tensor):
  
    device = model.device
    assert device != torch.device("cpu"), "CPU rendering is not supported."


    view, proj, pos = [torch.from_numpy(t).to(device=device, dtype=torch.float32) 
            for t in (camera.camera_t_world, camera.ndc_t_world, camera.position)]

    width, height = camera.image_size
    fovW, fovH = camera.fov

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fovW * 0.5),
        tanfovy=math.tan(fovH * 0.5),
        bg=bg_color.to(device),
        scale_modifier=1.0,
        viewmatrix=view.transpose(0, 1),
        projmatrix=proj.transpose(0, 1),
        sh_degree=model.sh_degree,
        campos=pos,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(model.positions, device=device)

    if torch.is_grad_enabled():
      means2D.requires_grad_(True).retain_grad()

    rendered_image, radii = rasterizer(
        means2D = means2D,
        means3D = model.positions,
        shs = model.sh_features,
        colors_precomp = None,
        opacities = model.opacity,
        scales = model.scaling,
        rotations = model.rotation)

    return RenderOutputs(
      image = rendered_image,
      radii = radii
    )