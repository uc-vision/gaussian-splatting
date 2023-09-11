

import torch
import math
from .gaussians import Gaussians

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_geometry import Camera

def render(camera:Camera, model : Gaussians, bg_color : torch.Tensor):
  
    fovH, fovW = camera.fov
    width, height = camera.image_size

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fovW * 0.5),
        tanfovy=math.tan(fovH * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=torch.from_numpy(camera.parent_t_camera),
        projmatrix=torch.from_numpy(camera.projection),
        sh_degree=model.sh_degree,
        campos=torch.from_numpy(camera.location),
        prefiltered=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
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
