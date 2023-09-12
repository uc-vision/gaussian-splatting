

import torch
import math
from .gaussians import Gaussians

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_geometry import Camera

def render(camera:Camera, model : Gaussians, bg_color : torch.Tensor):
  
    fovH, fovW = camera.fov
    width, height = camera.image_size

    device = model.device

    view, proj, pos = [torch.from_numpy(t).to(device=device, dtype=torch.float32) 
            for t in (camera.camera_t_parent, camera.unprojection, camera.location)]

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fovW * 0.5),
        tanfovy=math.tan(fovH * 0.5),
        bg=bg_color.to(device),
        scale_modifier=1.0,
        viewmatrix=view,
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
