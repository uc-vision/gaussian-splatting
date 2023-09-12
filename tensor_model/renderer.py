

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


def render(camera:Camera, model : Gaussians, bg_color : torch.Tensor):
  
    fovH, fovW = camera.fov
    width, height = camera.image_size

    device = model.device

    view, proj, pos = [torch.from_numpy(t).to(device=device, dtype=torch.float32) 
            for t in (camera.camera_t_parent, np.linalg.inv(camera.projection), camera.location)]

    spcam = to_camera(0, camera)
    print("view\n", spcam.world_view_transform, "\n", view)

    print("proj\n", spcam.full_proj_transform, "\n", proj)
    print(spcam.camera_center, pos)

    m44 = torch.eye(4, device=device, dtype=torch.float32)
    m44[:3, :3] = torch.from_numpy(camera.intrinsic).to(device=device, dtype=torch.float32)

    print(camera.image_size)
    print(torch.inverse(m44) @ spcam.projection_matrix.to(device))

    print(spcam.projection_matrix, "\n", camera.intrinsic)
  
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
