#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from torch.nn import functional as F

class ColorCorrection(nn.Module):
    def __init__(self, n:int):
      super(ColorCorrection, self).__init__()     

      self.weight1 = nn.Parameter(torch.eye(3).repeat(n, 1, 1).requires_grad_(True))
      self.bias1 = nn.Parameter(torch.zeros(n, 3).requires_grad_(True))

      self.gamma_offset = nn.Parameter(torch.zeros(n, 1).requires_grad_(True))

      self.weight2 = nn.Parameter(torch.eye(3).repeat(n, 1, 1).requires_grad_(True))
      self.bias2 = nn.Parameter(torch.zeros(n, 3).requires_grad_(True))


    def parameter_groups(self, lr):
        return [
            {'params': [self.weight1, self.bias1, self.weight2, self.bias2], 
             'lr': lr, "name": "weights"},
            {'params': [self.gamma_offset], 'lr': lr * 0.01, "name": "gamma", "weight_decay": 1e-6} 
        ]

    def forward(self, idx, image):
      return F.conv2d(image, self.weight1[idx].view(3, 3, 1, 1), self.bias1[idx])
      # x = torch.pow(x, 1 + self.gamma_offset[idx])
      # return F.conv2d(x, self.weight2[idx].view(3, 3, 1, 1), self.bias2[idx])

class ModifySH(nn.Module):
    def __init__(self, n:int, max_sh_degree : int):
      super(ModifySH, self).__init__()     
      features = (max_sh_degree + 1) ** 2

      self.scale = nn.Parameter(torch.ones(n, features, 3).requires_grad_(True))
      self.bias = nn.Parameter(torch.zeros(n, features, 3).requires_grad_(True))


    def parameter_groups(self, lr):
        return [
            {'params': [self.scale, self.bias], 'lr': lr, "name": "weights"},
        ]
    
    def forward(self, idx, sh):
      sh = sh * self.scale[idx] + self.bias[idx]     
      return sh
      #return F.conv1d(sh.permute(0, 2, 1), self.weight[idx].view(3, 3, 1)).permute(0, 2, 1)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.vs_gradient_accum = torch.empty(0)
        self.ws_gradient_accum = torch.empty(0)

        self.vis_count = torch.empty(0)
        self.optimizer = None
        self.image_optimizer = None
        self.correct_colors = None
        self.transform_sh = None
        self.setup_functions()
        self.num_images = 0

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.vs_gradient_accum,
            self.ws_gradient_accum,
            
            self.vis_count,
            self.optimizer.state_dict(),
            self.image_optimizer.state_dict(),
            self.num_images
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        vs_gradient_accum, 
        ws_gradient_accum,
        vis_count,
        opt_dict,
        image_opt_dict,
        num_images) = model_args
        
        self.training_setup(training_args, num_images)
        self.vs_gradient_accum = vs_gradient_accum
        self.ws_gradient_accum = ws_gradient_accum
        
        self.vis_count = vis_count
        self.optimizer.load_state_dict(opt_dict)
        self.image_optimizer.load_state_dict(image_opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_image_features(self, idx):
        features = self.get_features
        if self.transform_sh is not None:
          return self.transform_sh(idx, features)
        else:
          return features

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity) 
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud,  scales:torch.Tensor = None):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        if scales is None:
          dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
          scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        else:        
          scales = torch.log(scales).cuda()            

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def to_sparse(self, optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group['params']:
                  if param.grad is not None:
                    param.grad = param.grad.to_sparse()

    def step(self):
        self.to_sparse(self.optimizer)
        self.to_sparse(self.image_optimizer)
                
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.image_optimizer.step()
        self.image_optimizer.zero_grad(set_to_none = True)


    def training_setup(self, training_args, num_images):
        self.vs_gradient_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.ws_gradient_accum = torch.zeros(self.get_xyz.shape, device="cuda")
        
        self.vis_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.correct_colors = ColorCorrection(num_images)
        self.transform_sh = ModifySH(num_images, self.max_sh_degree)

        self.correct_colors.to(device="cuda")
        self.transform_sh.to(device="cuda")

        self.num_images = num_images

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr * training_args.feature_rest_lr_mul, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.SparseAdam(l, lr=0.001, eps=1e-15)

        self.image_optimizer = torch.optim.SparseAdam(
            self.correct_colors.parameter_groups(training_args.image_color_lr) 
            + self.transform_sh.parameter_groups(training_args.transform_sh_lr),
                     lr=0.001, eps=1e-15)



        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init,
                                                    lr_final=training_args.position_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.vs_gradient_accum = self.vs_gradient_accum[valid_points_mask]
        self.ws_gradient_accum = self.ws_gradient_accum[valid_points_mask]
        
        self.vis_count = self.vis_count[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def pad_zeros(self, tensor, n):
        shape = (n - tensor.shape[0],) + tensor.shape[1:]
        return torch.cat((tensor, torch.zeros(shape, device="cuda")), dim=0)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : F.normalize(new_rotation)}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.max_radii2D = self.pad_zeros(self.max_radii2D, self._xyz.shape[0])
        self.vis_count = self.pad_zeros(self.vis_count, self._xyz.shape[0])

        self.ws_gradient_accum = self.pad_zeros(self.ws_gradient_accum, self._xyz.shape[0])
        self.vs_gradient_accum = self.pad_zeros(self.vs_gradient_accum, self._xyz.shape[0])
        
    def sample_gaussians(self, mask, N):
        stds = self.get_scaling[mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[mask].repeat(N, 1)

        return new_xyz

    def densify_and_split(self, grads, grad_threshold, size_threshold, N=2):
        # Extract points that satisfy the gradient condition

        large_points = torch.max(self.get_scaling, dim=1).values > size_threshold
        selected_pts_mask = (grads >= grad_threshold) & large_points

        new_xyz = self.sample_gaussians(selected_pts_mask, N)
        new_scaling = self.scaling_inverse_activation(
             self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))

        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.prune_points(selected_pts_mask)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        return selected_pts_mask.sum()


    def densify_and_clone(self, grads, grad_threshold, size_threshold):

        small_points = torch.max(self.get_scaling, dim=1).values < size_threshold
        selected_pts_mask = (grads >= grad_threshold) & small_points

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

        return selected_pts_mask.sum()


    def densify(self, max_grad_clone, max_grad_split, split_size_threshold, min_vis_count=50):
        valid = self.vis_count > min_vis_count

        grads = self.vs_gradient_accum / self.vis_count
        grads[~valid] = 0.0
        
        self.vs_gradient_accum[valid] = 0.0
        self.ws_gradient_accum[valid] = 0.0
        self.vis_count[valid] = 0.0
                    
        cloned = self.densify_and_clone(grads, max_grad_clone, split_size_threshold)
        grads = self.pad_zeros(grads, self.get_xyz.shape[0])
        splits = self.densify_and_split(grads, max_grad_split, split_size_threshold)

        return dict(cloned=cloned, split=splits)

    def prune(self, min_opacity, max_screen_size):

        min_opacity = (self.get_opacity < min_opacity)
        big_points_vs = self.max_radii2D > (max_screen_size or torch.inf) 


        prune_mask = min_opacity | big_points_vs 
        self.prune_points(prune_mask.squeeze(1))

        self.max_radii2D.fill_(0.0)

        return dict(min_opacity=min_opacity.sum().item(),
                    big_points_vs=big_points_vs.sum().item())
    

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.vs_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter,:2], dim=-1)
        
        self.ws_gradient_accum[update_filter] += self._xyz.grad[update_filter]
        self.vis_count[update_filter] += 1


    def get_regularization_loss(self):
        scaling = self.get_scaling
        factors = (scaling.max(dim=1).values / scaling.min(dim=1).values) - 1.0

        return factors.mean() #(1 - self.get_opacity.mean()) 
        
