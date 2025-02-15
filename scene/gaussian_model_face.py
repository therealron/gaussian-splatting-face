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

from sys import implementation
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
import igl

class FullyConnectedMLP(nn.Module):
    def __init__(self, rot_size=4, 
                        scale_size=3, 
                        xyz_size= 3, 
                        hidden_size=256, 
                        depth=5,  
                        intermediate_size=128,
                        embedding_dim = 10,
                        expr_dim = 100):

        super(FullyConnectedMLP, self).__init__()
        self.rot_size = rot_size
        self.scale_size = scale_size
        self.xyz_size = xyz_size
        self.intermediate_size = intermediate_size
        self.embedding_dim = embedding_dim
        self.expr_dim = expr_dim
        layers = []
        output_size = rot_size + scale_size + xyz_size
        input_size = 3 + embedding_dim*2*3 + expr_dim

        self.input_size = input_size
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, intermediate_size))

        # Combine all layers
        self.layers = nn.Sequential(*layers)

        self.rot_head = nn.Linear(intermediate_size, rot_size)
        self.scale_head = nn.Linear(intermediate_size, scale_size)
        self.xyz_head = nn.Linear(intermediate_size, xyz_size)

    
    def positional_encoding(self,x, L):
        out = [x]

        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, canonical_template, expr_code):
        """
        canonical_xyz: (N,3)
        expr_code: (1, 100)

        """
        # print("self.input_size = ",self.input_size)
        
        
        batch_size = expr_code.shape[0]
        # canonical_xyz_features is of shape (N, num_pos_encodings)
        canonical_xyz_features = self.positional_encoding(canonical_template, self.embedding_dim)
        canonical_xyz_features = canonical_xyz_features.unsqueeze(0)
        # now it is of shape (B, N, num_pos_encodings)
        canonical_xyz_features = canonical_xyz_features.repeat(batch_size, 1, 1)

        expr_code = expr_code.unsqueeze(1)
        #of shape (B,N, expr_code)
        expr_code = expr_code.repeat(1, canonical_xyz_features.shape[1], 1)
        expr_code = expr_code.cuda()

        # x will be of shape (B, N, expr_code+num_pose_encodings)
        x = torch.cat([canonical_xyz_features.cuda(),expr_code.cuda() ], dim=2)
        # x will be of shape (B*N, expr_code+num_pose_encodings)
        x = x.view(-1, self.input_size )
        # print("x.shape = ",x.shape)

        intermediate_output = self.layers(x)

        # import pdb

        del_rot = self.rot_head(intermediate_output)
        del_rot = del_rot.view(batch_size,-1, self.rot_size) # eg. (B,N,4)
        
        del_scale = self.scale_head(intermediate_output)
        del_scale = del_scale.view(batch_size, -1, self.scale_size) # eg. (B,N,3)
        
        del_xyz = self.xyz_head(intermediate_output)
        del_xyz = del_xyz.view(batch_size, -1, self.xyz_size) # eg. (B,N,3)

        # del_scale = del_xyz.to(torch.float32)
        
        
        return del_xyz.to(torch.float32), del_scale, del_rot



class GaussianModelFace:

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
        self._canonical_xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.delta_mlp_model = None
        self.mlp_optimizer = None
        self._final_rotation = torch.empty(0)
        self._final_scale = torch.empty(0)
        self.setup_functions()
        # self.

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
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
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
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._final_scale)
        # return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._final_rotation)
        # return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz.to(torch.float32)
    

    @property
    def get_canonical_xyz(self):
        return self._canonical_xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        raise NotImplementedError
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_canonical_xyz.shape[0]), device="cuda")

    def create_from_flame(self,  spatial_lr_scale : float, flame_canonical_path: str = '/content/gaussian-splatting-face/scene/justin/canonical.obj'):
        
        self.spatial_lr_scale = spatial_lr_scale
        assert os.path.exists(flame_canonical_path), flame_canonical_path+ " does not exist!"
        # v, vt, _, faces, ftc, _ = igl.read_obj(flame_canonical_path)
        v, faces = igl.read_triangle_mesh(flame_canonical_path)
        v, faces = igl.upsample(v, faces)
        v, faces = igl.upsample(v, faces)
        v, faces = igl.upsample(v, faces)
        # v, faces = igl.upsample(v, faces)
        print("canonical xyz shape = ",v.shape)
        mm_to_m = 1e3
        v = v * mm_to_m
        
        fused_point_cloud = torch.tensor(v ).float().cuda()
        self._canonical_xyz = fused_point_cloud
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((v.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", v.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(v).float().cuda()), 0.0000001)
        
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_canonical_xyz.shape[0]), device="cuda")
        self.delta_mlp_model = FullyConnectedMLP().cuda()
        self.delta_mlp_model = self.delta_mlp_model.cuda()


    def normalize_quaternion(self, q):
        norm = torch.norm(q, p=2, dim=-1, keepdim=True)
        return q / norm

    
    def multiply_quaternions(self, q1, q2):
        """
        Multiplies two batches of quaternions.
        q1, q2: Tensors of shape (B, 4)
        """
        # Extract components
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        # Calculate the product
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        # Combine components into a single tensor
        return torch.stack((w, x, y, z), dim=1)

    def generate_dynamic_gaussians(self, tracked_mesh, flame_expr_params):
        """
        in each iteration just pass a single mesh
        tracked_mesh: (N,3)
        flame_expr_params: (1,100)
        """
        

        assert tracked_mesh.shape[0] == self._canonical_xyz.shape[0] # assert that they have the same # vertices
        
        assert flame_expr_params.shape[0] == 1 and flame_expr_params.shape[1]==100

        if self._canonical_xyz.shape[0]<1e6:
            del_u, del_scale, del_rot = self.delta_mlp_model(self._canonical_xyz, flame_expr_params)
            self._xyz = tracked_mesh + del_u[0]
            # self._rotation += del_rot[0]
            # self._final_rotation = self._rotation + del_rot[0]
            self._final_rotation = self.multiply_quaternions(self._rotation, del_rot[0])
            self._final_scale = self._scaling + del_scale[0]
        else:
            # do them in three passes
            # del_u_1, del_scale_1, del_rot_1 = self.delta_mlp_model(self._canonical_xyz, flame_expr_params)
            num_vertices = self._canonical_xyz.shape[0]

            # Calculate the size of each split
            split_size = num_vertices // 3

            # Split the mesh into three parts
            split1 = self._canonical_xyz[:split_size, :]
            split2 = self._canonical_xyz[split_size:2*split_size, :]
            split3 = self._canonical_xyz[2*split_size:, :]

            # Run the model on each part
            del_u_1_part1, del_scale_1_part1, del_rot_1_part1 = self.delta_mlp_model(split1, flame_expr_params)
            del_u_1_part2, del_scale_1_part2, del_rot_1_part2 = self.delta_mlp_model(split2, flame_expr_params)
            del_u_1_part3, del_scale_1_part3, del_rot_1_part3 = self.delta_mlp_model(split3, flame_expr_params)

            # Concatenate the results back together
            
            del_u_1 = torch.cat((del_u_1_part1, del_u_1_part2, del_u_1_part3), dim=1)
            del_scale_1 = torch.cat((del_scale_1_part1, del_scale_1_part2, del_scale_1_part3), dim=1)
            del_rot_1 = torch.cat((del_rot_1_part1, del_rot_1_part2, del_rot_1_part3), dim=1)


            self._xyz = tracked_mesh + del_u_1[0]
            # self._rotation += del_rot[0]
            # self._final_rotation = self._rotation + del_rot[0]
            self._final_rotation = self.multiply_quaternions(self._rotation, del_rot_1[0])
            self._final_scale = self._scaling + del_scale_1[0]
            

        # print("del_u.shape = ",del_u.shape)
        # print("del_rot.shape = ",del_rot.shape)
        # print("del_scale.shape = ",del_scale.shape)

        # normalized_del_rot 

        # random_number = random.randint(1, 100)
        # print("del_u.max() = ",del_u.max())
        # print("del_scale.max() = ",del_scale.max())
        # print("del_rot.max() = ",del_rot.max())
        # import pdb; pdb.set_trace();
        
        # print("del_scale.dtype = ",del_scale.dtype)
        # print("del_rot.dtype = ",del_rot.dtype)
        # import pdb; pdb.set_trace();
        
        

        # print("del_u.dtype = ",del_u.dtype)
        # print("del_scale.dtype a= ",del_scale.dtype)
        # print("del_rot.dtype = ",del_rot.dtype)
        # print("self._xyz.dtype = ",self._xyz.dtype)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_canonical_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_canonical_xyz.shape[0], 1), device="cuda")
        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.delta_mlp_model.parameters(), 'lr': 0.00016, "name": "mlp"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.001, eps=1e-15)

        # optim.Adam()
        # mlp_params = [{'params': self.delta_mlp_model.parameters(), 'lr': 0.01, "name": "mlp"}
        # ]/
        # self.mlp_optimizer = torch.optim.Adam(mlp_params, lr=0.001,eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
        
        # update the lr of the mlp model
        if iteration > 9999 and iteration%50000 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "mlp":
                    lr = param_group['lr'] / 10
                    param_group['lr'] = lr
                    return lr
        return None

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
        print("not saving ply")
        return
        mkdir_p(os.path.dirname(path))

        canonical_xyz = self._canonical_xyz.detach().cpu().numpy()
        normals = np.zeros_like(canonical_xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(canonical_xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((canonical_xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_model(self, path):
        mkdir_p(os.path.dirname(path))
        model = self.delta_mlp_model
        torch.save(model.state_dict(), path)
        print("Saved Checkpoint to ",path)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        raise NotImplementedError
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
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        raise NotImplementedError
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
        raise NotImplementedError
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        raise NotImplementedError
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        raise NotImplementedError
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_canonical_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_canonical_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_canonical_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        raise NotImplementedError
        n_init_points = self.get_canonical_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        raise NotImplementedError
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        raise NotImplementedError
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        raise NotImplementedError
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1