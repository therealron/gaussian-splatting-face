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
import torchvision
import os
from re import I
import igl
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
import numpy as np
from scene import Scene, GaussianModelFace
from utils.general_utils import safe_state
import torchvision
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.dataset_readers import readCamerasFromTransforms
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.graphics_utils import getProjectionMatrix
class ViewCamera:
  def __init__(self):
    self.FovX = 0.2225811228028787
    self.FovY = 0.2225811228028787
    self.image_height = 512
    self.image_width = 512
    self.c2w = np.array([[0.9994248320452279,-0.03332750212378886,0.0062736968987590225,-0.016231912076032987],
            [-0.03097184270523503,-0.9723530145588566,-0.23145305663677745,0.385889258502049],
            [0.013814000838468536,0.23112562731765812,-0.9728259921280814,1.7591296183106164],
            [0.0,0.0,0.0,1.0]
          ])
    self.w2c = np.linalg.inv(self.c2w)
    self.world_view_transform = torch.tensor(self.w2c, dtype=torch.float32).cuda()
    
    self.zfar = 100.0
    self.znear = 0.01
    self.camera_center = torch.tensor(self.c2w[:3, 3], dtype=torch.float32).cuda()
    self.full_proj_transform = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0,1).cuda()
    self.original_image = torchvision.io.read_image('/content/gaussian-splatting-face/scene/justin/rgb_0.png')
    self.original_image = self.original_image[:3,:,:].to(torch.float32).cuda()

def read_expr(file_path):
    # Path to your text file
    # file_path = 'path_to_your_file.txt'

    # Read the file and convert each line to a float
    with open(file_path, 'r') as file:
        float_list = [float(line.strip()) for line in file]

    # Convert the list of floats to a PyTorch tensor
    tensor = torch.tensor(float_list)

    # Reshape the tensor to size (1, 100)
    tensor = tensor.view(1, -1)
    return tensor


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModelFace(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    print()# /content/gaussian-splatting-face
    expr = read_expr('/content/gaussian-splatting-face/scene/justin/flame/expr/00000.txt')
    # tracked_mesh, _, _, _, _, _ = igl.read_obj('/content/gaussian-splatting-face/scene/justin/mesh_0.obj')
    tracked_mesh_v, tracked_mesh_f = igl.read_triangle_mesh('/content/gaussian-splatting-face/scene/justin/mesh_0.obj')
    tracked_mesh_v, tracked_mesh_f = igl.upsample(tracked_mesh_v, tracked_mesh_f)
    tracked_mesh_v, tracked_mesh_f = igl.upsample(tracked_mesh_v, tracked_mesh_f)
    # tracked_mesh_v, tracked_mesh_f = igl.upsample(tracked_mesh_v, tracked_mesh_f)
    tracked_mesh, tracked_mesh_f = igl.upsample(tracked_mesh_v, tracked_mesh_f)
    print("tracked_mesh_v.shape = ",tracked_mesh.shape)
    m_to_mm = 1e3
    tracked_mesh = tracked_mesh * m_to_mm
    tracked_mesh = torch.tensor(tracked_mesh, dtype=torch.float32)
    tracked_mesh = tracked_mesh.cuda()
    torch.autograd.set_detect_anomaly(True)

    # curr_cam_infos = readCamerasFromTransforms("/content/gaussian-splatting-face/scene/justin", "transforms.json", True, "")
    # import pdb; pdb.set_trace();







    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        gaussians.generate_dynamic_gaussians(tracked_mesh, expr)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # import pdb; pdb.set_trace();
        # viewpoint_cam = ViewCamera()
        # viewpoint_cam = curr_cam_infos[0]



        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        mask = torchvision.io.read_image('/content/gaussian-splatting-face/scene/justin/images/00000.png')[3:,:,:]

        # print("mask.shape = ",mask.shape)
        mask = mask != 0
        mask = mask.to(torch.int32).cuda()
        # import pdb; pdb.set_trace();
        # background = torch.zeros_like(background) + 1.0
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = image * 255.0 * mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() 
        gt_image = gt_image * 255.0
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss =  Ll1 # + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        
        # self.mlp_optimizer
        
        if iteration % 5000 ==0:
            img1 = image.cpu().detach()
            print("img1.max() = ",img1.max())
            print("img1.min() = ",img1.min())
            img2 = gt_image.cpu()
            print("img2.max() = ",img2.max())
            print("img2.min() = ",img2.min())
            img = torch.cat([img1, img2], dim=1)
            checkpoint_img_path = f'/content/gaussian-splatting-face/checkpoint_img_{iteration}.jpeg'
            # import pdb; pdb.set_trace();
            img = img.to(torch.uint8)
            print("img.max() = ",img.max())
            print("img.min() = ",img.min())
            # checkpoint_img_path
            torchvision.io.write_jpeg(img , checkpoint_img_path)
            print("Wrote ",checkpoint_img_path)
            # torch

        iter_end.record()

        with torch.no_grad():
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            # if iteration < opt.densify_until_iter:
                # # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                # if iteration % opt.opacity_reset_interval == 0 or dataset.white_background :
                #     print("here")
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # gaussians.optimizer.step()
                # gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                # gaussians.mlp_optimizer.step()
                # gaussians.mlp_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.zero_grad(set_to_none = True)
            

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
