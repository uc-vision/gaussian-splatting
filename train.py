7#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from pathlib import Path
import torch
from random import randint
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians:GaussianModel = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    per_image_l1 = {}

    opt.iterations = int(opt.iterations * opt.training_scale)
    opt.densify_until_iter = int(opt.densify_until_iter * opt.training_scale)
    opt.position_lr_max_steps = int(opt.position_lr_max_steps * opt.training_scale)
    opt.densify_from_iter = int(opt.densify_from_iter * opt.training_scale)
    
    gaussians.training_setup(opt, len(scene.getTrainCameras()))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

        # first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):     
        # if iteration % 10 == 0:
        #   torch.cuda.empty_cache()
   
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
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
        if iteration % opt.sh_inc_iterations == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam:Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.to(device='cuda', non_blocking=True)

        # gt_depth = viewpoint_cam.depth
        # Render

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # depth = render_pkg["depth"]

        # image = gaussians.correct_colors(viewpoint_cam.uid, image_raw)

        # Loss


        Ll1 = l1_loss(image, gt_image)
        reg_loss = torch.zeros(1, device="cuda")
        if opt.reg_gaussians > 0.0:
          reg_loss = opt.reg_gaussians * gaussians.get_regularization_loss()

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  + reg_loss  
        # loss += l1_loss(depth, gt_depth) * 0.1
        loss.backward()

        name = viewpoint_cam.image_name
        if name in per_image_l1:
          per_image_l1[name] = 0.8 * per_image_l1[name] + 0.2 * Ll1.detach().item()
        else:
          per_image_l1[name] = Ll1.detach().item()


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
            training_report(tb_writer, iteration, l1_loss, testing_iterations, scene, render, (pipe, background))
            if tb_writer:
              tb_writer.add_scalar('train/l1_loss', Ll1.item(), iteration)
              tb_writer.add_scalar('train/reg_loss', reg_loss.item(), iteration)

              tb_writer.add_scalar('train/total_loss', loss.item(), iteration)
              tb_writer.add_scalar('train/time', iter_start.elapsed_time(iter_end), iteration)
              
              tb_writer.add_scalar('points/total', scene.gaussians.get_xyz.shape[0], iteration)

              n_visible = visibility_filter.sum()
              tb_writer.add_scalar('points/visible', n_visible, iteration)
              tb_writer.add_scalar('points/percent_visible', 100.0 * (n_visible / scene.gaussians.get_xyz.shape[0]), iteration)

            if iteration in testing_iterations and len(per_image_l1) > 0:
              tb_writer.add_histogram('images/l1_loss', torch.Tensor(list(per_image_l1.values())), iteration)

             

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter, :] = torch.max(gaussians.max_radii2D[visibility_filter, :], radii[visibility_filter].unsqueeze(1))
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    densify_stats = gaussians.densify(max_grad_clone = opt.densify_grad_threshold, 
                                                      max_grad_split = opt.clone_split_ratio * opt.densify_grad_threshold,
                                                      split_size_threshold=opt.split_size_threshold)


                    size_threshold = opt.vs_threshold if iteration > opt.opacity_reset_interval else None
                    prune_stats = gaussians.prune(min_opacity=0.05, max_screen_size=size_threshold)

                    for k, v in prune_stats.items():
                      tb_writer.add_scalar(f'pruned/{k}', v, iteration)

                    for k, v in densify_stats.items():
                      tb_writer.add_scalar(f'densified/{k}', v, iteration)

                    torch.cuda.empty_cache()
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.step()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    # if not args.model_path:
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str=os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    if args.model_path == "":
        args.model_path = str(Path(args.source_path).parent / "gaussian")
    
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

def training_report(tb_writer, iteration, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):


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

                    l1_test += l1_loss(image, gt_image).mean()
                    psnr_test += psnr(image, gt_image).mean()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    iterations = [1, 500, 1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 50000, 60000]

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=iterations)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=iterations)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=iterations)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
