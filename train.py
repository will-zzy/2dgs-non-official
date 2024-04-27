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
import imageio
from omegaconf import OmegaConf
import cv2
from matplotlib import cm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
from random import randint
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

def get_grayscale_image_(img, data_range, cmap):
    img = convert_data(img)
    img = np.nan_to_num(img)
    if data_range is None:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img.clip(data_range[0], data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
    assert cmap in [None, 'jet', 'magma']
    if cmap == None:
        img = (img * 255.).astype(np.uint8)
        img = np.repeat(img[...,None], 3, axis=2)
    elif cmap == 'jet':
        img = (img * 255.).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif cmap == 'magma':
        img = 1. - img
        base = cm.get_cmap('magma')
        num_bins = 256
        colormap = LinearSegmentedColormap.from_list(
            f"{base.name}{num_bins}",
            base(np.linspace(0, 1, num_bins)),
            num_bins
        )(np.linspace(0, 1, num_bins))[:,:3]
        a = np.floor(img * 255.)
        b = (a + 1).clip(max=255.)
        f = img * 255. - a
        a = a.astype(np.uint16).clip(0, 255)
        b = b.astype(np.uint16).clip(0, 255)
        img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
        img = (img * 255.).astype(np.uint8)
    return img
def convert_data(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: convert_data(v) for k, v in data.items()}
    else:
        raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))

# def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
def training(config, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(config.gs_model)
    gaussians = GaussianModel(config.gs_model.sh_degree,config.gs_model.sigma)
    scene = Scene(config, gaussians)
    gaussians.training_setup(config.optimizer)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, config.optimizer)

    bg_color = [1, 1, 1] if config.gs_model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, config.optimizer.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, config.optimizer.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, config.pipeline.convert_SHs_python, config.pipeline.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, config.pipeline, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, config.gs_moded.source_path)
                if do_training and ((iteration < int(config.optimizer.iterations)) or not keep_alive):
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
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            config.pipeline.debug = True

        bg = torch.rand((3), device="cuda") if config.optimizer.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, config.pipeline, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        means3D, scales, sh ,colors =\
        render_pkg["means3D"], render_pkg["scales"], render_pkg["sh"], render_pkg["colrors"]
        
        image_write = torch.clamp(render_pkg["render"], 0.0, 1.0)
        depth_write = render_pkg['depth']
        opacity_map = render_pkg["opacity"]
        opacity_map[opacity_map<1e-4] = 1.0
        depth_write /= opacity_map
        # opacity_write = render_pkg['opacity']
        depth_write = get_grayscale_image_(depth_write,data_range=[0,20],cmap='jet')
        output_dir = os.path.join("/data3/zzy/public_data/tankandtemples/intermediate/Family/output",f"{0}")
        os.makedirs(output_dir,exist_ok=True)
        imageio.imwrite(os.path.join(output_dir,f"{0}_rgb.jpg"),(image_write.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir,f"{0}_depth.jpg"),depth_write)
        
        
        # Loss
        gt_image = viewpoint_cam.get_image.cuda()
        # gt_image = torch.zeros_like(image).cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - config.loss.lambda_ssim) * Ll1 + config.loss.lambda_ssim * (1.0 - ssim(image, gt_image))
        # loss = (1.0 - config.loss.lambda_ssim) * Ll1 
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == config.optimizer.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (config.pipeline, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < config.optimizer.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > config.optimizer.densify_from_iter and iteration % config.optimizer.densification_interval == 0:
                    size_threshold = 20 if iteration > config.optimizer.opacity_reset_interval else None
                    gaussians.densify_and_prune(config.optimizer.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % config.optimizer.opacity_reset_interval == 0 or (config.dataset.white_background and iteration == config.optimizer.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < config.optimizer.iterations:
                gaussians.optimizer.step()
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
        
        output_dir = os.path.join("/data3/zzy/public_data/tankandtemples/intermediate/Family/output",f"{iteration}")
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                os.makedirs(output_dir,exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_out["render"], 0.0, 1.0)
                    depth = render_out['depth']
                    gt_image = torch.clamp(viewpoint.get_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    depth = get_grayscale_image_(depth,data_range=None,cmap='jet')
                    imageio.imwrite(os.path.join(output_dir,f"{idx}_rgb.jpg"),(image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                    cv2.imwrite(os.path.join(output_dir,f"{idx}_depth.jpg"),depth)
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

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf

if __name__ == "__main__":
    # Set up command line argument parser
    torch.set_printoptions(precision=8,sci_mode=False)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1,500,7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # args = parser.parse_args(sys.argv[1:])
    parser.add_argument('--conf_path',default='./config/base.yaml')
    
    args, extras = parser.parse_known_args()
    cli_conf = OmegaConf.from_cli()
    config = load_config(args.conf_path,cli_args=extras)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.test_iterations = [i*500 for i in range(0,60)]
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    training(config, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
