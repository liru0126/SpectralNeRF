"""

2022-10-04 10:16:32  合并run_nerf和run_snerf，用一个额外的config 参数控制是否对fn输出rgb进行约束
同时默认存最好的模型用best spectral psnr和best spectral l1 distance

2022-09-11 15:36:43  snerf训练

"""

import os, sys

## debug
# os.environ['CUDA_VISIBLE_DEVICES']= '3'

import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import configargparse

import matplotlib.pyplot as plt

from mymodule.net_loss_define import fetch_net_loss_criternion
from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

import pdb

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import Params

np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # near = 2.0, far = 6.0
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # pdb.set_trace()

    k_extract = ['rgb_map', 'spectrum_rgb_maps', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, s_gt_imgs=None, savedir=None,
                render_factor=0, render_only=False, Unet=None):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
    # pdb.set_trace()

    fn_rgbs = []
    unet_rgbs = []
    spectrum_rgbs = []
    disps = []
    fn_t_psnr = []
    unet_t_psnr = []
    t_s_psnr = []
    t_s_l1_distance = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        if not render_only:
            rgb, spectrum_rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        else:
            if Unet is None:
                raise ValueError('Warning! Unet is not passes to the render function')
            rgb, spectrum_rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)

            # acc -> shape(256,256)
            # mask = acc[None, ...].repeat(3, 1, 1) > 0  # (3,256,256)
            # rgb用UNET融合spectrum rgb得到
            unet_rgb = Unet(spectrum_rgb.permute(2, 0, 1)[None, ...])  # (1,3,H,W)
            unet_rgb = unet_rgb[0]
            # unet_rgb[mask != 1] = 1.  # unet训练的时候没有管背景，需要手动mask掉，置为白色
            unet_rgb = unet_rgb.permute(1, 2, 0)  # (H,W,3)

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(rgb.cpu().numpy()/6)
            # plt.title('snerf + fn')
            # plt.subplot(1, 3, 2)
            # plt.imshow(unet_rgb.cpu().numpy()/6)
            # plt.title('snerf + unet')
            # plt.subplot(1, 3, 3)
            # plt.imshow(gt_imgs[i]/6)
            # plt.title('gt')
            # plt.show()

            unet_rgbs.append(unet_rgb.cpu().numpy())

        fn_rgbs.append(rgb.cpu().numpy())
        spectrum_rgbs.append(spectrum_rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        spectrum_num = int(spectrum_rgb.shape[2] / 3)

        if i == 0:
            print(rgb.shape, spectrum_rgb.shape, disp.shape)

        # 计算psnr
        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(fn_rgbs[-1] - gt_imgs[i])))
            fn_t_psnr.append(p)

            # 在render only模式下额外计算snerf + unet 的 psnr
            if unet_rgbs.__len__() > 1:
                unet_p = -10. * np.log10(np.mean(np.square(unet_rgbs[-1] - gt_imgs[i])))
                unet_t_psnr.append(unet_p)

        if s_gt_imgs is not None and render_factor == 0:
            for num in range(spectrum_num):
                p = -10. * np.log10(np.mean(
                    np.square(spectrum_rgb[..., num * 3:(num + 1) * 3].cpu().numpy() - s_gt_imgs[i * 11 + num])))
                t_s_psnr.append(p)
                s_l1_distance = np.mean(
                    np.abs(spectrum_rgb[..., num * 3:(num + 1) * 3].cpu().numpy() - s_gt_imgs[i * 11 + num]))
                t_s_l1_distance.append(s_l1_distance)

        if savedir is not None:
            rgb8 = to8b(fn_rgbs[-1])
            filename = os.path.join(savedir, 'fn_{:03d}.png'.format(i))
            # rgb8 = np.clip(rgbs[-1], 0, 1)
            # rgb8 = rgbs[-1]  # 保存exr的时候不clip
            # filename = os.path.join(savedir, '{:03d}.exr'.format(i))
            imageio.imwrite(filename, rgb8)

            # 在render only模式下额外保存snerf + unet的主观图结果
            if len(unet_rgbs) > 1:
                unet_rgb8 = to8b(unet_rgbs[-1])
                filename = os.path.join(savedir, 'unet_{:03d}.png'.format(i))
                imageio.imwrite(filename, unet_rgb8)

            for num, wavelength in enumerate(np.linspace(380, 780, spectrum_num, dtype=int)):
                s_rgb8 = to8b(spectrum_rgbs[-1][..., num * 3:(num + 1) * 3])
                filename = os.path.join(savedir, '{:03d}_s{}.png'.format(i, wavelength))
                # s_rgb8 = spectrum_rgbs[-1][..., num * 3:(num + 1) * 3]  # 保存exr的时候不进行clip
                # s_rgb8 = np.clip(spectrum_rgbs[-1][..., num * 3:(num + 1) * 3], 0, 1)
                # filename = os.path.join(savedir, '{:03d}_s{:03d}.exr'.format(i, num))
                imageio.imwrite(filename, s_rgb8)

    # pdb.set_trace()
    fn_rgbs = np.stack(fn_rgbs, 0)
    disps = np.stack(disps, 0)
    test_psnr = {}
    if gt_imgs is not None and render_factor == 0:
        fn_avg_psnr = np.mean(fn_t_psnr)
        test_psnr['fn_avg_test_psnr'] = fn_avg_psnr
        test_psnr['fn_test_psnr'] = fn_t_psnr

        # 在render only模式下额外保存snerf + unet的metrics
        if len(unet_rgbs) > 1:
            unet_avg_psnr = np.mean(unet_t_psnr)
            test_psnr['unet_avg_test_psnr'] = unet_avg_psnr
            test_psnr['unet_test_psnr'] = unet_t_psnr

    if s_gt_imgs is not None and render_factor == 0:
        avg_s_psnr = np.mean(t_s_psnr)
        test_psnr['avg_test_s_psnr'] = avg_s_psnr
        test_psnr['test_s_psnr'] = t_s_psnr
        avg_s_l1_distance = np.mean(t_s_l1_distance)
        test_psnr['avg_s_l1_distance'] = avg_s_l1_distance
        test_psnr['test_s_l1_distance'] = t_s_l1_distance
    return fn_rgbs, spectrum_rgbs, disps, test_psnr


def render_path_backup(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, s_gt_imgs=None, savedir=None,
                       render_factor=0):
    """
    2022-08-30 15:26:18  创建备份，改Unet
    """
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
    # pdb.set_trace()

    rgbs = []
    spectrum_rgbs = []
    disps = []
    t_psnr = []
    t_s_psnr = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, spectrum_rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        spectrum_rgbs.append(spectrum_rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        spectrum_num = int(spectrum_rgb.shape[2] / 3)

        if i == 0:
            print(rgb.shape, spectrum_rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            # print(p)
            t_psnr.append(p)

        if s_gt_imgs is not None and render_factor == 0:
            for num in range(spectrum_num):
                p = -10. * np.log10(np.mean(
                    np.square(spectrum_rgb[..., num * 3:(num + 1) * 3].cpu().numpy() - s_gt_imgs[i * 11 + num])))
                t_s_psnr.append(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            # rgb8 = np.clip(rgbs[-1], 0, 1)
            # rgb8 = rgbs[-1]  # 保存exr的时候不clip
            # filename = os.path.join(savedir, '{:03d}.exr'.format(i))
            imageio.imwrite(filename, rgb8)

            for num in range(spectrum_num):
                s_rgb8 = to8b(spectrum_rgbs[-1][..., num * 3:(num + 1) * 3])
                filename = os.path.join(savedir, '{:03d}_s{:03d}.png'.format(i, num))
                # s_rgb8 = spectrum_rgbs[-1][..., num * 3:(num + 1) * 3]  # 保存exr的时候不进行clip
                # s_rgb8 = np.clip(spectrum_rgbs[-1][..., num * 3:(num + 1) * 3], 0, 1)
                # filename = os.path.join(savedir, '{:03d}_s{:03d}.exr'.format(i, num))
                imageio.imwrite(filename, s_rgb8)

    # pdb.set_trace()
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    test_psnr = {}
    if gt_imgs is not None and render_factor == 0:
        avg_psnr = np.mean(t_psnr)
        test_psnr['avg_test_psnr'] = avg_psnr
        test_psnr['test_psnr'] = t_psnr
    if s_gt_imgs is not None and render_factor == 0:
        avg_s_psnr = np.mean(t_s_psnr)
        test_psnr['avg_test_s_psnr'] = avg_s_psnr
        test_psnr['test_s_psnr'] = t_s_psnr

    return rgbs, spectrum_rgbs, disps, test_psnr


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    # model = NeRF(D=args.netdepth, W=args.netwidth,
    #              input_ch=input_ch, output_ch=output_ch, skips=skips,
    #              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, spectrum_num=args.spectrum_num).cuda()
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                          spectrum_num=args.spectrum_num).cuda()

        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model_fine = torch.nn.DataParallel(model_fine, device_ids=range(torch.cuda.device_count()))

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    if args.test_checkpoint is not None:
        ckpts = [os.path.join(basedir, expname, args.test_checkpoint)]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'spectrum_num': args.spectrum_num,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, spectrum_num=11, raw_noise_std=0, white_bkgd=False, pytest=False, act_func="none"):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
                      -1)  # [N_rays, N_samples], torch.Size([32768, 64])

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    if act_func == 'none':
        ## 不要激活函数
        rgb = raw[..., :3]  # [N_rays, N_samples, 3] # torch.Size([32768, 64, 3])
        spectrum_rgbs = raw[..., 3:(spectrum_num + 1) * 3]

    else:
        rgb = eval("torch.nn.functional.{}(raw[...,:3])".format(
            act_func))  # [N_rays, N_samples, 3] # 2022-08-09 14:29:35  no sigmoid实验
        spectrum_rgbs = eval("torch.nn.functional.{}(raw[..., 3:(spectrum_num + 1) * 3])".format(act_func))

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # pdb.set_trace()

    alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    spectrum_rgb_maps = torch.sum(weights[..., None] * spectrum_rgbs, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
        spectrum_rgb_maps = spectrum_rgb_maps + (1. - acc_map[..., None])

    return rgb_map, spectrum_rgb_maps, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                spectrum_num,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False, act_func="none"):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)  # 从0-1之间均匀采64个点
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                        None]  # [N_rays, N_samples, 3], torch.Size([32768, 64, 3])

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, spectrum_rgb_maps, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, spectrum_num,
                                                                                    raw_noise_std, white_bkgd,
                                                                                    pytest=pytest, act_func=act_func)

    if N_importance > 0:
        rgb_map_0, spectrum_rgb_maps_0, disp_map_0, acc_map_0 = rgb_map, spectrum_rgb_maps, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, spectrum_rgb_maps, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d,
                                                                                        spectrum_num, raw_noise_std,
                                                                                        white_bkgd, pytest=pytest,
                                                                                        act_func=act_func)

    ret = {'rgb_map': rgb_map, 'spectrum_rgb_maps': spectrum_rgb_maps, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['spectrum_rgbs0'] = spectrum_rgb_maps_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        default="configs/20221004_spaceship_leaky_relu_1.0_weighted_spectral_loss_only.txt",
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--test_checkpoint", type=str, default=None,
                        help='load this checkpoint to test images')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_false',
                        help='render the test set instead of render_poses path')

    # debug mode
    # parser.add_argument("--render_only", action='store_false',
    #                     help='do not optimize, reload weights and render out render_poses path')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--act_func", type=str,
                        default="none", help='activation function for nerf output')
    parser.add_argument("--use_weighted_spectral_loss", action='store_true',
                        help='use weighted spectral loss')
    parser.add_argument("--no_rgb_loss", action='store_true', help='use fn output white light image rgb loss')

    # dataset options
    parser.add_argument("--image_format", type=str, default='.exr',
                        help='dataset image format')
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--spectrum_num", type=int, default=11,
                        help='number of specturm images')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--blender_near", type=float,
                        default=2., help='near distance')
    parser.add_argument("--blender_far", type=float,
                        default=6., help='far distance')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()


def train():
    parser = config_parser()
    args = parser.parse_args()
    args.expname = os.path.split(args.config)[-1]  # 直接用txt文件名作为实验文件夹命名

    # Load data
    K = None
    # pdb.set_trace()
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, s_images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res,
                                                                                args.testskip, args.spectrum_num,
                                                                                args.image_format)
        print('Loaded blender', images.shape, s_images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = args.blender_near
        far = args.blender_far

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            s_images = s_images[..., :3] * s_images[..., -1:] + (1. - s_images[..., -1:])
        else:
            images = images[..., :3]
            s_images = s_images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    sys.stdout = Logger(os.path.join(basedir, expname, 'log.txt'))
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
        "act_func": args.act_func
    }
    print("nerf output act_func: ", args.act_func)
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).cuda()

    # Short circuit if only rendering out from trained model
    if args.render_only:
        # pdb.set_trace()
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
                s_images = s_images[i_test[0] * args.spectrum_num:(i_test[-1] + 1) * args.spectrum_num]

            else:
                # Default is smoother render_poses path
                images = None
                s_images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            ## 加载训练好的Unet
            exp_path = r'experiments/exp30_mitsuba2_0927_3.0dataset/dataset#act_func/dataset_mitsuba_spaceship_mixed_native_texture_3.0_20220926/act_func_leaky_relu'
            params = Params(exp_path + os.sep + 'params.json')
            Unet, _, criternion = fetch_net_loss_criternion(params=params,
                                                            device='cuda' if torch.cuda.is_available() else 'cpu')

            checkpoint_path = os.path.join(exp_path, 'best_checkpoint.pth')
            checkpoint = torch.load(checkpoint_path,
                                    map_location='cpu')  # 读取checkpoint文件
            Unet.load_state_dict(checkpoint['net_state_dict'])
            Unet = Unet.cuda()
            Unet.eval()

            rgbs, spectrum_rgbs, _, test_psnr = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test,
                                                            gt_imgs=images, s_gt_imgs=s_images, savedir=testsavedir,
                                                            render_factor=args.render_factor,
                                                            render_only=args.render_only, Unet=Unet)
            test_metric = os.path.join(basedir, expname, 'test_metric.json')
            with open(test_metric, 'w') as file:
                json.dump(test_psnr, file)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    # pdb.set_trace()
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).cuda()
    poses = torch.Tensor(poses).cuda()
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).cuda()

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    best_spsnr = 0.
    best_s_L1_distance = 10000


    spectral_loss_weights = np.ones(args.spectrum_num)  # 光谱loss 权重
    if args.use_weighted_spectral_loss:
        spectral_loss_weights = [1., 3., 4., 3., 4., 4., 4., 2., 1., 1., 1.]  # 设置有内容的波段高权重
        print('\n', 'Use weighted spectral loss, weights:', spectral_loss_weights, '\n')

    if not args.no_rgb_loss:
        print('\n', 'Use snerf + fn output rgb loss\n')
    else:
        print('\n', 'Spectral loss only for training\n')

    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            s_target = s_images[img_i * 11:(img_i + 1) * 11]
            target = torch.Tensor(target).cuda()
            s_target = torch.Tensor(s_target).cuda()
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                s_target_s = torch.zeros([args.spectrum_num, N_rand, 3]).cuda()
                for num in range(args.spectrum_num):
                    s_target_s[num, ...] = s_target[num, ...][select_coords[:, 0], select_coords[:, 1]]

        #####  Core optimization loop  #####
        rgb, spectrums_rgbs, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                        verbose=i < 10, retraw=True,
                                                        **render_kwargs_train)

        # pdb.set_trace()
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)

        s_img_losses = 0
        # pdb.set_trace()
        s_psnrs = torch.zeros(args.spectrum_num).cuda()
        s_L1_distance = torch.zeros(args.spectrum_num).cuda()
        for num in range(args.spectrum_num):
            s_img_loss = img2mse(spectrums_rgbs[:, num * 3:(num + 1) * 3], s_target_s[num, ...])
            s_psnr = mse2psnr(s_img_loss)
            s_img_losses += spectral_loss_weights[num] * s_img_loss
            s_psnrs[num] = s_psnr
            s_L1_distance[num] = torch.nn.functional.l1_loss(spectrums_rgbs[:, num * 3:(num + 1) * 3],
                                                             s_target_s[num, ...]) * 255

        trans = extras['raw'][..., -1]
        psnr = mse2psnr(img_loss)
        L1_distance = torch.nn.functional.l1_loss(rgb, target_s) * 255

        s_psnr_avg = torch.mean(s_psnrs)
        s_L1_distance_avg = torch.mean(s_L1_distance)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            img_loss = img_loss + img_loss0

        s_img_loss0 = 0
        if 'spectrum_rgbs0' in extras:
            for num in range(args.spectrum_num):
                s_img_loss0 += spectral_loss_weights[num] * img2mse(extras['spectrum_rgbs0'][:, num * 3:(num + 1) * 3],
                                                                    s_target_s[num, ...])
            s_img_losses = s_img_losses + s_img_loss0
        loss = s_img_losses

        # 如果使用输出白光图像进行约束
        if not args.no_rgb_loss:
            loss += img_loss

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, spectrum_rgbs, disps, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).cuda(), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test],
                            s_gt_imgs=s_images[i_test[0] * args.spectrum_num:(i_test[-1] + 1) * args.spectrum_num],
                            savedir=testsavedir)
            print('Saved test set')

        # pdb.set_trace()
        # save the model with best spsnr
        # 2022-10-04 10:01:29  修正,spsnr对后续输出数据集更有价值,优先保存spsnr更好的模型
        if s_psnr_avg.item() > best_spsnr:
            best_spsnr = s_psnr_avg.item()

            # 储存当前最好的metric
            evaluate_metric = {
                "psnr": psnr.item(),
                "s_psnr": s_psnr_avg.item(),
                "L1_distance": L1_distance.item(),
                "s_L1_distance":s_L1_distance_avg.item()
            }


            path = os.path.join(basedir, expname, 'model_best.tar')

            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

            best_metric_file_path = os.path.join(basedir, expname, 'best_metric_psnr.json')
            with open(best_metric_file_path, 'w') as b_file:
                json.dump(evaluate_metric, b_file)

            print(f"Save best checkpoints with psnr {evaluate_metric['psnr']} / s_psnr {evaluate_metric['s_psnr']} at {path}")

        # save the model with best spectral avg l1 distance
        # 2022-10-04 10:01:29  修正,spectral avg l1 distance对后续输出数据集更有价值,优先保存spsnr更好的模型
        if s_L1_distance_avg.item() < best_s_L1_distance:
            best_s_L1_distance = s_L1_distance_avg.item()

            # 储存当前最好的metric
            evaluate_metric = {
                "psnr": psnr.item(),
                "s_psnr": s_psnr_avg.item(),
                "L1_distance": L1_distance.item(),
                "s_L1_distance":s_L1_distance_avg.item()
            }
            path = os.path.join(basedir, expname, 'model_best_l1_distance.tar')

            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

            best_metric_file_path = os.path.join(basedir, expname, 'best_metric_l1_distance.json')
            with open(best_metric_file_path, 'w') as b_file:
                json.dump(evaluate_metric, b_file)

            print(f"Save best checkpoints with psnr {evaluate_metric['psnr']} / s_psnr {evaluate_metric['s_psnr']} at {path}")

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i}  Loss: {loss.item()}  S_LOSS: {s_img_losses.item()} "
                f"L1_distance: {L1_distance.item()} s_L1_distance: {s_L1_distance_avg.item()} "
                f"PSNR: {psnr.item()}  S_PSNR: {s_psnr_avg.item()}")
        """ 
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
