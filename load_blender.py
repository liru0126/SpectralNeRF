import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import pdb

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    # pdb.set_trace()
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, spectrum_num=11, image_format='.png'):
    print('Using ', image_format, ' to train nerf')

    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_s_imgs = []
    all_poses = []
    counts = [0]
    # pdb.set_trace()
    for s in splits:
        meta = metas[s]
        imgs = []
        s_imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            # frame_file_path = frame['file_path'].split('mitsuba_space_ship_exr_rgba')[1]
            frame_file_path = frame['file_path']
            print(frame_file_path)
            for wavelength in np.linspace(380, 780, spectrum_num, dtype=np.int):
                # fname_s = basedir + os.sep + frame_file_path + '/{}.exr'.format(wavelength)
                fname_s = os.path.join(basedir, frame_file_path, str(wavelength) + image_format)
                fname_img_s = imageio.imread(fname_s)
                s_imgs.append(fname_img_s)
            poses.append(np.array(frame['transform_matrix']))
            fname = os.path.join(basedir, frame_file_path, 'full' + image_format)
            fname_img = imageio.imread(fname)
            imgs.append(fname_img)
        # pdb.set_trace()
        # imgs = np.array(imgs).astype(np.float32)  # keep all 4 channels (RGBA)
        # s_imgs = np.array(s_imgs).astype(np.float32)
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        s_imgs = (np.array(s_imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_s_imgs.append(s_imgs)
        all_poses.append(poses)

    # pdb.set_trace()

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    s_imgs = np.concatenate(all_s_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(1.0471976)
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 8.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4)) 
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        s_imgs_half_res = np.zeros((s_imgs.shape[0], H, W, 4))
        for i, s_img in enumerate(s_imgs):
            s_imgs_half_res[i] = cv2.resize(s_img, (W, H), interpolation=cv2.INTER_AREA)
        s_imgs = s_imgs_half_res

    return imgs, s_imgs, poses, render_poses, [H, W, focal], i_split
