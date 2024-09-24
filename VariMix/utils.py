"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import cv2
from os.path import join as ospj
import json
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn.functional as nnf
from torch.optim import lr_scheduler
from PIL import Image


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x): # [-1, 1] to [0, 1]
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0) # (256, 256, 3)
    temp_img = (((temp_img+1) / 2) * 255).astype("uint8")
    saliency = cv2.saliency.StaticSaliencyFineGrained_create() # pip install opencv_contrib_python==4.2.0.32
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imwrite("SaliencySource.png", temp_img)  # 保存图片
    cv2.imwrite("SaliencyMap.png", saliencyMap)  # 保存图片

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2, saliencyMap


def saliency_mix(images, labels, rand_index, lam, filename): # x is data, y is the labels
    labels_a = labels
    labels_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2, saliencyMap = saliency_bbox(images[rand_index[0]], lam)
    
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    saliencyMap = torch.tensor(saliencyMap, device=0)
    x_concat = [images]
    x_concat = torch.cat(x_concat).to(images.device)
    save_image(x_concat, images.size()[0], filename)
    return images, labels_a, labels_b, lam


def variety_mix(args, images, labels, alpha, nets): # x is data, y is the labels
    lam = np.random.beta(alpha, alpha)
    ref_index = torch.randperm(images.size()[0])
    x_src = images
    x_ref = images[ref_index, :]
    x_ref_resized = nnf.interpolate(x_ref, size=(args.stargan_size, args.stargan_size), mode='bicubic', align_corners=False)
    labels_src = labels
    labels_ref = labels[ref_index]

    # get the difference map of the source image and the generated image
    with torch.no_grad():
        s_ref = nets.style_encoder(x_ref_resized)
        x_fake = nets.generator(x_src, s_ref)
    x_fake_diff = torch.abs(x_fake - x_src)

    # get the binarization of the difference map
    x_binary_diff = []
    for i in range(x_fake_diff.shape[0]):
        x_binary_temp = x_fake_diff[i, :]
        max_value = torch.max(x_binary_temp)
        min_value = torch.min(x_binary_temp)
        x_binary_temp = ((x_binary_temp - min_value) / (max_value - min_value))
        x_binary_temp = torch.mean(x_binary_temp, dim=0, keepdim=True)
        x_binary_temp[x_binary_temp >= lam] = 1.0
        x_binary_temp[x_binary_temp < lam] = 0.0
        x_binary_diff.append(x_binary_temp)
    x_binary_diff = torch.stack(x_binary_diff, dim=0) # [batchsize, 1, img_size, img_size]

    # fuse the source image and the reference image
    image_fusion_src = x_src * x_binary_diff + x_ref * (1.0 - x_binary_diff)
    # fuse the reference image and the source image
    image_fusion_ref = x_ref * x_binary_diff + x_src * (1.0 - x_binary_diff)

    # get the fusion weight by calculate the interesting area
    lam_list = []
    for i in range(x_binary_diff.shape[0]):
        cnt = torch.sum(x_binary_diff[i, :]).item()
        ratio = 1 - (cnt / (images.size()[-1] * images.size()[-2]))
        lam_list.append(ratio)
    lam = torch.Tensor(lam_list).to(x_src.device)

    return image_fusion_src, image_fusion_ref, labels_src, labels_ref, lam


def translate_and_fusion(nets, x_src, labels, filename): # downstream classification
    lam = np.random.beta(2.0, 2.0)
    N, C, H, W = x_src.size()
    ref_index = torch.randperm(x_src.size()[0])
    print("src_label: ", labels)
    print("ref_label: ", labels[ref_index])
    x_ref = x_src[ref_index, :]

    images = x_src.clone()
    saliency_mix(images, labels, ref_index, lam, 'SaliencyMix.png') # x is data, y is the labels

    # get the difference map of the source image and the generated image
    s_ref = nets.style_encoder(x_ref)
    x_fake = nets.generator(x_src, s_ref)
    x_fake_diff = torch.abs(x_fake - x_src)

    # get the colormap of the difference map
    x_color_diff = []
    # x_fake_diff_norm = ((x_fake_diff - min_value) / (max_value - min_value)) * 255  # [batchsize, 3, img_size, img_size]
    for i in range(x_fake_diff.shape[0]):
        x_color_temp = x_fake_diff.data[i]
        max_value = torch.max(x_color_temp)
        min_value = torch.min(x_color_temp)
        x_color_temp = ((x_color_temp - min_value) / (max_value - min_value)) * 255            # [img_size, img_size, 3]
        x_color_temp = x_color_temp.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        x_color_temp = cv2.applyColorMap(x_color_temp.astype(np.uint8), cv2.COLORMAP_JET)
        x_color_temp= cv2.cvtColor(x_color_temp, cv2.COLOR_BGR2RGB)
        x_color_temp = np.float32(x_color_temp) / 255  # [0, 1]
        x_color_temp = (x_color_temp - 0.5) / 0.5      # [-1, 1]
        x_color_temp = torch.tensor(x_color_temp.transpose(2, 0, 1)).to(x_src.device)     # [3, img_size, img_size]
        x_color_diff.append(x_color_temp)
    x_color_diff = torch.stack(x_color_diff, dim=0) # [batchsize, 3, img_size, img_size]
    x_color_diff = x_color_diff + x_src

    # get the binarization of the difference map
    x_binary_diff = []
    for i in range(x_fake_diff.shape[0]):
        x_binary_temp = x_fake_diff[i, :]
        max_value = torch.max(x_binary_temp)
        min_value = torch.min(x_binary_temp)
        x_binary_temp = ((x_binary_temp - min_value) / (max_value - min_value)) # [3, img_size, img_size]
        x_binary_temp = torch.mean(x_binary_temp, dim=0, keepdim=True)          # [1, img_size, img_size]
        x_binary_temp[x_binary_temp >= lam] = 1.0
        x_binary_temp[x_binary_temp < lam] = 0.0
        x_binary_diff.append(x_binary_temp)
    x_binary_diff = torch.stack(x_binary_diff, dim=0) # [batchsize, 1, img_size, img_size]

    heatmap = []
    for b in range(x_binary_diff.shape[0]):
        x_diff_temp = x_binary_diff.data[b].cpu().numpy().transpose(1, 2, 0)
        x_diff_temp = (x_diff_temp - 0.5) / 0.5  # [-1, 1]
        x_diff_temp = torch.tensor(x_diff_temp.transpose(2, 0, 1)).to(x_src.device)
        x_diff_temp = x_diff_temp.repeat(3, 1, 1)
        heatmap.append(x_diff_temp)  
    heatmap = torch.stack(heatmap, dim=0) # [batchsize, 1, img_size, img_size]

    # fuse the source image and the reference image
    image_fusion_src = x_src * x_binary_diff + x_ref * (1.0 - x_binary_diff)
    # fuse the reference image and the source image
    image_fusion_ref = x_ref * x_binary_diff + x_src * (1.0 - x_binary_diff)

    x_concat = [x_src, x_ref, x_fake, x_color_diff, heatmap, image_fusion_src, image_fusion_ref]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat

@torch.no_grad()
def translate_and_reconstruct(nets, x_src, x_ref, filename):
    N, C, H, W = x_src.size()

    s_ref = nets.style_encoder(x_ref)
    x_fake = nets.generator(x_src, s_ref)

    s_src = nets.style_encoder(x_src)
    x_rec = nets.generator(x_fake, s_src)

    # obtain the heat map
    x_fake_diff = torch.abs(x_fake - x_src)
    max_value = torch.max(x_fake_diff)
    min_value = torch.min(x_fake_diff)
    x_fake_diff = ((x_fake_diff - min_value) / (max_value - min_value)) * 255  # [batchsize, 3, img_size, img_size]
    heatmap = []
    for b in range(x_fake_diff.shape[0]):
        x_diff_temp = x_fake_diff.data[b].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # [img_size, img_size, 3]
        x_diff_temp = cv2.applyColorMap(x_diff_temp, cv2.COLORMAP_JET)
        x_diff_temp = cv2.cvtColor(x_diff_temp, cv2.COLOR_BGR2RGB)
        x_diff_temp = np.float32(x_diff_temp) / 255  # [0, 1]
        x_diff_temp = (x_diff_temp - 0.5) / 0.5      # [-1, 1]
        x_diff_temp = torch.tensor(x_diff_temp.transpose(2, 0, 1)).to(x_src.device)     # [3, img_size, img_size]
        heatmap.append(x_diff_temp) 
    heatmap = torch.stack(heatmap, dim=0)  # [batchsize, 3, img_size, img_size]
    x_fake_diff = heatmap + x_src

    x_concat = [x_src, x_ref, x_fake, x_fake_diff, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        s_many = nets.mapping_network(z_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@torch.no_grad()
def translate_using_reference(nets, x_src, x_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    s_ref = nets.style_encoder(x_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src = inputs.x_src
    x_ref = inputs.x_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, x_src, x_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, x_src, x_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        print(s_ref.shape)
        x_fake = nets.generator(x_src, s_ref)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        s_many = nets.mapping_network(z_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255


def loss_visualize(epochs, tra_loss, val_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch Loss")
    plt.plot(np.arange(1, epochs+1), tra_loss, label='train_loss', color='r', linestyle='-')
    plt.plot(np.arange(1, epochs+1), val_loss, label='val_loss', linestyle='-', color='b')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(result_path + '/loss.png')
    plt.show()


def acc_visualize(epochs, tra_acc, val_acc, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch Accuracy")
    plt.plot(np.arange(1, epochs+1), tra_acc, label='train_acc', color='r', linestyle='-')
    plt.plot(np.arange(1, epochs+1), val_acc, label='val_acc', linestyle='-', color='b')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(result_path + '/accuracy.png')
    plt.show()


def update_learning_rate(optimizer, scheduler): 
    """
    Update learning rates for all the networks
    """
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))
    

def get_scheduler(optimizer, args):  # Return a learning rate scheduler
    """
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.n_epochs) / float(args.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler