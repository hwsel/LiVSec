import os.path
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_face_rgbd(args, idx, img, folder='livsec'):
    face_locator_start = time.perf_counter_ns()
    L = img.shape[1] // 2 + 1
    R = img.shape[1] - 1
    while L != R:
        mid = L + (R - L) // 2
        if np.sum(img[:, mid, :]) == 0:
            R = mid - 1
        else:
            L = mid + 1
    w = L - 270
    h = 140
    face_locator_end = time.perf_counter_ns()
    print('face locating binary search', face_locator_end-face_locator_start)

    half_h = int(img.shape[0]/2)
    rgb_face = img[h:h+args.img_size:, w:w+args.img_size, :] / 255.0
    d_face = img[h+half_h:h+args.img_size+half_h, w:w+args.img_size, :]

    rgb2hsv_start = time.perf_counter_ns()
    d_face_hsv = cv2.cvtColor(d_face, cv2.COLOR_RGB2HSV)
    d_face = d_face_hsv[:, :, 0] / 180.
    print('rgb2h', time.perf_counter_ns() - rgb2hsv_start)

    d_face_s = d_face_hsv[:, :, 1]
    d_face_v = d_face_hsv[:, :, 2]
    d_face[d_face == 0.0] = 1.0
    d_face = np.expand_dims(d_face, -1)

    # stack RGB and D
    rgbd_face = np.concatenate((rgb_face, d_face), axis=2)
    rgbd_face = np.float32(rgbd_face)
    rgbd_face = np.rot90(rgbd_face)

    # to normalized tensor
    a = time.perf_counter_ns()
    rgbd = (rgbd_face - 0.5) / 0.5  # normalize to range [-1, 1]
    rgbd = np.transpose(rgbd, (2, 0, 1))  # 4 x 256 x 256
    rgbd = torch.from_numpy(rgbd)
    torch.save(rgbd, os.path.join(args.data_path, folder, 'person0_pose'+str(idx)+'.pt'))
    print('torch save time', time.perf_counter_ns() - a)

    return h, w, d_face_s, d_face_v


def to_numpy(x):
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1).cuda()
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1).cuda()
    # Scale back to range [0, 1]
    x = (x * std) + mean
    x = x.squeeze(0).permute(1, 2, 0)
    return x.cpu().numpy()


def unnormalize(x):
    mu = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    mu = mu.to(x.device)
    std = std.to(x.device)
    x = (x * std) + mu
    return x


def display3(x_ref, x, x_adv, cosine_x, cosine_x_adv, idx):

    # x_ref = to_numpy(x_ref)
    # x = to_numpy(x)
    # x_adv = to_numpy(x_adv)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(10, 10)

    fontsize = 15

    axs[0, 0].imshow(x_ref[:, :, :3])
    axs[0, 0].set_title("Reference Input - RGB", fontsize=fontsize)
    axs[1, 0].imshow(x_ref[:, :, 3], cmap="RdYlBu")
    axs[1, 0].set_title("Reference Input - Depth", fontsize=fontsize)

    axs[0, 1].imshow(x[:, :, :3])
    axs[0, 1].set_title("User Input - RGB \n Similarity Score: " + str(cosine_x), fontsize=fontsize)
    axs[1, 1].imshow(x[:, :, 3], cmap="RdYlBu")
    axs[1, 1].set_title("User Input - Depth", fontsize=fontsize)

    axs[0, 2].imshow(x_adv[:, :, :3])
    axs[0, 2].set_title("Protected Face - RGB \n Similarity Score: " + str(cosine_x_adv), fontsize=fontsize)
    axs[1, 2].imshow(x_adv[:, :, 3], cmap="RdYlBu")
    axs[1, 2].set_title("Protected Face - Depth", fontsize=fontsize)

    for i in range(2):
        for j in range(3):
            axs[i, j].axis("off")

    # plt.show()
    plt.savefig('temp_img_save/result_'+str(idx) + '.png')


def display(x_ref, x, x_adv, cosine_x, cosine_x_adv, l2, lpips, idx, save=False):

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)

    axs[0, 0].imshow(x_ref[:, :, :3])
    axs[0, 0].set_title("X_ref - RGB")
    axs[1, 0].imshow(x_ref[:, :, 3], cmap="RdYlBu")
    axs[1, 0].set_title("X_ref - Depth")

    axs[0, 1].imshow(x[:, :, :3])
    axs[0, 1].set_title("X - RGB \n Cosine to X_ref: " + str(cosine_x), color="b")
    axs[1, 1].imshow(x[:, :, 3], cmap="RdYlBu")
    axs[1, 1].set_title("X - Depth")

    axs[0, 2].imshow(x_adv[:, :, :3])
    axs[0, 2].set_title(
        "X_adv - RGB \n Cosine to X_ref: " + str(cosine_x_adv) +
        "\nL2: " + str(l2)[:6] + '\nLPIPS: ' + str(lpips), color="b"
    )
    axs[1, 2].imshow(x_adv[:, :, 3], cmap="RdYlBu")
    axs[1, 2].set_title("X_adv - Depth")

    for i in range(2):
        for j in range(3):
            axs[i, j].axis("off")

    if save:
        plt.savefig('lpips_results/'+'dataset2_0_'+str(idx)+'.png')
        plt.close()
    else:
        plt.show()

## SSIM
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r"""Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(
        X, Y, data_range=data_range, win=win, size_average=False, K=K
    )
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    weights=None,
    K=(0.01, 0.03),
):

    r"""interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % (
        (win_size - 1) * (2 ** 4)
    )

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(
            X, Y, win=win, data_range=data_range, size_average=False, K=K
        )

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(
        mcs + [ssim_per_channel], dim=0
    )  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r"""class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channel, 1] + [1] * spatial_dims
        )
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r"""class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channel, 1] + [1] * spatial_dims
        )
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
