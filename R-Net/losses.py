# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import torch

    
def Dice(segment, label, return_dice=False):
    N = segment.shape[0]
    segment_flat = segment.view(N, -1)
    label_flat = label.view(N, -1)
    intersection = segment_flat * label_flat 
    dice = (2 * intersection.sum(1) + 0.001) / (segment_flat.sum(1) + label_flat.sum(1) + 0.001)
    if return_dice:
        return dice.mean()
    loss = 1 - dice.mean()
    return loss


def Gradient(feild):
    g_H = (feild[:, :, 1:, :, :]-feild[:, :, :-1, :, :]).abs()
    g_W = (feild[:, :, :, 1:, :]-feild[:, :, :, :-1, :]).abs()
    g_D = (feild[:, :, :, :, 1:]-feild[:, :, :, :, :-1]).abs()
    loss = g_H.mean() + g_W.mean() + g_D.mean()
    return loss


def Gradient_step(feild):
    g_H = (feild[:, :, 1:, :, :]-feild[:, :, :-1, :, :]).abs()
    g_W = (feild[:, :, :, 1:, :]-feild[:, :, :, :-1, :]).abs()
    g_D = (feild[:, :, :, :, 1:]-feild[:, :, :, :, :-1]).abs()
    g_H = 2 / (1 + torch.exp(-10*g_H)) - 1
    g_W = 2 / (1 + torch.exp(-10*g_W)) - 1
    g_D = 2 / (1 + torch.exp(-10*g_D)) - 1
    loss = g_H.mean() + g_W.mean() + g_D.mean()
    return loss


def Ncc(Ii, Ji):
    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to(Ii.device)

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -torch.mean(cc)


if __name__ == '__main__':
    pass

