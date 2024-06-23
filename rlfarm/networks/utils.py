import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.02


def activation(name):
    fn = None
    if name == 'relu':
        fn = F.relu
    elif name == 'elu':
        fn = F.elu
    elif name == 'leaky_relu':
        fn = lambda x: F.leaky_relu(x, negative_slope=LRELU_SLOPE)
    elif name == 'tanh':
        fn = torch.tanh
    elif name == 'prelu':
        return F.prelu
    return fn


def init_weights(net: nn.Module, activation: str = None):
        # if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #     # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        #     assert m.weight.size(2) == m.weight.size(3)
        #     m.weight.data.fill_(0.0)
        #     m.bias.data.fill_(0.0)
        #     mid = m.weight.size(2) // 2
        #     gain = init.calculate_gain('relu')
        #     init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
    for m in net.modules():
        if isinstance(m, nn.Linear) or \
           isinstance(m, nn.Conv2d) or \
           isinstance(m, nn.ConvTranspose2d):
            if activation is None:
                nn.init.xavier_uniform_(m.weight,
                                        gain=nn.init.calculate_gain('linear'))
                nn.init.zeros_(m.bias)
            elif activation == 'tanh':
                nn.init.xavier_uniform_(m.weight,
                                        gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)
            elif activation == 'lrelu':
                nn.init.kaiming_uniform_(m.weight, a=LRELU_SLOPE,
                                         nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            elif activation == 'relu':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            else:
                raise ValueError()


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def padding_same(conv_input_shape, kernel_sizes, stride_sizes, dilation_sizes):
    """Finds padding_sizes such that 
    conv_output_shape(conv_input_shape, kernel_sizes, padding_sizes, stride_sizes, dilation_sizes)
    equals conv_input_shape
    
    :param conv_input_shape: (h, w)
    """
    if isinstance(kernel_sizes[0], int): kernel_sizes = [(x, x) for x in kernel_sizes]
    if isinstance(stride_sizes[0], int): stride_sizes = [(x, x) for x in stride_sizes]
    if isinstance(dilation_sizes[0], int): dilation_sizes = [(x, x) for x in dilation_sizes]

    h = conv_input_shape[0]
    w = conv_input_shape[1]
    padding_sizes = []
    for (k, s, d) in zip(kernel_sizes, stride_sizes, dilation_sizes):
        pad_h = int(np.ceil((s[0]*(h - 1) - h + d[0]*(k[0] - 1) + 1) / 2))
        pad_w = int(np.ceil((s[1]*(w - 1) - w + d[1]*(k[1] - 1) + 1) / 2))
        padding_sizes.append((pad_h, pad_w))
    return padding_sizes


def conv_output_shape(
    conv_input_shape, kernel_sizes, padding_sizes, stride_sizes, dilation_sizes):
    """
    :param conv_input_shape: (h, w)
    """
    if isinstance(kernel_sizes[0], int): kernel_sizes = [(x, x) for x in kernel_sizes]
    if isinstance(padding_sizes[0], int): padding_sizes = [(x, x) for x in padding_sizes]
    if isinstance(stride_sizes[0], int): stride_sizes = [(x, x) for x in stride_sizes]
    if isinstance(dilation_sizes[0], int): dilation_sizes = [(x, x) for x in dilation_sizes]

    h = conv_input_shape[0]
    w = conv_input_shape[1]
    conv_h = [h]
    conv_w = [w]
    for (k, p, s, d) in zip(kernel_sizes, padding_sizes, stride_sizes, dilation_sizes):
        h = int(np.floor((h + 2*p[0] - d[0]*(k[0] - 1) - 1) / s[0] + 1))
        w = int(np.floor((w + 2*p[1] - d[1]*(k[1] - 1) - 1) / s[1] + 1))
        conv_h.append(h)
        conv_w.append(w)
    return (conv_h, conv_w)


def conv_transpose_output_shape(
    conv_input_shape, kernel_sizes, padding_sizes, stride_sizes, dilation_sizes, output_padding_sizes):
    """
    :param conv_input_shape: (h, w)
    """
    if isinstance(kernel_sizes[0], int): kernel_sizes = [(x, x) for x in kernel_sizes]
    if isinstance(padding_sizes[0], int): padding_sizes = [(x, x) for x in padding_sizes]
    if isinstance(stride_sizes[0], int): stride_sizes = [(x, x) for x in stride_sizes]
    if isinstance(dilation_sizes[0], int): dilation_sizes = [(x, x) for x in dilation_sizes]
    if isinstance(output_padding_sizes[0], int): output_padding_sizes = [(x, x) for x in output_padding_sizes]

    h = conv_input_shape[0]
    w = conv_input_shape[1]
    conv_h = [h]
    conv_w = [w]
    for (k, p, s, d, o) in zip(kernel_sizes, padding_sizes, stride_sizes, dilation_sizes, output_padding_sizes):
        h = (h - 1)*s[0] - 2*p[0] + d[0]*(k[0] - 1) + o[0] + 1
        w = (w - 1)*s[1] - 2*p[1] + d[1]*(k[1] - 1) + o[1] + 1
        conv_h.append(h)
        conv_w.append(w)
    return (conv_h, conv_w)


def normalization(fn_name, input_shape):
    """
    :param input_shape: (C) or (C,L) or (C,H,W)
    """
    num_channels = input_shape[0]
    fn = None
    if fn_name == 'batch':
        if len(input_shape)==3:
            fn = nn.BatchNorm2d(num_channels, affine=True) # Input: (N,C,H,W)
        else:
            fn = nn.BatchNorm1d(num_channels, affine=True) # Input: (N,C) or (N,C,L)            
    elif fn_name == 'instance':
        if len(input_shape)==2:
            fn = nn.InstanceNorm1d(num_channels, affine=True) # Input: (N,C,L)
        elif len(input_shape)==3:
            fn = nn.InstanceNorm2d(num_channels, affine=True) # Input: (N,C,H,W) 
    elif fn_name == 'layer':
        from_axis = 1
        fn = nn.LayerNorm(input_shape[from_axis:], elementwise_affine=True) # Input: (N,∗)
    elif fn_name == 'group':
        num_groups = 1 # num_groups = 1: equivalent to LayerNorm along all axes: nn.LayerNorm(input_shape)
                       # num_groups = num_channels: equivalent to either InstanceNorm1d(num_channels) or InstanceNorm2d(num_channels)
        fn = nn.GroupNorm(num_groups, num_channels, affine=True) # Input: (N,C,*)
    return fn