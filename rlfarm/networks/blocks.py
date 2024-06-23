from typing import List

import torch
import torch.nn as nn

from rlfarm.networks.utils import conv_output_shape
from rlfarm.networks.utils import normalization, activation
from rlfarm.utils.logger import Summary, HistogramSummary, ImageSummary, ParamSummary


def log_block(block, prefix) -> List[Summary]:
    summaries = []

    for k, v in block.outputs.items():
        summaries.append(HistogramSummary(prefix + '/%s_hist' % k, v))
        if len(v.shape) > 2:
            summaries.append(ImageSummary(prefix + '/%s_img' % k, v[0]))

    for i in range(len(block.layers)):
        summaries.append(ParamSummary(
            prefix + '/' + block.tag + '%s' % (i + 1), block.layers[i]))

    return summaries


class Conv2DBlock(nn.Module):
    def __init__(self, input_shape, tag='conv',
        filters=[40,40], kernels=[3,3], strides=[2,2], paddings=[2,2], dilations=[1,1],
        norm=None, act='relu', drop_rate=0.0, 
    ):
        """Convolution layers followed by flatten operator

        :param input_shape: (h, w, c)
        """
        super().__init__()
        assert len(input_shape) == 3
        self.activation = act

        ### layers   
        num_layers = len(kernels)
        channels = [input_shape[2]] + filters 
        conv_h, conv_w = conv_output_shape(
            input_shape[:2], kernels, paddings, strides, dilations)
        self.layers, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(channels[i], channels[i+1],
                kernel_size=kernels[i], stride=strides[i], padding=paddings[i], dilation=dilations[i]))
            self.norms.append(normalization(norm, [channels[i+1], conv_h[i+1], conv_w[i+1]]))
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None # TODO should avoid model reuse
        self.act = activation(act)

        ### logging
        self.output_dim = channels[-1] * conv_h[-1] * conv_w[-1]
        self.tag = tag
        self.outputs = dict()

    def forward(self, x):
        """
        :param x: tensor of shape (N, h, w, c)
        """
        x = x.permute(0,3,1,2) # (N, c, h, w)
        self.outputs[self.tag + '_input'] = x

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if norm is not None: x = norm(x)
            if self.act is not None: x = self.act(x)
            if self.drop is not None: x = self.drop(x)
            self.outputs[self.tag + '%s' % (i + 1)] = x
            
        x = torch.flatten(x, 1)
        return x


class ConvTranspose2DBlock(nn.Module):
    def __init__(self, output_shape, tag='deconv',
        filters=[40,40], kernels=[3,3], strides=[2,2], paddings=[2,2], dilations=[1,1],
        norm=None, act='relu', drop_rate=0.0, 
    ):
        """Unflatten operator followed by ConvTranspose layers

        :param output_shape: (h, w, c)
        """
        super().__init__()
        self.activation = act

        ### layers   
        num_layers = len(kernels)
        channels = filters + [output_shape[2]]
        conv_h, conv_w = conv_output_shape(
            output_shape[:2], kernels, paddings, strides, dilations)
        conv_h, conv_w = conv_h[::-1], conv_w[::-1]
        self.layers, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ConvTranspose2d(channels[i], channels[i+1],
                kernel_size=kernels[i], stride=strides[i], padding=paddings[i], dilation=dilations[i]))
            self.norms.append(normalization(norm, [channels[i+1], conv_h[i+1], conv_w[i+1]]))
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.act = activation(act)

        ### logging
        self.input_dim = channels[0] * conv_h[0] * conv_w[0]
        self._init_c, self._init_h, self._init_w = channels[0], conv_h[0], conv_w[0]
        self._output_shape = [output_shape[2], output_shape[0], output_shape[1]]
        self.tag = tag
        self.outputs = dict()

    def forward(self, x):
        """
        :param x: tensor of shape (N, c)
        :return: tensor of shape (N, h, w, c)
        """
        batch_size = x.shape[0]
        x = x.view(-1, self._init_c, self._init_h, self._init_w) # (N, c, h, w)
        self.outputs[self.tag + '_input'] = x

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            if i == len(self.layers) - 1:
                x = layer(x, output_size = [batch_size] + self._output_shape)
            else:
                x = layer(x)
            if norm is not None: x = norm(x)
            if self.act is not None: x = self.act(x)
            if self.drop is not None: x = self.drop(x)
            self.outputs[self.tag + '%s' % (i + 1)] = x

        x = x.permute(0,2,3,1) # (N, h, w, c)    
        return x


# class ConvUpsample2DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes, strides,
#                  norm=None, activation=None):
#         super(Conv2DUpsampleBlock, self).__init__()
#         layer = [Conv2DBlock(
#             in_channels, out_channels, kernel_sizes, strides=1, norm, activation)]
#         if strides > 1:
#             layer.append(nn.Upsample(
#                 scale_factor=strides, mode='bilinear',
#                 align_corners=False))
#         convt_block = Conv2DBlock(
#             out_channels, out_channels, kernel_sizes, strides=1, norm, activation)
#         layer.append(convt_block)
#         self.conv_up = nn.Sequential(*layer)

#     def forward(self, x):
#         return self.conv_up(x)


class DenseBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, tag='fc',
        hidden_nodes=[64,64],
        norm=None, act='relu', drop_rate=0.0,
    ):
        """Fully connected layers
        """
        super().__init__()
        self.activation = act

        ### layers   
        nodes = [input_dim] + hidden_nodes
        if output_dim is not None: nodes += [output_dim]
        num_layers = len(nodes) - 1 
        self.layers, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(nodes[i], nodes[i+1]))   
            self.norms.append(normalization(norm, [nodes[i+1]]))
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.act = activation(act)

        ### logging
        self.output_dim = nodes[-1]
        self.tag = tag
        self.outputs = dict()

    def forward(self, x):
        """
        :param x: tensor of shape (N, c)
        """
        self.outputs[self.tag + '_input'] = x

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if norm is not None: x = norm(x)
            if self.act is not None: x = self.act(x)
            if self.drop is not None: x = self.drop(x)
            self.outputs[self.tag + '%s' % (i + 1)] = x

        return x


class IdBlock(nn.Module):
    def __init__(self, tag='id',
    ):
        super().__init__()
        self.activation = None

        ### logging
        self.tag = tag
        self.outputs = dict()
        self.layers = nn.ModuleList()

    def forward(self, x):
        return x