import os
from typing import List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from rlfarm.utils.logger import Summary, HistogramSummary
from rlfarm.networks.builder import Network
from rlfarm.vision.model import create_encoder, create_resnet_basic_block
from rlfarm.vision.model import NB_CAMERAS, STATE_DIM, NB_CLASS_SEGMENTATION


class Decoder(Network):
    def __init__(self, weight):
        super(Decoder, self).__init__()

        tasks = ["reach"]
        self.tasks = tasks
        self.latent_channels = 8
        self.latent_size = 128
        nb_images_input = 2
        aux_hidden_size = 128
        depth = True

        self.decoder_fc = nn.Sequential(
                nn.Linear(32, self.latent_size),
                nn.ReLU(inplace=True),
            )

        self.fc_regression = nn.ModuleDict()
        for task in tasks:
            self.fc_regression[task] = self.auxiliary_linear_regression(
                32*nb_images_input, aux_hidden_size, task)

        self.decoder_segmentation = self.auxiliary_decoder(NB_CLASS_SEGMENTATION)

        self.depth = depth
        if self.depth:
            self.decoder_depth = self.auxiliary_decoder(1)

        self.load_my_state_dict(torch.load(os.path.join(weight), map_location=torch.device("cpu")))

    def auxiliary_linear_regression(self, input_size, hidden_size, task):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, STATE_DIM[task][0])
        )

    def auxiliary_decoder(self, out_channels):
        return nn.Sequential(
            create_resnet_basic_block(8, 8, self.latent_channels, 16), 
            create_resnet_basic_block(32, 32, 16, 16), 
            create_resnet_basic_block(64, 64, 16, 16),
            create_resnet_basic_block(128, 128, 16, 8),
            nn.Conv2d(8, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        )

    def forward(self, x, detach=False, full_output=True):
        if detach:
            raise NotImplementedError()
        
        batch_size = x.shape[0] # N, 64
        out_seg, out_dep = None, None

        if full_output:
            encoding = torch.cat([x[:, im_idx*32:(im_idx+1)*32] for im_idx in range(NB_CAMERAS)], 0) # 2N*32

            encoding = self.decoder_fc(encoding) # 2N*128
            encoding = encoding.reshape(-1, self.latent_channels, 4, 4)  # 2N * latent_channels * 4 * 4

            out_seg = self.decoder_segmentation(encoding) # 2N * NB_CLASS_SEGMENTATION * 128 * 128
            out_dep = self.decoder_depth(encoding) if self.depth else None # 2N * 1 * 128 * 128

            # concatenate in the channel dimension rather than by batch
            out_seg = torch.cat([out_seg[im_idx*batch_size:(im_idx+1)*batch_size, ...] for im_idx in range(NB_CAMERAS)], 1)
            if self.depth:
                out_dep = torch.cat([out_dep[im_idx*batch_size:(im_idx+1)*batch_size, ...] for im_idx in range(NB_CAMERAS)], 1)

        # out_states = []
        for i, task in enumerate(self.tasks):
            out_state = self.fc_regression[task](x)#[task_splits[i]:task_splits[i+1]])
            # out_states.append(out_state)
        # out_state = torch.cat(out_states, 0)

        return out_seg, out_dep, out_state

    def copy_weights_from(self, source):
        raise NotImplementedError()

    def log(self, prefix) -> List[Summary]:
        summaries = []
        for name, param in self.named_parameters():
            summaries.append(HistogramSummary(prefix + '/%s_hist' % name, param))
        return summaries

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name in self.state_dict().keys():
            param = state_dict["_model." + name]
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class Encoder(Network):
    def __init__(self, weight):
        super(Encoder, self).__init__()
        self.encoder = create_encoder(8, "resnet", False)
        self.encoder_fc = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
            )
        # self._model_contrastive = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 5), 
        # )

        self.load_my_state_dict(torch.load(os.path.join(weight), map_location=torch.device("cpu")))

    def forward(self, x, detach=False, full=False):
        batch_size = x.shape[0]
        x = x.permute(0,3,1,2) # N*6*128*128
        x = torch.cat([x[:, im_idx*3:(im_idx+1)*3, :, :] for im_idx in range(NB_CAMERAS)], 0) # 2N*3*128*128
        x = self.encoder(x).reshape(-1, 128) # 2N*128
        x = self.encoder_fc(x) # 2N*32
        x = torch.cat([x[im_idx*batch_size:(im_idx+1)*batch_size, ...] for im_idx in range(NB_CAMERAS)], 1) # N*64
        if full:
            x = self._model_contrastive(x) # N*5
        if detach: 
            x = x.detach()     
        return x

    def copy_weights_from(self, source):
        import gc
        assert len(gc.get_referrers(self.encoder)) == 1
        self.encoder = source.encoder
        self.encoder_fc = source.encoder_fc

    def log(self, prefix) -> List[Summary]:
        summaries = []
        for name, param in self.named_parameters():
            summaries.append(HistogramSummary(prefix + '/%s_hist' % name, param))
        return summaries

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name in own_state.keys():

            new_name = name
            if not "_model_contrastive" in name:
                new_name = "_model." + name
            param = state_dict[new_name]

            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)