from typing import List, Dict, Tuple

import torch
import torch.nn as nn

from rlfarm.networks.builder import make_network
from rlfarm.utils.logger import Summary


class ValueNetwork(nn.Module):
    def __init__(self, state_shape: dict, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict):
        super().__init__()
        self._rgb = 'rgb_state' in state_shape.keys()
        in_enc  = state_shape['rgb_state'] if self._rgb else state_shape['low_dim_state'][0]
        in_net = state_shape['low_dim_state'][0] if self._rgb else 0

        encoder_kwargs['input_dim'] = in_enc
        network_kwargs['input_dim'] = in_net + encoder_kwargs.get('output_dim', encoder_kwargs['input_dim'])
        network_kwargs['output_dim'] = 1

        self._encoder_class, self._encoder_kwargs = encoder_class, encoder_kwargs
        self._network_class, self._network_kwargs = network_class, network_kwargs

        self.outputs = {}

    def build(self):
        self.encoder = make_network(self._encoder_class, self._encoder_kwargs)
        self.network = make_network(self._network_class, self._network_kwargs)

    def forward(self, x: dict, detach_encoder=False, track_outputs=False):
        if 'encoded_state' in x:
            lat = x['encoded_state']
        else:
            x_enc  = x['rgb_state'] if self._rgb else x['low_dim_state']     
            lat = self.encoder(x_enc, detach=detach_encoder)

        if track_outputs:
            self.outputs['encoded_state'] = lat.cpu()

        x_net = x['low_dim_state'] if self._rgb else None 
        x = torch.cat((lat, x_net), 1) if self._rgb else lat

        return self.network(x)

    def log(self, prefix) -> List[Summary]:
        summaries = []
        summaries += self.encoder.log(prefix + '/encoder')
        summaries += self.network.log(prefix + '/network')
        return summaries


def make_value_network(encoder_class: str, encoder_kwargs: dict,
                       network_class: str, network_kwargs: dict,
                       state_shape: Dict[str, Tuple]):
    return ValueNetwork(state_shape, encoder_class, encoder_kwargs, network_class, network_kwargs)