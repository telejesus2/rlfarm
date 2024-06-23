from typing import List, Dict, Tuple

import torch
import torch.nn as nn

from rlfarm.networks.builder import make_network
from rlfarm.utils.logger import Summary

AVAILABLE_Q_FUNCTIONS = [
    'discrete',
    'continuous',
    'continuous-double',
    'continuous-double-shared-encoder',
]


class QDiscreteNetwork(nn.Module):
    def __init__(self, state_shape: dict, action_dim: int, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict):
        super().__init__()
        self._rgb = 'rgb_state' in state_shape.keys()
        in_enc  = state_shape['rgb_state'] if self._rgb else state_shape['low_dim_state'][0]
        in_net = state_shape['low_dim_state'][0] if self._rgb else 0

        encoder_kwargs['input_dim'] = in_enc
        network_kwargs['input_dim'] = in_net + encoder_kwargs.get('output_dim', encoder_kwargs['input_dim'])
        network_kwargs['output_dim'] = action_dim

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


class QContinuousNetwork(nn.Module):
    def __init__(self, state_shape: dict, action_dim: int, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict):
        super().__init__()
        self._rgb = 'rgb_state' in state_shape.keys()
        in_enc  = state_shape['rgb_state'] if self._rgb else state_shape['low_dim_state'][0]
        in_net = state_shape['low_dim_state'][0] if self._rgb else 0

        encoder_kwargs['input_dim'] = in_enc
        network_kwargs['input_dim'] = in_net + action_dim + encoder_kwargs.get('output_dim', encoder_kwargs['input_dim'])
        network_kwargs['output_dim'] = 1

        self._encoder_class, self._encoder_kwargs = encoder_class, encoder_kwargs
        self._network_class, self._network_kwargs = network_class, network_kwargs

        self.outputs = {}

    def build(self):
        self.encoder = make_network(self._encoder_class, self._encoder_kwargs)
        self.network = make_network(self._network_class, self._network_kwargs)

    def forward(self, x: dict, u=None, detach_encoder=False, track_outputs=False):
        if 'encoded_state' in x:
            lat = x['encoded_state']
        else:
            x_enc  = x['rgb_state'] if self._rgb else x['low_dim_state']     
            lat = self.encoder(x_enc, detach=detach_encoder)

        if track_outputs:
            self.outputs['encoded_state'] = lat.cpu()

        x_net = x['low_dim_state'] if self._rgb else None 
        x = torch.cat((lat, x_net, u), 1) if self._rgb else torch.cat((lat, u), 1)

        return self.network(x)

    def log(self, prefix, log_encoder=True) -> List[Summary]:
        summaries = []
        if log_encoder: summaries += self.encoder.log(prefix + '/encoder')
        summaries += self.network.log(prefix + '/network')
        return summaries


# TODO should implement self.outputs and track_outputs (only necessary if some algorithm uses q during act())
class DoubleQContinuousNetwork(nn.Module):
    def __init__(self, q1: QContinuousNetwork, q2: QContinuousNetwork,
                 shared_encoder = False):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self._shared_encoder = shared_encoder

    def build(self):
        self.q1.build()
        self.q2.build()
        if self._shared_encoder:
            import gc
            assert len(gc.get_referrers(self.q2.encoder)) == 1
            self.q2.encoder = self.q1.encoder
            self.encoder = self.q1.encoder

    def forward(self, x: dict, u=None, q1=True, q2=True, detach_encoder=False):
        if q1 and not q2:
            return self.q1(x, u, detach_encoder=detach_encoder)
        elif q2 and not q1:
            return self.q2(x, u, detach_encoder=detach_encoder)
        elif q1 and q2:
            return self.q1(x, u, detach_encoder=detach_encoder), \
                   self.q2(x, u, detach_encoder=detach_encoder)

    def log(self, prefix) -> List[Summary]:
        summaries = []
        summaries += self.q1.log(prefix + '/q1')
        summaries += self.q2.log(prefix + '/q2', log_encoder = not self._shared_encoder)
        return summaries


def make_q_network(class_: str, 
                   encoder_class: str, encoder_kwargs: dict,
                   network_class: str, network_kwargs: dict,
                   state_shape: Dict[str, Tuple], action_dim: int):
    assert class_ in AVAILABLE_Q_FUNCTIONS

    if class_ == 'discrete':
        return QDiscreteNetwork(
            state_shape, action_dim, encoder_class, encoder_kwargs, network_class, network_kwargs)
    elif class_ == 'continuous':
        return QContinuousNetwork(
            state_shape, action_dim, encoder_class, encoder_kwargs, network_class, network_kwargs)
    elif class_ == 'continuous-double':
        q1 = QContinuousNetwork(
            state_shape, action_dim, encoder_class, encoder_kwargs, network_class, network_kwargs)
        q2 = QContinuousNetwork(
            state_shape, action_dim, encoder_class, encoder_kwargs, network_class, network_kwargs)
        return DoubleQContinuousNetwork(q1, q2, shared_encoder=False)
    elif class_ == 'continuous-double-shared-encoder':
        q1 = QContinuousNetwork(
            state_shape, action_dim, encoder_class, encoder_kwargs, network_class, network_kwargs)
        q2 = QContinuousNetwork(
            state_shape, action_dim, encoder_class, encoder_kwargs, network_class, network_kwargs)
        return DoubleQContinuousNetwork(q1, q2, shared_encoder=True)