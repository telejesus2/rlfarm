from typing import List

import torch
import torch.nn as nn

from rlfarm.agents.sacae.sacae import SACAE
from rlfarm.functions.q_function import make_q_network
from rlfarm.functions.policy import make_policy_network
from rlfarm.networks.builder import make_network
from rlfarm.utils.logger import Summary


class CurlNetwork(nn.Module):
    def __init__(self, z_dim):
        super(CurlNetwork, self).__init__()
        self._z_dim = z_dim

    def build(self, encoder_net, encoder_target_net, device):
        self._encoder_net = encoder_net
        self._encoder_target_net = encoder_target_net 
        self.W = torch.rand(self._z_dim, self._z_dim, requires_grad=True, device=device)

    def parameters(self):
        return list(self._encoder_net.parameters()) + [self.W]

    def encode(self, x, detach=False, ema=False):
        """
        :param ema: exponential moving average
        """
        if ema:
            with torch.no_grad():
                z_out = self._encoder_target_net(x)
        else:
            z_out = self._encoder_net(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim, B)
        logits = torch.matmul(z_a, Wz)  # (B, B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class SimpleNetwork(nn.Module):
    def __init__(self, network_class: str, network_args: dict):
        super().__init__()
        self._network_class, self._network_args = network_class, network_args

    def build(self):
        self.network = make_network(self._network_class, self._network_args)

    def forward(self, x):
        return self.network(x)

    def log(self, prefix) -> List[Summary]:
        return self.network.log(prefix + '/network')


def make_agent(action_space, action_min_max, state_shape, action_dim, config):
    # critic
    q_config = config['agent']['critic']
    q_net = make_q_network(
        q_config['class'], 
        q_config['encoder']['class'], q_config['encoder']['kwargs'] or {}, 
        q_config['network']['class'], q_config['network']['kwargs'] or {},
        state_shape, action_dim,
    )

    # actor
    pi_config = config['agent']['actor']
    pi_net = make_policy_network(
        pi_config['class'],
        pi_config['encoder']['class'], pi_config['encoder']['kwargs'] or {}, 
        pi_config['network']['class'], pi_config['network']['kwargs'] or {},
        state_shape, action_dim,
        action_space, action_min_max,
    )

    z_dim = q_config['encoder']['kwargs']['output_dim']
    encoder_config = config['agent']['encoder']
    curl_config = config['agent']['curl']

    # decoder
    decoder_config = config['agent']['decoder']
    decoder_config['kwargs']['input_dim'] = z_dim
    decoder_config['kwargs']['output_dim'] = state_shape['rgb_state']
    decoder_net = SimpleNetwork(decoder_config['class'], decoder_config['kwargs'])

    return SACAE(
        action_space,
        action_min_max,
        pi_net,
        q_net,
        pi_config['optimizer']['class'],
        pi_config['optimizer']['kwargs'],
        q_config['optimizer']['class'],
        q_config['optimizer']['kwargs'],
        # new wrt parent
        decoder_net=decoder_net,
        encoder_opt_class=encoder_config['optimizer']['class'],
        encoder_opt_kwargs=encoder_config['optimizer']['kwargs'],
        decoder_opt_class=decoder_config['optimizer']['class'],
        decoder_opt_kwargs=decoder_config['optimizer']['kwargs'],
        curl_net=CurlNetwork(z_dim),
        cpc_opt_class=curl_config['optimizer']['class'],
        cpc_opt_kwargs=curl_config['optimizer']['kwargs'],
        **config['agent']['kwargs'],
    )