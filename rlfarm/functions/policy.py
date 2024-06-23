from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from rlfarm.networks.builder import make_network
from rlfarm.utils.logger import Summary
from rlfarm.envs.env import ActionSpace

AVAILABLE_POLICIES = [
    'gaussian',
    'squashed-gaussian',
    'soft-gaussian',
    'softmax',
    'deterministic'
]


class PolicyNetwork(nn.Module):
    def __init__(self, action_space: ActionSpace, state_shape: dict, action_dim: int, 
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
        self._action_space = action_space

        self.outputs = {}

    def build(self):
        self.encoder = make_network(self._encoder_class, self._encoder_kwargs)
        self.network = make_network(self._network_class, self._network_kwargs)

    def _postprocess(self, x):
        return self._action_space.normalize(x)

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


class SoftmaxPolicyNetwork(PolicyNetwork):
    def forward(self, x: dict, detach_encoder=False, deterministic=False, track_outputs=False):
        """
        :return: full distribution if training, deterministic action otherwise
        """
        out = super().forward(x, detach_encoder=detach_encoder, track_outputs=track_outputs)
        out = self._postprocess(out)
        probs = F.softmax(out, dim=-1)
        if deterministic:
            return torch.argmax(probs, dim=-1)
        else:
            return Categorical(probs)


class GaussianPolicyNetwork(PolicyNetwork):
    def __init__(self, action_space: ActionSpace, state_shape: dict, action_dim: int, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict):
        super().__init__(action_space, state_shape, action_dim,
                         encoder_class, encoder_kwargs, network_class, network_kwargs)
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self._log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, x: dict, detach_encoder=False, deterministic=False, track_outputs=False):
        """
        :return: full distribution if training, deterministic action otherwise
        """
        mu = super().forward(x, detach_encoder=detach_encoder, track_outputs=track_outputs)
        mu = self._postprocess(mu)
        if deterministic:
            return mu
        else:
            std = torch.exp(self._log_std)
            return Independent(Normal(mu, std), 1)  # if norm = Normal(mu, std) and normI = Independent(norm, 1)
                                                    # then norm.log_prob(ac).sum(axis=-1) = normI.log_prob(ac)
    def to(self, device):
        self._log_std = self._log_std.to(device)
        return super().to(device)


class SquashedPolicyNetwork(PolicyNetwork):
    def __init__(self, action_space: ActionSpace, state_shape: dict, action_dim: int, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict,
                 action_min_max: np.ndarray):
        super().__init__(action_space, state_shape, action_dim,
                         encoder_class, encoder_kwargs, network_class, network_kwargs)
        self._center = torch.tensor(action_min_max[1] + action_min_max[0]) / 2
        self._scale = torch.tensor(action_min_max[1] - action_min_max[0]) / 2

    def _squash(self, x):
        return torch.tanh(x) * self._scale + self._center

    def to(self, device):
        self._center = self._center.to(device)
        self._scale = self._scale.to(device)
        return super().to(device)


class SquashedGaussianPolicyNetwork(SquashedPolicyNetwork):
    def __init__(self, action_space: ActionSpace, state_shape: dict, action_dim: int, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict,
                 action_min_max: np.ndarray):
        super().__init__(action_space, state_shape, 2 * action_dim, 
                         encoder_class, encoder_kwargs, network_class, network_kwargs,
                         action_min_max)

    def forward(self, x: dict, detach_encoder=False, deterministic=False, track_outputs=False):
        """
        :return: full distribution if training, deterministic action otherwise
        """
        out = super().forward(x, detach_encoder=detach_encoder, track_outputs=track_outputs)
        ac_dim = out.shape[-1] // 2
        mu = self._squash(out[:, :ac_dim])
        mu = self._postprocess(mu)
        if deterministic:
            return mu
        else:
            log_std = out[:, ac_dim:] * self._scale
            std = torch.exp(log_std)
            return Independent(Normal(mu, std), 1)


class SoftGaussianPolicyNetwork(SquashedPolicyNetwork):
    def __init__(self, action_space: ActionSpace, state_shape: dict, action_dim: int, 
                 encoder_class: str, encoder_kwargs: dict, 
                 network_class: str, network_kwargs: dict,
                 action_min_max: np.ndarray):
        super().__init__(action_space, state_shape, 2 * action_dim, 
                         encoder_class, encoder_kwargs, network_class, network_kwargs,
                         action_min_max)

    def forward(self, x: dict, full_output=False, detach_encoder=False, deterministic=False, track_outputs=False):
        """
        :return: if deterministic, deterministic action (N, ac_dim)
                 else, sampled action (N, ac_dim) + (optionally) log prob (N, 1) and log std (N, 1)
        """
        mu, log_std = super().forward(
            x, detach_encoder=detach_encoder, track_outputs=track_outputs).chunk(2, dim=-1)

        log_std = torch.clamp(log_std, -20, 2)  # why this?
        # self._center = torch.tensor(self.log_std_max + self.log_std_min) / 2
        # self._scale = torch.tensor(self.log_std_max - self.log_std_min) / 2
        # log_std =  log_std * self._scale + self._center
        std = torch.exp(log_std)

        if deterministic:
            return self._postprocess(self._squash(mu))
        else:
            normal = Normal(mu, std)
            ac = normal.rsample()     # returns mu + eps * std with eps sampled from N(0,I)
            if not full_output:
                return self._postprocess(self._squash(ac))

            # compute logprob, and then apply correction for tanh squashing and scaling
            logprob = normal.log_prob(ac)
            # TODO should apply tanh to ac here ?
            logprob -= 2 * (np.log(2) - ac - F.softplus(- 2 * ac))      # more stable
            # logprob -= torch.log(1 - torch.tanh(ac).pow(2) + 1e-6)    # less stable
            # logprob /= self._scale                                    # omit this?
            logprob = logprob.sum(axis=-1)
            # TODO should apply squash (without tanh) to ac here ?
            return self._postprocess(self._squash(ac)), logprob.view(-1, 1), log_std


class DeterministicPolicyNetwork(SquashedPolicyNetwork):
    def forward(self, x: dict, detach_encoder=False, track_outputs=False):
        """
        :return: deterministic action
        """
        out = super().forward(x, detach_encoder=detach_encoder, track_outputs=track_outputs)
        return self._postprocess(self._squash(out))


def make_policy_network(class_: str, 
                        encoder_class: str, encoder_kwargs: dict,
                        network_class: str, network_kwargs: dict,
                        state_shape: Dict[str, Tuple], action_dim: int,
                        action_space: ActionSpace, action_min_max: np.ndarray = None):
    assert class_ in AVAILABLE_POLICIES

    if class_ == 'softmax':
        return SoftmaxPolicyNetwork(
            action_space, state_shape, action_dim,
            encoder_class, encoder_kwargs, network_class, network_kwargs)
    elif class_ == 'gaussian':
        return GaussianPolicyNetwork(
            action_space, state_shape, action_dim,
            encoder_class, encoder_kwargs, network_class, network_kwargs)
    elif class_ == 'squashed-gaussian':
        return SquashedGaussianPolicyNetwork(
            action_space, state_shape, action_dim,
            encoder_class, encoder_kwargs, network_class, network_kwargs, action_min_max)
    elif class_ == 'soft-gaussian':
        return SoftGaussianPolicyNetwork(
            action_space, state_shape, action_dim,
            encoder_class, encoder_kwargs, network_class, network_kwargs, action_min_max)
    elif class_ == 'deterministic':
        return DeterministicPolicyNetwork(
            action_space, state_shape, action_dim,
            encoder_class, encoder_kwargs, network_class, network_kwargs, action_min_max)