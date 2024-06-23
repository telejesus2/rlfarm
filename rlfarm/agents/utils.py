import copy

import torch
import numpy as np
from scipy.spatial.transform import Rotation

REPLAY_BONUS = 1e-6
PRIORITIES = 'priorities'


def make_target_net(net):
    target_net = copy.deepcopy(net)
    for p in target_net.parameters():
        p.requires_grad = False
    return target_net

def grad_step(loss, opt, clip_params=None, clip_val=None, retain_graph=None):
    opt.zero_grad()
    # if clip_val is not None and clip_params is not None:
    #     for param in clip_params:
    #         param.register_hook(lambda grad: grad.clamp_(-clip_val, clip_val)) 
    loss.backward(retain_graph=retain_graph)
    clip_grad_norm(clip_params, clip_val)
    opt.step()

def clip_grad_norm(clip_params=None, clip_val=None):
    if clip_val is not None and clip_params is not None:
        torch.nn.utils.clip_grad_norm_(clip_params, clip_val)

def soft_update(net, target_net, tau):
    net_parameters_names = dict(net.named_parameters()).keys()
    with torch.no_grad():
        for (name, param), (_, target_param) in zip(
                net.state_dict().items(), target_net.state_dict().items()):
            if name in net_parameters_names:
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            else: # copy tensors such as batch_norm.running_average
                target_param.copy_(param)

    # for param, target_param in zip(net.parameters(), target_net.parameters()):
    #     target_param.data.copy_(
    #         tau * param.data + (1 - tau) * target_param.data
    #     )

def get_loss_weights(probs, beta=1.0):
    loss_weights = 1.0 / torch.sqrt(probs + REPLAY_BONUS) # TODO why sqrt?
    loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights

def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat)

def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc

def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()

def stack_on_channel(x):
    # expect (B, T, C, ...) where T typically corresponds to history of past states
    return x # TODO
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)