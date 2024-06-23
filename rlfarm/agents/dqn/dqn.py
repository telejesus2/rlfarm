import os
import logging
import copy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from rlfarm.agents.agent import Agent
from rlfarm.utils.transition import ActResult
from rlfarm.agents.utils import grad_step, make_target_net, soft_update, get_loss_weights
from rlfarm.functions.optimizer import make_optimizer
from rlfarm.utils.logger import Summary, ScalarSummary

REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.5


class DQN(Agent):
    def __init__(
            self,
            action_min_max,
            q_net: torch.nn.Module,
            q_opt_class: str,
            q_opt_kwargs: dict,

            exploration: TODO = None,
            double_q: bool = False,
            critic_tau: float = 0.005,
            critic_grad_clip: float = 20.0,
            gamma: float = 0.99,
            n_steps: int = 1,
            target_update_freq: int = 2,
    ):
        self._critic_tau = critic_tau
        self._critic_grad_clip = critic_grad_clip
        self._gamma = gamma
        self._n_steps = n_steps
        self._double_q = double_q
        self._target_update_freq = target_update_freq

        # networks
        self._q_net = q_net
        self._q_opt_class = q_opt_class
        self._q_opt_kwargs = q_opt_kwargs

        # exploration strategy
        self._exploration

    def build(self, training: bool, device: torch.device) -> None:
        self._device = device

        self._q_net = copy.deepcopy(self._q_net)
        self._q_net.build()
        self._q_net = self._q_net.to(device).train(training)

        if not training:
            for p in self._q_net.parameters():
                p.requires_grad = False
        else:
            self._q_target_net = make_target_net(self._q_net)
            self._q_target_net = self._q_target_net.to(device).train(False)
            soft_update(self._q_net, self._q_target_net, 1)

            self._q_opt = make_optimizer(self._q_opt_class, self._q_opt_kwargs,
                self._q_net.parameters())          

            logging.info('# DQN Critic Params: %d' % sum(
                p.numel() for p in self._q_net.parameters() if p.requires_grad))

    @torch.no_grad()
    def act(self, step: int, state: dict, deterministic=False, explore=False) -> ActResult:
        # if explore:
            # if self._frame < self.learning_starts or random.random() < self.exploration.value(self._frame):
            #     ac = self.env.action_space.sample()
        # else:
        q = self._q_net(state)
        action = torch.argmax(q, axis=1)
        return ActResult(action)

    def update(self, step: int, sample: dict) -> dict:
        states = sample['states']				# dict: each entry of shape (N, ob_dim)
        next_states = sample['next_states']		# dict: each entry of shape (N, ob_dim)
        actions = sample['actions'].long().view(-1, 1)	    # shape (N, 1)
        rewards = sample['rewards'].view(-1, 1)				# shape (N, 1)
        done_mask = sample['terminals'].view(-1, 1)  		# shape (N, 1)

        # update critic
        self._update_critic(sample, states, next_states, actions, rewards, done_mask)

        # update the target network
        if step % self._target_update_freq == 0:
            soft_update(self._q_net, self._q_target_net, self._critic_tau)

        # logging
        self._main_summaries = [
            ScalarSummary('train/batch_reward', rewards.mean()),
        ]

        if 'sampling_probabilities' in sample['info']:
            return {
                'priority': self._new_priority ** REPLAY_ALPHA
            }
        return {}

    def _update_critic(self, sample, states, next_states, actions, rewards, done_mask):
        # compute q values
        q_values = self._q_net(states).gather(1, actions)	# shape (N, 1)

        # compute targets
        with torch.no_grad():
            if not self._double_q:
                q_targets_next = self._q_target_net(next_states).max(1, keepdim=True)[0]
            else:
                q_tmp = self._q_net(next_states)
                best_actions = torch.argmax(q_tmp, axis=1, keepdim=True)
                q_targets_next = self._q_target_net(next_states).gather(1, best_actions)
            q_targets = rewards + (1 - done_mask) * (self._gamma ** self._n_steps) * q_targets_next

        # compute loss and update network
        loss_weights = get_loss_weights(sample, REPLAY_BETA)
        q_delta = F.smooth_l1_loss(q_values, q_targets, reduction='none').mean(1)
        loss = (q_delta * loss_weights).mean()
        grad_step(loss, self._q_opt,
                  list(self._q_net.parameters()), self._critic_grad_clip)

        # logging
        self._critic_summaries = [
            ScalarSummary('train_critic/loss', loss),
            ScalarSummary('train_critic/q_values_mean', q_values.mean().item()),
        ]

        # adjust replay priorities
        new_priority = torch.sqrt(q_delta / 2. + 1e-10) # TODO is this correct?
        self._new_priority = (new_priority / torch.max(new_priority)).detach()

    def update_summaries(self, log_scalar_only=True) -> List[Summary]:
        summaries = self._main_summaries \
                  + self._critic_summaries
        if not log_scalar_only:
            summaries += self._q_net.log('train_critic')
        return summaries
                  
    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._q_net.load_state_dict(torch.load(
            os.path.join(savedir, 'q.pt'), map_location=self._device))

    def save_weights(self, savedir: str):
        torch.save(self._q_net.state_dict(),
            os.path.join(savedir, 'q.pt'))