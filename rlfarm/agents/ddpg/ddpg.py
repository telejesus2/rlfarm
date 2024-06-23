import os
import logging
import copy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from rlfarm.agents.agent import Agent
from rlfarm.envs.env import ActionSpace
from rlfarm.utils.transition import ActResult
from rlfarm.agents.utils import grad_step, make_target_net, soft_update, get_loss_weights
from rlfarm.agents.utils import REPLAY_BONUS, PRIORITIES
from rlfarm.functions.optimizer import make_optimizer
from rlfarm.utils.logger import Summary, ScalarSummary
from rlfarm.buffers.replay.const import *
from rlfarm.utils.scheduler import PiecewiseSchedule


class DDPG(Agent):
    def __init__(
            self,
            action_space: ActionSpace,
            action_min_max,
            policy_net: torch.nn.Module,
            q_net: torch.nn.Module,
            policy_opt_class: str,
            policy_opt_kwargs: dict,
            q_opt_class: str,
            q_opt_kwargs: dict,
            # optional
            critic_tau: float = 0.005,
            actor_tau: float = 0.005,
            critic_grad_clip: float = 20.0,
            actor_grad_clip: float = 20.0,
            gamma: float = 0.99,
            target_update_freq: int = 2,
            actor_update_freq: int = 2,
            shared_encoder: bool = False,
            normalize_priorities: bool = True,
            replay_alpha: float = 0.7,
            replay_beta: float = 0.5,
            init_weightsdir: str = None,
            lambda_bc: float =  0,
            lambda_nstep: float = 0,
            q_filter: bool = False,
            replay_demo_bonus: float = 0,
            # specific to sail
            lambda_sail: float = 0,
            clip_sail: float = 0,
    ):
        self._action_space = action_space
        self._critic_tau = critic_tau
        self._critic_grad_clip = critic_grad_clip
        self._actor_tau = actor_tau
        self._actor_grad_clip = actor_grad_clip
        self._gamma = gamma
        self._target_update_freq = target_update_freq
        self._actor_update_freq = actor_update_freq
        self._shared_encoder = shared_encoder
        self._normalize_priorities = normalize_priorities
        self._replay_alpha = replay_alpha
        self._replay_beta = replay_beta
        self._init_weightsdir = init_weightsdir

        # from rlfarm.agents.sacfd
        self._lambda_bc = lambda_bc
        self._q_filter = q_filter
        self._replay_demo_bonus = replay_demo_bonus
        self._lambda_nstep = lambda_nstep

        # exploration TODO
        self._exploration = PiecewiseSchedule(
            [(0, 1), (300000 * 0.1, 0.02)],
            outside_value=0.02)
        self._action_min_max = action_min_max

        # networks
        self._policy_net = policy_net
        self._q_net = q_net
        self._policy_opt_class = policy_opt_class
        self._policy_opt_kwargs = policy_opt_kwargs
        self._q_opt_class = q_opt_class
        self._q_opt_kwargs = q_opt_kwargs

        # self-advantage imitation learning
        self._lambda_sail = lambda_sail
        self._clip_sail = clip_sail

    def build(self, training: bool, device: torch.device) -> None:
        self._device = device

        self._policy_net = copy.deepcopy(self._policy_net)
        self._policy_net.build()
        self._policy_net = self._policy_net.to(device).train(training)

        if not training:
            for p in self._policy_net.parameters():
                p.requires_grad = False
        else:
            self._policy_target_net = make_target_net(self._policy_net)
            self._policy_target_net = self._policy_target_net.to(device).train(False)
            soft_update(self._policy_net, self._policy_target_net, 1)
            
            self._q_net = copy.deepcopy(self._q_net)
            self._q_net.build()
            self._q_target_net = make_target_net(self._q_net)
            self._q_net = self._q_net.to(device).train(training)
            self._q_target_net = self._q_target_net.to(device).train(False)
            soft_update(self._q_net, self._q_target_net, 1)

            if self._shared_encoder:
                self._policy_net.encoder.copy_weights_from(self._q_net.encoder)

            self._q_opt = make_optimizer(self._q_opt_class, self._q_opt_kwargs,
                self._q_net.parameters())
            self._policy_opt = make_optimizer(self._policy_opt_class, self._policy_opt_kwargs,
                self._policy_net.parameters())            

            logging.info('# DDPG Critic Params: %d' % sum(
                p.numel() for p in self._q_net.parameters() if p.requires_grad))
            logging.info('# DDPG Actor Params: %d' % sum(
                p.numel() for p in self._policy_net.parameters() if p.requires_grad))

        if self._init_weightsdir is not None:
            logging.info('Loading initial weights.')
            self.load_weights(self._init_weightsdir, training=training)

    def encoder(self):
        return self._policy_net.encoder

    @torch.no_grad()
    def act(self, step: int, state: dict, deterministic=False, explore=False, track_outputs=False) -> ActResult:
        if explore:
            action = self._action_space.sample()
        else:
            action = self._policy_net(state)
            if not deterministic: # TODO depending of the action space we might not want to clip the action, maybe it should be handled by self._action_space.apply_noise() ?
                action = action + torch.FloatTensor(self._action_space.action_size).normal_(
                                    mean=0, std=self._exploration.value(step)).to(self._device)
                # action = torch.clamp(action, 
                #             min=torch.tensor(self._action_min_max[0]), max=torch.tensor(self._action_min_max[1]))
                action = torch.max(torch.min(action, torch.tensor(self._action_min_max[1]).to(self._device)),
                                                     torch.tensor(self._action_min_max[0]).to(self._device))
        return ActResult(action)

    def update(self, step: int, sample: dict, warmup: bool = False) -> dict:
        states = sample[STATE]                   # dict: each entry of shape (N, ob_dim)
        next_states = {k: v[:,0] for k,v in sample[NEXT_STATE].items()}      # dict: each entry of shape (N, ob_dim)
        actions = sample[ACTION]                 # shape (N, ac_dim)
        rewards = sample[REWARD][:,0].view(-1, 1)     # shape (N, 1)
        done_mask = sample[TERMINAL][:,0].view(-1, 1) # shape (N, 1)
        n_steps = sample[N_STEPS][:,0].view(-1, 1)    # shape (N, 1)
        demo_mask = sample[DEMO].view(-1, 1)     # shape (N, 1)
        self._prioritized = SAMPLING_PROBABILITIES in sample
        loss_weights = get_loss_weights(sample[SAMPLING_PROBABILITIES], self._replay_beta
            ).to(self._device) if self._prioritized else 1.0 # shape (N) or scalar

        if self._lambda_nstep > 0 and self._lambda_sail > 0:
            assert sample[N_STEPS].shape[1] == 3
        elif self._lambda_nstep > 0 or self._lambda_sail > 0:
            assert sample[N_STEPS].shape[1] == 2
        else:
            assert sample[N_STEPS].shape[1] == 1

        # double lookahead
        next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn = None, None, None, None
        if self._lambda_nstep > 0:
            assert torch.all(torch.ge(sample[N_STEPS][:,1], sample[N_STEPS][:,0]))
            next_states_tpn = {k: v[:,1] for k,v in sample[NEXT_STATE].items()}         # dict: each entry of shape (N, ob_dim)
            rewards_tpn = sample[REWARD][:,1].view(-1, 1)     # shape (N, 1)
            done_mask_tpn = sample[TERMINAL][:,1].view(-1, 1) # shape (N, 1)
            n_steps_tpn = sample[N_STEPS][:,1].view(-1, 1)    # shape (N, 1)

        # compute returns for sail
        returns = None
        if self._lambda_sail > 0:
            assert torch.all(torch.ge(sample[N_STEPS][:,-1], sample[N_STEPS][:,-2]))
            returns = sample[REWARD][:,-1].view(-1, 1)     # shape (N, 1)

        # update critic
        self._update_critic(states, next_states, actions, rewards, done_mask, n_steps, loss_weights,
            next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn,
            returns)

        # update actor and temperature
        if step % self._actor_update_freq == 0:
            self._update_actor(states, actions, demo_mask, loss_weights)

        # update the target networks
        if step % self._target_update_freq == 0:
            soft_update(self._q_net, self._q_target_net, self._critic_tau)
            soft_update(self._policy_net, self._policy_target_net, self._actor_tau)

        # logging
        self._main_summaries = [
            ScalarSummary('train/batch_reward', rewards.mean()),
        ]

        if self._prioritized:
            self._new_priorities += REPLAY_BONUS + (demo_mask.view(-1) * self._replay_demo_bonus)
            if self._normalize_priorities: self._new_priorities /= torch.max(self._new_priorities)
            return {
                PRIORITIES: self._new_priorities ** self._replay_alpha
            }
        return {}

    def _compute_critic_targets(
            self, states, actions, next_states, rewards, done_mask, n_steps, returns,
    ):
        with torch.no_grad():
            # SAIL
            rewards_sail = torch.zeros_like(rewards)
            if self._lambda_sail > 0:
                q_tmp = self._q_target_net(states, actions) # shape (N, 1)
                rewards_sail = torch.max(returns, q_tmp) - self._q_target_net(states, self._policy_target_net(states)) # shape (N, 1)
                if self._clip_sail > 0:
                    rewards_sail = torch.clamp(rewards_sail, min=-self._clip_sail, max=self._clip_sail)

            # compute next actions
            next_actions = self._policy_target_net(next_states)

            # compute q targets
            q_targets_next = self._q_target_net(next_states, next_actions)
            q_targets = rewards + self._lambda_sail * rewards_sail + (1 - done_mask) * (
                self._gamma ** n_steps) * q_targets_next
        return q_targets

    def _compute_critic_loss(self, q_values, q_targets, loss_weights):
        q_delta = F.mse_loss(q_values, q_targets, reduction='none') # shape (N, 1)
        q_loss = (q_delta.mean(1) * loss_weights).mean()
        return q_loss, q_delta

    def _update_critic(self, states, next_states, actions, rewards, done_mask, n_steps, loss_weights,
                       next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn,
                       returns):
        # compute targets
        q_targets = self._compute_critic_targets(
            states, actions, next_states, rewards, done_mask, n_steps, returns)
        if self._lambda_nstep > 0:
            q_targets_tpn = self._compute_critic_targets(
                states, actions, next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn, returns)

        # compute q values
        q_values = self._q_net(states, actions) # shape (N, 1)

        # update critic
        critic_loss, q_delta = self._compute_critic_loss(q_values, q_targets, loss_weights)
        if self._lambda_nstep > 0:
            critic_loss_tpn, _ = self._compute_critic_loss(
                q_values, q_targets_tpn, loss_weights)
            critic_loss += self._lambda_nstep * critic_loss_tpn
        grad_step(critic_loss, self._q_opt,
                  list(self._q_net.parameters()), self._critic_grad_clip)

        # logging
        self._critic_summaries = [
            ScalarSummary('train_critic/loss', critic_loss),
            ScalarSummary('train_critic/q_values_mean', q_values.mean().item()),
        ]

        # adjust replay priorities
        if self._prioritized:
            new_priorities = q_delta.mean(1)
            self._new_priorities = new_priorities.detach()

    def _update_actor(self, states, actions, demo_mask, loss_weights, detach_encoder=False):
        # temporally freeze q-networks 
        for p in self._q_net.parameters():
            p.requires_grad = False

        # compute ddpg loss
        greedy_actions = self._policy_net(states, detach_encoder=detach_encoder)
        q_values = self._q_net(states, greedy_actions) # shape (N, 1)
        ddpg_loss_ew = - q_values.view(-1) # shape (N)
        ddpg_loss = (ddpg_loss_ew * loss_weights).mean()
        actor_loss = ddpg_loss

        # add bc loss    
        if self._lambda_bc > 0:
            mask = demo_mask

            if self._q_filter:
                q1_values_demo, q2_values_demo = self._q_net(states, actions) # shape (N, 1)
                q_values_demo = torch.min(q1_values_demo, q2_values_demo)
                q_mask = q_values_demo > q_values # shape (N, 1)
                mask = q_mask * demo_mask

            bc_loss_ew = torch.square(
                mask * (greedy_actions - actions)
            ).sum(1) # shape (N)
            bc_loss = (bc_loss_ew * loss_weights).sum()

            actor_loss += self._lambda_bc * bc_loss

        # update actor
        grad_step(actor_loss, self._policy_opt, 
                  list(self._policy_net.parameters()), self._actor_grad_clip)

        # unfreeze q-networks
        for p in self._q_net.parameters():
            p.requires_grad = True

        # logging
        self._actor_summaries = [
            ScalarSummary('train_actor/loss', actor_loss),
            ScalarSummary('train_actor/actions', greedy_actions.mean()),
        ]

    def update_summaries(self, log_scalar_only=True) -> List[Summary]:
        summaries = self._main_summaries \
                  + self._critic_summaries \
                  + self._actor_summaries
        if not log_scalar_only:
            summaries += self._policy_net.log('train_actor') \
                       + self._q_net.log('train_critic')
        return summaries
                  
    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str, training: bool = False):
        self._policy_net.load_state_dict(torch.load(
            os.path.join(savedir, 'pi.pt'), map_location=self._device))
        if training:
            self._q_net.load_state_dict(torch.load(
                os.path.join(savedir, 'q.pt'), map_location=self._device))

    def save_weights(self, savedir: str):
        torch.save(self._policy_net.state_dict(),
            os.path.join(savedir, 'pi.pt'))
        torch.save(self._q_net.state_dict(),
            os.path.join(savedir, 'q.pt'))