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


class SAC(Agent):
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
            critic_grad_clip: float = 20.0,
            actor_grad_clip: float = 20.0,
            gamma: float = 0.99,
            alpha: float = 0.2,
            alpha_auto_tune: bool = True,
            alpha_lr: float = 0.0001,
            target_entropy: float =-2.,
            target_update_freq: int = 2,
            actor_update_freq: int = 2,
            shared_encoder: bool = False,
            action_prior: str = "uniform",
            normalize_priorities: bool = True,
            replay_alpha: float = 0.7,
            replay_beta: float = 0.5,
            init_weightsdir: str = None,
    ):
        self._action_space = action_space
        self._critic_tau = critic_tau
        self._critic_grad_clip = critic_grad_clip
        self._actor_grad_clip = actor_grad_clip
        self._gamma = gamma
        self._target_update_freq = target_update_freq
        self._actor_update_freq = actor_update_freq
        self._shared_encoder = shared_encoder
        self._action_prior = action_prior
        self._normalize_priorities = normalize_priorities
        self._replay_alpha = replay_alpha
        self._replay_beta = replay_beta
        self._init_weightsdir = init_weightsdir

        # networks
        self._policy_net = policy_net
        self._q_net = q_net
        self._policy_opt_class = policy_opt_class
        self._policy_opt_kwargs = policy_opt_kwargs
        self._q_opt_class = q_opt_class
        self._q_opt_kwargs = q_opt_kwargs

        # alpha (entropy regularization)
        self._alpha = alpha
        self._alpha_auto_tune = alpha_auto_tune
        self._alpha_lr = alpha_lr
        self._target_entropy = target_entropy
        if target_entropy is None:
            self._target_entropy = - np.prod(len(action_min_max[0])).item() # heuristic from paper

    def build(self, training: bool, device: torch.device) -> None:
        self._device = device

        self._policy_net = copy.deepcopy(self._policy_net)
        self._policy_net.build()
        self._policy_net = self._policy_net.to(device).train(training)

        if not training:
            for p in self._policy_net.parameters():
                p.requires_grad = False
        else:
            self._q_net = copy.deepcopy(self._q_net)
            self._q_net.build()
            self._q_target_net = make_target_net(self._q_net)
            self._q_net = self._q_net.to(device).train(training)
            self._q_target_net = self._q_target_net.to(device).train(False)
            soft_update(self._q_net, self._q_target_net, 1)

            if self._shared_encoder:  # assumes q shares encoder
                self._policy_net.encoder.copy_weights_from(self._q_net.encoder)

            self._q_opt = make_optimizer(self._q_opt_class, self._q_opt_kwargs,
                self._q_net.parameters())
            self._policy_opt = make_optimizer(self._policy_opt_class, self._policy_opt_kwargs,
                self._policy_net.parameters())            

            self._log_alpha = 0
            if self._alpha_auto_tune:
                self._log_alpha = torch.tensor(
                    (np.log(self._alpha)), dtype=torch.float,
                    requires_grad=True, device=device)
                self._alpha_opt = torch.optim.Adam(
                    [self._log_alpha], lr=self._alpha_lr)
            else:
                self._alpha = torch.tensor(
                    self._alpha, dtype=torch.float,
                    requires_grad=False, device=device)

            logging.info('# SAC Critic Params: %d' % sum(
                p.numel() for p in self._q_net.parameters() if p.requires_grad))
            logging.info('# SAC Actor Params: %d' % sum(
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
            action = self._policy_net(state, deterministic=deterministic, track_outputs=track_outputs)
        result = ActResult(action)
        
        if track_outputs and not explore:
            result.state.update(self._policy_net.outputs)

        return result

    @property
    def alpha(self):
        return self._log_alpha.exp() if self._alpha_auto_tune else self._alpha

    def update(self, step: int, sample: dict, warmup: bool = False) -> dict:
        states = sample[STATE]                   # dict: each entry of shape (N, ob_dim)
        next_states = {k: v[:,0] for k,v in sample[NEXT_STATE].items()}         # dict: each entry of shape (N, ob_dim)
        actions = sample[ACTION]                 # shape (N, ac_dim)
        rewards = sample[REWARD][:,0].view(-1, 1)     # shape (N, 1)
        done_mask = sample[TERMINAL][:,0].view(-1, 1) # shape (N, 1)
        n_steps = sample[N_STEPS][:,0].view(-1, 1)    # shape (N, 1)
        self._prioritized = SAMPLING_PROBABILITIES in sample
        loss_weights = get_loss_weights(sample[SAMPLING_PROBABILITIES], self._replay_beta
            ).to(self._device) if self._prioritized else 1.0 # shape (N) or scalar

        assert sample[N_STEPS].shape[1] == 1

        # update critic
        self._update_critic(states, next_states, actions, rewards, done_mask, n_steps, loss_weights)

        # update actor and temperature
        if step % self._actor_update_freq == 0:
            self._update_actor_and_temperature(states, loss_weights)

        # update the target networks
        if step % self._target_update_freq == 0:
            soft_update(self._q_net, self._q_target_net, self._critic_tau)

        # logging
        self._main_summaries = [
            ScalarSummary('train/batch_reward', rewards.mean()),
        ]

        if self._prioritized:
            self._new_priorities += REPLAY_BONUS
            if self._normalize_priorities: self._new_priorities /= torch.max(self._new_priorities)
            return {
                PRIORITIES: self._new_priorities ** self._replay_alpha
            }
        return {}

    def _compute_critic_targets(self, next_states, rewards, done_mask, n_steps):
        with torch.no_grad():
            # compute next actions
            next_actions, logprobs, _ = self._policy_net(next_states, full_output=True)

            # compute q targets
            q1_targets_next, q2_targets_next = self._q_target_net(next_states, next_actions)
            q_targets_next = torch.min(q1_targets_next, q2_targets_next)
            q_targets = rewards + (1 - done_mask) * (self._gamma ** n_steps) * (
                        q_targets_next - self.alpha.detach() * logprobs)
        return q_targets

    def _compute_critic_loss(self, q1_values, q2_values, q_targets):
        # q1_delta = F.smooth_l1_loss(q1_values, q_targets, reduction='none')
        # q2_delta = F.smooth_l1_loss(q2_values, q_targets, reduction='none')
        q1_delta = F.mse_loss(q1_values, q_targets, reduction='none') # shape (N, 1)
        q2_delta = F.mse_loss(q2_values, q_targets, reduction='none')
        q1_loss, q2_loss = q1_delta.mean(1), q2_delta.mean(1) # shape (N)
        return q1_loss, q2_loss

    def _update_critic(self, states, next_states, actions, rewards, done_mask, n_steps, loss_weights):
        # compute targets
        q_targets = self._compute_critic_targets(next_states, rewards, done_mask, n_steps)

        # compute q values
        q1_values, q2_values = self._q_net(states, actions) # shape (N, 1)

        # update critic
        q1_loss_ew, q2_loss_ew = self._compute_critic_loss(q1_values, q2_values, q_targets)
        q1_loss, q2_loss = (q1_loss_ew * loss_weights).mean(), (q2_loss_ew * loss_weights).mean()       
        critic_loss = q1_loss + q2_loss
        grad_step(critic_loss, self._q_opt,
                  list(self._q_net.parameters()), self._critic_grad_clip)

        # logging
        self._critic_summaries = [
            ScalarSummary('train_critic/loss', critic_loss),
            ScalarSummary('train_critic/q1_loss', q1_loss),
            ScalarSummary('train_critic/q2_loss', q2_loss),
            ScalarSummary('train_critic/q1_values_mean', q1_values.mean().item()),
            ScalarSummary('train_critic/q2_values_mean', q2_values.mean().item()),
        ]

        # adjust replay priorities
        if self._prioritized:
            # new_priorities = abs((q1_loss_ew + q2_loss_ew) / 2.)
            new_priorities = torch.sqrt((q1_loss_ew + q2_loss_ew) / 2.)
            self._new_priorities = new_priorities.detach()

    def _compute_actor_loss(self, states, actions, logprobs):
        q1_values, q2_values = self._q_net(states, actions)
        q_values = torch.min(q1_values, q2_values)
        actor_loss = - (q_values - self.alpha.detach() * logprobs).view(-1) # shape (N)

        # add prior term
        if self._action_prior == "normal":
            policy_prior = MultivariateNormal(
                loc=torch.zeros(self._action_space.action_size),
                scale_tril=torch.diag(torch.ones(self._action_space.action_size)))
            prior_logprobs = policy_prior.log_prob(actions) # shape (N)
        elif self._action_prior == "uniform":
            prior_logprobs = 0.0
        actor_loss -= prior_logprobs

        return actor_loss, q_values

    def _update_actor_and_temperature(self, states, loss_weights, detach_encoder=False):
        # temporally freeze q-networks 
        for p in self._q_net.parameters():
            p.requires_grad = False

        # compute loss
        greedy_actions, logprobs, logstd = self._policy_net(states, full_output=True,
            detach_encoder=detach_encoder)
        actor_loss_ew, _ = self._compute_actor_loss(states, greedy_actions, logprobs)

        # update actor
        actor_loss = (actor_loss_ew * loss_weights).mean()
        grad_step(actor_loss, self._policy_opt, 
                  list(self._policy_net.parameters()), self._actor_grad_clip)

        # unfreeze q-networks
        for p in self._q_net.parameters():
            p.requires_grad = True

        # logging
        entropy = 0.5 * logstd.shape[1] * (1.0 + np.log(2 * np.pi)) + logstd.sum(dim=-1)
        self._actor_summaries = [
            ScalarSummary('train_actor/loss', actor_loss),
            ScalarSummary('train_actor/entropy', entropy.mean()),
            ScalarSummary('train_actor/actions', greedy_actions.mean()),
            ScalarSummary('train_actor/logprobs', logprobs.mean()),
        ]
        self._actor_summaries += self._action_space.log_actions(greedy_actions, 'train_actor')

        # adjust temperature
        self._update_temperature(logprobs)

    def _update_temperature(self, logprobs):
        if self._alpha_auto_tune:
            alpha_loss = (self.alpha * (- logprobs - self._target_entropy).detach()).mean()
            grad_step(alpha_loss, self._alpha_opt)

            self._actor_summaries += [
                ScalarSummary('train_alpha/loss', alpha_loss),
                ScalarSummary('train_alpha/value', self.alpha),            
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