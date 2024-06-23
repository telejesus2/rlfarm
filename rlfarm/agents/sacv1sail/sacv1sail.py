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


class SACv1SAIL(Agent):
    def __init__(
            self,
            action_space: ActionSpace,
            action_min_max,
            policy_net: torch.nn.Module,
            q_net: torch.nn.Module,
            v_net: torch.nn.Module,
            policy_opt_class: str,
            policy_opt_kwargs: dict,
            q_opt_class: str,
            q_opt_kwargs: dict,
            v_opt_class: str,
            v_opt_kwargs: dict,
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
            use_v: bool = False,
            # same as in sacfd.py
            lambda_bc: float =  0,
            lambda_nstep: float = 0,
            q_filter: bool = False,
            replay_demo_bonus: float = 0,
            replay_lambda_actor: float = 0,
            # specific to sail
            lambda_sail: float = 0,
            clip_sail: float = 0,
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
        self._use_v = use_v

        # networks
        self._policy_net = policy_net
        self._q_net = q_net
        self._v_net = v_net
        self._policy_opt_class = policy_opt_class
        self._policy_opt_kwargs = policy_opt_kwargs
        self._q_opt_class = q_opt_class
        self._q_opt_kwargs = q_opt_kwargs
        self._v_opt_class = v_opt_class
        self._v_opt_kwargs = v_opt_kwargs

        # alpha (entropy regularization)
        self._alpha = alpha
        self._alpha_auto_tune = alpha_auto_tune
        self._alpha_lr = alpha_lr
        self._target_entropy = target_entropy
        if target_entropy is None:
            self._target_entropy = - np.prod(len(action_min_max[0])).item() # heuristic from paper

        # overcoming exploration in reinforcement learning with demonstrations
        self._lambda_bc = lambda_bc
        self._q_filter = q_filter

        # leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards
        self._replay_demo_bonus = replay_demo_bonus
        self._replay_lambda_actor = replay_lambda_actor
        self._lambda_nstep = lambda_nstep

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
            self._q_net = copy.deepcopy(self._q_net)
            self._q_net.build()
            self._q_target_net = make_target_net(self._q_net)
            self._q_net = self._q_net.to(device).train(training)
            self._q_target_net = self._q_target_net.to(device).train(False)
            soft_update(self._q_net, self._q_target_net, 1)

            self._v_net = copy.deepcopy(self._v_net)
            self._v_net.build()
            self._v_target_net = make_target_net(self._v_net)
            self._v_net = self._v_net.to(device).train(training)
            self._v_target_net = self._v_target_net.to(device).train(False)
            soft_update(self._v_net, self._v_target_net, 1)

            if self._shared_encoder:  # assumes q shares encoder
                self._policy_net.encoder.copy_weights_from(self._v_net.encoder)
                self._q_net.encoder.copy_weights_from(self._v_net.encoder)

            self._q_opt = make_optimizer(self._q_opt_class, self._q_opt_kwargs,
                self._q_net.parameters())
            self._policy_opt = make_optimizer(self._policy_opt_class, self._policy_opt_kwargs,
                self._policy_net.parameters())
            self._v_opt = make_optimizer(self._v_opt_class, self._v_opt_kwargs,
                self._v_net.parameters())      

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

            logging.info('# SAC Q-function Params: %d' % sum(
                p.numel() for p in self._q_net.parameters() if p.requires_grad))
            logging.info('# SAC Actor Params: %d' % sum(
                p.numel() for p in self._policy_net.parameters() if p.requires_grad))
            logging.info('# SAC V-function Params: %d' % sum(
                p.numel() for p in self._v_net.parameters() if p.requires_grad))

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
            self._update_actor_and_temperature(states, actions, demo_mask, loss_weights)

        # update the target networks
        if step % self._target_update_freq == 0:
            soft_update(self._q_net, self._q_target_net, self._critic_tau)
            soft_update(self._v_net, self._v_target_net, self._critic_tau)

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
                q1_tmp, q2_tmp = self._q_target_net(states, actions)
                q_tmp = torch.min(q1_tmp, q2_tmp)  # shape (N, 1)
                rewards_sail = torch.max(returns, q_tmp) - self._v_target_net(states) # shape (N, 1)
                # rewards_sil = torch.max(returns - self._v_target_net(states), 0)
                if self._clip_sail > 0:
                    rewards_sail = torch.clamp(rewards_sail, min=-self._clip_sail, max=self._clip_sail)

            next_actions, logprobs, _ = self._policy_net(next_states, full_output=True)
            if self._use_v:
                targets_next = self._v_target_net(next_states)
            else:
                q1_targets_next, q2_targets_next = self._q_target_net(next_states, next_actions)
                targets_next = torch.min(q1_targets_next, q2_targets_next)

            q_targets = rewards + self._lambda_sail * rewards_sail + (1 - done_mask) * (
                self._gamma ** n_steps) * (targets_next - self.alpha.detach() * logprobs)
        return q_targets

    def _compute_critic_loss(self, q1_values, q2_values, q_targets):
        # q1_delta = F.smooth_l1_loss(q1_values, q_targets, reduction='none')
        # q2_delta = F.smooth_l1_loss(q2_values, q_targets, reduction='none')
        q1_delta = F.mse_loss(q1_values, q_targets, reduction='none') # shape (N, 1)
        q2_delta = F.mse_loss(q2_values, q_targets, reduction='none')
        q1_loss, q2_loss = q1_delta.mean(1), q2_delta.mean(1) # shape (N)
        return q1_loss, q2_loss

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
        q1_values, q2_values = self._q_net(states, actions) # shape (N, 1)

        # update critic
        q1_loss_ew, q2_loss_ew = self._compute_critic_loss(q1_values, q2_values, q_targets)
        q1_loss, q2_loss = (q1_loss_ew * loss_weights).mean(), (q2_loss_ew * loss_weights).mean()
        if self._lambda_nstep > 0:
            q1_loss_tpn_ew, q2_loss_tpn_ew = self._compute_critic_loss(
                q1_values, q2_values, q_targets_tpn)
            q1_loss_tpn = (q1_loss_tpn_ew * loss_weights).mean()
            q2_loss_tpn = (q2_loss_tpn_ew * loss_weights).mean()
            q1_loss += self._lambda_nstep * q1_loss_tpn
            q2_loss += self._lambda_nstep * q2_loss_tpn
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
            new_priorities = ((q1_loss_ew + q2_loss_ew) / 2.).pow(2)
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

        return actor_loss, q_values, prior_logprobs

    def _update_actor_and_temperature(
        self, states, actions, demo_mask, loss_weights, detach_encoder=False):

        # temporally freeze q-networks 
        for p in self._q_net.parameters():
            p.requires_grad = False

        # compute sac loss
        greedy_actions, logprobs, logstd = self._policy_net(states, full_output=True,
            detach_encoder=detach_encoder)
        sac_loss_ew, q_values, prior_logprobs = self._compute_actor_loss(
            states, greedy_actions, logprobs)
        sac_loss = (sac_loss_ew * loss_weights).mean()
        actor_loss = sac_loss

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

        # compute value function loss
        v_values = self._v_net(states) # shape (N, 1)
        with torch.no_grad():
            v_targets = q_values - (self.alpha.detach() * logprobs) + prior_logprobs # shape (N, 1)
        v_loss_ew = F.mse_loss(v_values, v_targets, reduction='none').mean(1) # shape (N)

        # update value function
        v_loss = (v_loss_ew * loss_weights).mean()
        grad_step(v_loss, self._v_opt, 
                  list(self._v_net.parameters()), self._critic_grad_clip)
        
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
            ScalarSummary('train_critic/v_loss', v_loss),
        ]
        if self._lambda_bc > 0:
            self._actor_summaries += [
                ScalarSummary('train_actor/bc_loss', bc_loss),
                ScalarSummary('train_actor/sac_loss', sac_loss),
                ScalarSummary('train_actor/bc_mask_kept_proportion', mask.float().mean()),
            ]

        # adjust temperature
        self._update_temperature(logprobs)

        # adjust replay priorities
        if self._prioritized:
            new_priorities = sac_loss_ew.pow(2)
            self._new_priorities += self._replay_lambda_actor * new_priorities.detach()

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
                       + self._q_net.log('train_q') \
                       + self._v_net.log('train_v')
        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str, training: bool = False):
        self._policy_net.load_state_dict(torch.load(
            os.path.join(savedir, 'pi.pt'), map_location=self._device))
        if training:
            self._q_net.load_state_dict(torch.load(
                os.path.join(savedir, 'q.pt'), map_location=self._device))
            self._v_net.load_state_dict(torch.load(
                os.path.join(savedir, 'v.pt'), map_location=self._device))

    def save_weights(self, savedir: str):
        torch.save(self._policy_net.state_dict(),
            os.path.join(savedir, 'pi.pt'))
        torch.save(self._q_net.state_dict(),
            os.path.join(savedir, 'q.pt'))
        torch.save(self._v_net.state_dict(),
            os.path.join(savedir, 'v.pt'))
