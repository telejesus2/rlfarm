from typing import List, Tuple

import numpy as np
import torch

from rlfarm.agents.sac.sac import SAC
from rlfarm.envs.env import ActionSpace
from rlfarm.agents.utils import grad_step, soft_update, get_loss_weights
from rlfarm.agents.utils import REPLAY_BONUS, PRIORITIES
from rlfarm.utils.logger import Summary, ScalarSummary
from rlfarm.buffers.replay.const import *


class SACfD(SAC):
    def __init__(
            self,
            action_space: ActionSpace,
            action_min_max: Tuple[np.ndarray],
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
            init_weightsdir: str = '',
            # new wrt parent
            lambda_bc: float =  0,
            lambda_nstep: float = 0,
            q_filter: bool = False,
            replay_demo_bonus: float = 0,
            replay_lambda_actor: float = 0,
    ):
        super(SACfD, self).__init__(
            action_space, action_min_max,
            policy_net, q_net, policy_opt_class, policy_opt_kwargs, q_opt_class, q_opt_kwargs,
            critic_tau, critic_grad_clip, actor_grad_clip, gamma,
            alpha, alpha_auto_tune, alpha_lr, target_entropy,
            target_update_freq, actor_update_freq, shared_encoder, action_prior,
            normalize_priorities, replay_alpha, replay_beta, init_weightsdir,
        )
        # overcoming exploration in reinforcement learning with demonstrations
        self._lambda_bc = lambda_bc
        self._q_filter = q_filter

        # leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards
        self._replay_demo_bonus = replay_demo_bonus
        self._replay_lambda_actor = replay_lambda_actor
        self._lambda_nstep = lambda_nstep

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

        # double lookahead
        next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn = None, None, None, None
        if self._lambda_nstep > 0:
            assert sample[N_STEPS].shape[1] == 2
            assert torch.all(torch.ge(sample[N_STEPS][:,1], sample[N_STEPS][:,0]))
            next_states_tpn = {k: v[:,1] for k,v in sample[NEXT_STATE].items()}         # dict: each entry of shape (N, ob_dim)
            rewards_tpn = sample[REWARD][:,1].view(-1, 1)     # shape (N, 1)
            done_mask_tpn = sample[TERMINAL][:,1].view(-1, 1) # shape (N, 1)
            n_steps_tpn = sample[N_STEPS][:,1].view(-1, 1)    # shape (N, 1)
        else:
            assert sample[N_STEPS].shape[1] == 1

        # update critic
        self._update_critic(states, next_states, actions, rewards, done_mask, n_steps, loss_weights,
            next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn)

        # update actor and temperature
        if step % self._actor_update_freq == 0:
            self._update_actor_and_temperature(states, actions, demo_mask, loss_weights)

        # update the target networks
        if step % self._target_update_freq == 0:
            soft_update(self._q_net, self._q_target_net, self._critic_tau)

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

    def _update_critic(self, states, next_states, actions, rewards, done_mask, n_steps, loss_weights,
                       next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn):
        # compute targets
        q_targets = self._compute_critic_targets(next_states, rewards, done_mask, n_steps)
        if self._lambda_nstep > 0:
            q_targets_tpn = self._compute_critic_targets(
                next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn)

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

    def _update_actor_and_temperature(
        self, states, actions, demo_mask, loss_weights, detach_encoder=False):

        # temporally freeze q-networks 
        for p in self._q_net.parameters():
            p.requires_grad = False

        # compute sac loss
        greedy_actions, logprobs, logstd = self._policy_net(states, full_output=True,
            detach_encoder=detach_encoder)
        sac_loss_ew, q_values = self._compute_actor_loss(states, greedy_actions, logprobs)
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