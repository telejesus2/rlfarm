import os
import logging
import copy
from typing import List

import torch
import torch.nn.functional as F

from rlfarm.agents.sacfd.sacfd import SACfD
from rlfarm.envs.env import ActionSpace
from rlfarm.agents.utils import grad_step, soft_update, get_loss_weights, clip_grad_norm
from rlfarm.agents.utils import REPLAY_BONUS, PRIORITIES
from rlfarm.agents.data_augs import random_cutout_torch as random_cutout
from rlfarm.functions.optimizer import make_optimizer
from rlfarm.utils.logger import Summary, ScalarSummary, ImageSummary
from rlfarm.buffers.replay.const import *


class SACAE(SACfD):
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
            init_weightsdir: str = '',
            lambda_bc: float =  0,
            lambda_nstep: float = 0,
            q_filter: bool = False,
            replay_demo_bonus: float = 0,
            replay_lambda_actor: float = 0,
            # new wrt parent
            encoder_tau: float = 0.005,
            use_rec: bool = True,
            decoder_net: torch.nn.Module = None,
            encoder_opt_class: str = None,
            encoder_opt_kwargs: dict = None,
            decoder_opt_class: str = None,
            decoder_opt_kwargs: dict = None,
            encoder_grad_clip: float = 20.0,
            decoder_grad_clip: float = 20.0,
            decoder_latent_lambda=1e-6,
            rec_update_freq: int = 1,
            replay_lambda_rec: float = 1.,
            rec_loss: str = 'mse',
            use_cpc: bool = True,
            curl_net: torch.nn.Module = None,
            cpc_opt_class: str = None,
            cpc_opt_kwargs: dict = None,
            cpc_grad_clip: float = 20.0,
            cpc_update_freq: int = 1,
    ):
        super(SACAE, self).__init__(
            action_space, action_min_max,
            policy_net, q_net, policy_opt_class, policy_opt_kwargs, q_opt_class, q_opt_kwargs,
            critic_tau, critic_grad_clip, actor_grad_clip, gamma,
            alpha, alpha_auto_tune, alpha_lr, target_entropy,
            target_update_freq, actor_update_freq, shared_encoder, action_prior,
            normalize_priorities, replay_alpha, replay_beta, init_weightsdir,
            lambda_bc, lambda_nstep, q_filter, replay_demo_bonus, replay_lambda_actor,
        )
        # assert use_rec or use_cpc
        self._encoder_tau = encoder_tau

        # rec loss
        self._use_rec = use_rec
        self._encoder_grad_clip = encoder_grad_clip
        self._decoder_grad_clip = decoder_grad_clip
        self._decoder_latent_lambda = decoder_latent_lambda
        self._decoder_net = decoder_net
        self._encoder_opt_class = encoder_opt_class
        self._encoder_opt_kwargs = encoder_opt_kwargs
        self._decoder_opt_class = decoder_opt_class
        self._decoder_opt_kwargs = decoder_opt_kwargs
        self._rec_update_freq = rec_update_freq
        self._replay_lambda_rec = replay_lambda_rec
        assert rec_loss in ['mse', 'l1']
        self._rec_loss = F.mse_loss if rec_loss == 'mse' else F.l1_loss

        # cpc loss
        self._use_cpc = use_cpc
        self._cpc_opt_class = cpc_opt_class
        self._cpc_opt_kwargs = cpc_opt_kwargs
        self._cpc_grad_clip = cpc_grad_clip
        self._curl_net = curl_net
        self._cpc_update_freq = cpc_update_freq

    def build(self, training: bool, device: torch.device) -> None:
        super(SACAE, self).build(training, device)

        if training:
            if self._use_rec:
                self._decoder_net = copy.deepcopy(self._decoder_net)
                self._decoder_net.build()
                self._decoder_net = self._decoder_net.to(device).train(training)
            
                self._encoder_opt = make_optimizer(self._encoder_opt_class, self._encoder_opt_kwargs,
                    self._q_net.encoder.parameters())
                self._decoder_opt = make_optimizer(self._decoder_opt_class, self._decoder_opt_kwargs,
                    self._decoder_net.parameters())

                logging.info('# SAC Decoder Params: %d' % sum(
                    p.numel() for p in self._decoder_net.parameters() if p.requires_grad))

            if self._use_cpc:
                self._curl_net.build(self._q_net.encoder, self._q_target_net.encoder, device)
                self._cpc_opt = make_optimizer(self._cpc_opt_class, self._cpc_opt_kwargs,
                    self._curl_net.parameters())

    def update(self, step: int, sample: dict, warmup: bool = False) -> dict:
        """during warmup, train only the vision networks"""
        
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

        self._critic_summaries, self._actor_summaries = [], [] 
        if not warmup:
            # update critic
            self._update_critic(states, next_states, actions, rewards, done_mask, n_steps, loss_weights,
                next_states_tpn, rewards_tpn, done_mask_tpn, n_steps_tpn)

            # update actor and temperature
            if step % self._actor_update_freq == 0:
                if self._shared_encoder: # set grad to None to avoid weight decay on encoder
                    self._policy_opt.zero_grad(set_to_none=True)
                self._update_actor_and_temperature(
                    states, actions, demo_mask, loss_weights, detach_encoder=self._shared_encoder)

        # update encoder
        if self._use_cpc and step % self._cpc_update_freq == 0:
            self._contastive_predictive_coding_update(states)

        # update decoder and encoder
        if self._use_rec and  step % self._rec_update_freq == 0:
            self._reconstruction_update(states, next_states)

        # update the target networks
        if step % self._target_update_freq == 0:
            soft_update(self._q_net.q1.network, self._q_target_net.q1.network, self._critic_tau)
            soft_update(self._q_net.q2.network, self._q_target_net.q2.network, self._critic_tau)
            soft_update(self._q_net.encoder, self._q_target_net.encoder, self._encoder_tau)

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

    def _compute_reconstruction_loss(self, rgb_states):
        latent = self._q_net.encoder(rgb_states)
        rgb_rec = self._decoder_net(latent)
        latent_loss = (0.5 * latent.pow(2).sum(1)).mean() * self._decoder_latent_lambda
        rec_loss = self._rec_loss(rgb_rec, rgb_states, reduction='none')
        # TODO coordconv in decoder arch
        rec_loss = rec_loss.mean(-1).mean(-1).mean(-1)
        return rec_loss, latent_loss, latent, rgb_rec

    def _reconstruction_update(self, states, next_states):
        rec_loss, latent_loss, latent, rgb_rec = self._compute_reconstruction_loss(
            states['rgb_state'])
        rec_loss_tp1, latent_loss_tp1, _, _, = self._compute_reconstruction_loss(
            next_states['rgb_state'])
        rec_loss += rec_loss_tp1
        latent_loss += latent_loss_tp1
        total_loss = latent_loss + rec_loss.mean()

        # update decoder and encoder
        self._encoder_opt.zero_grad()
        self._decoder_opt.zero_grad()
        total_loss.backward()
        clip_grad_norm(list(self._q_net.encoder.parameters()), self._encoder_grad_clip)
        clip_grad_norm(list(self._decoder_net.parameters()), self._decoder_grad_clip)   
        self._encoder_opt.step()
        self._decoder_opt.step()

        # logging
        self._rec_summaries = [
            ScalarSummary('train_decoder/loss', total_loss),
            ScalarSummary('train_decoder/latent_loss', latent_loss),
            ScalarSummary('train_decoder/reconstruction_loss', rec_loss.mean()),
            ScalarSummary('train_decoder/latent', latent.mean()),
            ScalarSummary('train_decoder/latent_min', latent.min()),
            ScalarSummary('train_decoder/latent_max', latent.max()),
        ]
        self._rec_image_summaries = [ # TODO should remove because redondant
            ImageSummary('train_decoder/rgb_reconstruction',
                torch.clamp((rgb_rec[:1] + 1.0) / 2.0, 0, 1).permute(0,3,1,2))
        ]

        # adjust replay priorities
        if self._prioritized:
            # self._new_priorities += (rec_loss / torch.max(rec_loss)).detach() # TODO dont normalize here
            new_priorities = rec_loss
            self._new_priorities += self._replay_lambda_rec * new_priorities.detach()

    def _contastive_predictive_coding_update(self, states):
        with torch.no_grad():
            positives = random_cutout(states['rgb_state'])
            anchors = random_cutout(states['rgb_state'].clone())

        z_a = self._curl_net.encode(anchors)
        z_pos = self._curl_net.encode(positives, ema=True)
        
        logits = self._curl_net.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self._device)
        loss = F.cross_entropy(logits, labels)        
        grad_step(loss, self._cpc_opt, self._curl_net.parameters(), self._cpc_grad_clip)

        # logging
        self._cpc_summaries = [
            ScalarSummary('train_curl/loss', loss)]
        self._cpc_image_summaries = [
            ImageSummary('train_curl/rgb_augmentation',
                ((positives[:1] + 1.0) / 2.0).permute(0,3,1,2))]

    def update_summaries(self, log_scalar_only=True) -> List[Summary]:
        summaries = super(SACAE, self).update_summaries(log_scalar_only)
        if self._use_cpc:
            summaries += self._cpc_summaries
            if not log_scalar_only:
                summaries += self._cpc_image_summaries
        if self._use_rec:
            summaries += self._rec_summaries
            if not log_scalar_only:
                summaries += self._decoder_net.log('train_decoder') \
                           + self._rec_image_summaries
        return summaries

    def load_weights(self, savedir: str, training: bool = False):
        super(SACAE, self).load_weights(savedir, training)
        if training:
            if self._use_cpc:
                self._curl_net.W = torch.load(
                    os.path.join(savedir, 'curl.pt'), map_location=self._device)
            if self._use_rec:
                self._decoder_net.load_state_dict(torch.load(
                    os.path.join(savedir, 'decoder.pt'), map_location=self._device))

    def save_weights(self, savedir: str):
        super(SACAE, self).save_weights(savedir)
        if self._use_cpc:
            torch.save(self._curl_net.W, os.path.join(savedir, 'curl.pt'))
        if self._use_rec:
            torch.save(self._decoder_net.state_dict(), os.path.join(savedir, 'decoder.pt'))