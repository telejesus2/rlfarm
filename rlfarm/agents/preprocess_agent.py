from typing import List

import torch
import torchvision

from rlfarm.agents.agent import Agent
from rlfarm.utils.transition import ActResult
from rlfarm.utils.logger import Summary, ScalarSummary, ImageSummary, HistogramSummary
from rlfarm.buffers.replay.const import *


class PreprocessAgent(Agent):
    def __init__(self,
                 agent: Agent,
                 demos: bool = False):
        self._agent = agent
        self._demos = demos

    def build(self, training: bool, device: torch.device):
        self._agent.build(training, device)
        self._device = self._agent._device

    def encoder(self):
        return self._agent.encoder()

    def _norm_rgb_(self, x):
        # scale images from [0, 255] to [-1, 1]
        return (x.float() / 255.0) * 2.0 - 1.0

    def update(self, step: int, sample: dict, warmup: bool = False) -> dict:
        n_steps = sample[N_STEPS].shape[1]
        for k in sample[STATE].keys():
            if 'rgb' in k:
                sample[STATE][k] = self._norm_rgb_(sample[STATE][k])
                sample[NEXT_STATE][k] = torch.stack(
                    [self._norm_rgb_(sample[NEXT_STATE][k][:,n]) for n in range(n_steps)], 1)
        
        self._sample = sample
        return self._agent.update(step, sample, warmup)

    def act(self, step: int, state: dict, deterministic=False, explore=False, track_outputs=False) -> ActResult:
        for k, v in state.items():
            if 'rgb' in k:
                state[k] = self._norm_rgb_(v)        
        act_res = self._agent.act(step, state, deterministic, explore, track_outputs)
        if self._demos:
            act_res.replay_elements.update({DEMO: False})
        return act_res

    def update_summaries(self, log_scalar_only=True) -> List[Summary]:
        prefix = 'inputs'
        tile = lambda x: torchvision.utils.make_grid(x.permute(0,3,1,2))
        states, next_states = self._sample[STATE], self._sample[NEXT_STATE]

        sums = [
            ScalarSummary('%s/low_dim_state_mean' % prefix,
                    states['low_dim_state'].mean()),
            ScalarSummary('%s/low_dim_state_min' % prefix,
                    states['low_dim_state'].min()),
            ScalarSummary('%s/low_dim_state_max' % prefix,
                    states['low_dim_state'].max()),
            ScalarSummary('%s/timeouts' % prefix,
                    self._sample[TIMEOUT].float().mean()),
        ]

        if self._demos:
            demo_f = self._sample[DEMO].float()
            demo_proportion = demo_f.mean()
            sums.extend([
                ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
            ])

        if not log_scalar_only:
            sums.extend([
                HistogramSummary('%s/low_dim_state' % prefix,
                        states['low_dim_state']),
                HistogramSummary('%s/low_dim_state_tp1' % prefix,
                        next_states['low_dim_state']),
            ])
            for k, v in states.items():
                if 'rgb' in k:
                    # scale images from [-1, 1] to [0, 1]
                    v = (v + 1.0) / 2.0
                    sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v)))
            if SAMPLING_PROBABILITIES in self._sample:
                sums.extend([
                    HistogramSummary('replay/priority',
                        self._sample[SAMPLING_PROBABILITIES]),
                ])

        sums.extend(self._agent.update_summaries(log_scalar_only))
        return sums

    def act_summaries(self) -> List[Summary]:
        return self._agent.act_summaries()

    def load_weights(self, savedir: str, training: bool = False):
        self._agent.load_weights(savedir, training)

    def save_weights(self, savedir: str):
        self._agent.save_weights(savedir)

    def reset(self) -> None:
        self._agent.reset()