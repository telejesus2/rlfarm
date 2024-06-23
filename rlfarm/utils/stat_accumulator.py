from multiprocessing import Lock
from typing import List
from collections import deque
import itertools

import numpy as np
from rlfarm.utils.logger import Summary, ScalarSummary
from rlfarm.utils.transition import ReplayTransition


class StatAccumulator(object):
    def step(self, transition: ReplayTransition, eval: bool):
        pass

    def pop(self) -> List[Summary]:
        pass

    def peak(self) -> List[Summary]:
        pass

    def reset(self) -> None:
        pass


class Metric(object):
    def __init__(self, max_len):
        self._max_len = max_len
        self._previous = deque(maxlen=max_len)
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        self._previous.clear()

    def _slice(self, len):
        current_len = self.__len__()
        if current_len == 0:
            return [0]
        if len is None or len > current_len:
            return self._previous
        return list(itertools.islice(self._previous, 
                    current_len - len, current_len))

    def min(self, len=None):
        return np.min(self._slice(len))

    def max(self, len=None):
        return np.max(self._slice(len))

    def mean(self, len=None):
        return np.mean(self._slice(len))

    def median(self, len=None):
        return np.median(self._slice(len))

    def std(self, len=None):
        return np.std(self._slice(len))

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]


class _SimpleAccumulator(StatAccumulator):
    def __init__(self, prefix, eval_video_fps: int = 30,
                 mean_only: bool = True,
                 metrics_len: list = [10, 100],
                 to_print=False):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()
        self._episode_returns = Metric(max(metrics_len))
        self._episode_lengths = Metric(max(metrics_len))
        self._episode_success = Metric(max(metrics_len))
        self._episode_failure = Metric(max(metrics_len))
        self._episode_timeout = Metric(max(metrics_len)) 
        self._episode_error = Metric(max(metrics_len)) 
        self._metrics_len = metrics_len
        self._metrics_len.sort(reverse=True)
        self._summaries = []
        self._transitions = self._prev_transitions = 0
        self._episodes = self._prev_episodes = 0
        self._max_stats = {}
        self._to_print = to_print

        self._metric_names = ["return", "length", "success", "failure", "timeout", "error"]
        self._metrics = [
            self._episode_returns, self._episode_lengths, self._episode_success,
            self._episode_failure, self._episode_timeout, self._episode_error]

    def step(self, transition: ReplayTransition):
        with self._lock:
            self._transitions += 1
            self._episode_returns.update(transition.reward)
            self._episode_lengths.update(1)
            if transition.terminal:
                if transition.success: self._episode_success.update(1)
                if transition.failure: self._episode_failure.update(1)
                if transition.error: self._episode_error.update(1)
                if transition.timeout: self._episode_timeout.update(1)
                self._episodes += 1
                for m in self._metrics:
                    m.next()
            # self._summaries.extend(list(transition.summaries)) # TODO (jesus) this keeps growing and growing

    def _get(self) -> List[Summary]:
        sums = []

        if self._mean_only:
            stat_keys = ["mean"]
        else:
            stat_keys = ["min", "max", "mean", "median", "std"]
        for name, metric in zip(self._metric_names, self._metrics):
            for stat_key in stat_keys:
                for i, length in enumerate(self._metrics_len):
                    sum_name = '%s/%s/%s/%d' % (self._prefix, name, stat_key, length)
                    sum_value = getattr(metric, stat_key)(length)
                    sums.append(ScalarSummary(sum_name, sum_value, to_print=self._to_print))
                    # we track the max of the mean return over the longest length (i == 0)
                    if i == 0 and name=="return":
                        if len(metric) < length:
                            self._max_stats[sum_name] = -np.inf
                        else:
                            self._max_stats[sum_name] = max(sum_value,
                                self._max_stats.get(sum_name, -np.inf))
        sums.append(ScalarSummary(
            '%s/total_transitions' % self._prefix, self._transitions, to_print=self._to_print))
        sums.append(ScalarSummary(
            '%s/new_transitions' % self._prefix, self._transitions - self._prev_transitions))
        sums.append(ScalarSummary(
            '%s/total_episodes' % self._prefix, self._episodes, to_print=self._to_print))
        sums.append(ScalarSummary(
            '%s/new_episodes' % self._prefix, self._episodes - self._prev_episodes))
        for k, v in self._max_stats.items():
            sums.append(ScalarSummary(k + '/max', v, to_print=self._to_print))
        self._prev_transitions = self._transitions
        self._prev_episodes = self._episodes
        sums.extend(self._summaries)
        return sums

    def peak(self) -> List[Summary]:
        return self._get()

    def pop(self) -> List[Summary]:
        data = []
        if len(self._episode_returns) > 1:
            data = self._get()
            self._reset_data()
        return data

    def _reset_data(self):
        with self._lock:
            self._episode_returns.reset()
            self._episode_lengths.reset()
            self._summaries.clear()
    
    def reset(self):
        self._transitions = self._prev_transitions = 0
        self._episodes = self._prev_episodes = 0
        self._reset_data()


class SingleTaskAccumulator(StatAccumulator):
    def __init__(self, 
                 eval_video_fps: int = 30,
                 mean_only: bool = True,
                 metrics_len: list = [10, 100]):
        self._train_acc = _SimpleAccumulator('train_envs', 
            eval_video_fps, mean_only=mean_only, metrics_len=metrics_len, to_print=True)
        self._eval_acc = _SimpleAccumulator('eval_envs', 
            eval_video_fps, mean_only=mean_only, metrics_len=metrics_len)

    def step(self, transition: ReplayTransition, eval: bool):
        if eval:
            self._eval_acc.step(transition)
        else:
            self._train_acc.step(transition)

    def pop(self) -> List[Summary]:
        return self._train_acc.pop() + self._eval_acc.pop()

    def peak(self) -> List[Summary]:
        return self._train_acc.peak() + self._eval_acc.peak()
    
    def reset(self) -> None:
        self._train_acc.reset()
        self._eval_acc.reset()


class MultiTaskAccumulator(StatAccumulator):
    def __init__(self, num_tasks,
                 eval_video_fps: int = 30,
                 mean_only: bool = True,
                 metrics_len: list = [10, 100]):
        self._train_acc = _SimpleAccumulator('train_envs', 
            eval_video_fps, mean_only=mean_only, metrics_len=metrics_len, to_print=True)
        self._eval_acc = _SimpleAccumulator('eval_envs', 
            eval_video_fps, mean_only=mean_only, metrics_len=metrics_len)
        self._train_acc_per_task = [_SimpleAccumulator('train_envs_task_%d' % (i), 
            eval_video_fps, mean_only=mean_only, metrics_len=metrics_len)
            for i in range(num_tasks)]
        self._eval_acc_per_task = [_SimpleAccumulator('eval_envs_task_%d' % (i),
            eval_video_fps, mean_only=mean_only, metrics_len=metrics_len)
            for i in range(num_tasks)]

    def step(self, transition: ReplayTransition, eval: bool):
        task_id = transition.info["task_id"]
        if eval:
            self._eval_acc_per_task[task_id].step(transition)
            self._eval_acc.step(transition)
        else:
            self._train_acc_per_task[task_id].step(transition)
            self._train_acc.step(transition)

    def pop(self) -> List[Summary]:
        combined = self._train_acc.pop() + self._eval_acc.pop()
        for acc in self._train_acc_per_task + self._eval_acc_per_task:
            combined.extend(acc.pop())
        return combined

    def peak(self) -> List[Summary]:
        combined = self._train_acc.peak() + self._eval_acc.peak()
        for acc in self._train_acc_per_task + self._eval_acc_per_task:
            combined.extend(acc.peak())
        return combined

    def reset(self) -> None:
        self._train_acc.reset()
        self._eval_acc.reset()
        [acc.reset() for acc in self._train_acc_per_task + self._eval_acc_per_task]