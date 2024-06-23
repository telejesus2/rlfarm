import csv
import os
import logging
from termcolor import colored
from typing import Any
from collections import OrderedDict

import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def list_representer(dumper, data):
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
yaml.add_representer(list, list_representer)

def save_config(dir, config):
    with open(os.path.join(dir, 'config.yaml'), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(file):
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return None


class Summary(object):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

class ScalarSummary(Summary):
    def __init__(self, name: str, value: Any, to_print: bool = False):
        super(ScalarSummary, self).__init__(name, value)
        self.to_print = to_print

class HistogramSummary(Summary):
    pass

class ParamSummary(Summary):
    pass

class ImageSummary(Summary):
    pass

class VideoSummary(Summary):
    def __init__(self, name: str, value: Any, fps: int = 30):
        super(VideoSummary, self).__init__(name, value)
        self.fps = fps


class Logger(object):
    def __init__(self,
                 logdir: str,
                 save_tb=True,
                 save_csv=True,
                 log_scalar_frequency=1000,
                 log_array_frequency=10000,
                 log_console_frequency=1000,
                 action_repeat=1):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        print(colored("Logging data to %s"%logdir, 'green'))
        assert log_console_frequency % log_scalar_frequency == 0
        assert log_array_frequency % log_scalar_frequency == 0
        self._save_tb = save_tb
        self._save_csv = save_csv
        self.log_scalar_frequency = log_scalar_frequency
        self.log_array_frequency = log_array_frequency
        self._log_console_frequency = log_console_frequency
        self._action_repeat = action_repeat

        self._field_names = None
        self._field_names_to_print = []
        self._prev_row_data = self._row_data = OrderedDict()
        if save_tb:
            tb_dir = os.path.join(logdir, 'tb')
            self._tb_writer = SummaryWriter(tb_dir)
        if save_csv:
            self._csv_file = os.path.join(logdir, 'data.csv')

    def close(self):
        if self._save_tb:
            self._tb_writer.close()

    def _should_log(self, step, log_frequency):
        return step % log_frequency == 0

    def _update_step(self, step):
        return step * self._action_repeat

    def add_summaries(self, step, summaries):
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary) and \
                                    self._should_log(step, self.log_scalar_frequency):
                    step = self._update_step(step)
                    self._add_scalar(step, summary)
                elif self._save_tb and \
                                    self._should_log(step, self.log_array_frequency):
                    step = self._update_step(step)
                    if isinstance(summary, HistogramSummary):
                        self._tb_writer.add_histogram(summary.name, summary.value, step)
                    elif isinstance(summary, ImageSummary):
                        # Assumes (n,c,h,w) or (c,h,w)
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 3 else
                             summary.value[0])
                        assert v.ndim == 3
                        self._tb_writer.add_image(summary.name, v[:3,:,:], step)
                    elif isinstance(summary, VideoSummary):
                        v = torch.from_numpy(np.array(summary.value))
                        v = v.unsqueeze(0)
                        self._tb_writer.add_video(summary.name, v, step, fps=summary.fps)
                    elif isinstance(summary, ParamSummary):
                        name, param = summary.name, summary.value
                        self._tb_writer.add_histogram(name + '_w', param.weight.data, step)
                        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
                            self._tb_writer.add_histogram(name + '_w_g', param.weight.grad.data, step)
                        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
                            self._tb_writer.add_histogram(name + '_b', param.bias.data, step)
                            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                                self._tb_writer.add_histogram(name + '_b_g', param.bias.grad.data, step)
            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e

    def _add_scalar(self, step, summary: ScalarSummary):
        if self._save_tb:
            self._tb_writer.add_scalar(summary.name, summary.value, step)
        if len(self._row_data) == 0:
            self._row_data['step'] = step
        self._row_data[summary.name] = summary.value.item() if isinstance(
            summary.value, torch.Tensor) else summary.value
        if summary.to_print and summary.name not in self._field_names_to_print:
            self._field_names_to_print += [summary.name]

    def end_iteration(self, step):
        if len(self._row_data) > 0:
            names = self._field_names or self._row_data.keys()
            if self._field_names is not None:
                if not np.array_equal(self._field_names, self._row_data.keys()):
                    # Special case when we are logging faster than new
                    # summaries are coming in, so some entries are missing.
                    missing_keys = list(set(self._field_names) - set(
                        self._row_data.keys()))
                    for mk in missing_keys:
                        self._row_data[mk] = self._prev_row_data[mk]

            if self._save_csv:
                with open(self._csv_file, mode='a+') as csv_f:
                    writer = csv.DictWriter(csv_f, fieldnames=names)
                    if self._field_names is None:
                        writer.writeheader()
                    writer.writerow(self._row_data)  

            self._field_names = names
            self._prev_row_data = self._row_data
            self._dump_to_console(step)              
            self._row_data = OrderedDict()

    def _dump_to_console(self, step): #, data, prefix):
        if self._should_log(step, self._log_console_frequency):
            # prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
            # print(prefix + " ***** iteration %i" % self._row_data['iteration'])
            print(" ***** iteration %i ***** " % self._row_data['step'], self.logdir)
            key_lens = [len(key) for key in self._field_names]
            max_key_len = max(15,max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-"*n_slashes)
            for key in self._field_names_to_print:
                val = self._row_data.get(key, "")
                if hasattr(val, "__float__"): valstr = "%8.3g"%val
                else: valstr = val
                print(fmt%(key, valstr))
            print("-"*n_slashes)