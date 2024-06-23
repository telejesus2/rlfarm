import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import numpy as np


def plot_grad_flow(named_parameters, fname):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    lgd = plt.legend([Line2D([0], [0], color="c", lw=4),
                      Line2D([0], [0], color="b", lw=4),
                      Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(fname + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def _iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            print(fn)
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_input), f"{fn} grad_input={grad_input} grad_output={grad_output}"
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_output), f"{fn} grad_input={grad_input} grad_output={grad_output}"
            
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    _iter_graph(var.grad_fn, hook_cb)