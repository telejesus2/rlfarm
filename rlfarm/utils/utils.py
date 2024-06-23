import torch
import numpy as np
import matplotlib.pyplot as plt

# def normalize(x):
#     return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)


def normalize(values, mean=0., std=1.):
    values = (values - values.mean()) / (values.std() + 1e-8)
    return mean + (std + 1e-8) * values


def expand_shape(dim, shape):
    return (dim, shape) if np.isscalar(shape) else (dim, *shape)


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def show_image(im):
    plt.imshow(im, interpolation='nearest')
    plt.show()


def batch_from_obs(obs, batch_size=32):
	"""Converts a pixel obs (C,H,W) to a batch (B,C,H,W) of given size"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def assert_vars_change(old_params, params, should_change=True):
    """Check if variables have changed

    Usage:  old_params = [(name, p.clone()) for (name, p) in net.named_parameters()]
            <training_step>
            assert_vars_change(old_params, list(net.named_parameters()), should_change=True)
    """
    for (_, p0), (name, p1) in zip(old_params, params):
        try:
            if should_change:
                assert not torch.equal(p0, p1)
            else:
                assert torch.equal(p0, p1)
        except AssertionError:
            raise AssertionError("{var_name} {msg}".format(
                var_name=name,
                msg='did not change!' if should_change else 'changed!'))


# stolen from https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/math_util.py
# e.g. explained_variance(values, returns)
def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


# stolen from https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/math_util.py
def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y-ypred)/vary
    out[vary < 1e-10] = 0
    return out
