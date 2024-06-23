import torch.optim as optim


def make_optimizer(class_, kwargs, params):
    if class_ == 'adam':
        return optim.Adam(params, **kwargs)
    if class_ == 'sgd':
        return optim.SGD(params, **kwargs)