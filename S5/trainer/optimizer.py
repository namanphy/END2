import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def sgd_optimizer(model, lr=0.01, l2_factor=0, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_factor)


def adam_optimizer(model, lr=0.01):
    return optim.Adam(model.parameters(), lr=lr)


def step_lr_scheduler(optimizer, step_size=None, gamma=0.15):
    if step_size is None:
        raise Exception('step size value must be provided with valid integer')
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
