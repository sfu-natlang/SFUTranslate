from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from configuration import cfg


def get_a_new_optimizer(optim_name,  lr, model_params):
    if optim_name == "sgd":
        op = optim.SGD(model_params, lr=float(lr), momentum=float(cfg.learning_momentum))
    elif optim_name == "asgd": # pretty fast round initially
        op = optim.ASGD(model_params, lr=float(lr))
    elif optim_name == "adam":
        op = optim.Adam(model_params, lr=float(lr))
    elif optim_name == "adadelta":
        op = optim.Adadelta(model_params, lr=float(lr))
    elif optim_name == "adagrad":
        op = optim.Adagrad(model_params, lr=float(lr))
    elif optim_name == "rmsprop":
        op = optim.RMSprop(model_params, lr=float(lr))
    else:
        raise ValueError("Optimizer: {} is not implemented".format(optim_name))
    return op, ReduceLROnPlateau(op, mode='max', patience=int(cfg.lr_decay_patience_steps), verbose=True,
                                 factor=float(cfg.lr_decay_factor), threshold=float(cfg.lr_decay_threshold),
                                 min_lr=float(cfg.lr_decay_min))


def adjust_learning_rate(optimizer_, gamma):
    """Multiplies the learning rate by gamma"""
    for param_group in optimizer_.param_groups:
        past = param_group['lr']
        param_group['lr'] = max(param_group['lr'] * gamma, float(cfg.lr_decay_min))
        print("Updating the learning rate from {:.2f} to {:.2f}".format(past, param_group['lr']))
