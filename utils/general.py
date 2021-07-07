import os, re, yaml
import logging
import torch
import importlib
from torch.utils.tensorboard import SummaryWriter
from utils import losses as custom_loss

def get_attr_by_name(func_str):
    """
    Load function by full name
    :param func_str:
    :return: fn, mod
    """
    module_name, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func, module, func_name

def model_loader(config):
    model_dict = config.get('model')
    func, _, _ = get_attr_by_name(model_dict['model.class'])
    return func(**model_dict)

def get_optimizer(config):
    cfg =  config.get("optimizer")
    optimizer_name = cfg["name"]
    try:
        optimizer = getattr(torch.optim, optimizer_name,\
            "The optimizer {} is not available".format(optimizer_name))
    except:
        optimizer = getattr(torch.optim, optimizer_name,\
            "The optimizer {} is not available".format(optimizer_name))
    del cfg['name']
    return optimizer, cfg

def get_lr_scheduler(config):
    cfg = config.get("scheduler")
    scheduler_name = cfg["name"]
    try:
        # if the lr_scheduler comes from torch.optim.lr_scheduler package
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name,\
            "The scheduler {} is not available".format(scheduler_name))
    except:
        # use custom loss
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name,\
            "The scheduler {} is not available".format(scheduler_name))
    del cfg['name']
    return scheduler, cfg

def get_loss_fn(config):
    loss_function = config["train"]["loss"]
    try:
        # if the loss function comes from nn package
        criterion = getattr(torch.nn, loss_function, "The loss {} is not available".format(loss_function))
    except:
        # use custom loss
        criterion = getattr(custom_loss, loss_function, "The loss {} is not available".format(loss_function))
    return criterion

def make_dir_epoch_time(base_path, session_name, time_str):
    """
    make a new dir on base_path with epoch_time
    :param base_path:
    :return:
    """
    new_path = os.path.join(base_path, session_name + "_" + time_str)
    os.makedirs(new_path, exist_ok=True)
    return new_path

def yaml_loader(yaml_file):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=loader) # cfg dict
    return config

def log_initilize(log_dir):
    log_file = os.path.join(log_dir, "model_logs.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create error file handler and set level to error
    handler = logging.FileHandler(log_file, "a", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.terminator = "\n"
    logger.addHandler(handler)
    return logger

def make_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer