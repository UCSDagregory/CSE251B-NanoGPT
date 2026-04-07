import torch.nn as nn

helper_impl = None

def registerCreateModel(func):
    global helper_impl
    helper_impl = func

def createModel(*args, **kwargs) -> nn.Module:
    if helper_impl is None:
        raise NotImplementedError("No implementation registered for createModel")
    return helper_impl(*args, **kwargs)