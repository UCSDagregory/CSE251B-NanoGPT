import torch.nn as nn

helper_impl = None

def registerCreateModel(func):
    global helper_impl
    helper_impl = func

def CreateModel(*args, **kwargs) -> nn.Module:
    if helper_impl is None:
        raise NotImplementedError("No implementation registered for CreateModel")
    return helper_impl(*args, **kwargs)

opt_impl = None
def registerCreateOptimizer(func):
    global opt_impl
    opt_impl = func

def CreateOptimizer(*args):
    if opt_impl is None:
        raise NotImplementedError("No implementation registered for CreateOptimizer")
    return opt_impl(*args)