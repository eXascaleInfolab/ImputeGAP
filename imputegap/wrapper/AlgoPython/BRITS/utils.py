# ===============================================================================================================
# SOURCE: https://github.com/caow13/BRITS
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://papers.nips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
# ===============================================================================================================

import torch
from torch.autograd import Variable

#def to_var(var, device=None):
#    if torch.is_tensor(var):
#        var = Variable(var)
#        if torch.cuda.is_available():
#            var = var.cuda()
#        return var
#    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
#        return var
#    if isinstance(var, dict):
#        for key in var:
#            var[key] = to_var(var[key])
#        return var
#    if isinstance(var, list):
#        var = map(lambda x: to_var(x), var)
#        return var

def to_var(var, device=None):
    # Tensor: transfer to the requested device.
    if torch.is_tensor(var):
        var = Variable(var)
        if device is not None:
            var = var.to(device)
        return var

    # Primitive values do not have a device.
    if isinstance(var, (int, float, str)):
        return var

    # Recursively transfer tensors inside dictionaries.
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device=device)
        return var

    # Recursively transfer tensors inside lists.
    if isinstance(var, list):
        return [to_var(x, device=device) for x in var]

    return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz, device="cpu"):
    #x = Variable(torch.zeros(sz))
    #if torch.cuda.is_available():
    #    x = x.cuda()
    #return x
    # Create the tensor directly on the requested device.
    return Variable(torch.zeros(sz, device=device))
