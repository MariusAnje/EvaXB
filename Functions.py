import torch
from torch import autograd
from torch import nn

class QuantFunction(autograd.Function):
    @staticmethod
    def forward(ctx, N, input, input_range=None):
        integer_range = pow(2, N) - 1
        if input_range is None:
            det = input.abs().max() / integer_range
        else:
            det = input_range / integer_range
        if det == 0:
            return input
        else:
            return (input/det).round().clamp(-integer_range, integer_range) * det

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None

def sepMM(in_vect, w_mat, A_quant, array_size):
    if type(array_size) == tuple:
        rows, cols = array_size
    else:
        rows = array_size
    in_split = torch.split(in_vect, split_size_or_sections=rows, dim=1)
    w_split = torch.split(w_mat, split_size_or_sections=rows, dim=1)
    res = 0
    for i in range(len(in_split)):
        res += A_quant(in_split[i].mm(w_split[i].T))
    return res