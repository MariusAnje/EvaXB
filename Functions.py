import torch
from torch import autograd
from torch import nn
import numpy as np

class MappingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_tensor, mapping, total):
        sign = in_tensor.sign()
        res = torch.zeros_like(in_tensor)
        x = in_tensor.abs().long()
        scale = 1
        while not (x == 0).all():
            res += scale * mapping(x % total).view(sign.size())
            scale *= total
            x = torch.div(x, total, rounding_mode='floor')
        return res * sign

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


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


def mapping_func(in_tensor, mapping):
    if mapping is not None:
        total = mapping.shape[0]
        sign = in_tensor.sign()
        res = torch.zeros_like(in_tensor)
        x = in_tensor.abs().long()
        scale = 1
        while not (x == 0).all():
            res += scale * torch.nn.functional.embedding(x % total, mapping).view(sign.size())
            scale *= total
            x = torch.div(x, total, rounding_mode='floor')
        return res * sign
    else:
        return in_tensor

class QuantMappingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, N, input, input_range=None, mapping=None):
        integer_range = pow(2, N) - 1
        if input_range is None:
            det = input.abs().max() / integer_range
        else:
            det = input_range / integer_range
        if det == 0:
            return input
        else:
            int_input = (input/det).round().clamp(-integer_range, integer_range)
            mapped_input = mapping_func(int_input, mapping)
            return mapped_input * det

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None, None

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

def cal_output_size(activation, weight, stride=1, padding=0):
    input_size = (activation.shape[2], activation.shape[3])
    kernel_size = (weight.shape[2], weight.shape[3])
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    # Calculate output size along each dimension
    output_size = tuple(
        ((input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i]) + 1
        for i in range(len(input_size))
    )

    return output_size

def confused_padding(activation, padding):
    assert isinstance(padding, tuple)
    new_padding = ()
    for i in padding:
        new_padding += (i,i)
    padded_act = nn.functional.pad(activation, new_padding, "constant", 0)
    return padded_act

# @torch.compile
def sepConv2d(activation, weight, A_quant, array_size, padding="same", stride=1):
    # Only supports group = 1, stride = 1
    # Only supports kernel size with odd numbers
    # TODO: support stride other than 1
    # TODO: support other kernel sizes

    if not (isinstance(padding, int) or isinstance(padding, tuple)):
        if padding == "same":
            padding = (weight.shape[2]//2, weight.shape[3]//2)
    elif isinstance(padding, int):
        padding = (padding, padding)

    input_channels = activation.shape[1]
    output_channels = weight.shape[0]
    output_act_size = (activation.shape[0], weight.shape[0]) + cal_output_size(activation, weight, padding=padding, stride=stride)
    output_act = torch.zeros(output_act_size, device=weight.device)
    padded_act = confused_padding(activation, padding)
    for i in range(int(np.ceil(input_channels/array_size))):
        start, end = i * array_size, (i+1) * array_size
        # Only support kernel size of odd numbers, e.g., 3
        for j in range(weight.shape[2]):
            e_j = j + output_act_size[2] * stride[0]
            for k in range(weight.shape[3]):
                e_k = k + output_act_size[3] * stride[1]
                psum = nn.functional.conv2d(padded_act[:,start:end,j:e_j,k:e_k], weight[:,start:end,j:j+1,k:k+1], stride=stride)
                output_act += A_quant(psum)
                
    return output_act