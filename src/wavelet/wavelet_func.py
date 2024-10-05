"""
This code is adapted from https://github.com/VinAIResearch/WaveDiff/blob/main/DWT_IDWT/DWT_IDWT_Functions.py
We use the 2D DWT and IDWT functions.
"""
from typing import Tuple

import torch
from torch.autograd import Function

class DWTFunction(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH) -> Tuple[torch.Tensor, None, None, None, None]:
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(
            matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None

class IDWTFunction(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1) -> torch.Tensor:
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()),
                      torch.matmul(input_LH, matrix_High_1.t()))
        H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()),
                      torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), L),
                           torch.matmul(matrix_High_0.t(), H))
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None