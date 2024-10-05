import torch
import torch.nn as nn

class ConcatenateLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return torch.cat(list(x), dim=1)

class SplitLayer(nn.Module):
    def __init__(self, split_sizes):
        super().__init__()
        
        self.split_sizes = split_sizes

    def forward(self, x):
        current_pos = 0
        chunks = []
        for l in self.split_sizes:
            chunks.append(
                x[:,current_pos : current_pos+l]
            )
            current_pos = l
        return tuple(chunks)

class DualNet(nn.Module):
    def __init__(self, n_dual):
        super().__init__()

        self.n_dual = n_dual
        self.n_channels = n_dual + 2

        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_dual, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, h, Op_f, g):

        x = self.input_concat_layer(h, Op_f, g)
        x = h + self.block(x)
        return x

class PrimalNet(nn.Module):
    def __init__(self, n_primal):
        super(PrimalNet, self).__init__()

        self.n_primal = n_primal
        self.n_channels = n_primal + 1

        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_primal, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, f, OpAdj_h):

        x = self.input_concat_layer(f, OpAdj_h)
        x = f + self.block(x)
        return x