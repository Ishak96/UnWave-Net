import torch
from torch import nn

class CNNRegularization(nn.Module):
    def __init__(self, in_chans: int = 1, n_convs: int = 3, 
                 n_filters: int = 48, kernel_size: int = 5,
                 back_to_input: bool = True):
        super().__init__()

        # Compute the stride and padding to keep the same size
        stride = 1
        padding = (kernel_size - 1) // 2

        self.back_to_input = back_to_input
        curr_filters = in_chans
        self.convs = nn.ModuleList([])
        for _ in range(n_convs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=curr_filters, out_channels=n_filters, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.ReLU(inplace=True),
                )
            )
            curr_filters = n_filters

        if self.back_to_input:
            self.out_conv = nn.Conv2d(in_channels=curr_filters, out_channels=in_chans, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        if self.back_to_input:
            x_out = self.out_conv(x)
        else:
            x_out = x
        return x_out
    
if __name__ == '__main__':
    # test
    x = torch.randn(1, 1, 256, 256)
    
    model = CNNRegularization(n_convs=3, filters=48, kernel_size=5)

    y = model(x)
    print(y.shape)
