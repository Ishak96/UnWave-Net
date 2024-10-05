import torch
from torch import nn

from src.physics import NCCCST
from src.regularizer import WaveFormer

class GradientFunction(nn.Module):
    def __init__(self,
                 x_model: nn.Module,
    ):
        super().__init__()

        self.x_model = x_model
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self,
                physics_model: NCCCST,
                x: torch.Tensor,
                y: torch.Tensor,
    ):
        current_y = physics_model.forward_operator(x)

        # Compute the regularization term on the current x and y
        regularization_x = self.x_model(x)

        # Compute the data consistency term
        data_residual_term = current_y - y

        # Compute the bp of data residual term
        bp_data_residual_term = physics_model.adjoint_operator(data_residual_term)

        direction = -self.alpha * bp_data_residual_term + regularization_x

        return direction

class UnWaveNet_Block(nn.Module):
    def __init__(self,
                 x_model: nn.Module,
    ):
        super().__init__()

        self.gradient_function = GradientFunction(x_model)

    def forward(self,
                physics_model: NCCCST,
                current_x: torch.Tensor,
                y: torch.Tensor,
    ):
        # Compute the gradient
        grad = self.gradient_function(physics_model, current_x, y)

        # Update the x
        new_x = current_x + grad

        return new_x

class UnWaveNet(nn.Module):
    def __init__(self,
                 num_cascades: int = 12,
                 wavename="haar",
                 dim=96,
                 in_chans=1,
                 img_size=256,
                 **kwargs,
    ):
        super().__init__()

        self.cascades = nn.ModuleList(
            [
                UnWaveNet_Block(
                    x_model=WaveFormer(wavename=wavename, dim=dim, in_chans=in_chans, img_size=img_size)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(self,
                physics_model: NCCCST,
                y: torch.Tensor,
    ):

        pred_x = physics_model.adjoint_operator(y).unsqueeze(0)

        for cascade in self.cascades:
            pred_x = cascade(physics_model, pred_x, y)

        return pred_x
