import torch
from torch import nn

import math

from loguru import logger

class NCCCST():
    def __init__(self, 
                 matrix_path: str,
                 size: int = 256,
                 num_detectors: int = 150,
                 emax: int = 100,
    ):  
        self.y_shape = (1, num_detectors, emax)
        self.x_shape = (1, size, size)

        self.read_matrix(matrix_path)

    def read_matrix(self, matrix_path: str):
        logger.info(f"Reading matrix from {matrix_path}")
        radon_file = matrix_path + "/A_256x256.pt"
        radonT_file = matrix_path + "/AT_256x256.pt"
        # load to cuda if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.load(radon_file, map_location=device)
        self.AT = torch.load(radonT_file, map_location=device)
        logger.info(f"Radon matrix shape: {self.A.shape}")
        logger.info(f"Pseudo inverse shape: {self.AT.shape}")

    def forward_operator(self, x: torch.Tensor, squeeze: bool = False):
        if squeeze:
            return torch.sparse.mm(self.A, x.reshape(self.x_shape[1] ** 2, 1).to(self.A.device)).reshape(self.y_shape).squeeze().to(x.device)
        return torch.sparse.mm(self.A, x.reshape(self.x_shape[1] ** 2, 1).to(self.A.device)).reshape(self.y_shape).to(x.device)
    
    def adjoint_operator(self, y: torch.Tensor, squeeze: bool = False):
        if squeeze:
            return torch.mm(self.AT, y.reshape(self.y_shape[1] * self.y_shape[2], 1).to(self.AT.device)).reshape(self.x_shape).squeeze().to(y.device)
        return torch.mm(self.AT, y.reshape(self.y_shape[1] * self.y_shape[2], 1).to(self.AT.device)).reshape(self.x_shape).to(y.device)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    size = 256
    num_detectors = 150
    emax = 100
    matrix_path = f"../../radon/{num_detectors}"
    
    cst = NCCCST(matrix_path=matrix_path, size=size, num_detectors=num_detectors, emax=emax)
    
    x = torch.zeros((1, size, size))
    x[0, 100:150, 100:150] = 1
    y = cst.forward_operator(x)
    x_recon = cst.adjoint_operator(y)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(x[0, ...], cmap="gray")
    plt.title("x")
    plt.subplot(1, 3, 2)
    plt.imshow(y[0, ...], cmap="gray")
    plt.title("y")
    plt.subplot(1, 3, 3)
    plt.imshow(x_recon[0, ...].detach().numpy(), cmap="gray")
    plt.title("x_recon")
    plt.show()
