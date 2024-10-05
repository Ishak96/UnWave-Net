import sys
sys.path.append("../")
import time

import torch
from src.regularizer import WaveFormer
    
model = WaveFormer(wavename="haar", dim=48, in_chans=1, img_size=256).cuda()
x = torch.zeros(1, 1, 256, 256).cuda()
x[0, 0, 50:200, 50:200] = 1.0

t_start = time.time()
y = model(x)
t_end = time.time()

print(f"WaveFormer: {y.shape} in {t_end - t_start} seconds")
# Error compute
print(f"Error: {torch.functional.F.mse_loss(x, y)}")