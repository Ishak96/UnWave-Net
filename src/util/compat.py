import pytorch_lightning as pl

def is_lightning_2() -> bool:
    return pl.__version__.startswith("2.")