import traceback

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./config/", config_name="train.yaml")
def my_app(cfg: DictConfig) -> None:

    func = hydra.utils.call(cfg.runners, cfg=cfg, _recursive_=False)

    try:
        out = func()
        return out
    except:
        print(traceback.format_exc())

if __name__ == "__main__":
    my_app()