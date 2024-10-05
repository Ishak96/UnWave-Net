import sys
from functools import partial

from loguru import logger

from src.util.dist import get_rank, get_world_size, is_main_process

def setup_loguru() -> None:
    # https://github.com/Delgan/loguru/issues/109#issuecomment-508912347

    def loguru_filter(record) -> bool:
        """True if the message should be logged"""
        rank = get_rank()
        world_size = get_world_size()
        is_master = is_main_process()

        # add rank info to loguru record
        record["extra"]["rank"] = str(rank)
        record["extra"]["world_size"] = str(world_size - 1)

        return is_master

    format_loguru = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
        " | "
        "<level>{level: <8}</level>"
        " | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        " | "
        " <green>[{extra[rank]}/{extra[world_size]}]</green>"
        " - "
        "<level>{message}</level>"
    )

    logger.remove()
    logger.add(sys.stderr, filter=loguru_filter, format=format_loguru)

def builder(name: str, cfg, *args, **kwargs):
    """Builds a function with arguments and keyword arguments"""
    from .train import train

    if name == "train":
        func = train
    else:
        raise ValueError(f"Unknown function name: {name}")

    partial_func = partial(func, cfg, *args, **kwargs)
    return partial_func    