"""I/O Utils.
"""
from pathlib import Path

import yaml
from pytorch_lightning.utilities.parsing import AttributeDict


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GRAY = "\033[90m"


def color(txt: str, c="GRAY"):
    return getattr(bcolors, c) + txt + bcolors.ENDC


def to_str(path: Path):
    if isinstance(path, Path):
        return str(path)
    return path


def load_config(yaml_file: str):
    """Load a YAML config file."""
    with open(yaml_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return AttributeDict(**cfg)
