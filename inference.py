import os
import warnings

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm.rich import tqdm


def inference(config):
    pass

@hydra.main(version_base=None, config_path="configs", config_name="dino.yaml")
def main(config):
    OmegaConf.resolve(config)

    inference(config)
    