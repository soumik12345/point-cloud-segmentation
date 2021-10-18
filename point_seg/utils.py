import os
import wandb
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt


class LearningRateDecay:
    def plot(self, epochs, title: str = "Learning Rate Schedule") -> None:
        """Compute the set of learning rates for each corresponding epoch"""
        lrs = [self(i) for i in epochs]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):
    """
    Reference: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
    """

    def __init__(self, initial_lr: float, drop_every: int, decay_factor: float) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.drop_every = drop_every
        self.decay_factor = decay_factor

    def __call__(self, epoch: int) -> float:
        exp = np.floor((1 + epoch) / self.drop_every)
        new_lr = self.initial_lr * (self.decay_factor ** exp)
        return new_lr


def init_wandb(project_name, experiment_name, wandb_api_key, config: Dict):
    """Initialize Wandb
    Args:
        project_name: project name on Wandb
        experiment_name: experiment name on Wandb
        wandb_api_key: Wandb API Key
    """
    if project_name is not None and experiment_name is not None:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
