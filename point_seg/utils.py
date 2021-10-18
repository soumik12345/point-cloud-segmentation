import numpy as np
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
