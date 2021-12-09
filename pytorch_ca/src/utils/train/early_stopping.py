from abc import abstractmethod

import numpy as np


class StoppingCriteria:
    def __init__(self):
        pass

    @abstractmethod
    def stop(self, epoch, epoch_loss):
        pass


class DefaultStopping(StoppingCriteria):
    def __init__(self):
        super().__init__()

    def stop(self, epoch:int, epoch_loss:float):
        if np.isnan(epoch_loss):
            raise Exception("Loss is NaN")
        if epoch_loss > 5 and epoch > 2 or epoch_loss > 0.25 and epoch == 40:
            raise Exception("Loss is too high")
