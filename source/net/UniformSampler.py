from abc import ABC, abstractmethod

import random
import numpy as np
import torch as th

class UniformSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = th.ones([num_timesteps])

    def sample(self, batch_size,  use_fp16=False):
        indices = th.randint(0, self.num_timesteps, (batch_size, ))
        if use_fp16:
            weights = th.ones_like(indices).half()
        else:
            weights = th.ones_like(indices).float()
        return indices, weights