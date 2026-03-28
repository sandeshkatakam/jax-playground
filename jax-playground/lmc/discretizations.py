import jax
import optax
import probax
from abc import ABC, abstractmethod

class Discretization(ABC):
    def __init__(self, dt: float):
        self.dt = dt
    def step(self, x, drift, diffusion, rng_key):
        pass


class Euler_Maruyama(Discretization):
    def __init__(self, step_size, dt: float):
        super().__init__(dt)
        self.step_size = step_size


    def step(self, x, drift, diffusion, rng_key):
        