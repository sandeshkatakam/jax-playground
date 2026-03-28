import jax
import optax

import flax
from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def sample(self, params, rng_key):
        pass

    @abstractmethod
    def log_prob(self, params):
        pass

    @abstractmethod
    def _initialize_params(self, rng_key):
        pass

    @abstractmethod
    def _update_params(self, params, rng_key):
        pass
