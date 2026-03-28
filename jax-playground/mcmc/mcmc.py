import jax
import probax

import optax

from .samplers import Sampler


class MCMC(Sampler):
    def __init__(self, sampler: Sampler, max_steps: int = 1000):
        self._rng_key = jax.random.PRNGKey(0)
        self.max_steps: int = 1000
        self._sampler = sampler
        self.params = self._sampler._initialize_params(self._rng_key)

    def sample(self, params, rng_key):

        pass

    def update_params(self, params, rng_key):
        pass

    def log_prob(self, params):
        pass


def metropolis_hastings(target_log_prob, proposal_dist, initial_params, num_samples, rng_key):
    samples = []
    current_params = initial_params
    rng_key = jax.random.PRNGKey(0)
    current_log_prob = target_log_prob(current_params)

    for i in range(1, num_samples + 1):
        rng_key, subkey = jax.random.split(rng_key)
        proposed_params = proposal_dist.sample(subkey)
        proposed_log_prob = target_log_prob(proposed_params)

        acceptance_ratio = jnp.exp(proposed_log_prob - current_log_prob)
        accept = jax.random.uniform(subkey) < acceptance_ratio

        current_params = jnp.where(accept, proposed_params, current_params)
        current_log_prob = jnp.where(accept, proposed_log_prob, current_log_prob)

        samples.append(current_params)
