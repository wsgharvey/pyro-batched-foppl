from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.distributions import Normal, MultivariateNormal

from batch_pyro.utils import flatten


class BatchAdapter(WarmupAdapter):
    def __init__(self, *args, batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def step(self, t, z, accept_prob):
        r"""
        Identical to WarmupAdapter func except from flattenting z differently.
        """
        if t >= self._warmup_steps or self._adaptation_disabled:
            return
        window = self._adaptation_schedule[self._current_window]
        num_windows = len(self._adaptation_schedule)
        mass_matrix_adaptation_phase = self.adapt_mass_matrix and \
            (0 < self._current_window < num_windows - 1)
        if self.adapt_step_size:
            self._update_step_size(accept_prob.item())
        if mass_matrix_adaptation_phase:
            # Different stuff
            batch_flat = flatten(z, self.batch_size)
            for z_flat in batch_flat:
                self._mass_matrix_adapt_scheme.update(z_flat.detach())
        if t == window.end:
            if self._current_window == num_windows - 1:
                self._current_window += 1
                self._end_adaptation()
                return

            if self._current_window == 0:
                self._current_window += 1
                return

            if mass_matrix_adaptation_phase:
                self.inverse_mass_matrix = self._mass_matrix_adapt_scheme.get_covariance()
                if self.adapt_step_size:
                    self.reset_step_size_adaptation()

            self._current_window += 1

    def _update_r_dist(self):
        loc = self._inverse_mass_matrix.new_zeros(self._inverse_mass_matrix.size(0))
        if self.is_diag_mass:
            self._r_dist = Normal(
                loc.repeat(self.batch_size, 1),
                self._inverse_mass_matrix.rsqrt())
        else:
            r_dist_dim = loc.shape[0] * self.batch_size
            self._r_dist = MultivariateNormal(
                loc.repeat(self.batch_size, 1),
                precision_matrix=self._inverse_mass_matrix)
