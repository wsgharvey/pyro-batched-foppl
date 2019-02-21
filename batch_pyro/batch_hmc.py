import torch
from torch.distributions import biject_to, constraints

import pyro
from pyro.infer.mcmc import HMC
import pyro.poutine as poutine
from pyro.distributions.util import eye_like
from pyro.util import torch_isnan, optional
from pyro.poutine.subsample_messenger import _Subsample
from pyro.infer import config_enumerate

from batch_pyro.batch_adaptation import BatchAdapter
from batch_pyro.batch_integrator import velocity_verlet
from batch_pyro.utils import flatten, unpack, BatchTraceEinsumEvaluator


class BatchHMC(HMC):
    def __init__(self, *args,
                 batch_size=10,
                 step_size=1,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 target_accept_prob=0.8,
                 full_mass=False,
                 **kwargs):
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        self._adapter = BatchAdapter(
            batch_size=batch_size,
            step_size=step_size,
            adapt_step_size=adapt_step_size,
            target_accept_prob=target_accept_prob,
            adapt_mass_matrix=adapt_mass_matrix,
            is_diag_mass=not full_mass)

    def _sample_r(self, name):
        r_dist = self._adapter.r_dist
        r_flat = pyro.sample(name, r_dist)
        assert r_flat.shape[0] == self.batch_size
        r = unpack(r_flat, self._r_shapes, self.batch_size)
        return r, r_flat

    def _kinetic_energy(self, r):
        r_flat_batch = flatten(r, self.batch_size)
        if self.inverse_mass_matrix.dim() == 2:
            return 0.5 * torch.tensor([self.inverse_mass_matrix.matmul(r_flat).dot(r_flat)
                                       for r_flat in r_flat_batch])
        else:
            return 0.5 * torch.tensor([self.inverse_mass_matrix.dot(r_flat ** 2)
                                       for r_flat in r_flat_batch])

    def _initialize_model_properties(self):
        if self.max_plate_nesting is None:
            self._guess_max_plate_nesting()
        # Wrap model in `poutine.enum` to enumerate over discrete latent sites.
        # No-op if model does not have any discrete latents.
        self.model = poutine.enum(config_enumerate(self.model),
                                  first_available_dim=-1 - self.max_plate_nesting)
        if self._automatic_transform_enabled:
            self.transforms = {}
        trace = poutine.trace(self.model).get_trace(*self._args, **self._kwargs)
        self._prototype_trace = trace
        for name, node in trace.iter_stochastic_nodes():
            if isinstance(node["fn"], _Subsample):
                continue
            if node["fn"].has_enumerate_support:
                self._has_enumerable_sites = True
                continue
            site_value = node["value"]
            if node["fn"].support is not constraints.real and self._automatic_transform_enabled:
                self.transforms[name] = biject_to(node["fn"].support).inv
                site_value = self.transforms[name](node["value"])
            self._r_shapes[name] = site_value.shape
            self._r_numels[name] = site_value.numel()
        self._trace_prob_evaluator = BatchTraceEinsumEvaluator(trace,
                                                               self._has_enumerable_sites,
                                                               self.max_plate_nesting)
        mass_matrix_size = int(sum(self._r_numels.values())/self.batch_size)
        if self._adapter.is_diag_mass:
            initial_mass_matrix = site_value.new_ones(mass_matrix_size)
        else:
            initial_mass_matrix = eye_like(site_value, mass_matrix_size)
        self._adapter.configure(self._warmup_steps,
                                inv_mass_matrix=initial_mass_matrix,
                                find_reasonable_step_size_fn=self._find_reasonable_step_size)
        self._initialize_step_size()  # this method also caches z and its potential energy

    def _find_reasonable_step_size(self):
        """
        Literally copy-pasted from non-batch HMC, but velocity_verlet now
        refers to a batch-friendly version (and has a new argument).
        """
        step_size = self.step_size

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        z, potential_energy, z_grads = self._fetch_from_cache()
        r, _ = self._sample_r(name="r_presample_0")
        energy_current = self._kinetic_energy(r) + potential_energy
        z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
            z, r, self._potential_energy, self.inverse_mass_matrix, step_size, self.batch_size, z_grads=z_grads)
        energy_new = self._kinetic_energy(r_new) + potential_energy_new
        avg_delta_energy = (energy_new - energy_current).sum()/self.batch_size
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN` which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample site).
        direction = 1 if self._direction_threshold < -avg_delta_energy else -1

        # define scale for step_size: 2 for increasing, 1/2 for decreasing
        step_size_scale = 2 ** direction
        direction_new = direction
        # keep scale step_size until accept_prob crosses its target
        # TODO: make thresholds for too small step_size or too large step_size
        t = 0
        while direction_new == direction:
            t += 1
            step_size = step_size_scale * step_size
            r, _ = self._sample_r(name="r_presample_{}".format(t))
            energy_current = self._kinetic_energy(r) + potential_energy
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
                z, r, self._potential_energy, self.inverse_mass_matrix, step_size, self.batch_size, z_grads=z_grads)
            energy_new = self._kinetic_energy(r_new) + potential_energy_new
            avg_delta_energy = (energy_new - energy_current).sum()/self.batch_size
            direction_new = 1 if self._direction_threshold < -avg_delta_energy else -1
        return step_size

    def sample(self, trace):
        z, potential_energy, z_grads = self._fetch_from_cache()
        r, _ = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r) + potential_energy

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(z, r, self._potential_energy,
                                                                              self.inverse_mass_matrix,
                                                                              self.step_size,
                                                                              self.batch_size,
                                                                              self.num_steps,
                                                                              z_grads=z_grads)
            # apply Metropolis correction.
            energy_proposal = self._kinetic_energy(r_new) + potential_energy_new
        delta_energy = energy_proposal - energy_current
        # Set accept prob to 0.0 if delta_energy is `NaN` which may be
        # the case for a diverging trajectory when using a large step size.
        if torch_isnan(delta_energy):
            accept_prob = delta_energy.new_tensor(0.0)
        else:
            accept_prob = (-delta_energy).exp().clamp(max=1.)
        rand = torch.rand(self.batch_size)
        accepted = rand < accept_prob
        self._accept_cnt += accepted.sum()/self.batch_size

        # select accepted zs to get z_new
        transitioned_z = {}
        for name in z:
            assert len(z_grads[name].shape) == 2
            assert z_grads[name].shape[0] == self.batch_size
            assert len(z[name].shape) == 2
            assert z[name].shape[0] == self.batch_size
            old_val = z[name]
            old_grad = z_grads[name]
            new_val = z[name]
            new_grad = z_grads_new[name]
            val_dim = old_val.shape[1]
            accept_val = accepted.view(self.batch_size, 1).repeat(1, val_dim)
            transitioned_z[name] = torch.where(accept_val,
                                               new_val,
                                               old_val)
            transitioned_grads = torch.where(accept_val,
                                             new_grad,
                                             old_grad)

        self._cache(transitioned_z,
                    potential_energy,
                    transitioned_grads)

        if self._t < self._warmup_steps:
            self._adapter.step(self._t, transitioned_z, accept_prob)

        self._t += 1

        # get trace with the constrained values for `z`.
        z = transitioned_z.copy()
        for name, transform in self.transforms.items():
            z[name] = transform.inv(z[name])
        return self._get_trace(z)
