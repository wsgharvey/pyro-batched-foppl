import torch

from pyro.infer.mcmc.util import TraceEinsumEvaluator
from opt_einsum import shared_intermediates
from pyro.ops.contract import contract_to_tensor


def flatten(r_or_z, batch_size):
    return torch.cat([r_or_z[name].reshape(batch_size, -1)
                      for name in sorted(r_or_z)], dim=1)


def unpack(r_flat, r_shapes, batch_size):
    r = {}
    pos = 0
    for name in sorted(r_shapes):
        next_pos = pos + int(r_shapes[name].numel()/batch_size)
        r[name] = r_flat[:, pos:next_pos].reshape(r_shapes[name])
        pos = next_pos
    assert pos*batch_size == r_flat.numel()
    return r


class BatchTraceEinsumEvaluator(TraceEinsumEvaluator):
    def log_prob(self, model_trace):
        """
        almost identical to that of TraceEinsumEvaluator but
        uses log_prob instead of log_prob_sum
        """
        if not self.has_enumerable_sites:
            log_prob = 0
            for name in model_trace.stochastic_nodes:
                dist = model_trace.nodes[name]['fn']
                value = model_trace.nodes[name]['value']
                site_log_prob = dist.log_prob(value)
                log_prob = log_prob + site_log_prob
            return log_prob
        log_probs = self._get_log_factors(model_trace)
        with shared_intermediates() as cache:
            return contract_to_tensor(log_probs, self._enum_dims, cache=cache)
