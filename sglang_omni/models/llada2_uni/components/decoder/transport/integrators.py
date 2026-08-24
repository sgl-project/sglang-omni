# Adapted from inclusionAI/LLaDA2.0-Uni decoder/transport/integrators.py.
import torch as th
from torchdiffeq import odeint

from .utils import get_lin_function, time_shift


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        do_shift=False,
        time_shifting_factor=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.do_shift = do_shift
        self.t = th.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (
                self.t + time_shifting_factor - time_shifting_factor * self.t
            )
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        x = x.float()
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            model_output = self.drift(x, t, model, **model_kwargs).float()
            return model_output

        t = self.t.to(device)
        if self.do_shift:
            mu = get_lin_function(y1=0.5, y2=1.15)(x.shape[1])
            t = time_shift(mu, 1.0, t)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
