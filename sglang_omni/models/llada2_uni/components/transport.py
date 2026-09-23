import enum
import math
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import torch as th

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def mean_flat(x):
    return th.mean(x, dim=list(range(1, len(x.size()))))


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def expand_t_like_x(t, x):
    dims = (
        [1] * len(x[0].size()) if isinstance(x, (list, tuple)) else [1] * len(x.size())
    )
    return t.view(t.size(0), *dims)


def time_shift(mu: float, sigma: float, t: th.Tensor):
    t = 1 - t
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    t = 1 - t
    return t


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


# ---------------------------------------------------------------------------
# Flow path: Linear (ICPlan)
# ---------------------------------------------------------------------------


class ICPlan:
    """Linear Interpolation Plan: x_t = t * x1 + (1-t) * x0."""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        return t, 1

    def compute_sigma_t(self, t):
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1 / t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
        }
        return choices.get(form, norm)

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * velocity - mean) / var

    def compute_mu_t(self, t, x0, x1):
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        if isinstance(x1, (list, tuple)):
            return [alpha_t[i] * x1[i] + sigma_t[i] * x0[i] for i in range(len(x1))]
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        if isinstance(x1, (list, tuple)):
            return [d_alpha_t * x1[i] + d_sigma_t * x0[i] for i in range(len(x1))]
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


# ---------------------------------------------------------------------------
# ODE solver
# ---------------------------------------------------------------------------


class ODE:
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
        assert t0 < t1
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
        from torchdiffeq import odeint

        x = x.float()
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            return self.drift(x, t, model, **model_kwargs).float()

        t = self.t.to(device)
        if self.do_shift:
            mu = get_lin_function(y1=0.5, y2=1.15)(x.shape[1])
            t = time_shift(mu, 1.0, t)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        return odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        snr_type,
        do_shift,
        seq_len,
    ):
        if path_type != PathType.LINEAR:
            raise ValueError("Image decoding supports only Linear transport")
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = ICPlan()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.snr_type = snr_type
        self.do_shift = do_shift
        self.seq_len = seq_len

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0, t1 = 0, 1
        eps = train_eps if not eval else sample_eps
        if (
            type(self.path_sampler) in [ICPlan]
            and self.model_type != ModelType.VELOCITY
        ):
            t0 = eps
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif type(self.path_sampler) in [ICPlan]:
            t0 = 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def time_shift(self, mu: float, sigma: float, t: th.Tensor):
        return time_shift(mu, sigma, t)

    def get_drift(self):
        if self.model_type == ModelType.VELOCITY:

            def drift_fn(x, t, model, **model_kwargs):
                return model(x, t, **model_kwargs)

        else:
            raise NotImplementedError(
                "Only velocity prediction is supported for inference"
            )

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape
            return model_output

        return body_fn

    def get_score(self):
        if self.model_type == ModelType.VELOCITY:
            return lambda x, t, model, **kwargs: (
                self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
            )
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class Sampler:
    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def sample_ode(
        self,
        *,
        sampling_method="euler",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=False,
        time_shifting_factor=None,
        stochast_ratio=0.0,
        generator=None,
    ):
        if stochast_ratio == 0.0:

            def drift(x, t, model, **kwargs):
                return self.drift(x, t, model, **kwargs)

            t0, t1 = self.transport.check_interval(
                self.transport.train_eps,
                self.transport.sample_eps,
                sde=False,
                eval=True,
                reverse=reverse,
                last_step_size=0.0,
            )
            _ode = ODE(
                drift=drift,
                t0=t0,
                t1=t1,
                sampler_type=sampling_method,
                num_steps=num_steps,
                atol=atol,
                rtol=rtol,
                do_shift=do_shift,
                time_shifting_factor=time_shifting_factor,
            )
            return _ode.sample

        # DDPM-style re-noising (for turbo decoder)
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        path_sampler = self.transport.path_sampler

        def _sample(init, model, **model_kwargs):
            t_steps = th.linspace(t0, t1, num_steps + 1, dtype=th.float64).to(init)
            x_cur = init.to(th.float64)

            for t_cur, t_next in pairwise(t_steps):
                t_batch = (
                    th.ones(x_cur.size(0), device=x_cur.device, dtype=x_cur.dtype)
                    * t_cur
                )
                v = model(x_cur, t_batch, **model_kwargs)

                t_exp = expand_dims(t_batch, x_cur.dim())
                alpha_t, d_alpha_t = path_sampler.compute_alpha_t(t_exp)
                sigma_t, d_sigma_t = path_sampler.compute_sigma_t(t_exp)
                denom = sigma_t * d_alpha_t - d_sigma_t * alpha_t
                x1_hat = (sigma_t * v - d_sigma_t * x_cur) / denom
                x0_hat = (d_alpha_t * x_cur - alpha_t * v) / denom

                t_next_batch = th.ones_like(t_batch) * t_next
                t_next_exp = expand_dims(t_next_batch, x_cur.dim())
                alpha_next, _ = path_sampler.compute_alpha_t(t_next_exp)
                sigma_next, _ = path_sampler.compute_sigma_t(t_next_exp)

                noi = th.randn(
                    x_cur.shape,
                    dtype=x_cur.dtype,
                    device=x_cur.device,
                    generator=generator,
                )
                x_cur = alpha_next * x1_hat + sigma_next * (
                    x0_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
                )
            return [x_cur]

        return _sample


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    snr_type="uniform",
    do_shift=True,
    seq_len=1024,
):
    if path_type != "Linear" or prediction != "velocity" or loss_weight is not None:
        raise ValueError(
            "Image decoding supports only Linear/velocity transport without loss weighting"
        )
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    loss_type = WeightType.VELOCITY if loss_weight == "velocity" else WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_type_enum = path_choice[path_type]

    if path_type_enum in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0
        sample_eps = 0

    return Transport(
        model_type=model_type,
        path_type=path_type_enum,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        snr_type=snr_type,
        do_shift=do_shift,
        seq_len=seq_len,
    )
