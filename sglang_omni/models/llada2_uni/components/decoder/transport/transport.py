# Adapted from inclusionAI/LLaDA2.0-Uni decoder/transport/transport.py.
import enum

import torch as th

from . import path
from .integrators import ode
from .utils import expand_dims


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


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
        path_options = {
            PathType.LINEAR: path.ICPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
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
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0, t1 = 0, 1
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        return velocity_ode

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        assert self.model_type == ModelType.VELOCITY
        return lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
            model(x, t, **kwargs), x, t
        )


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=False,
        time_shifting_factor=None,
        stochast_ratio=0.0,  # 新增参数，0.0=纯ODE，1.0=完全重加噪
    ):
        if stochast_ratio == 0.0:
            # 原有逻辑不变
            drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs)
            t0, t1 = self.transport.check_interval(
                self.transport.train_eps,
                self.transport.sample_eps,
                sde=False,
                eval=True,
                reverse=reverse,
                last_step_size=0.0,
            )
            _ode = ode(
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

        else:
            # 新增：DDPM风格重加噪采样
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
                # t0→t1: noise(t=0) → data(t=1)
                t_steps = th.linspace(t0, t1, num_steps + 1, dtype=th.float64).to(init)
                x_cur = init.to(th.float64)

                for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
                    t_batch = (
                        th.ones(x_cur.size(0), device=x_cur.device, dtype=x_cur.dtype)
                        * t_cur
                    )

                    # 1. 模型预测 velocity
                    v = model(x_cur, t_batch, **model_kwargs)

                    # 2. 直接从流匹配公式还原 x̂₁ 和 x̂₀，避免除以 alpha_t 的奇点
                    # 联立 x_t = alpha_t*x1 + sigma_t*x0 与 v = d_alpha_t*x1 + d_sigma_t*x0
                    t_exp = expand_dims(t_batch, x_cur.dim())
                    alpha_t, d_alpha_t = path_sampler.compute_alpha_t(t_exp)
                    sigma_t, d_sigma_t = path_sampler.compute_sigma_t(t_exp)
                    denom = sigma_t * d_alpha_t - d_sigma_t * alpha_t  # =1 for ICPlan
                    x1_hat = (sigma_t * v - d_sigma_t * x_cur) / denom
                    x0_hat = (d_alpha_t * x_cur - alpha_t * v) / denom

                    # 3. 按 t_next 重加噪
                    t_next_batch = th.ones_like(t_batch) * t_next
                    t_next_exp = expand_dims(t_next_batch, x_cur.dim())
                    alpha_next, _ = path_sampler.compute_alpha_t(t_next_exp)
                    sigma_next, _ = path_sampler.compute_sigma_t(t_next_exp)

                    noi = th.randn_like(x_cur)
                    x_cur = alpha_next * x1_hat + sigma_next * (
                        x0_hat * ((1 - stochast_ratio) ** 0.5)
                        + noi * (stochast_ratio**0.5)
                    )

                return [x_cur]

            return _sample
