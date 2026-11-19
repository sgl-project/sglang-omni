# SPDX-License-Identifier: Apache-2.0
"""Optional block-FP8 dense backend for Drama 3.

Enabled by ``--tts_engine.factory.fp8_dense true``. Nothing here runs on the
default path: ``prepare_fp8_dense_backend()`` is the only entry point, and it
imports ``fish_scales_ops`` lazily so a plain install never needs the library.

When enabled, bf16 dense linears are quantized to 128x128 block-FP8 at weight
load time (the same dialect as drama-3's ``bsgemm-moe`` experts) and served by
the library's GEMM. The sm_90 kernels are JIT-compiled by NVRTC on first use;
``FsoDenseLinearMethod`` compiles them once during weight loading instead, so
the first request never pays for it. Any layer the backend cannot serve falls
back to the inherited bf16 path.
"""

from __future__ import annotations

import glob
import logging
import os
import site
import sysconfig
from functools import lru_cache

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

NVRTC_ENV = "FSO_JIT_NVRTC_LIB"


def _site_roots() -> list[str]:
    roots = [sysconfig.get_paths()["purelib"]]
    try:
        roots.extend(site.getsitepackages())
    except Exception:  # pragma: no cover - virtualenvs without getsitepackages
        pass
    return list(dict.fromkeys(roots))


def _ensure_nvrtc() -> str | None:
    """Point the sm_90 JIT at the CUDA 13 NVRTC torch ships.

    An in-place ``fish_scales_ops`` build has no bundled ``_nvrtc/`` copy. When
    the caller has not chosen a library, reuse the one installed with torch
    instead of requiring another environment variable. Libraries of another
    toolkit are never guessed: NVRTC 13.0 is what the kernels are built for.
    """
    if os.environ.get(NVRTC_ENV):
        return os.environ[NVRTC_ENV]
    for root in _site_roots():
        patterns = (
            os.path.join(root, "nvidia", "cu13", "lib", "libnvrtc.so.13"),
            os.path.join(root, "nvidia", "*", "lib", "libnvrtc.so.13"),
        )
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                builtins = os.path.join(
                    os.path.dirname(path), "libnvrtc-builtins.so.13.0"
                )
                if os.path.isfile(builtins):
                    os.environ[NVRTC_ENV] = path
                    return path
    return None


@lru_cache(maxsize=1)
def prepare_fp8_dense_backend() -> bool:
    """Check the optional backend and set up its JIT. False keeps bf16 dense."""
    try:
        import fish_scales_ops as fso
    except Exception as exc:
        logger.warning(
            "fp8_dense was requested but fish_scales_ops is unavailable (%s); "
            "using bf16 dense linears.",
            exc,
        )
        return False
    try:
        supported = fso.dense.supported("bsfp8")
    except Exception as exc:
        logger.warning(
            "fp8_dense was requested but the backend could not be probed (%s); "
            "using bf16 dense linears.",
            exc,
        )
        return False
    if not supported:
        logger.warning(
            "fp8_dense was requested but this GPU cannot serve bsfp8; "
            "using bf16 dense linears."
        )
        return False

    nvrtc = _ensure_nvrtc()
    if nvrtc is None:
        logger.warning(
            "fp8_dense: no CUDA 13 NVRTC found; set %s to a "
            "libnvrtc.so.13 or run scripts/vendor_nvrtc.py in the "
            "fish-scales-ops tree.",
            NVRTC_ENV,
        )
    else:
        logger.info("fp8_dense: sm_90 JIT NVRTC %s", nvrtc)
    # describe() loads the JIT compiler and never raises; its last line names
    # the compiler in use, so a broken NVRTC is visible at startup.
    try:
        logger.info(
            "fp8_dense: %s", fso.dense.describe().splitlines()[-1]
        )
    except Exception:  # pragma: no cover - describe() itself is defensive
        pass
    return True


class FsoDenseLinearMethod(UnquantizedLinearMethod):
    """Serve one bf16 dense linear through block-FP8 GEMM, with bf16 fallback."""

    _warmup_done = False
    _fallback_count = 0
    _degraded_logged = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        try:
            import fish_scales_ops as fso

            weight = fso.dense.prepare_weight(
                layer.weight.data, format="bsfp8"
            )
        except Exception as exc:
            layer.fso_weight = None
            FsoDenseLinearMethod._fallback_count += 1
            if FsoDenseLinearMethod._fallback_count == 1:
                logger.warning(
                    "fp8_dense: a layer fell back to bf16 (%s); later "
                    "fallbacks are logged at debug level.",
                    exc,
                )
            else:
                logger.debug("fp8_dense: layer fell back to bf16 (%s)", exc)
            return
        layer.fso_weight = weight

        if not FsoDenseLinearMethod._warmup_done:
            FsoDenseLinearMethod._warmup_done = True
            try:
                # Compile the sm_90 kernels now rather than on the first
                # request. Four rows are enough for the decode tile.
                x = torch.zeros(
                    4,
                    layer.weight.shape[1],
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                fso.dense.linear(x, weight)
                torch.cuda.synchronize()
                logger.info("fp8_dense: JIT kernels compiled at startup")
            except Exception as exc:
                logger.warning(
                    "fp8_dense: JIT warmup failed (%s); kernels will compile "
                    "on first use.",
                    exc,
                )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = getattr(layer, "fso_weight", None)
        if weight is None:
            return super().apply(layer, x, bias)
        try:
            import fish_scales_ops as fso

            if x.dim() != 2:
                leading = x.shape[:-1]
                y = fso.dense.linear(
                    x.reshape(-1, x.shape[-1]), weight, bias=bias
                )
                return y.reshape(*leading, -1)
            return fso.dense.linear(x, weight, bias=bias)
        except Exception as exc:
            layer.fso_weight = None
            if not FsoDenseLinearMethod._degraded_logged:
                FsoDenseLinearMethod._degraded_logged = True
                logger.warning(
                    "fp8_dense: GEMM failed (%s); this layer now uses bf16.",
                    exc,
                )
            return super().apply(layer, x, bias)
