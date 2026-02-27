# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph support for DualAR models — removed after profiling.

Explicit CUDA graphs (2-graph and mega-graph approaches) were explored but
torch.compile(mode="reduce-overhead") proved superior (~9.3x vs 2.6x best
graph approach).  See fish_audio.md for profiling results.

This file is intentionally empty.  The torch.compile path lives in
``dual_ar.py:decode_one_token`` and is activated via ``--compile``.
"""
