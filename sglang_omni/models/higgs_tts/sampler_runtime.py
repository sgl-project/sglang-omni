# SPDX-License-Identifier: Apache-2.0
"""Sampler runtime state for Higgs TTS."""

from __future__ import annotations

import torch

from sglang_omni.models.higgs_tts.sampler import K_MAX, HiggsBatchedSamplerState


class HiggsSamplerRuntime:
    def __init__(
        self,
        *,
        max_batch_size: int,
        num_codebooks: int,
        device: torch.device,
    ) -> None:
        self._max_batch_size = int(max_batch_size)
        pool_size = self._max_batch_size + 1
        self._sampler_pool = HiggsBatchedSamplerState(
            max_batch_size=pool_size,
            num_codebooks=num_codebooks,
            device=device,
        )
        self._padding_row = self._max_batch_size
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(self._max_batch_size))
        self._output_codes: dict[str, list[torch.Tensor]] = {}

        self._cg_row_indices = torch.zeros(pool_size, dtype=torch.long, device=device)
        self._cg_temperature = torch.ones(pool_size, dtype=torch.float32, device=device)
        self._cg_top_p = torch.ones(pool_size, dtype=torch.float32, device=device)
        self._cg_top_k_buf = torch.full(
            (pool_size,),
            K_MAX,
            dtype=torch.long,
            device=device,
        )
        self._cg_codes_BN = torch.zeros(
            pool_size, num_codebooks, dtype=torch.long, device=device
        )
        # Note(Jiaxin): Packs codes_BN | was_done | active_generation_done into one buffer.
        self._cg_collect_staging = torch.zeros(
            pool_size, num_codebooks + 2, dtype=torch.long, device=device
        )
        self._cg_was_done = torch.zeros(pool_size, dtype=torch.bool, device=device)

        self._cg_active_delay_count = torch.zeros(
            pool_size, dtype=torch.int32, device=device
        )
        self._cg_active_eoc_countdown = torch.full(
            (pool_size,), -1, dtype=torch.int32, device=device
        )
        self._cg_active_generation_done = torch.zeros(
            pool_size, dtype=torch.bool, device=device
        )
        self._cg_active_last_codes = torch.zeros(
            pool_size, num_codebooks, dtype=torch.long, device=device
        )

    def acquire_row(self, req_id: str) -> int:
        """Allocate or look up the sampler-pool row for ``req_id``. Idempotent."""
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            raise RuntimeError(
                f"HiggsTTSModel sampler pool exhausted (max_batch_size="
                f"{self._max_batch_size}); raise ``max_batch_size`` or limit "
                f"concurrent requests."
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._sampler_pool.reset_row(row)
        return row

    def release_row(self, req_id: str) -> None:
        """Return ``req_id``'s row to the free pool and drop its output codes."""
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)

    def reset_request(self, req_id: str) -> None:
        self.release_row(req_id)

    def get_output_codes(
        self,
        req_id: str,
        *,
        num_codebooks: int,
        device: torch.device,
    ) -> torch.Tensor:
        codes = self._output_codes.get(req_id)
        if not codes:
            return torch.empty(
                (0, num_codebooks),
                dtype=torch.long,
                device=device,
            )
        return torch.stack(codes, dim=0).to(torch.long)

    def append_sampled_codes(
        self,
        req_ids: list[str],
        codes_BN: torch.Tensor,
        was_done_cpu: list[bool],
    ) -> None:
        for b, req_id in enumerate(req_ids):
            if was_done_cpu[b]:
                continue
            self._output_codes.setdefault(req_id, []).append(codes_BN[b])

    def pack_decode_collect_staging(self, n_real: int) -> torch.Tensor:
        """Scatter CG shadow state back into the pool and pack collect staging."""
        rows_t = self._cg_row_indices[:n_real]
        pool = self._sampler_pool
        pool.delay_count[rows_t] = self._cg_active_delay_count[:n_real]
        pool.eoc_countdown[rows_t] = self._cg_active_eoc_countdown[:n_real]
        pool.generation_done[rows_t] = self._cg_active_generation_done[:n_real]
        pool.last_codes[rows_t] = self._cg_active_last_codes[:n_real]

        # Note(Jiaxin): pack the 3 tensors so a single D2H pulls them all back.
        num_codebooks = self._cg_codes_BN.shape[1]
        staging = self._cg_collect_staging
        staging[:n_real, :num_codebooks] = self._cg_codes_BN[:n_real]
        staging[:n_real, num_codebooks] = self._cg_was_done[:n_real]
        staging[:n_real, num_codebooks + 1] = self._cg_active_generation_done[:n_real]
        return staging
