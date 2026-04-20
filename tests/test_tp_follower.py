# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TP follower worker."""
from __future__ import annotations

import pickle
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestFollowerRegistration(unittest.TestCase):
    def test_register_delegates_to_shared_helper(self):
        from sglang_omni.engines.tp.follower import register_omni_models

        with patch(
            "sglang_omni.models.sglang_registry.register_omni_models_in_sglang"
        ) as mock_reg:
            register_omni_models()
            mock_reg.assert_called_once()


class TestFollowerBatchFlow(unittest.TestCase):
    def _make_sanitized_batch(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        original = types.SimpleNamespace()
        original.reqs = [MagicMock(), MagicMock()]
        original.input_ids = torch.tensor([10, 20, 30])
        original.seq_lens = torch.tensor([3])
        original.sampling_info = MagicMock()
        return make_follower_batch(original)

    def test_sanitized_batch_has_none_reqs(self):
        follower = self._make_sanitized_batch()
        self.assertIsNone(follower.reqs)

    def test_sanitized_batch_has_none_sampling_info(self):
        follower = self._make_sanitized_batch()
        self.assertIsNone(follower.sampling_info)

    def test_stop_signal_survives_pickle(self):
        data = pickle.dumps([None])
        result = pickle.loads(data)
        self.assertIsNone(result[0])

    def _call_patch_batch(self, batch, vocab_size=32000):
        import torch

        mock_sbi_cls = MagicMock()
        mock_sbi_cls.side_effect = lambda **kw: types.SimpleNamespace(**kw)
        with (
            patch(
                "sglang_omni.engines.tp.follower.SamplingBatchInfo",
                mock_sbi_cls,
                create=True,
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.sampling.sampling_batch_info": MagicMock(
                        SamplingBatchInfo=mock_sbi_cls
                    )
                },
            ),
        ):
            from sglang_omni.engines.tp.follower import patch_batch_for_follower

            device = torch.device("cpu")
            patch_batch_for_follower(batch, device, vocab_size=vocab_size)

    def test_patch_batch_fills_reqs(self):
        batch = self._make_sanitized_batch()
        self._call_patch_batch(batch)
        self.assertEqual(batch.reqs, [])

    def test_patch_batch_creates_sampling_stub(self):
        batch = self._make_sanitized_batch()
        self._call_patch_batch(batch, vocab_size=32000)

        si = batch.sampling_info
        self.assertIsNotNone(si)
        self.assertTrue(si.is_all_greedy)
        self.assertEqual(si.vocab_size, 32000)

    def test_relocate_batch_tensors(self):
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3], device="cpu")
        batch.seq_lens = torch.tensor([3], device="cpu")

        target = torch.device("cpu")
        relocate_batch_tensors(batch, target)
        self.assertEqual(batch.input_ids.device, target)
        self.assertEqual(batch.seq_lens.device, target)

    def test_relocate_moves_input_embeds(self):
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        batch = types.SimpleNamespace()
        batch.input_embeds = torch.randn(3, 128)
        target = torch.device("cpu")
        relocate_batch_tensors(batch, target)
        self.assertEqual(batch.input_embeds.device, target)

    def test_relocate_moves_deepstack_tensors(self):
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        batch = types.SimpleNamespace()
        batch.tp_deepstack_visual_embeds = [torch.randn(2, 64), torch.randn(2, 64)]
        batch.tp_visual_pos_masks = torch.tensor([True, False])
        target = torch.device("cpu")
        relocate_batch_tensors(batch, target)
        for t in batch.tp_deepstack_visual_embeds:
            self.assertEqual(t.device, target)
        self.assertEqual(batch.tp_visual_pos_masks.device, target)

    def test_sync_page_table_writes_to_pool(self):
        import torch

        from sglang_omni.engines.tp.follower import sync_page_table

        pool = types.SimpleNamespace()
        pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)

        batch = types.SimpleNamespace()
        batch.req_pool_indices = torch.tensor([2])
        batch.seq_lens = torch.tensor([3])
        batch.tp_page_table_rows = [
            torch.tensor([10, 11, 12], dtype=torch.int32),
        ]

        sync_page_table(batch, pool)

        expected = torch.tensor([10, 11, 12], dtype=torch.int32)
        assert torch.equal(pool.req_to_token[2, 0:3], expected)

    def test_sync_page_table_noop_without_snapshot(self):
        import torch

        from sglang_omni.engines.tp.follower import sync_page_table

        pool = types.SimpleNamespace()
        pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        batch = types.SimpleNamespace()
        batch.req_pool_indices = torch.tensor([0])
        batch.seq_lens = torch.tensor([1])

        sync_page_table(batch, pool)

        assert pool.req_to_token.sum() == 0


class TestFollowerGpuAssignment(unittest.TestCase):
    def test_spawn_followers_passes_model_config(self):
        from sglang_omni.engines.tp.follower import spawn_followers

        started = []

        class DummyProcess:
            def __init__(self, target, args, daemon):
                self.target = target
                self.args = args
                self.daemon = daemon
                self.pid = 12345

            def start(self):
                started.append(self.args)

        class DummyContext:
            def Process(self, target, args, daemon):
                return DummyProcess(target, args, daemon)

        server_args = types.SimpleNamespace(gpu_id_step=1)
        with patch(
            "sglang_omni.engines.tp.follower.mp.get_context",
            return_value=DummyContext(),
        ):
            processes = spawn_followers(
                server_args=server_args,
                nccl_port=23456,
                base_gpu_id=0,
                tp_size=2,
                model_arch_override="BailingMoeV2ForCausalLM",
                weight_prefix="thinker.",
            )

        self.assertEqual(len(processes), 1)
        self.assertEqual(
            started[0],
            (
                1,
                1,
                server_args,
                23456,
                "BailingMoeV2ForCausalLM",
                "thinker.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
