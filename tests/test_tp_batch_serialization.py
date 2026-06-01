# SPDX-License-Identifier: Apache-2.0
"""Regression tests for TP follower batch serialization (pickle safety)."""
import pickle
import types
import unittest
import weakref
from enum import Enum, auto

import torch

from sglang_omni.engines.tp.serialization import make_follower_batch


class _FakeModality(Enum):
    IMAGE = auto()

    @classmethod
    def from_str(cls, value):
        return cls[value.upper()]

    @staticmethod
    def all():
        return [_FakeModality.IMAGE]


def _make_mock_batch():
    """Create a mock ModelWorkerBatch-like object with unpicklable fields."""
    batch = types.SimpleNamespace()
    batch.input_ids = torch.tensor([1, 2, 3])
    batch.seq_lens = torch.tensor([3])

    # sampling_info with weakref (the actual bug trigger)
    # Need a class that supports weakrefs (object/list don't in Python 3.12+)
    class _Referent:
        pass

    target = _Referent()
    penalizer = types.SimpleNamespace(batch_ref=weakref.ref(target))
    batch.sampling_info = types.SimpleNamespace(
        penalizer_orchestrator=penalizer,
        sampling_info_done=None,
    )

    batch.reqs = [lambda: None]
    return batch


class TestMakeFollowerBatch(unittest.TestCase):
    def test_make_follower_batch_is_pickle_safe(self):
        batch = _make_mock_batch()
        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)
        self.assertTrue(torch.equal(restored.input_ids, batch.input_ids))
        self.assertTrue(torch.equal(restored.seq_lens, batch.seq_lens))
        self.assertIsNone(restored.sampling_info)
        self.assertIsNone(restored.reqs)

    def test_make_follower_batch_preserves_tensors(self):
        batch = _make_mock_batch()
        follower = make_follower_batch(batch)
        self.assertTrue(torch.equal(follower.input_ids, batch.input_ids))
        self.assertTrue(torch.equal(follower.seq_lens, batch.seq_lens))

    def test_make_follower_batch_nulls_unsafe_fields(self):
        batch = _make_mock_batch()
        follower = make_follower_batch(batch)
        self.assertIsNone(follower.sampling_info)
        self.assertIsNone(follower.reqs)

    def test_make_follower_batch_leaves_original_intact(self):
        batch = _make_mock_batch()
        original_sampling_info = batch.sampling_info
        original_reqs = batch.reqs
        make_follower_batch(batch)
        self.assertIs(batch.sampling_info, original_sampling_info)
        self.assertIs(batch.reqs, original_reqs)

    def test_stop_signal_broadcast_still_works(self):
        data = pickle.dumps(None)
        self.assertIsNone(pickle.loads(data))

    def test_shape_primitives_are_pickle_safe(self):
        """TP shape attrs on the batch must not break pickle verification."""
        import sglang_omni.engines.tp.serialization as ser

        ser._pickle_verified = False
        batch = _make_mock_batch()
        batch.tp_input_embeds_shape = (3, 128)
        batch.tp_input_embeds_dtype = torch.bfloat16
        batch.tp_deepstack_shapes = [(2, 128)]
        batch.tp_deepstack_dtype = torch.bfloat16
        batch.tp_visual_pos_mask_shape = (3,)

        follower = make_follower_batch(batch)
        restored = pickle.loads(pickle.dumps(follower))
        self.assertEqual(restored.tp_input_embeds_shape, (3, 128))
        self.assertEqual(restored.tp_input_embeds_dtype, torch.bfloat16)
        self.assertEqual(restored.tp_deepstack_shapes, [(2, 128)])
        self.assertEqual(restored.tp_visual_pos_mask_shape, (3,))

        ser._pickle_verified = False

    def test_mm_flag_is_pickle_safe(self):
        import sglang_omni.engines.tp.serialization as ser

        ser._pickle_verified = False
        batch = _make_mock_batch()
        batch.tp_has_mm_payload = True

        follower = make_follower_batch(batch)
        restored = pickle.loads(pickle.dumps(follower))
        self.assertTrue(restored.tp_has_mm_payload)

        ser._pickle_verified = False

    def test_new_unpicklable_field_raises_clear_error(self):
        """Verify the runtime safety net catches new unpicklable fields."""
        import sglang_omni.engines.tp.serialization as ser

        ser._pickle_verified = False

        batch = _make_mock_batch()
        batch.bad_callback = lambda: None

        with self.assertRaises(RuntimeError) as ctx:
            make_follower_batch(batch)
        self.assertIn("bad_callback", str(ctx.exception))

        ser._pickle_verified = False


class TestCpuSanitizer(unittest.TestCase):
    """Rank 0 must move tensors to CPU before broadcast and leave original batch intact."""

    def test_nested_tensors_rebuilt_without_mutating_original(self):
        class MMInput:
            pass

        mm = MMInput()
        mm.mrope_positions = torch.tensor([0, 1, 2])
        mm.mrope_position_delta = torch.tensor([1])

        batch = _make_mock_batch()
        page_table_rows = [torch.tensor([10, 11, 12], dtype=torch.int32)]
        batch.tp_page_table_rows = page_table_rows
        batch.multimodal_inputs = [mm]

        follower = make_follower_batch(batch)

        self.assertIsNot(follower.tp_page_table_rows, page_table_rows)
        self.assertIs(batch.tp_page_table_rows, page_table_rows)
        self.assertTrue(torch.equal(follower.tp_page_table_rows[0], page_table_rows[0]))
        self.assertIsNot(follower.multimodal_inputs[0], mm)
        self.assertIs(batch.multimodal_inputs[0], mm)
        self.assertTrue(
            torch.equal(
                follower.multimodal_inputs[0].mrope_positions, mm.mrope_positions
            )
        )
        self.assertTrue(
            torch.equal(
                follower.multimodal_inputs[0].mrope_position_delta,
                mm.mrope_position_delta,
            )
        )

    def test_enum_fields_are_not_recursed_into(self):
        mm_item = types.SimpleNamespace()
        mm_item.modality = _FakeModality.IMAGE
        mm_item.feature = torch.tensor([1.0])

        batch = _make_mock_batch()
        batch.multimodal_inputs = [types.SimpleNamespace(mm_items=[mm_item])]

        follower = make_follower_batch(batch)
        restored = pickle.loads(pickle.dumps(follower))

        self.assertIs(
            restored.multimodal_inputs[0].mm_items[0].modality,
            _FakeModality.IMAGE,
        )
        self.assertTrue(
            torch.equal(
                restored.multimodal_inputs[0].mm_items[0].feature,
                torch.tensor([1.0]),
            )
        )

    def test_cycle_handled_via_memo(self):
        batch = _make_mock_batch()
        a = {"tensor": torch.tensor([1.0])}
        a["self"] = a
        batch.cycle = a

        follower = make_follower_batch(batch)

        self.assertIsNot(follower.cycle, a)
        self.assertIs(follower.cycle["self"], follower.cycle)


class TestInputEmbedsPickle(unittest.TestCase):
    def test_input_embeds_survives_follower_batch_round_trip(self):
        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.input_embeds = torch.randn(3, 128)

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertIsNotNone(restored.input_embeds)
        self.assertTrue(torch.equal(restored.input_embeds, batch.input_embeds))

    def test_input_embeds_none_when_not_set(self):
        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        embeds = getattr(restored, "input_embeds", None)
        self.assertIsNone(embeds)


class TestDeepstackPickle(unittest.TestCase):
    def test_deepstack_payload_survives_round_trip(self):
        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.input_embeds = torch.randn(3, 128)
        batch.tp_deepstack_visual_embeds = [torch.randn(2, 64), torch.randn(2, 64)]
        batch.tp_visual_pos_masks = torch.tensor([True, False, True])

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertEqual(len(restored.tp_deepstack_visual_embeds), 2)
        self.assertTrue(
            torch.equal(
                restored.tp_deepstack_visual_embeds[0],
                batch.tp_deepstack_visual_embeds[0],
            )
        )
        self.assertTrue(
            torch.equal(restored.tp_visual_pos_masks, batch.tp_visual_pos_masks)
        )


class TestShapeAttrsPickle(unittest.TestCase):
    """Shape primitives let follower alloc NCCL receive buffers pre-broadcast."""

    def test_shape_attrs_survive_pickle_round_trip(self):
        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.tp_input_embeds_shape = (3, 128)
        batch.tp_input_embeds_dtype = torch.bfloat16
        batch.tp_deepstack_shapes = [(2, 128), (2, 128)]
        batch.tp_deepstack_dtype = torch.bfloat16
        batch.tp_visual_pos_mask_shape = (3,)

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertEqual(restored.tp_input_embeds_shape, (3, 128))
        self.assertEqual(restored.tp_input_embeds_dtype, torch.bfloat16)
        self.assertEqual(restored.tp_deepstack_shapes, [(2, 128), (2, 128)])
        self.assertEqual(restored.tp_deepstack_dtype, torch.bfloat16)
        self.assertEqual(restored.tp_visual_pos_mask_shape, (3,))

    def test_shape_attrs_absent_for_text_only(self):
        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertIsNone(getattr(restored, "tp_input_embeds_shape", None))
        self.assertIsNone(getattr(restored, "tp_deepstack_shapes", None))
        self.assertIsNone(getattr(restored, "tp_visual_pos_mask_shape", None))

    def test_shape_attrs_with_only_input_embeds(self):
        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.tp_input_embeds_shape = (3, 64)
        batch.tp_input_embeds_dtype = torch.float16
        batch.tp_deepstack_shapes = None
        batch.tp_deepstack_dtype = None
        batch.tp_visual_pos_mask_shape = None

        follower = make_follower_batch(batch)
        restored = pickle.loads(pickle.dumps(follower))

        self.assertEqual(restored.tp_input_embeds_shape, (3, 64))
        self.assertIsNone(restored.tp_deepstack_shapes)
        self.assertIsNone(restored.tp_visual_pos_mask_shape)


def test_attach_page_table_snapshot():
    import torch

    from sglang_omni.engines.tp.serialization import attach_page_table_snapshot

    pool = types.SimpleNamespace()
    pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
    pool.req_to_token[2, 0:5] = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)

    batch = types.SimpleNamespace()
    batch.req_pool_indices = torch.tensor([2])
    batch.seq_lens = torch.tensor([5])

    attach_page_table_snapshot(batch, pool)

    assert hasattr(batch, "tp_page_table_rows")
    assert len(batch.tp_page_table_rows) == 1
    assert torch.equal(
        batch.tp_page_table_rows[0],
        torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
    )


def test_page_table_snapshot_survives_pickle():
    import pickle

    import torch

    from sglang_omni.engines.tp.serialization import attach_page_table_snapshot

    pool = types.SimpleNamespace()
    pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
    pool.req_to_token[1, 0:3] = torch.tensor([7, 8, 9], dtype=torch.int32)

    batch = types.SimpleNamespace()
    batch.req_pool_indices = torch.tensor([1])
    batch.seq_lens = torch.tensor([3])
    batch.sampling_info = None
    batch.reqs = None

    attach_page_table_snapshot(batch, pool)
    data = pickle.dumps(batch)
    restored = pickle.loads(data)

    assert len(restored.tp_page_table_rows) == 1
    assert torch.equal(
        restored.tp_page_table_rows[0],
        torch.tensor([7, 8, 9], dtype=torch.int32),
    )


if __name__ == "__main__":
    unittest.main()
