import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


def _runner():
    runner = MiniCPMOThinkerModelRunner.__new__(MiniCPMOThinkerModelRunner)
    runner._pending_hidden = {}
    return runner


@pytest.mark.parametrize("speech_enabled", [False, True])
def test_bootstrap_wires_abort_cleanup(monkeypatch, speech_enabled):
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.models.minicpm_o import bootstrap as minicpm_bootstrap
    from sglang_omni.models.minicpm_o import request_builders, thinker_model_runner
    from sglang_omni.scheduling import bootstrap
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    runner = _runner()
    runner._pending_hidden = {"aborted": [torch.ones(4)], "other": [torch.zeros(4)]}
    monkeypatch.setattr(
        thinker_model_runner, "MiniCPMOThinkerModelRunner", lambda *args: runner
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            object(),
            None,
            None,
            None,
            SimpleNamespace(model_path="model", vocab_size=100),
        ),
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        request_builders,
        "make_thinker_scheduler_adapters",
        lambda **kwargs: (None, None),
    )

    scheduler_signature = inspect.signature(OmniScheduler.__init__)

    def init_scheduler(scheduler, **kwargs):
        scheduler_signature.bind(scheduler, **kwargs)
        scheduler._abort_callback = kwargs.get("abort_callback")

    monkeypatch.setattr(OmniScheduler, "__init__", init_scheduler)
    scheduler = minicpm_bootstrap.create_thinker_scheduler(
        SimpleNamespace(disable_cuda_graph=True), speech_enabled=speech_enabled
    )
    assert scheduler._abort_callback == runner.reset_request
    scheduler._run_abort_callback("aborted")
    scheduler._run_abort_callback("aborted")
    scheduler._run_abort_callback("unknown")
    assert set(runner._pending_hidden) == {"other"}


def test_finish_flushes_cloned_hidden_states_and_abort_drops_them():
    runner = _runner()
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    scheduler_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="finished")]
    )
    outputs = {"finished": SimpleNamespace(extra={"hidden_states": hidden})}
    runner.post_process_outputs(None, scheduler_output, outputs)
    expected = hidden[-1].clone()
    hidden.zero_()
    assert outputs["finished"].extra == {}

    request_data = SimpleNamespace(extra_model_outputs={})
    runner.on_request_finished("finished", request_data)
    runner.reset_request("finished")
    assert not runner._pending_hidden
    sequence = request_data.extra_model_outputs["hidden_states_seq"]
    assert len(sequence) == 1
    torch.testing.assert_close(sequence[0], expected)
    assert sequence[0].device.type == "cpu"

    runner._pending_hidden["aborted"] = [torch.ones(4)]
    runner.reset_request("aborted")
    aborted_data = SimpleNamespace(extra_model_outputs={})
    runner.on_request_finished("aborted", aborted_data)
    assert aborted_data.extra_model_outputs == {}
    assert not runner._pending_hidden


def test_middle_prefill_chunk_does_not_capture_hidden_or_advance_state():
    runner = _runner()
    middle_req = SimpleNamespace(
        request_id="middle",
        data=SimpleNamespace(req=SimpleNamespace(inflight_middle_chunks=1)),
    )
    final_req = SimpleNamespace(
        request_id="final",
        data=SimpleNamespace(req=SimpleNamespace(inflight_middle_chunks=0)),
    )
    scheduler_output = SimpleNamespace(requests=[middle_req, final_req])
    outputs = {
        "middle": SimpleNamespace(extra={"hidden_states": torch.ones(1, 4)}),
        "final": SimpleNamespace(extra={"hidden_states": torch.zeros(1, 4)}),
    }

    runner.post_process_outputs(None, scheduler_output, outputs)

    assert set(runner._pending_hidden) == {"final"}
    assert outputs["middle"].extra == {}
    assert outputs["final"].extra == {}
    assert runner.finalize_skip_rids(scheduler_output) == {"middle"}


def test_single_request_prefill_preserves_all_hidden_rows():
    class ForwardMode:
        def is_extend(self):
            return True

    req = SimpleNamespace(extend_range=SimpleNamespace(length=3))
    scheduler_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="req")],
        batch_data=SimpleNamespace(
            reqs=[req],
            forward_mode=ForwardMode(),
        ),
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    selected = SGLangOutputProcessor._slice_per_request_tensor(
        hidden,
        request_index=0,
        scheduler_output=scheduler_output,
    )

    torch.testing.assert_close(selected, hidden)
