# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the runtime-resolution channel.

The invariants under test:

* a field the runtime filled in (configured ``None``, resolved concrete) is
  reported; a field that passed through unchanged is not — a resolution is a
  change of value, not an echo;
* ``require_resolved`` refuses an unresolved "auto" with a message naming
  the safe derivation site, and passes concrete values through;
* runtime observations never become patches: ``SourceKind.RUNTIME`` has no
  layer, so patch construction from it fails structurally;
* provenance renders runtime entries — on touched paths as an extra line,
  on untouched paths without the historical ``KeyError``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.provenance import ProvenanceMap
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime_resolution import (
    RuntimeResolution,
    RuntimeResolutionRecord,
    capture_runtime_resolutions,
    require_resolved,
)
from sglang_omni.config.schema import PipelineConfig

MEM_FRACTION = "stages.thinker.engine.mem_fraction_static"
CHUNK = "stages.thinker.engine.chunked_prefill_size"

CLI = ConfigSource(SourceKind.CLI_DOTTED, "command line")

HARDWARE = "sglang ServerArgs hardware resolution"


def server_args(**fields):
    """A stand-in for a constructed ServerArgs: just resolved attributes."""
    return SimpleNamespace(**fields)


def runtime(path, resolved, configured=None):
    return RuntimeResolution(
        path=path, configured=configured, resolved=resolved, origin=HARDWARE
    )


class TestCapture:
    def test_reports_a_field_the_runtime_filled_in(self):
        resolutions = capture_runtime_resolutions(
            {"chunked_prefill_size": None},
            server_args(chunked_prefill_size=2048),
        )
        assert [(r.path, r.configured, r.resolved) for r in resolutions] == [
            ("chunked_prefill_size", None, 2048)
        ]

    def test_absent_key_counts_as_auto(self):
        """ServerArgs defaults the field to None itself; not writing it is
        the same request as writing null."""
        resolutions = capture_runtime_resolutions(
            {}, server_args(chunked_prefill_size=8192)
        )
        assert resolutions[0].configured is None
        assert resolutions[0].resolved == 8192

    def test_explicit_value_passing_through_is_not_reported(self):
        resolutions = capture_runtime_resolutions(
            {"chunked_prefill_size": 4096},
            server_args(chunked_prefill_size=4096),
        )
        assert resolutions == []

    def test_explicit_value_the_runtime_rewrote_is_reported(self):
        """SGLang force-disables the field to -1 under some configurations;
        that rewrite is exactly what an operator needs to see."""
        resolutions = capture_runtime_resolutions(
            {"chunked_prefill_size": 4096},
            server_args(chunked_prefill_size=-1),
        )
        assert [(r.configured, r.resolved) for r in resolutions] == [(4096, -1)]

    def test_missing_attribute_is_skipped(self):
        assert capture_runtime_resolutions({}, server_args()) == []

    def test_origin_names_the_resolver(self):
        (resolution,) = capture_runtime_resolutions(
            {}, server_args(mem_fraction_static=0.85)
        )
        assert resolution.origin == HARDWARE

    def test_cuda_graph_max_bs_reads_the_nested_decode_config(self):
        """ServerArgs has no cuda_graph_max_bs attribute; the omni override
        is aliased at construction and the resolved value lands on
        cuda_graph_config.decode.max_bs."""
        args = server_args(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=160))
        )
        (resolution,) = capture_runtime_resolutions({}, args)
        assert (resolution.path, resolution.resolved) == ("cuda_graph_max_bs", 160)


class TestRequireResolved:
    def test_passes_a_concrete_value_through(self):
        assert require_resolved(2048, field_name="chunked_prefill_size") == 2048

    def test_refuses_auto_and_names_the_safe_site(self):
        with pytest.raises(ValueError, match="finalize_runtime_derived"):
            require_resolved(None, field_name="chunked_prefill_size")

    def test_error_names_the_field(self):
        with pytest.raises(ValueError, match="chunked_prefill_size"):
            require_resolved(None, field_name="chunked_prefill_size")


class TestRecord:
    def test_record_and_get(self):
        record = RuntimeResolutionRecord()
        record.record("thinker", [runtime("chunked_prefill_size", 2048)])
        assert record.get("thinker")[0].resolved == 2048

    def test_unknown_stage_is_empty(self):
        assert RuntimeResolutionRecord().get("talker") == []

    def test_get_returns_a_copy(self):
        record = RuntimeResolutionRecord()
        record.record("thinker", [runtime("chunked_prefill_size", 2048)])
        record.get("thinker").clear()
        assert len(record.get("thinker")) == 1


class TestNeverAPatch:
    def test_runtime_source_has_no_layer(self):
        """The structural guarantee: SourceKind.RUNTIME is deliberately
        absent from the layer table, so a runtime observation cannot be
        turned into a patch and can never enter precedence."""
        with pytest.raises(KeyError):
            ConfigSource(SourceKind.RUNTIME, "launch").layer


class TestProvenanceRuntimeChannel:
    def resolved_with_patch(self, pipeline_config: PipelineConfig):
        patch = ConfigPatch.create(MEM_FRACTION, "0.8", CLI, root=type(pipeline_config))
        return ConfigResolver(pipeline_config).resolve(ConfigPatchSet([patch]))

    def test_untouched_path_no_longer_key_errors(self):
        provenance = ProvenanceMap()
        provenance.record_runtime(CHUNK, runtime(CHUNK, 2048))
        text = provenance.explain(CHUNK)
        assert f"{CHUNK} = 2048" in text
        assert "runtime (sglang ServerArgs hardware resolution)" in text

    def test_unknown_path_still_key_errors(self):
        with pytest.raises(KeyError):
            ProvenanceMap().explain(CHUNK)

    def test_paths_includes_runtime_only_paths(self):
        provenance = ProvenanceMap()
        provenance.record_runtime(CHUNK, runtime(CHUNK, 2048))
        assert provenance.paths() == [CHUNK]
        assert provenance.runtime_resolved(CHUNK)
        assert not provenance.touched(CHUNK)

    def test_touched_path_appends_a_runtime_line(self, pipeline_config: PipelineConfig):
        resolved = self.resolved_with_patch(pipeline_config)
        resolved.provenance.record_runtime(
            MEM_FRACTION, runtime(MEM_FRACTION, 0.85, configured=0.8)
        )
        text = resolved.provenance.explain(MEM_FRACTION)
        lines = text.splitlines()
        # The launch has the last word: headline and final line report it,
        # with the patch history intact in between.
        assert lines[0] == f"{MEM_FRACTION} = 0.85"
        assert "[winner]" in text
        assert lines[-1].strip() == (
            "0.85  <- runtime (sglang ServerArgs hardware resolution)"
        )

    def test_no_runtime_line_without_a_runtime_entry(
        self, pipeline_config: PipelineConfig
    ):
        resolved = self.resolved_with_patch(pipeline_config)
        assert "runtime" not in resolved.provenance.explain(MEM_FRACTION)
