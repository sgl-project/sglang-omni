# SPDX-License-Identifier: Apache-2.0
"""Reuse private Ascend KV for one-position VoiceChat units."""

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule, get_spec
from sglang.srt.session.streaming_session import StreamingSession


class VoiceChatAscendStreamingSession(StreamingSession):
    def try_match_prefix(self, params: MatchPrefixParams) -> MatchResult | None:
        request = params.req
        slot = self.find_active_slot(request)
        if slot is None:
            return None
        else:
            pass

        native_session = request.session
        record = slot.kv
        prefix_tokens = len(params.key)
        # note (celestial-micha): The retained Mamba state belongs to the last completed unit.
        if (
            self.cache.disable
            and self.cache.tree_components
            in ((ComponentType.FULL,), (ComponentType.FULL, ComponentType.MAMBA))
            and not self.cache.enable_mamba_extra_buffer
            and get_parallel().attn_dcp_size == 1
            and get_disagg().disaggregation_mode == "null"
            and get_schedule().disable_overlap_schedule
            and get_spec().speculative_algorithm is None
            and prefix_tokens > 0
            and prefix_tokens == native_session.committed_origin_len
            and prefix_tokens == record.kv_committed_len == record.kv_allocated_len
            and len(request.full_untruncated_fill_ids) == prefix_tokens + 1
            and request.sampling_params.max_new_tokens == 1
            and request.retraction_count == 0
            and record.cache_protected_len == record.max_evicted_seqlen == 0
            and record.holds_mamba == self.cache.is_mamba_enabled
            and record.mamba_ping_pong_track_buffer is None
            and not record.mamba_needs_clear
            and record.mamba_cow_src_index is None
            and record.retraction_backup is None
        ):
            request.kv = record
            return MatchResult(
                device_prefix_len=prefix_tokens,
                last_device_node=slot.virtual_node,
                last_host_node=slot.virtual_node,
                best_match_node=slot.virtual_node,
                cache_protected_len=0,
            )
        else:
            if request.kv is record:
                request.detach_kv()
            else:
                pass
            self.release_session(native_session.session_id)
            return None
