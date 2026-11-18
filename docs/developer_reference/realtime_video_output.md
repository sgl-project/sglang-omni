# Realtime video output

The shared `/v1/realtime` WebSocket supports a video-capable deployment through SGLang extension events. Each video delta contains one independently decodable JPEG or PNG frame. This is a frame transport, not an H.264 stream or a model integration. Existing text/audio deployments do not acquire video capabilities automatically.

## Negotiation and output events

Request video alongside any supported output modalities:

```json
{"type":"session.update","session":{"output_modalities":["audio","video"]}}
```

Check `session.updated.session.output_modalities` and `session.updated.session.sglang.granted.output_video_format`. The latter is `{"type":"image/jpeg"}` or `{"type":"image/png"}` and is fixed by the deployment. Unsupported modalities follow the existing negotiation policy: a partial grant reports rejections, and a request with no supported modality fails. Clients must inspect the grant.

A frame event has the following form:

```json
{
  "type": "sglang.response.video.delta",
  "event_id": "evt_...",
  "response_id": "resp_0",
  "item_id": "item_0",
  "output_index": 0,
  "format": "image/jpeg",
  "delta": "<base64-encoded JPEG frame>",
  "sglang": {
    "media_time": {"t_start_ms": 1000.0, "duration_ms": 41.666666666666664}
  }
}
```

The adapter emits `ResponseStarted` before media, `VideoFinished` when the video stream ends, and `ResponseFinished(has_video=True, ...)` to finish the response. These become `response.created`, `sglang.response.video.done`, and `response.done`; the final message includes an `output_video` content descriptor. The video extensions identify one video stream by response and item, without assuming a content index for text/audio that may arrive later. They are not standard OpenAI Realtime events.

The optional `sglang.unit_id` and `sglang.chunk_seq` describe the producing input unit and delivery sequence. They do not determine output presentation time. Video frames always carry their own finite nonnegative start time and positive duration; the WebSocket preserves them even when an input unit is attached.

## Connecting a producer

A native realtime adapter declares `Capabilities(output_modalities=("audio", "video"), max_granted_modalities=2, output_video_format="image/jpeg")` and emits `VideoDelta(response_id, item_id, encoded_frame, "image/jpeg", t_start_ms, duration_ms)` through its existing output sink. The runtime checks the declared format. Media encoding belongs to the model/codec stage; the API does not decode or re-encode frames.

For a Session pipeline, add `video_output=VideoPort("image/jpeg")` to its existing `SessionPorts`. `SessionBridge` then accepts `OutputChunk` values with `modality="video"`, `format="image/jpeg"`, an encoded frame as the byte payload, and the frame's presentation time and duration. Use `turn.started`/`turn.done`/`turn.interrupted` controls consistently with the existing response lifecycle. Frame `eos` does not finish the entire response independently of its audio. Graph composition requires one published video format.

Video-enabled SessionBridge deployments also preserve their audio chunks' `t_start_ms` and `duration_ms` in `response.output_audio.delta.sglang.media_time`. Native adapters can provide the same optional fields on `AudioDelta`. The producer must use one output clock for both modalities, measured in milliseconds from a consistent session media origin. An audio/video graph must provide authoritative output timestamps; the existing audio-oriented `shared_timeline=True` rewriting is not an audio/video clock and should remain disabled for this path.

## Buffering, cancellation, and limits

The existing outbound byte budget counts base64 video bytes and event metadata. Overflow uses the existing session failure/cleanup path. A cancelled response drops its queued audio and video deltas; closing a session preserves completion events for visible video while purging unsent frames. Already delivered frames cannot be withdrawn: clients should discard pending playback for a cancelled response. Changes to requested modalities apply to new responses, preserving each active response's negotiated modalities.

Clients need a video renderer and a bounded playback queue, ideally synchronized to the output audio clock. The existing realtime playground does not yet render these video events. JPEG/PNG keeps frames independently decodable but adds bandwidth and base64 overhead; this change makes no codec or real-time throughput claim. It also does not add stereo audio negotiation, H3 generation, KV rollback, or shared audio/video graph pacing.

Run the CPU protocol and bridge tests with `python -m pytest tests/unit_test/serve/test_realtime_video.py tests/unit_test/serve/test_realtime_duplex_session.py tests/unit_test/serve/test_realtime_duplex_runtime.py tests/unit_test/serve/test_session_bridge.py`. The tests exercise the real ASGI WebSocket path with deterministic media producers; they do not validate model generation quality or GPU performance.
