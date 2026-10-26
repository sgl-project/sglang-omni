# Nemotron ASR runtime

For deployment and client examples, see the
[Nemotron ASR cookbook](../cookbook/nemotron3_5_asr.md).

## Input processing

The 20 ms native unit is an input operation, not a model window or a first
text latency guarantee. Deployment callers can configure `native_unit_ms`
on `create_realtime_deployment`. Short units complete without waiting for a
future window. Long appends drain all currently runnable windows.

## Scheduling

Each ASR replica owns one engine and model thread. Different sessions share
model batches while retaining separate attention, convolution and RNN-T
caches. Concurrent hook threads and one inbox bridge thread feed the engine.
If `session_max_concurrency` is smaller than `max_batch_size`, the public
path cannot submit a full batch. Replicas do not share batches. Offline and streaming work
alternate at batch boundaries when both are ready. A long offline forward
cannot be preempted. `max_batch_wait_ms` is a collection deadline, not an
end-to-end latency bound.

## Resource limits

Model task, PCM, cache reservation, token history and text budgets are
bounded. Usage reports actual persistent cache/history/PCM bytes. Batch
temporaries and shared weights are measured separately at process level.
When an explicit internal-session token limit is reached, later PCM is
validated and counted toward duration without further inference or buffering.
Budget errors after stage acceptance terminate the stream. They are not a
promise that retrying the same seq is safe. Public admission rejection follows
the shared runtime's sequence/retry contract.

## Persistent encoder state pool

Streaming preallocates a bounded pool for attention and convolution history.
Each session leases one slot, and batches address state by slot ID, preserving
stable device storage across audio windows.

Capacity is limited by `max_open_sessions` and `max_state_bytes`. A budget too
small for one slot fails at startup. Completion, cancellation, failure and
shutdown release slots, which are cleared before reuse. The pool is the sole
streaming encoder state representation. Offline transcription is unchanged.
