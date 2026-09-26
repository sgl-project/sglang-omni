# CosyVoice3 continuous decode (draft)

## Goal

Run several sequential AR forwards before reading sampled tokens on the host. The next forward consumes the preceding token on the GPU. At the end of a bounded window, the host commits tokens only through the first stop token for each request.

This differs from SGLang's historical `num_continuous_decode_steps`: that loop postponed receiving new requests but still processed each decode result and checked EOS on the host after every step. Merely enabling that option, or repeating Omni's current `run_batch`, does not remove the per-token readback.

## Required state transition

1. Reserve each inner step's KV slot and advance its position before the next forward.
2. Update sampling history on the device after each sampled token, including Cosy's repetition penalty and minimum generation length. A fixed request seed is paired with the changing token position.
3. Record each window's sampled tokens and the first of Cosy's stop/control tokens. Commit only the accepted prefix to output, streaming chunks, and request state.
4. Release KV according to the accepted length. Drain the window before admitting a conflicting prefill, abort, or audio chunk boundary.

The first implementation should use a bounded window for one Cosy request. Multi-request filtering and merge behavior are a separate step. The feature remains off until codec tokens and WAV output pass a real-service A/B comparison.

## Existing evidence and limits

On one A100, a readback-boundary replay using saved Cosy stop positions took 116/68/61 ms for 24 short C1 requests with one/four/eight-step readback windows. With 24 longer C1 requests it took 414/241/209 ms. C16 and C32 fixed-wave replays showed the same boundary trend. Their stop positions are inferred as one step after the recorded speech-token count. The included benchmark and anonymized step counts reproduce this synthetic CUDA measurement; it excludes Cosy forwards, KV, Flow, vocoder, scheduling, and streaming. It is evidence that the boundary is worth testing, **not** a serving speedup or a correctness result.

The saved single-step service trace contains 72 C1 requests at one, two, and four times the original text length. All completed through the normal stop token. Their request-mean full-WAV RTFs were 0.114, 0.099, and 0.093 respectively. No continuous-decode service output exists yet.

An offline stop-boundary replay used those 72 C1 requests and the saved C16/C32 speech-token counts. The first non-final audio flush is at 28 speech tokens; later flushes are 25 tokens apart. A four-step window can cross a flush by up to three tokens. Ending the four-step window at each flush removes that delay. Waiting for a full audio chunk can overshoot EOS by up to 24 model steps. This checks only boundary accounting, not generated-token parity:

| Trace | Four-step extra steps after EOS | Four-step capped at chunk | Next-chunk window |
| --- | ---: | ---: | ---: |
| C1, 72 requests | 115 / 18,881 (0.61%) | 82 (0.43%) | 835 (4.42%) |
| C16, 64 requests | 54 / 8,002 (0.67%) | 118 (1.47%) | 615 (7.69%) |
| C32, 64 requests | 119 / 8,009 (1.49%) | 67 (0.84%) | 483 (6.03%) |
