# Full-duplex audio examples

Run commands from the repository root.

## MiniCPM-o

Download the checkpoint and start the server:

```bash
hf download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5

sgl-omni serve --config examples/full_duplex/minicpmo.yaml \
  --model-path models/MiniCPM-o-4_5 --enable-realtime
```

The server accepts live voice and video conversations on `/v1/realtime`. See [docs/cookbook/minicpm_o.md](../../docs/cookbook/minicpm_o.md) for a client example and the per-session settings, and [playground/realtime](../../playground/realtime/README.md) for a browser page.

Two configs are provided:

| Config | Use it for |
|---|---|
| `minicpmo.yaml` | Normal serving; replies are sampled the way the MiniCPM-o demo samples them |
| `minicpmo-parity.yaml` | Repeatable output for regression and parity recordings; it differs in greedy sampling, `top_k: 100`, and running the thinker and talker without CUDA graphs |

Settings you may want to change in the config:

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 2 | Conversations served at the same time. Startup warms up perception at each batch size up to this value |
| `reference_audio` | checkpoint default | Voice used when a session sends no reference |
| `speech_state_bytes_per_session` | 2 GiB | Memory the speech stage may hold per conversation; a conversation that needs more is closed and the others keep running |
| `stages.thinker/talker.engine.enable_torch_compile` | `false` | Compiles every decode graph batch size; adds minutes to startup |
| `sampling.*` | see file | Default sampling for sessions that do not set their own |
| `vision.*` | see file | Limits on camera frames per unit (1 s of audio) |

One conversation can hold 8192 tokens of history, which is the model's limit. When a conversation fills it, the server sends a `context_exhausted` error and closes that session.
