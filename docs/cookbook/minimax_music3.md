# MiniMax Music 3

[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) is a text-to-music model from MiniMax. Give it lyrics and a style description; it returns 32 kHz stereo song audio, vocals and instrumental together.

It is not a TTS model with a music voice. Generation runs in two stages. A Qwen3 backbone predicts one audio frame per decode step, and each frame is eight RVQ codebooks deep: the backbone's own `lm_head` emits `c0`, then a four-layer depth decoder walks `c1..c7`, each code conditioning the next. Every 200 frames are handed to a flow-matching DIT that solves a VAE latent, which a DAC-style decoder turns into waveform. So the knobs that shape a TTS request do nothing here — there is no reference speaker, no `voice`, no `temperature`. What you control is the lyrics, the caption, the seed, and the length.

| Component | Spec |
|---|---|
| Backbone | Qwen3 decoder (36 L, hidden=4096, GQA 32/8, vocab 200k) |
| Depth decoder | 4 L, hidden=4096, 16 heads; 7 residual codebooks of 1024 |
| Frame | 8 codebooks; `c0` in the backbone vocabulary, `c1..c7` in the depth decoder |
| AR guidance | Classifier-free, scale 1.5, `c0` masked to the conditioned branch's top 50 |
| Acoustic stage | Flow-matching DIT (36 L, dim=2048, 32 heads) + DAC decoder, 512x upsampling |
| Solver | Euler, 30 steps, acoustic CFG scale 1.7 |
| Window | 200 frames per acoustic chunk, 100-frame hop |
| Frame rate | 25 frames/second |
| Context length | 10,240 tokens |
| Output | 32 kHz stereo WAV |

## Prerequisites

Install `sglang-omni` from source as in [Installation](../get_started/installation.md).

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install -v -e .   # drop -e for a non-editable install
```

**Single GPU** (colocate both stages):

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

**Dual GPU** (AR on the first device, DIT/DAV on the second):

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

Default optimizations that are on without further flags: backbone decode CUDA graph, RVQ depth CUDA graph, compiled DIT blocks, compiled DAV decoder, and batched seeded sampling.

Classifier-free guidance is on in both stages and has no flag. See [Guidance](#guidance) for what it costs you, because the AR half changes how much a request occupies.

## Generating Music

Two fields carry the request. `input` is the lyrics; `instructions` is the caption that describes style, instrumentation, tempo and mood. Both are required, and both matter: the caption is what decides genre and arrangement, and the lyrics are what gets sung.

Structure tags in the lyrics (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`) steer the arrangement and are part of the prompt contract.

**Put a tag on its own line.** Normalization keeps only the tags on a line that starts with one and drops the rest of that line, so lyrics written next to a tag are silently lost:

```text
"[Verse]\nWalking down the street"   ->   [start] [verse] Walking down the street
"[Verse] Walking down the street"    ->   [start] [verse]
```

The second form generates a song with that line missing, and nothing warns you.

### A first song

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nSunday morning, quiet and slow\nSunlight through the half-drawn blinds\nCoffee cooling by the window\nNothing on my mind\n[Chorus]\nTake it easy, take it slow\nWe got nowhere else to go",
    "instructions": "A melancholic lo-fi hip-hop track with a mellow piano riff, soft vinyl crackle, and a slow steady drum beat at 85 BPM",
    "seed": 42,
    "max_new_tokens": 750
  }' \
  --output song_1.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964122/song_1.wav" type="audio/wav">
</audio>

`max_new_tokens` counts audio frames at 25 frames per second, so 750 frames is **at most** 30 seconds. It is a cap, not a target: the model ends the song itself when it emits the audio-end token. The request above typically fills the cap (~30 seconds). Raise the cap when you want room for an earlier natural ending; a response that stops short of the cap is the model finishing, not a truncation.

The response body is a 32 kHz stereo WAV.

In Python:

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Verse]\nSunday morning, quiet and slow\nSunlight through the half-drawn blinds\n[Chorus]\nTake it easy, take it slow\nWe got nowhere else to go",
        "instructions": "A melancholic lo-fi hip-hop track with a mellow piano riff at 85 BPM",
        "seed": 42,
        "max_new_tokens": 750,
    },
    timeout=600,
)
resp.raise_for_status()
with open("song_2.wav", "wb") as f:
    f.write(resp.content)
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964204/song_2.wav" type="audio/wav">
</audio>


Or with the OpenAI client, since this is the standard speech endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

with client.audio.speech.with_streaming_response.create(
    model="MiniMaxAI/MiniMax-Music3",
    voice="default",
    input="[Verse]\nSunday morning, quiet and slow\nSunlight through the half-drawn blinds\n[Chorus]\nTake it easy, take it slow",
    instructions="A melancholic lo-fi hip-hop track with a mellow piano riff at 85 BPM",
    response_format="wav",
    extra_body={"seed": 42, "max_new_tokens": 750},
) as response:
    response.stream_to_file("song_3.wav")
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964231/song_3.wav" type="audio/wav">
</audio>


### Writing the caption

The caption is the strongest control you have. Vague captions give generic arrangements; naming instruments, tempo and production character gives you the track you asked for.

```bash
# Genre, instrumentation, tempo, and a production note
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Chorus]\nWe are the fire that never dies\nBurning bright against the sky",
    "instructions": "An energetic arena rock anthem with distorted electric guitars, punchy live drums and a soaring male vocal at 130 BPM, wide stereo image, lightly compressed",
    "seed": 7,
    "max_new_tokens": 750
  }' \
  --output rock_1.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964267/rock_1.wav" type="audio/wav">
</audio>

Long structured captions work too. The model was trained on descriptions that cover global attributes, emotional progression, and vocal detail, and it uses all of it:

```python
import requests

caption = """Basic Attributes: bpm is 92, key is E minor, Electric Blues / Blues Rock.
Emotional Progression: confident and gritty from the outset, building tension through
call-and-response verses before releasing into an extended lead guitar solo.
Sonics: live and organic, warm mid-range, tube grit, relatively uncompressed so the
drums keep their natural attack.
Vocal: male baritone, gravelly and textured, conversational phrasing in the verses
turning melodic in the refrain."""

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Verse]\nI came up on a dirt road, nothing but a name\n[Chorus]\nAnd the thunder rolls the same",
        "instructions": caption,
        "seed": 11,
        "max_new_tokens": 1500,
    },
    timeout=600,
)
resp.raise_for_status()
with open("blues_1.wav", "wb") as f:
    f.write(resp.content)
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964323/blues_1.wav" type="audio/wav">
</audio>

The caption and lyrics are cleaned deterministically before tokenizing — Markdown residue is stripped and `<|tag value|>` forms are rewritten to `tag is value` — and the tokenized prompt is capped at 5,000 tokens.

### Instrumental and short clips

Lyrics are required and must be non-empty, so ask for an instrumental in the caption and keep the lyric line minimal:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Intro]\n(instrumental)",
    "instructions": "An instrumental ambient piece, no vocals: warm analog pads, slow evolving texture, distant piano, 70 BPM",
    "seed": 3,
    "max_new_tokens": 250
  }' \
  --output ambient_1.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964363/ambient_1.wav" type="audio/wav">
</audio>

250 frames caps the clip at 10 seconds, which is the fastest way to audition a caption before committing to a full-length render. Short caps are usually reached rather than ended early, so these come back at the full length.

### Reproducibility and variations

A request is deterministic in its seed: the same lyrics, caption, seed and length return byte-identical audio. Changing only the seed gives another take of the same song — the way to explore alternatives without touching the prompt.

```python
import requests

for seed in (1, 2, 3):
    resp = requests.post(
        "http://localhost:8000/v1/audio/speech",
        json={
            "model": "MiniMaxAI/MiniMax-Music3",
            "input": "[Verse]\nCity lights are calling out my name",
            "instructions": "A dreamy synthwave track with analog pads and a driving bassline at 110 BPM",
            "seed": seed,
            "max_new_tokens": 500,
        },
        timeout=600,
    )
    resp.raise_for_status()
    with open(f"take_{seed}.wav", "wb") as f:
        f.write(resp.content)
```

Omitting `seed` uses `0`, which is still deterministic — it is a fixed seed, not a random one.


<audio controls>
  <source src="https://github.com/user-attachments/files/30964419/take_3.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/user-attachments/files/30964420/take_2.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/user-attachments/files/30964418/take_1.wav" type="audio/wav">
</audio>

### Reference outputs

Five captions across five genres, rendered on a single H200 with the defaults this page documents and nothing else set. Each request is given in full, so pasting one back reproduces its clip.

Four of the five fill the 750-frame cap at 30.0 seconds. The J-pop one stops itself at 26.2 seconds, which is `max_new_tokens` behaving as a cap rather than a target — the model ended the song.

**Lo-fi hip-hop**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nWalking down the empty street at midnight\nStreetlights flicker like a broken dream\nI've got nothing but the sound of my own heartbeat\nEchoing through the silent concrete stream\n[Chorus]\nAnd I keep on walking\nTill the morning finds me\nLeave the night behind me",
  "instructions": "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, soft vinyl crackle, dusty boom-bap drums with a laid-back swing, warm upright bass. Intimate bedroom production, gentle tape saturation, no bright cymbals.",
  "seed": 1,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 00_lofi_hiphop.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963578/00_lofi_hiphop.wav" type="audio/wav">
</audio>

**J-pop**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nMorning light is spilling through the curtain\nEvery colour waking up with me\n[Chorus]\nRun into the day and never look back\nEverything we wanted is ahead of us",
  "instructions": "A cheerful J-pop song at 128 BPM in C major: bright acoustic piano, chiming electric guitar, punchy four-on-the-floor drums, and a clear female lead vocal. Polished modern pop production, wide stereo, energetic and uplifting.",
  "seed": 2,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 01_jpop_bright.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963575/01_jpop_bright.wav" type="audio/wav">
</audio>

**Synthwave, instrumental**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Intro]\n(instrumental)\n[Outro]\n(instrumental)",
  "instructions": "A moody synthwave instrumental at 100 BPM in D minor: pulsing analog bass arpeggio, gated reverb drum machine, wide atmospheric pads, and a soaring lead synth melody. Retro 1980s production, heavy chorus effect, cinematic and nocturnal.",
  "seed": 3,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 02_synthwave_moody.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963577/02_synthwave_moody.wav" type="audio/wav">
</audio>

**Acoustic folk**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nI came up on a dirt road, nothing but a name\nCarried all my summers in a canvas bag\n[Chorus]\nAnd the river keeps on running\nLike it never learned to stay",
  "instructions": "A gentle acoustic folk ballad at 76 BPM in G major: fingerpicked steel-string guitar, soft brushed snare, subtle cello underneath, and a warm male vocal close to the microphone. Sparse and organic, natural room sound, very little compression.",
  "seed": 4,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 03_acoustic_folk.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963579/03_acoustic_folk.wav" type="audio/wav">
</audio>

**Cinematic orchestral**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Intro]\n(instrumental)\n[Chorus]\nRise above the ashes of the fallen sky\nWe were never meant to say goodbye",
  "instructions": "An epic cinematic orchestral piece at 90 BPM in E minor: sweeping string ostinato, powerful brass swells, timpani and taiko percussion, and a distant choir. Wide concert-hall reverb, dynamic build from restrained to triumphant, no drum kit.",
  "seed": 5,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 04_orchestral_epic.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963576/04_orchestral_epic.wav" type="audio/wav">
</audio>


## Guidance

Both stages run classifier-free guidance, and neither is optional. There is no request field and no serve flag for it: the scale, the mask width and the solver's own scale are fixed in the model, because this is how the reference implementation samples this checkpoint, and dropping guidance makes the model follow your caption and lyrics noticeably less closely.

The two halves are separate mechanisms that happen to share a name:

- **The AR stage** runs each request twice per decode step. One row sees your real prompt; the other sees the same prompt with the whole caption-and-lyrics span, delimiters included, overwritten by `<|audio_cfg|>` — only the chat markers and `<|audio_start|>` survive. Every code is then drawn from `uncond + (cond - uncond) * 1.5`. For `c0` the candidate set is additionally masked to the conditioned row's own top 50 before the ordinary top-k sampling runs, so the guided draw cannot wander outside what the conditioned branch already considered plausible.
- **The acoustic stage** applies its own guidance at scale 1.7 inside every one of the 30 solver steps.

The consequence you have to plan for is on the AR side: **one request occupies two rows in the engine, not one.** Both rows hold their own KV cache for the whole song, so a request costs twice the KV it would unguided. The visible follow-on is in [Concurrency](#concurrency).

What it does *not* change is the request contract or where the randomness comes from: the guided draw is still keyed by the same per-request seed and frame position, so seeding behaves as described above. Guidance does shift *which* codes get drawn, so audio from a build without it is not comparable take-for-take — expect a different arrangement and a different length from the same seed, not a cleaned-up version of the same song.

## Concurrency

The server batches continuously. Admission defaults to 16 concurrent requests (`max_running_requests=16`), which is **32 decode rows**, because guidance gives every request a second row. Raise admission at serve time with `--max-running-requests`; the row count, the decode CUDA graph and the RVQ depth graph are all derived from it:

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000 \
  --max-running-requests 32
```

That serves 32 concurrent requests as 64 rows. Do not pass `--cuda-graph-max-bs` here: this model computes the cap itself so the graphs always cover the doubled batch, and a value you supply is discarded rather than honoured.

Send clients in parallel rather than in sequence:

```python
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    ("[Verse]\nMorning breaks over the harbour", "A gentle acoustic folk song with fingerpicked guitar at 90 BPM"),
    ("[Chorus]\nDance until the record stops", "A disco house track with four-on-the-floor drums and funk guitar at 122 BPM"),
    ("[Verse]\nSnow falls quiet on the pines", "A cinematic orchestral piece with strings and soft horns at 60 BPM"),
]


def render(index_and_prompt):
    index, (lyrics, caption) = index_and_prompt
    resp = requests.post(
        "http://localhost:8000/v1/audio/speech",
        json={
            "model": "MiniMaxAI/MiniMax-Music3",
            "input": lyrics,
            "instructions": caption,
            "seed": index,
            "max_new_tokens": 750,
        },
        timeout=900,
    )
    resp.raise_for_status()
    with open(f"track_{index}.wav", "wb") as f:
        f.write(resp.content)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(render, enumerate(PROMPTS)))
```

Concurrency is where this model is most efficient. The depth decoder costs the same per step whether one request or eight share the batch, so its per-request share falls as concurrency rises.

That flatness is also why guidance is cheap in time and expensive in memory. The AR step is bound by reading the backbone's weights rather than by arithmetic, so the second row rides along in the same pass with little extra wall time, while doubling the KV each request holds. Size the pool for the rows, not the requests.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | (required) | Lyrics. Non-empty. Use `[Verse]` / `[Chorus]` / `[Bridge]` / `[Outro]` on their own lines |
| `instructions` | string | (required) | Caption describing genre, instrumentation, tempo, mood and production. Non-empty |
| `seed` | int | `0` | Non-negative 64-bit integer. Fixes the output for a given request |
| `max_new_tokens` | int | `9000` | Cap on audio frames, 25 per second. Maximum 9,000 (six minutes). The model may finish earlier |
| `response_format` | string | `"wav"` | Output container |
| `stream` | bool | `false` | Must be `false`; this model's external API is non-streaming |

Reaching the cap returns the audio generated so far. Ending on the audio-end token stops the song early.

## Parameters That The Model Rejects

The request is refused rather than silently ignored, so a mistake is visible immediately:

| Parameter | Reason |
|---|---|
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | Sampling is fixed: guidance at scale 1.5, then top-k 50 with a per-request seeded draw. Rejected when set explicitly |
| `voice` | There is no speaker to select; the vocal comes from the caption |
| `ref_audio`, `ref_text`, `language`, `task_type` | No reference-audio conditioning and no language tag in this contract |
| `speed` | Only `1.0`. Tempo belongs in the caption, e.g. "at 92 BPM" |
| `stream: true` | Not supported. This model's external API is non-streaming |

## Notes

**Length and cost.** Generation time scales with `max_new_tokens`. A 10-second clip is the cheapest way to iterate on a caption; render the full song once the style is right.

**Prompt sensitivity.** The prompt's token ids and the position of `<|audio_start|>` seed the backbone KV cache, so whitespace and tag rewrites are not cosmetic — they change the audio. If you need a result to be reproducible, keep the lyrics and caption byte-identical.

**Memory.** `mem_fraction_static` defaults to `0.50` and budgets the SGLang backbone KV cache only; the acoustic stage's DIT and DAV weights sit outside that fraction. It is a share of total device memory, so the absolute KV pool shrinks on smaller cards automatically. Lowering it does not reliably help — measured at `0.35` the pipeline is about 6% slower. Budget against rows rather than requests: guidance means the pool has to hold two sequences per concurrent request, so raising `--max-running-requests` costs KV twice as fast as the number suggests.

**Attention backends.** The DIT accepts `auto`, `torch_sdpa`, `fa` and `sage_attn` through `runtime_overrides`; `torch_sdpa` is the default and measured the fastest of them for these shapes. `cache_dit` is an approximate, default-off option that trades audio quality for speed, and reducing the solver's `dit_steps` below 30 is measurably a quality reduction rather than a free win.
