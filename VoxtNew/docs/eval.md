# Evaluating the curated models (M3)

VoxtNew ships a few curated models with parameters we tune, so their quality is measured, not assumed. This page has the eval set, how to run it, the numbers from mac-air, and the decisions they led to. The models and their settings are in [llm-models.md](llm-models.md) and `internal/omni/models.go`; how VoxtNew differs from Voxt is in [behavior.md](behavior.md).

## The eval set

`testdata/eval/`: 29 short clips (169 s, 5.3 MB of 16 kHz mono WAV) and two long ones joined from them.

| set | clips | what | example |
|---|---|---|---|
| `zh-filler` | 5 | Chinese dictation with fillers and self-corrections | 嗯，那个，我们下周二，呃不对，下周三下午三点开个会，讨论一下预算。 |
| `zh-plain` | 1 | Chinese without either | 今天的天气很好，我们下午去公园散步，晚上一起吃饭吧。 |
| `en` | 5 | English, two with fillers and corrections | Let's ship the release on Friday, no wait, on Thursday morning. |
| `mix` | 4 | Chinese with English words | 我们的服务用 Kubernetes 部署，CPU 占用太高了，需要 scale out。 |
| `dict` | 5 | the dictionary's terms (made-up product names, jargon) | VoxtNew 的新版本用 MLX 在本地跑语音模型。 |
| `dict2` | 9 | the user's own terms (Voxt, Omni, SGLang), people's names (陈喆, 韩煜, 周芷若, 欧阳明远), a term at the end of the sentence, the terms said as a list, and a sentence ending in a filler (#24) | 我们用 Omni。 |
| `long` | 2 | 95 s (13 Chinese and mixed clips) and 104 s (14 clips, starting in English), 0.8 s of silence between clips | |

- **Each clip** has its words (`text`), the expected clean-up (`clean`) and checks: phrases the clean-up keeps (`must`) or drops (`mustNot`: fillers, what a self-correction replaced), the same for an English translation of the Chinese clips (`mustEn`, `mustNotEn`), its dictionary terms (`terms`), and for the dictionary clips the transcript a speech model wrote without the dictionary (`heard`).
- **The dictionary** of every run but #24's is the set's nine terms (VoxtNew, MLX, AudioSmith, Pixcall, ActNow, Silero, SGLang, Qwen3, Auralink), with one alias ("Or Link" for Auralink).
- **The voices** are Qwen3-TTS (0.6B CustomVoice, Apache-2.0: Vivian, Serena and Uncle_Fu for Chinese, Ryan and Aiden for English), run by AudioSmith's engine through `internal/tools/evalsynth`. That tool writes files and never plays them. macOS `say` was the first idea, but its voices' terms do not clearly allow redistributing their output. The clips are therefore studio-clean synthetic speech: no noise, no accents, no real hesitation. Real dictation will score worse.
- **Each clip was checked** against both speech models. Takes that the TTS truncated or garbled were made again with another seed (`zh-filler-2`, `zh-filler-5`, `dict-5`). Two quirks remain:
  - the TTS reads 部署 in `dict-4` so that both models hear 注入;
  - "UI" in `mix-4` sometimes comes out as 友爱.
- **The speech sample** `testdata/speech/sample-zh-en.wav` was made again the same way. The old take said "Auralink"; the new one says "VoxtNew", as its golden text does, though its voice says it close to "Fox New", which is what both models write, even with the dictionary.

### Scores (`internal/eval`)

- **Error rate.** Both texts are normalised first: full-width characters, case, Chinese numerals as digits ("三百五十" = "350"), English number words ("twelve" = 12), and punctuation and fillers dropped from both sides. The error rate is then edits over the reference's tokens. A token is a Chinese character or a Latin word. So the rate is a CER on Chinese, a WER on English, and both on mixed speech.
- **Terms written** counts the clip's dictionary terms that the text writes exactly as the dictionary does.
- **Clean-up checks.** An answer has a fault when:
  - fillers are left;
  - a correction is missed: it keeps a replaced phrase, or loses a kept one;
  - it invents more than one token found in neither the input nor the expected answer (numbers aside);
  - it changes the language: its share of Chinese moves more than 30 points from the input's;
  - it copies the example: 4 Chinese characters or 5 words in a row from compose's example answer, absent from the input.

## Running it

On mac-air, from the persistent models and runtimes:

| path | what | refresh |
|---|---|---|
| `~/.cache/huggingface/hub` (`$HUB`) | the six catalogue models' snapshots at their pinned commits, about 9 GB, in the ActNow family's model library (shared with the user's AudioSmith and Dicta; claimed as `dev.hehehai.voxtnew`); the tests find them through the catalogue (`models.Installed`) | download in VoxtNew, or `internal/models` `TestRealMoveToHub` from per-model folders; never delete them by hand |
| `~/voxtnew-runtime/{omni,llm}/` | `scripts/build-omni.sh --out` and `scripts/build-llm.sh --out` from `voxtnew/dev` | its `README.md` |

```sh
# the whole eval, about 37 minutes; tables in build/eval (asr.md, llm.md, glossary.md, long.md, rewrite.md)
MAC_DIR=code/VoxtNew-xyz scripts/mac.sh 'VOXTNEW_EVAL=1 VOXTNEW_OMNI_RUNTIME=$HOME/voxtnew-runtime/omni \
  VOXTNEW_LLM=$HOME/voxtnew-runtime/llm/voxtnew-llm VOXTNEW_EVAL_OUT=build/eval go test -count=1 -timeout 2h -run TestEval -v .'
# quick: a clip of each kind, checked against floors (part of scripts/mac-regress.sh)
… VOXTNEW_EVAL_QUICK=1 …
# some clips only
… VOXTNEW_EVAL_ONLY=dict-1,dict-5 …
# other language models: a list, or all = every catalogue language model in $HUB (the borrowed
# llm.qwen3, llm.gemma4 when another ActNow app installed them); default VoxtNew's own three
… VOXTNEW_EVAL_LLM=all …   … VOXTNEW_EVAL_LLM=llm.qwen3,llm.gemma4 …
```

The tests are:
- `TestEvalASR`: `eval_asr_test.go`;
- `TestEvalLLM` and `TestEvalGlossary`: `eval_llm_test.go` and `eval_glossary_test.go`;
- `TestEvalLong` (the long clips: Transcribe whole and in pieces, the prompt cache's hits, long spoken Translate; the full eval only): `eval_long_test.go`;
- `TestEvalRewrite` (Rewrite on 8 cases, F1): `eval_rewrite_test.go`;
- `TestEvalAppEnhancement` and `TestEvalDictionary`: `eval_e2e_test.go`;
- `TestEvalDictASR` and `TestEvalDictLLM` (the dictionary's echo and worth, #24; `VOXTNEW_EVAL_ASR` picks the speech models): `eval_dictasr_test.go`, `eval_dictllm_test.go`, with the dictionaries in `eval_dict_test.go`.

The language-model tests also write the process's memory after each model's chats (`llm.md`: load time, footprint and its peak, MLX's peak).

They log a `CONTENTION` line when another `voxtnew-llm` or `qwen3_asr_server` was running before they started, since that skews the times. The numbers below come from runs without one at their start. mac-air is shared, though: other apps' engines (AudioSmith, Dicta) and another thread's runs were seen during the day, with swap in use. Read the times as upper bounds, and compare them only with runs on the same machine. `scripts/mac-regress.sh` (manual-test.md §10.6) runs the quick eval with the other Mac tests.

## Speech recognition

mac-air (M4, 24 GB, macOS 26.7), 2026-10-17; Finals with the eval dictionary as the context bias.

### Error rate by language hint

The user's main language is Chinese in all of these. "main zh" is Voxt's policy and VoxtNew's: the main language is always the hint. "none" lets the model detect the language. "main en" is the hint a user whose main language is English sends.

| clips (tokens) | `asr.accurate` main zh | none | main en | `asr.fast` main zh | none | main en |
|---|---|---|---|---|---|---|
| zh-filler (118) | 1.7% | 1.7% | 1.7% | 2.5% | 2.5% | 1.7% |
| zh-plain (23) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| en (72) | **0.0%** | 0.0% | 0.0% | **1.4%** | 1.4% | 1.4% |
| mix (70) | 2.9% | 2.9% | 0.0% | 11.4% | 11.4% | 8.6% |
| dict (59) | 11.9% | 11.9% | 11.9% | 20.3% | 20.3% | 20.3% |
| long-zh, 95 s | 1.6% | 1.6% | 1.6% | 6.3% | 6.3% | 6.3% |
| long-all, 104 s, starts in English | **2.9%** | **60.5%** | 60.1% | 4.2% | 3.8% | 3.8% |
| all (832) | **2.6%** | 19.1% | 18.8% | 6.0% | 5.9% | 5.5% |

- **The main language as the hint costs nothing on English.** Every English clip came out in English under "zh", on both models. Qwen3-ASR does not translate English to fit a Chinese hint.
- **Detection can go badly wrong on a mixed dictation.** Without a hint, `asr.accurate` took the long clip that starts in English for English, then translated all of its Chinese into English (60% errors). `asr.fast` did not.
- **A wrong hint does the same.** With the hint "en", `asr.fast` wrote two Chinese clips in English: 39% errors on `zh-filler`. That was in an earlier run with other takes of `zh-filler-2` and `zh-filler-5`; the final takes did not show it.

**Decision: keep (a), Voxt's policy.** The main language is the hint even when the user lists other languages (`session.speechLang`, `TestSpeechLang`). Option (b), no hint when other languages are set, would gain nothing on these clips and loses whole sentences on mixed speech. The open risk is the mirror case: a user whose main language is English and who dictates Chinese on `asr.fast` (Decisions needed in the handoff).

### The dictionary as context bias

Terms written exactly, of the 10 in the `dict` clips:

| | with the dictionary | without |
|---|---|---|
| `asr.accurate` | **7** | 4 |
| `asr.fast` | **4** | 3 |

- "VoxtNew" is fixed by the bias on `asr.accurate`: without it the model writes "Vox Tunew" or "Vox New".
- "SGLang" is fixed too: "S G Long" without it.
- "ActNow" (written "Act Now") and "Auralink" ("Or Link") are not fixed with nine terms in the context.
- In a whole session with only three terms, `asr.accurate` did write "Auralink". A small dictionary biases harder.
- `asr.fast` gains less from the bias.

### Times

| | `asr.accurate` (1.7B 8-bit) | `asr.fast` (0.6B 4-bit) |
|---|---|---|
| first Final (cold: the server starts and loads) | 1.16 s | 0.68 s |
| warm Final, short clips (6.5 s of audio on average) | median 648 ms, max 974 ms | median 218 ms, max 322 ms |
| Final of a long clip (95–104 s) | 8.6 s | 2.8 s |
| live: first partial after the speech starts (clip sent in real time) | median 153 ms, max 167 ms | median 69 ms, max 86 ms |
| live: the Final after Stop | median 672 ms, max 1.1 s | median 228 ms, max 327 ms |

- **Long dictation waits.** The live session's Final runs again over all the audio (Voxt's "offline Final is authoritative", behavior.md 3.7). A 100-second dictation on `asr.accurate` therefore waits 8.6 s after Stop.
- **Idle unload.** The 10-minute idle unload stays. A cold start costs 0.7–1.2 s, and it overlaps the user's speech, because the opener warms the model when the session starts.

## Clean-up with the small language models

Transcribe's clean-up of each clip's words, as said, with the user's main language from the clip and the eval dictionary as the glossary. Before M3, compose sent the clean-up rules alone; M3 adds an example turn (`compose.Example`). The example is a short dictation and its clean answer, sent as an earlier user and assistant message, one per template set:

| set | example input | example answer |
|---|---|---|
| zh-Hans | 呃，帮我把那个周报，嗯，发给小王，不对，是发给小李，今天下班前。 | 帮我把周报发给小李，今天下班前。 |
| en | Um, can you send the, uh, the weekly report to Tom, sorry, I mean to Anna, before five? | Can you send the weekly report to Anna before 5? |

### Transcribe, 20 short clips

| model | prompt | no fault | fillers left | corrections missed | copied the example | error vs expected | first token (median) | total (median) |
|---|---|---|---|---|---|---|---|---|
| `llm.balanced` | rules only | 18 | 0 | 2 | – | 7.0% | 1.89 s | 2.66 s |
| `llm.balanced` | **with the example** | **18** | 0 | 2 | 0 | **5.6%** | 2.05 s | 2.72 s |
| `llm.fast` | rules only | 14 | 4 | 6 | – | 13.6% | 0.93 s | 1.17 s |
| `llm.fast` | **with the example** | **16** | **1** | **3** | 0 | **9.9%** | 1.01 s | 1.23 s |

- **Both models keep two faults.** 就是说 is kept as a word. 那个报告周五之前发给我就行，不对，是周四之前 keeps both days: a correction at the end of the sentence.
- **`llm.fast` also has its own faults.** With the example, it once dropped the number it should have kept ("三百，五百" for "三百五十").

### Spoken Translate into English, 6 Chinese clips

| model | prompt | no fault | median total |
|---|---|---|---|
| `llm.balanced` | rules only | 4 | 3.37 s |
| `llm.balanced` | **with the example** | **6** | 3.64 s |
| `llm.fast` | rules only | 2 | 1.35 s |
| `llm.fast` | **with the example** | **6** | 1.38 s |

The faults without the example were the self-corrections kept and "uh" left in. No answer copied the example. A check that took three shared words for a copy ("Please send the report to me by Thursday") was too strict: it now needs five words in a row.

**Decision: the example turn is in.** It applies to Transcribe and spoken Translate, with Cleanup on, for both template sets. For Translate, the example is sent only when the target is English or Simplified Chinese, the two languages the example's answer is written in.
- `llm.fast` gains the most: its faultless clean-ups go from 14 to 16 of 20, and its translations from 2 to 6 of 6.
- `llm.balanced`'s translations go from 4 to 6 of 6. Its clean-ups stay at 18 of 20, a little closer to the expected text.
- It costs about 80–160 ms more of the first token: the example is about 60 more prompt tokens.
- compose's golden prompts show the turn (`internal/compose/testdata/golden`).

### Long dictation (two clips, 95–104 s; M3b, `TestEvalLong`)

The two long clips by their words: `long-zh` (441 characters, Chinese with English words) and `long-all` (742 characters, whole sentences alternating between English and Chinese, starting in English). M3 found that `llm.balanced` translated all of `long-all` into English. M3b measured where that starts, and how to cut the text so that it does not.

**Where a whole clean-up fails.** Whole clean-ups of the long clips' first 3, 5, 7… clips (`llm.balanced`):

| text | characters | result |
|---|---|---|
| `long-zh`, first 3 to 11 clips | 99–362 | no fault (error 7–13%) |
| `long-all`, first 3 clips | 192 | returned as it was: every filler kept |
| `long-all`, first 5 to 13 clips | 297–716 | translated into English (45–55% errors) |

The trigger is sentences that switch language, not length: Chinese with English words is fine at any length measured, while 192 characters of alternating sentences already fail.

**How to cut.** `long-all` on `llm.balanced`, each piece with the same system prompt and example turn:

| cut | requests | faults | error vs expected |
|---|---|---|---|
| whole | 1 | translated into English | 58.5% |
| pieces of at most 300 characters at sentence ends | 3 | the first piece (English, Chinese, English…) translated | 25.1% |
| pieces of at most 150 characters | 6 | two pieces translated | 32.4% |
| 300-character pieces, the English template (main language "en") | 3 | translated, fillers kept | 42.0% |
| **where whole sentences change language (and at most 300 characters)** | 10 | one self-correction at a sentence's end kept (周五…不对，周四, as on the short clip) | **6.3–7.7%** |

**Decision: Transcribe's clean-up is cut where the language changes, at any length, and by length over 400 characters** (`compose.LanguagePieces`, `BuildParts`; behavior.md 5.31). Since 10-17 the length cut starts at 300 characters (below, "Long clean-ups under memory pressure").
- A sentence counts as Chinese when most of its words are Chinese characters, so "我们的服务用 Kubernetes 部署" stays with the Chinese.
- A dictation in one language under 400 characters is one request, the same as before: all 20 short clips, as said and as heard, are one piece.
- On `long-zh` (one language, 441 characters: two pieces) `llm.balanced` scores the same whole and in pieces: no fault, 7.3%.

**What it costs, with M2b's prompt cache.** Every piece has the same system prompt and example turn, and the session warms that prefix up while the user speaks (`warmLLM`). So every piece's request starts from the cache: 10 of 10 hits on `long-all`, 2 of 2 on `long-zh`. Times on mac-air with the cache, after the warm-up, as the session runs:

| `llm.balanced` | whole | in pieces |
|---|---|---|
| `long-all` (742 characters) | 10.7 s, translated into English | **13.8 s** (10 requests), language kept; the first piece's answer after **1.6 s** |
| `long-zh` (441 characters) | 11.6 s | 12.0 s (2 requests); the first after 5.6 s |

- **Without the cache** (`voxtnew-llm` before M2b, under load), the 10 pieces took 63 s against 12.8 s whole: the cache is what makes the language cuts affordable.
- **The overlay shows the cleaned pieces as they come** (the session's Draft), so the first words appear after 1.6 s instead of 11 s.
- **Each request has its own timeout** (15 s then, behavior.md 5.5). A whole clean-up of 12.8 s no longer risks the session's timeout: M3's 8 s would have inserted the raw text. Since 10-17 the deadlines follow the answer's progress (below).
- `llm.fast` in pieces: `long-all` 4.8 s (whole 3.7 s); `long-zh` 4.1 s, with fewer fillers left than whole (6.4% against 16.4% error), still not clean.

**Spoken Translate stays one request.** Into English, in 300-character pieces against whole:

| model | whole | pieces |
|---|---|---|
| `llm.balanced` | `long-zh` faultless; `long-all` keeps two replaced phrases (Friday, Shanghai); 14 s | the same faults; 17–38 s |
| `llm.fast` | `long-all` faultless; `long-zh` copies the example's sentence | `long-all` keeps fillers and three replaced phrases; `long-zh` copies the example too |
| Hy-MT2 | self-corrections kept as spoken (as on short clips); 12–13 s | the same; 12–14 s |

The translation's target is one language, so it has no language to lose. A whole translation was as good or better, and faster.

**`llm.fast` does not clean up long text.** It keeps nearly every filler on Chinese from about 100 characters up (3 short clips joined), whole or in pieces. Pieces do not help it. It cleans one short dictation well (16 of 20, above), so the 300-character pieces are no fix for it. Users who dictate long texts need `llm.balanced`.

### Long clean-ups under memory pressure (10-17, `TestEvalLong`)

On 10-17 the user's clean-ups timed out twice on mac-air (a 366-character dictation as one request, and the first of two pieces of a 502-character one), and the text went in as recognised. The Mac's swap was 17.1 of 18.4 GB used, with macOS's memory-pressure level still "normal"; `voxtnew-llm` had 80–97 MB resident of a 3.4 GB footprint: the model's weights were swapped out and read back while it wrote. It wrote 10–14 tokens a second instead of 19–26, its first token after 2 s instead of 0.2–0.5 s, so a 366-character answer took 20 s or more against the fixed 15 s.

`TestEvalLong` now also reports each run's slowest request, its longest wait for a first token and its longest wait between two tokens (`long.md`). On mac-air with the same kind of pressure (swap 15.3–15.9 of 16.4–17.4 GB used, level "normal"; the user's VoxtNew with its speech model, Dicta's engine and the browsers running), `llm.balanced`:

| | whole requests | in pieces |
|---|---|---|
| rate | 5.7–15.6 tokens/s (13 normally, 6–10 in bad moments) | 4.8–12.8 |
| longest first token | 2.7 s | 3.2 s |
| longest wait between two tokens | **0.52 s** | 0.33 s |
| slowest request | **27.5 s** (`long-zh[:12]`, 410 characters, 5.7 tokens/s); 11 whole requests, all of 410 characters or more (two of them Translate's), took 15.1–27.5 s | 14.2 s |

So a fixed 15 s cut answers that were coming steadily, and nothing in the stream ever paused for more than half a second. **Decision: the deadlines follow progress** (behavior.md 5.5): the first token within 15 s (plus 5 ms a character of input), at most 5 s between two tokens, and the request in all within the first token's deadline plus 100 ms a character (at least 20 s for a piece, 60 s for Translate and Rewrite). Every request above passes them: the stall and first-token deadlines are ten times the longest waits measured, and the cap allows about 8 tokens a second (a 410-character piece: 15 + 2 + 41 = 58 s against 27.5 s).

**Pieces from 300 characters instead of 400** (one language). The same Chinese text, its first 9 to 12 clips, whole and in pieces (greedy sampling, so the answers were the same in three runs; the times are the three runs'):

| `long-zh` first clips | characters | whole | 2 pieces | first piece's text in the overlay | total, whole / pieces |
|---|---|---|---|---|---|
| [:9] | 292 | no fault, 10.8% | (one piece) | – | 9.8–12.9 s |
| [:10] | 336 | no fault, 9.6% | 周五…不对，周四 kept, 13.8% | 6.2–8.9 s | 11.4–13.4 / 11.2–15.3 s |
| [:11] | 362 | no fault, 8.4% | 周五 kept, 12.1% | 5.5–8.0 s | 10.9–11.7 / 11.8–15.6 s |
| [:12] | 410 | 周五 kept, 15.6% | no fault, 7.8% | 9.1–14.2 s | 13.7–27.5 / 17.0–25.0 s |

- The only fault either way is the self-correction at a sentence's end that the model also misses on its short clip: it moves with where the text is cut, and is kept twice in pieces and once whole.
- The first cleaned text reaches the overlay in about half the time; the whole clean-up takes 0.5–3.5 s longer (the second request's first token).
- A piece that fails keeps half as much text as recognised.
- `long-zh` (441 characters, two pieces at either threshold) and the language-cut runs of `long-all` are unchanged: no fault and 6.3% (`long-all` in 10 pieces). `TestEvalLLM` quick (short clips, one piece) is unchanged.

Every slow request now logs the Mac's memory (behavior.md 5.5), so the next report shows the cause.

### Long recordings on the speech model (10-17, `TestEvalLongASR`)

The user's 4m51s dictation failed after 30 s of recognition and lost everything (behavior.md 5.5a). `TestEvalLongASR` (`eval_longasr_test.go`) takes the long clips and a synthetic one of 4m46s (the 20 clips, then the long ones' clips again, 0.8 s apart) on `asr.accurate`, three ways: one Final over the whole audio (`whole`, as after Stop before), a live session sent in real time with one Final after Finish (`live, one Final`), and a live session recognised in pieces while it records (`live, pieces`, behavior.md 3.7a). mac-air was swapping hard (swap 14–19 of 15–20 GB used, level "normal", load 18; the user's VoxtNew with its own speech model, browsers and other apps running), so the times are upper bounds.

| clip | way | error | wait after Stop | real-time factor | preview: longest gap |
|---|---|---|---|---|---|
| long-zh, 95 s | whole | 1.6% | 13.0–13.7 s | 0.14 | – |
| | live, one Final | 1.6% | 13.1 s | 0.14 | 3.3 s |
| | live, pieces (2) | 1.6% / 2.4% (two runs) | **3.8 / 5.4 s** | | 6.0 / 8.2 s |
| long-all, 104 s | whole | 2.9% | 14.4–15.5 s | 0.15 | – |
| | live, one Final | 2.9% | 25.8 s | 0.25 | 4.5 s |
| | live, pieces (3) | 2.9% | **2.4 s** | | 16.4 s |
| long-synth, 4m46s | whole | 3.2% | 43.8 s | 0.15 | – |
| | live, one Final | 3.2% / 3.2% | 54.4 s / **32.3 s** ¹ | 0.19 / 0.11 | 18.2 / 6.3 s |
| | live, pieces (10) ¹ | **2.7%** | **2.6 s** | | 5.1 s |

¹ A later run, with less pressure (swap 2.6–4.7 of 4–6 GB used): the two lines to compare. Even there one Final over 4m46s took 32 s, over the old 30 s limit.

- **The cause, confirmed.** One Final over 4m46s takes 44–54 s on this Mac: over the old 30 s limit. A piece of 24–36 s took 3.4–13.3 s (real-time factor up to 0.43) while the live preview shared the server.
- **The new limit** is 60 s plus 1 s per second of recording (5m51s for 4m51s): about three times the slowest real-time factor seen. A model that stops answering is still caught by the stall watch (15 s without a byte).
- **Quality.** The pieces are cut in Silero's pauses, so no word is cut. On `long-all` and the synthetic clip the error rate is the whole Final's; on `long-zh` one run of two was 0.8 points worse (a few characters), with the same cuts: noise between runs (the server's one worker interleaves the preview's decodes).
- **The preview.** The server decodes one request at a time, so a piece holds the preview's next decode: under this memory pressure its longest gap went from 3–4.5 s to 6–16 s (the synthetic clip had 18 s gaps without pieces too). On a Mac with free memory a piece takes 1–3 s.
- **Memory.** The pieces add no model and little memory: the speech server's largest footprint over the synthetic clip was 9.9 GB with one Final and 10.2 GB with pieces. That footprint is large either way. #20 found the cause in MLX's buffer cache and capped it: see [the next section](#speech-server-memory-10-1718-20).

### Speech server memory (10-17/18, #20)

The footprints above (about 10 GB) came almost entirely from MLX's cache of freed buffers.
- **Within one request** the server keeps every freed buffer:
  - the KV cache, which grows by 256 tokens at a time;
  - the audio encoder's and the prefill's activations.
- **Between requests** it returns the cache only when its queue is empty. A live session queues its next decode as soon as the last one ends whenever a decode took more than its 1 s interval, so during a long recording the cache is often not returned. Each decode's segment has a different length, so few buffers are reused, and the cache grew to **8.1 GB** over the 286 s clip.

All measured on mac-air with `asr.accurate`, using an instrumented runtime that logs MLX's active, cache and peak memory per request. The 286 s synthetic clip is used throughout. Swap was at 75–90%, so the times are noisy.

**One Final over 286 s, in one request** (`ChunkSeconds` 1200, as before):

| cache limit | weights | peak active | cache at the end | server footprint | Final | error |
|---|---|---|---|---|---|---|
| none (before) | 2.35 GB | 5.07 GB | **6.24 GB** | **8.5–8.6 GB** | 48–60 s | 3.2% |
| **512 MB** | 2.35 GB | 5.07 GB | 0.51 GB | 5.3 GB | 58 s (78 s with swap at 90%) | 3.2% |
| 0 | 2.35 GB | 5.07 GB | 0 | 5.1 GB | 77 s | 3.2% |

**The same with 60 s chunks and the 512 MB limit** (now; 10-18, swap 76%):

| clip | error (60 s chunks) | before (one request) | server footprint | Final |
|---|---|---|---|---|
| long-zh, 95 s | 1.6% | 1.6% | 3.59 GB | 9.2 s |
| long-all, 104 s | 2.9% | 2.9% | 3.57 GB | 9.4 s |
| long-synth, 286 s | **2.2%** (edits 9/0/6) | 3.2% (10/5/7) | **3.68 GB** | 27 s |

**Live sessions over 286 s, sent in real time** (10-18; both runs with 60 s chunks):

| way | cache limit | error | server footprint | largest cache | live decode median / p90 | wait after Stop |
|---|---|---|---|---|---|---|
| live, one Final | none | 2.2% | **10.29 GB** | **8.1 GB** | 744 / 1710 ms | 24.4 s |
| live, one Final | **512 MB** | 2.2% | **3.92 GB** | 0.5 GB | 657 / 1209 ms | 22.3 s |
| live, pieces | none | 2.7% | **9.66 GB** | | | 2.2 s |
| live, pieces | **512 MB** | 2.7% | **3.73 GB** | | | 2.1 s |

The decode times are per request over both ways of each run.

- **Fix 1: the cache limit.** VoxtNew starts the speech model with `--cache-limit-mb 512` (`ASRTuning.CacheMB`, behavior.md 3.5a), a new server flag. Without the flag the server keeps MLX's own limit, as Voxt runs it.
  - Decoding reuses buffers of a few sizes, so 512 MB costs no speed: the live decodes were if anything faster, with less swap. A limit of 0 frees every buffer and was 27% slower.
  - The text is the same in every pair. The limit changes where buffers come from, not the computation.
- **Fix 2: 60 s chunks.** The activations grow with the audio of one request: the encoder peaks at +2.7 GB over the weights for 286 s (about 9.4 MB per second of audio), and the prefill of 3770 tokens at +1 GB.
  - A Final over more than 60 s (Transcribe again, the live session's Final when pieces failed) is now cut near each minute (`ChunkSeconds` 60, behavior.md 3.6), the same way Voxt cuts at 1200 s.
  - The error rate held or improved: no word was dropped on the 286 s clip, against 5 before. The whole Final was faster too.
  - A dictation recognised in pieces (3.7a) was already in requests of 20–45 s.
- **Result:** a 5-minute dictation peaks at about 3.7–3.9 GB, against 9.7–10.3 GB before: the 2.35 GB of weights, the 512 MB cache, and one request's activations.

### Spoken Translate with Hy-MT2 (`llm.translate`)

Hy-MT2 takes its own prompt and no clean-up rules (behavior.md 3.13). Raw speech, the same speech without its fillers, and `llm.balanced`'s clean-up first:

| input | no fault | fillers left | corrections missed | median total |
|---|---|---|---|---|
| raw speech | 2 | 2 ("well", "Ah", "um") | 3 | 0.55 s |
| **without fillers** (`compose.StripFillers`) | 3 | **0** | 3 | 0.46 s |
| `llm.balanced`'s clean-up first | 5 | 0 | 1 | 0.44 s, plus the clean-up's 2.7 s and loading two models |

**Decision: Hy-MT2 gets the speech without its fillers.**
- `compose.StripFillers` is deterministic and needs no model. It drops 嗯 呃 额 唔 and um uh er hmm anywhere. It drops 啊, 哦, 那个 and 就是说 only at the start of a clause, before a pause or a correction.
- The session puts that text in `llm.Request.Translate` for speech; a selection is sent as it is.
- **Clean-up first was rejected.** It wins on self-corrections, but `voxtnew-llm` holds one model at a time, so every spoken translation would load the 4B model and Hy-MT2 in turn: about 1.5 s of loading and 2.7 s of clean-up for a 0.5 s translation.
- **The trade-off for the user.** Self-corrections reach Hy-MT2 as spoken ("next Tuesday, no, next Wednesday"). A user who wants them resolved picks a general model for Translate: `llm.balanced` with the example got 6 of 6 in 3.6 s, `llm.fast` 6 of 6 in 1.4 s.

### The glossary on misheard terms (`TestEvalGlossary`)

Each dictionary clip's transcript as a speech model wrote it without the dictionary (`heard`), cleaned up with the eval dictionary as the glossary. Since M3b the session first replaces the dictionary's aliases (`dict.Replace`: "Or Link" becomes Auralink, behavior.md 7.9a), and the glossary of a dictionary of up to 30 terms is in the dictionary's order (M2b's decision: the same system prompt for every dictation, which the prompt cache keeps):

| model | glossary order | as heard | aliases replaced first (the session) | terms put in where none was said |
|---|---|---|---|---|
| `llm.balanced` | the terms the text mentions first (M3) | 9 of 10 | **10 of 10** | 0 |
| `llm.balanced` | the dictionary's order (M3b) | 8 of 10 | **9 of 10** | 0 |
| `llm.fast` | the terms the text mentions first (M3) | 5 of 10 | 6 of 10 | 0 |
| `llm.fast` | the dictionary's order (M3b) | 5 of 10 | 6 of 10 | 0 |

- `llm.balanced` turns "Vox Tunew", "Vox New", "S G Long", "Pixcal" and "Act Now" into the dictionary's spellings.
- **It keeps "Or Link" as a name** although that is Auralink's alias: a more insistent glossary wording changed nothing (M3). The alias replacement fixes it with no model at all.
- **The dictionary's order costs one term.** On `dict-5`, "Voxnu" stays "Voxnu" when Auralink, the term the text mentions, is no longer listed first (greedy decoding: the same every run); the other nine answers are the same. The 20 short clips' clean-ups and translations score the same with either order (the tables above are from M3b's run).
- `llm.fast` mostly ignores the glossary, as M2 saw.

### Rewrite (`TestEvalRewrite`, F1)

Eight Rewrite requests as the session builds them (compose, Cleanup on), each sampled with two seeds (one in quick mode): six selections with a spoken instruction (polite and shorter, fix the grammar, a bulleted list, formal, shorter, friendlier; three English, three Chinese, one of them with an English instruction) and two messages written from an instruction alone (the answer card's JSON, English and Chinese). An answer has a fault when it loses a fact the text gave, keeps what the instruction asked to fix ("ridiculous", "buyed"), changes the language, is not shorter when asked, is no list when asked, opens with a preamble ("Sure, here is…"), or, from an instruction alone, is not the JSON answer with a title. Tone and wording are not scored: `rewrite.md` lists every answer. Quick mode checks `llm.balanced` against a floor of 60%.

| model | no fault | selections | answers (JSON) | first token (median) | total (median) |
|---|---|---|---|---|---|
| `llm.balanced` | 15 of 16 | 11 of 12 | 4 of 4 | 0.72 s | 2.5 s |
| `llm.fast` | 14 of 16 | 10 of 12 | 4 of 4 | 0.19 s | 0.76 s |
| `llm.qwen3` | 15 of 16 | 11 of 12 | 4 of 4 | 0.51 s | 1.3 s |
| `llm.gemma4` | **16 of 16** | 12 of 12 | 4 of 4 | 0.45–0.53 s | 1.7–1.8 s |

- The faults: "not shorter" on the polite-and-shorter case (`llm.balanced`, `llm.qwen3`, `llm.fast`, one seed each; they added a sentence of politeness; `llm.qwen3` the same in two runs), "ridiculous" kept once by `llm.fast`, and `llm.fast` dropping "bought" when fixing the grammar.
- Every JSON answer parsed. Written from an instruction alone, every model sometimes adds a reason nobody gave ("由于时间安排冲突") and signs off "[Your Name]".
- The first token is the miss's: Rewrite has no warm-up, and each case's two seeds share one prompt (the second is a hit).

### The borrowed models (F1)

`llm.qwen3` (Qwen3-4B-Instruct-2507, AudioSmith's and Dicta's) and `llm.gemma4` (Gemma 4 E4B, Dicta's) ran the same eval beside VoxtNew's own models (`VOXTNEW_EVAL_LLM=all`, 2026-10-17, Dicta and AudioSmith resident). This first table is with compose's wording before F1; the one after "Compose's wording" below is with the new one. Their profiles and times are in [llm-models.md](llm-models.md#borrowed-models-family-f1).

| | `llm.balanced` | `llm.fast` | `llm.qwen3` | `llm.gemma4` |
|---|---|---|---|---|
| Transcribe, 20 clips, no fault: example / rules only | 18 / 18 | 15 / 14 | 17 / 18 | **20** / 17 |
| error vs expected (example) | 5.6% | 12.6% | **4.6%** | 5.3% |
| spoken Translate, 6 clips, no fault: example / rules only | 6 / 4 | 6 / 2 | 6 / 5 | 5 / 4 |
| glossary, terms written: as heard / aliases replaced | 8 / 9 of 10 | 5 / 6 | 8 / 8 | **10 / 10** |
| median total, Transcribe (prompt cache hit) | 0.79 s | 0.36 s | 0.59 s | 0.71 s |
| footprint / peak | 3.8 / 4.8 GB | 1.4 / 2.1 GB | 3.1 / 3.8 GB | 4.5 / 5.0 GB |

- `llm.qwen3`'s faults: 那个报告周五之前发给我就行，不对，是周四之前 kept both days (as `llm.balanced` does); "VoxtNew" written "Voxt New" in `dict-4`; "9 a.m." counted as two invented tokens (a false alarm).
- `llm.gemma4` with the example has none. Its translation fault is the same late correction, kept ("before Friday. Wait, before Thursday."). Without the example it leaves 呃 and 就是说 and one English correction, as the others do: the example turn helps it too.
- Neither ever changed the language or copied the example.

**Long dictation** (`TestEvalLong`, the same clips as M3b's; compare with `llm.balanced`'s tables above):

| | `llm.qwen3` | `llm.gemma4` |
|---|---|---|
| `long-zh` whole (441 characters) | "schedule1" and "voxt" invented, 9.6%; 5.7 s | **no fault, 2.3%**; 7.0 s |
| `long-zh` in pieces (2) | "VoxtNew 里" came out "Voント", 6.8%; 6.2 s | lost a phrase, 13.2%; 7.1 s |
| `long-all` whole (742 characters, alternating sentences) | **translated into Chinese**, 54% | **no fault, 2.9%**; 9.6 s |
| `long-all` in pieces (10, cut where the language changes) | **every English piece translated into Chinese**, 48%; 6.8 s, the first piece after 0.5 s | "um" and one correction kept, 4.8%; 10.4 s, the first piece after 1.2 s |
| `long-all`, first 3…13 clips whole | translated into Chinese at every length (192–716 characters) | no fault but "um" kept twice (0.6–7.6%) |
| spoken Translate whole | kept "Friday" / "Shanghai", 6.5–6.8 s | kept 3–5 replaced phrases, 8.7–10.1 s |

- **Gemma 4 keeps the languages of a mixed dictation**, whole, where `llm.balanced` turns it into English (58.5%) and needs the language cuts. With the cuts (the session's way) it is as good.
- **Qwen3-4B-Instruct-2507 translates English into Chinese** when the main language is Chinese: the whole English sentences of `long-all`, and each English piece alone. The short eval's English clips ran with English as the main language, so the next check runs them with Chinese.

**English dictation with Chinese as the main language.** TestEvalLLM now also cleans up the English clips (`en-1`…`en-5`, `dict-3`, `dict-5`) with `MainLanguage: zh` (the Chinese template set: a Chinese user dictating an English sentence; variant "example, main zh"). The first run, the five `en` clips, with compose's wording then:

| | `llm.balanced` | `llm.fast` | `llm.qwen3` | `llm.gemma4` |
|---|---|---|---|---|
| no fault | **5 of 5** | 1 of 5 | **0 of 5** | 3 of 5 |
| translated into Chinese | 0 | 2 | **5** | 0 |
| other faults | – | fillers kept ("um", "you know"), 1 correction | – | "um" kept once, 1 correction |

All 20 short clips with Chinese as the main language (the English clips and the English `dict` clips included): `llm.balanced` 17, `llm.fast` 11, `llm.qwen3` 11, `llm.gemma4` 18 of 20. `llm.qwen3` translated every English clip; `llm.fast` translated two.

**Compose's wording: "write in the language spoken"** (F1). compose already said 保持说话时的语言混合 (keep the mix of languages) and that the main language 不是翻译的目标语言 (is not a language to translate into). Four wordings were tried on all 20 short clips, under each main language (scratch runs, greedy, the example on; no fault of 20):

| wording | `llm.balanced` zh / en | `llm.fast` | `llm.qwen3` | `llm.gemma4` |
|---|---|---|---|---|
| as it was | 17 / 17 | 11 / **7** | **11** / 15 | 18 / 18 |
| A: the clean-up's mix line says English stays English, Chinese stays Chinese | 18 / – | 16 / – | 14 / – | 18 / – |
| C: the same, as "write in the language spoken" | 18 / 17 | 14 / 11 | 15 / 16 | 18 / 18 |
| D: the main-language line adds "when the user speaks English, write English; Chinese, Chinese" (Transcribe only) | 18 / 17 | 12 / 8 | 18 / 15 | 18 / 18 |
| **C and D** | **18 / 17** | **15 / 11** | **18 / 16** | **18 / 18** |

- With English as the main language, `llm.fast` translated six Chinese clips into English (7 of 20); this is not the borrowed models' problem, but the same one.
- **Decision (proposed to the parent, F1): C and D are in** (`internal/compose/templates/*/cleanup.txt`, `input-speech.txt`: D only under `FOR_TRANSCRIBE`, so Translate's prompts are unchanged; the Transcribe goldens changed). The eval with them, every model:

| | `llm.balanced` | `llm.fast` | `llm.qwen3` | `llm.gemma4` |
|---|---|---|---|---|
| Transcribe, 20 clips, no fault (example) | 18 (was 18) | **17** (15) | 17 (17) | 20 (20) |
| error vs expected | 6.0% (5.6%) | **8.6%** (12.6%) | 4.6% (4.6%) | 5.3% (5.3%) |
| English clips with Chinese as the main language, no fault | 7 of 7 | 4 of 7 (one still translated) | **7 of 7** (0 of 5) | 5 of 7 |
| glossary, terms written: as heard / aliases replaced | 8 / 9 | 6 / 7 (5 / 6) | 9 / 9 (8 / 8) | 10 / 10 |
| spoken Translate | unchanged (its prompt is) | | | |

Long dictation with the new wording (`TestEvalLong`, the four models): in pieces, as the session sends them, `llm.qwen3` now keeps every English piece in English: `long-all` 6.3% with one late correction kept, the same as `llm.balanced` (6.3%), 8.3 s against 10.7 s. Whole, it now turns `long-all` into English (60%), as `llm.balanced` does (58.5%): the language cuts are what keeps a mixed dictation's languages, for both. `llm.gemma4` in pieces is unchanged (4.8%); whole it now leaves a few "um" and "uh" (5.8%, was 2.9%). `llm.balanced` and `llm.fast` are as M3b measured them (`long-all` in pieces 6.3% and 10.1%).


**Decision: both are usable for clean-up, Translate and Rewrite** as alternatives of `llm.balanced` and `llm.fast`, with compose's new wording; their `Features` stay unrestricted. Without the wording, `llm.qwen3` would not be usable for Transcribe by anyone who dictates English with Chinese as the main language.

## The dictionary: echoed, and what it is worth (#24)

On 10-19 a user's dictation ended in "…这个Bot的这个文本处理还是很拉的，嗯。Voxt，Omni，SGLang。". Those are the three terms of their dictionary, which nobody said. The dictation was 77 s long. It was recognised in two pieces while recording; at Stop, the rest (5.6 s, probably "嗯" and silence) went to the speech model with the dictionary as its context, `Voxt\nOmni\nSGLang`, the most recently changed term first. The echo is that context, in the same order.

`TestEvalDictASR` (`eval_dictasr_test.go`) measures the echo and the dictionary's worth, on mac-air (10-20, swap 83–92% used, 45–60% of the memory free; another project's models were running), with four dictionaries (`eval_dict_test.go`):

- **small**: the user's 3 terms;
- **eval**: the set's 9;
- **medium**: 30 (the eval's, the dict2 clips' names and products, other product names);
- **large**: 130 (medium plus tools, people's names and Chinese jargon).

The clips are the 29 short clips, as recorded (0.3 s of silence after the speech) and with 3 s of room noise more (-55 dBFS, a dictation stopped late), and two recordings without speech (1.5 s of silence or of noise, also with the 3 s more). "Raw" is the context as the model writes it, before the fix. "Guarded" is what VoxtNew now delivers. A term put in is one that neither the clip's words nor its transcript without a dictionary have.

### Where the echo comes from

Before the fix, the scratch runs and the table below (raw) show:

- **Audio without speech echoes the context, every time.** That held on both models, in 2 of 2 recordings with every dictionary and pad: 64 of 64 runs per model in the first matrix (silence or noise, 1.5 s, plus 0, 0.5, 1, 2, 3 or 5 s of silence, or 1 or 3 s of noise). The echo is the whole context, or a run of it in order (the large dictionary's from 灰度发布 on), or a single term ("VoxtNew。", "Voxt。"). Its separators vary: the newlines, commas, 、 or spaces. Without a context the same audio gives "嗯。", nothing, or a hallucination ("在《西游记》中，孙悟空是齐天大圣。").
- **Speech followed by silence does not echo.** We tried 31 clips × 8 pads (0–5 s of silence, 1 or 3 s of noise) × 4 dictionaries on both models. No term came after the speech.
- **The format does not matter.** On `asr.fast` we tried four layouts: one term per line (Voxt's and ours), commas, a Chinese sentence ("可能出现的专有名词：Voxt、Omni、SGLang。") and an English one. All four echo on audio without speech, and the sentence layouts echo their sentence too. This is likely why Voxt dropped its sentence templates for the bare list. Upstream Qwen3-ASR puts the context as the system message, free-form, and filters nothing.

So the user's rest was most likely a stretch with no speech but a filler. VoxtNew's fix is a guard on each request's text (`omni/echo.go`, behavior.md 3.6a). It applies when the text ends in a sentence of context terms: after a sentence's end, or as the whole text, with only list marks between the terms. In that case it recognises the same audio again without the context, and drops the terms unless that transcript has them.

### The speech models (raw / guarded)

| model | dictionary | error (as recorded; +3 s noise) | terms written, with (without) the dictionary | speech: answers with a term put in | no speech: answers with terms, raw → guarded | median Final |
|---|---|---|---|---|---|---|
| `asr.accurate` | none | 7.7%; 9.0% | 8 of 26; 6 of 26 | | | 748 ms; 852 ms |
| | small (3) | 4.4%; 3.7% | 10 of 11 (1); 11 of 11 (1) | 0; 0 of 29 | 2 → 0; 2 → 0 | 732–766 ms |
| | eval (9) | 3.9%; 4.8% | 10 of 13 (4); 8 of 13 (3) | 0; 1 | 2 → 0; 2 → 0 | 777–914 ms |
| | medium (30) | 3.1%; 5.0% | 21 of 26 (8); 21 of 26 (6) | 2; 2 | 2 → 0; 2 → 0 | 937–1045 ms |
| | large (130) | 3.1%; 3.5% | 21 of 26 (8); 20 of 26 (6) | 2; 3 | 2 → 0; 2 → 0 | 1388–1537 ms |
| `asr.fast` | none | 9.4%; 9.0% | 8 of 26; 8 of 26 | | | 272 ms; 317 ms |
| | small (3) | 7.9%; 9.6% | 6 of 11 (1); 6 of 11 (1) | 0; 1 | 2 → 0; 2 → 0 | 256–337 ms |
| | eval (9) | 8.6%; 9.9% | 4 of 13 (3); 4 of 13 (3) | 0; 0 | 2 → 0; 2 → 0 | 277–347 ms |
| | medium (30) | 7.5%; 7.0% | 18 of 26 (8); 16 of 26 (8) | 1; 2 | 2 → 0; 2 → 0 | 318–400 ms |
| | large (130) | 10.3%; 13.4% | 16 of 26 (8); 17 of 26 (8) | 1; 1 | 2 → 0; 2 → 0 | 493–549 ms |

- **The guard removes every echo and changes nothing else.** On audio without speech the echo went from 32 of 32 runs to 0 of 32 across both models. On speech the raw and guarded transcripts were the same in every row (error, terms, put-in), and the times did not move: when a text does not end in terms, no second request is made. When it does, the same audio is recognised once more without the context: on a short stretch without speech that is one more short request (about 0.3 s on `asr.accurate`).
- **The dictionary helps, on `asr.accurate` most.** It wrote 10 of the user's 11 terms against 1 without, and 21 of 26 with the medium or large dictionary against 8. Its error rate on all 29 clips fell from 7.7% to 3.1–4.4%. On `asr.fast` it went from 8 of 26 to 16–18 of 26 with the larger dictionaries, but only 6 of 11 with the small one. Its error rate improves a little with up to 30 terms and gets worse with 130 (10.3–13.4% against 9.0–9.4%).
- **A term can pull in a near one.** "VoxtNew" becomes "Voxt New" (or "Voxt 新版本" on `asr.fast`) when the dictionary has Voxt: that is the 1–3 answers with a term put in. The same happens inside the long live session (`long-zh` in pieces, with the small dictionary). The alias "Voxt New" on the term VoxtNew turns it back (dict.Replace). On one take, `asr.accurate` with 9 terms and 3 s of noise wrote the spoken list of dict2-6 ("我说三个词：Voxt，Omni，SGLang") as "我说三个词，VoxtNew。"; the guard leaves it (it is after a comma, within speech).
- **A long context costs time.** With 130 terms, `asr.accurate`'s Final takes about twice as long (1.4–1.5 s against 0.75–0.85 s for 6.5 s clips); with 30 terms it takes 0.1–0.2 s more. `asr.fast` goes from 0.3 s to 0.5 s.
- **`asr.fast` takes the 30 most recently changed terms** (follow-up, `ASRTuning.ContextTerms`; `asr.accurate` keeps Voxt's 500). With the large dictionary its guarded rows are now the medium dictionary's: error 10.3% → 7.5% as recorded and 13.4% → 7.0% with the noise, terms written 16–17 → 16–18 of 26, median Final 721 → 353 ms and 845 → 546 ms (the raw rows send the whole list as a free prompt, as before). The same run's other rows are unchanged.
- **Long live session.** `long-zh` with 3 s of noise, recognised in pieces, with the small dictionary: no echo either way (`asr.accurate` 5.6%, `asr.fast` 6.3%).

### The language models and the glossary (`TestEvalDictLLM`)

Every short clip's words, as said, go through Transcribe's clean-up (29 answers per model and dictionary) and the Chinese ones through spoken Translate into English (21). The glossary is each of the three dictionaries; past 30 terms, only the terms the text mentions are sent (behavior.md 5.16). The models are VoxtNew's three and the two borrowed ones (`llm.qwen3`, `llm.gemma4`). The translation model `llm.translate` takes no glossary.

| model | answers with a term put in (transcribe, translate; 3 dictionaries) | terms kept (transcribe; translate) | median first token: small, medium (30), large (130) |
|---|---|---|---|
| `llm.balanced` | 0 of 150 | 26 of 26; 16 of 20 | 238, 242, **1798 ms** |
| `llm.fast` | 0 of 150 | 26 of 26; 16 of 20 | 110, 110, **556 ms** |
| `llm.qwen3` | 0 of 150 | 26 of 26; 16 of 20 | 222, 249, **1712 ms** |
| `llm.gemma4` | 0 of 150 | 26 of 26; 16 of 20 | 222, 221, **994 ms** |
| `llm.translate` | (no glossary) | –; 16 of 20 | |

- **No model puts a term in.** The glossary never made a model add a term that was not said: 0 of 600 answers, with every dictionary size and both tasks.
- **Every Latin term is kept.** In translations the 4 missing are the Chinese names, transliterated (陈喆 → Chen Zhe, 周芷若 → Zhou Zhiruo), which is right for English.
- **No model removes the echo.** Given the user's transcript with the echo at its end, all four general models keep it ("…很拉的。Voxt、Omni、SGLang。"). The fix has to be in recognition.
- **A dictionary over 30 terms costs the prompt cache.** Its glossary holds only the terms the text mentions, so the system prompt changes from one dictation to the next, and the cached prefix stops before the glossary. The first token comes 0.5–1.6 s later than with 30 terms (`llm.balanced` 0.24 → 1.8 s). Up to 30 terms the glossary is the same each time and costs nothing measurable.
  The follow-up below fixes this.

### A large dictionary's glossary in the user message (#24 follow-up)

A dictionary of more than 30 terms now keeps its rule in the system prompt, the same for every dictation ("the user message may open with a `<glossary>` block…; never output it"). Its terms that the text mentions go in a `<glossary>` block that opens the last user message, after the system prompt and the example, which the prompt cache keeps (behavior.md 5.16). A dictionary of up to 30 terms is unchanged: the whole glossary stays in the system prompt.

The runs were on mac-air with `TestEvalDictLLM` and `TestEvalGlossary`, before (5bdbd174) and after. The user's VoxtNew had its models loaded throughout, so times are upper bounds.

| | before | after |
|---|---|---|
| `llm.balanced`, first token, 130 terms (transcribe; translate) | 2383; 1852 ms | **494; 501 ms** |
| `llm.balanced`, first token, 30 terms | 374; 419 ms | 306; 409 ms |
| `llm.balanced`, glossary on misheard terms, 130 terms: terms written (as heard / aliases replaced), first token | 5 / 6 of 10, 1671–1875 ms | **6 / 6 of 10, 422–501 ms** |
| `llm.fast`, first token, 130 terms (transcribe; translate) | 819; 929 ms | **168; 174 ms** |
| `llm.fast`, first token, 30 terms | 170; 184 ms | 118; 131 ms |
| `llm.fast`, glossary on misheard terms, 130 terms | 3 / 4 of 10, 587–685 ms | 3 / 4 of 10, 170–181 ms |
| terms put in (both models, 3 dictionaries, 300 answers) | 0 | 0 |

- **The glossary is still obeyed.** Terms written with 130 terms are as before or one better. With 9 terms nothing changed (8 / 9 and 6 / 7 of 10). Translations keep 16 of 20 terms (the 4 transliterated names).
- **The glossary's rule belongs in the system prompt.** In a first layout the rule was inside the user message's block too. `llm.fast` then wrote the whole block as its answer once (dict-5, 130 terms). With the rule in the system prompt ("never output it") that did not happen in 300 answers.
- With 130 terms a dictation's first token still comes about 0.1–0.25 s after 30 terms' (`llm.balanced` 0.49 against 0.31 s): the block and the input are new on every dictation.


## App enhancement and the dictionary, whole sessions on the real models

These are `eval_e2e_test.go`.

**Setup:**
- the app's own Library and runtimes over the real `$HUB` (`asr.accurate`, Silero and `llm.balanced`; the app's family directory is a temporary one);
- the fake helper plays an eval clip as the recording, with an app in front;
- each session runs in an app of its own, quit before the next one.

Each session took about 6.5 s end to end, cold starts included: ASR 0.7–1.0 s and LLM 2.3–2.5 s after Stop.

| test | session | heard | result |
|---|---|---|---|
| App enhancement | `en-3`, Slack in front: the group "Casual chat: write everything in lowercase letters, with no period at the end." (`transcribe:cleanup+scene`) | Please remind me to call the dentist tomorrow at nine in the morning. | please remind me to call the dentist tomorrow at 9 in the morning |
| | the same in TextEdit: no group (`transcribe:cleanup`) | (the same) | Please remind me to call the dentist tomorrow at 9 in the morning. |
| Dictionary | `dict-1`, the dictionary on | VoxtNew的新版本用MLX在本地跑语音模型。 | VoxtNew 的新版本用 MLX 在本地跑语音模型。 |
| | the terms not sent to the speech model (glossary only) | Vox Tunew的新版本用MLX在本地跑语音模型。 | **VoxtNew** 的新版本用 MLX 在本地跑语音模型。 |
| | the dictionary off | Vox Tunew的新版本用MLX在本地跑语音模型。 | Vox Tunew 的新版本用 MLX 在本地跑语音模型。 |
| Dictionary alias (M3b) | `dict-5`, Auralink with the alias "Or Link", no speech hints, the clean-up off | Or Link and Voxnu share the same prompt composer. | **Auralink** and Voxnu share the same prompt composer. |

These sessions are sensitive to load. While two other threads ran their models on mac-air (swap 14–17 of 18 GB), the alias session failed twice: the speech model wrote "I'm not sure." and "I'm not a good person." for the whole `dict-5` recording, its usual output on near-silence. It passed 3 of 3 on a quieter Mac. Once, the glossary-only session heard "Vox Tunio" and `llm.balanced` left it.

## Not measured here

- Noise, real voices and accents.
- Recordings from a microphone.
- Sessions that run through the helper's real capture.
- Rewrite's tone and wording (`TestEvalRewrite` checks facts, language, length and form; the answers are read by hand).
- Translation into languages other than English.
- `TestMacE2EOmni` covers the live path with the real helper on the speech sample; scripts/mac-regress.sh runs it.
