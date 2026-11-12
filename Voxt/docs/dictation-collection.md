# Collecting dictations for a formalized ASR benchmark

We are building a benchmark from real Voxt dictations: given a recording, how close does a system get to text that is ready to send? Each item has three parts, all of which Voxt 2.0 already keeps in its history: the recording, the recognised text (`raw_text`) and the text Voxt delivered after the language model's clean-up (`result`). Hand-corrected reference text comes later, in a separate round.

Everything stays on your Mac until you hand in a zip yourself.

## 1. Set up Voxt now

Voxt keeps the texts of every dictation but **no recordings by default**, and recordings start only from the moment you turn them on.

1. Use Voxt 2.0 (2.0.0 or later). Voxt 1.x keeps only the final text of each dictation, so its history cannot be used. When you upgrade, the setup guide's **Import from Voxt 1.x** (从旧版 Voxt 导入) brings over your dictionary, models and language model setups; paste the API keys again on the Language models page.
2. History → **History settings** (历史设置) → turn on **Keep recordings** (保存录音). Settings → Privacy has the same switch.
3. In the same sheet, set **Keep entries for** (保留时长) to 1 year, or turn off **Clean up automatically** (自动清理旧记录). By default entries and their recordings are deleted after 3 months.
4. Dictate with Transcribe as usual, with **Clean up with a language model** (用语言模型整理) on. Translate and Rewrite are left out of the benchmark.

If tapping right Option does nothing, a key remapper such as Karabiner-Elements may be changing that key; record another shortcut on the Transcribe page.

## 2. Export

After a week or two, run the script with the Python that comes with macOS (the first `python3` may ask to install the command line developer tools):

```sh
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/main/Voxt/scripts/collect-dictations.py | python3 -
```

or `python3 Voxt/scripts/collect-dictations.py` from a checkout. It needs no packages and works while Voxt is running.

1. It reads `~/Library/Application Support/VoxtNew/voxtnew.db` read-only (or the folder Settings → Storage moved it to) and prints how many entries it can export, with a count and reason for every entry it leaves out.
2. It opens a page on `127.0.0.1` that lists each dictation with its recording, recognised text and delivered text. Untick anything you do not want to share, one entry at a time or a whole app at once.
3. Enter a short alias, tick the consent box and press **Pack**. The zip lands in `~/Downloads` and Finder shows it.

## 3. Hand in

Send the zip to Jiaxuan Luo ([@luojiaxuan](https://github.com/luojiaxuan)). Exporting again later is fine: an entry handed in twice is kept once, from the newest zip. The collected data stays in a private dataset for the group.

## What the zip holds

Only the ticked Transcribe entries that finished with text and a kept recording. For each:

- the recording, `audio/<id>.wav`: 16 kHz mono 16-bit PCM, the microphone signal as Voxt kept it, before the high-pass, gain and silence trimming that the speech model's copy went through;
- `raw_text` and `result`, with `asr_status` and `llm_status` (`llm_status` is `failed` when the language model errored and the recognised text went out instead, `off` when none ran);
- the app in front, the browser tab's host, the matched app enhancement group, the speech and language model ids, and the stage times.

`manifest.json` adds your alias, the consent text, the Voxt version, per-file SHA-256 sums, counts of what was left out, and the current clean-up settings: Voxt keeps no prompt per entry, so the zip carries the transcribe switches, the custom prompt, the main language, the model behind each provider id and the matched groups' scene prompts. The page shows these settings before you pack.

The zip never holds selected text, Translate or Rewrite entries, entries imported from Voxt 1.x, notes, meetings, dictionary terms, provider addresses, API keys, error messages, or the app a result was pasted into.

## Reading the data

- `raw_text` is the speech model's last pass over the whole recording, with the spoken end word removed, before dictionary aliases and the language model. The language model saw `raw_text` after dictionary replacement, in pieces of at most 300 characters.
- `result` is what Voxt delivered. Edits the user made afterwards in the target app are not recorded, so `result` is a system output and the references have to come from the later round.
- **Transcribe again** (重新识别) rewrites an entry's texts with the settings of that moment and keeps its date and recording.
- Where these facts come from in the code: `internal/store/store.go` (the entry), `internal/session/run.go` and `run_listen.go` (what is stored), `internal/session/run_audio.go` (the recording).
