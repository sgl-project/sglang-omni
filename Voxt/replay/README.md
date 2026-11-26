# Replay harness

The replay harness measures dictation apps on a Mac the way a user meets them. It plays recordings in real time into a virtual microphone, presses the app's shortcut, and timestamps every change of a focused text field, so it times the whole interaction, from the stop key to the last edit of the inserted text, and keeps the text the app wrote. The driver and the sink only press keys and read a text field; `run_cells.py` also writes the settings each cell names into the app under test and puts the original settings back when the plan ends.

| File | What it is |
| --- | --- |
| `build.sh` | Builds `build/replay-driver` and `build/ReplaySink.app` with `swiftc` |
| `Sources/Sink/main.swift` | `replay-sink`: one plain-text field that logs every change of its contents |
| `Sources/Driver/` | `replay-driver`: `run` (one segment after another), `mem` (memory alone) and `list-devices` |
| `Sources/Common/` | The clock, JSON lines and the Unix socket both programs share |
| `run_cells.py` | Runs a plan of cells unattended; `--check` checks the plan and the Mac without pressing a key |
| `app_settings.py` | Saves, resets and sets the settings of the apps under test for `run_cells.py` |
| `plan.example.json` | A plan template |
| `aasr_sample.py` | Draws a stratified sample of AASR-Bench and writes the driver's manifest for it |
| `score.py` | Error rates per language and median waits of one run |
| `hanzi.py` | Traditional to simplified Chinese with ICU's Hant-Hans transform in macOS |
| `tests/` | Unit tests of scoring and sampling, and an end-to-end self-test that presses no key |

## Requirements

- An Apple Silicon Mac with the Xcode Command Line Tools and Python 3.10 or later.
- A virtual audio device the app under test can record from, such as BlackHole 2ch, set as the system's default input. The driver plays into it by name (`--device`, default `BlackHole 2ch`) and never changes a system setting; it records the default input and output devices in `results.jsonl.meta.jsonl`.
- Accessibility access for the terminal that runs the driver, which posts the shortcut as HID-level key events. Without it the driver exits before sending any event.

## Build

```bash
./build.sh
```

Some Command Line Tools installs keep a stale `usr/include/swift/module.modulemap` that breaks every Swift import; `build.sh` hides it from this build with a VFS overlay and leaves the toolchain alone.

## One run

1. Write a manifest: one JSON object per line with `id`, `wav` and `lang` (`en`, `zh`, `mixed` or `none`), and `reference` for scoring. `aasr_sample.py` writes one for AASR-Bench (below).
2. Set up the app under test with the shortcut you pass as `--key`.
3. Start the sink and bring its window to the front. The socket path must fit in 103 bytes.
   ```bash
   open -n build/ReplaySink.app --args --log "$PWD/run/sink.log" --socket "$HOME/.replay-sink.sock"
   ```
4. Run the driver:
   ```bash
   build/replay-driver run --manifest run/manifest.jsonl --out run/results.jsonl \
     --sink-socket "$HOME/.replay-sink.sock" --sink-log run/sink.log --key right_option \
     --mem-app Voxt --mem-out run/mem.jsonl
   ```
   `--trigger toggle --presses 2` double-taps to start and stop (also `--press-ms` and `--press-gap-ms`); the default holds the key while the audio plays. `--modifiers` holds modifier keys around `--key`, which also takes a virtual key code. `--settle-s 10` ends a segment's observation once the field holds text that has not changed for 10 seconds and the stop key is 10 seconds past; the default observes to the deadline. It must be 0 or more than 2 s, since a row whose text changed within 2 s of the end of observation is a timeout.
5. Score: `python3 score.py --manifest run/manifest.jsonl --results run/results.jsonl [--keep-fillers] [--json out.json]`.

A driver fault ends the run with exit code 2 after writing the segment's row. When the fault comes between the start and the stop gesture, the driver first stops that dictation (with `toggle`, it sends the stop gesture) and waits until the field has been still for 3 s, so the next segment does not inherit the recording. Running the same command again skips the segments that already have a fault-free row.

## A plan of cells

A cell is one system (an app in one configuration) and one language block: `en` holds the English segments, `zh` the Chinese and mixed ones. For each cell, `run_cells.py` quits every app in the plan, puts back the settings saved before the plan and sets what the cell changes, so a field the cell does not name keeps the user's value whatever ran before. For Voxt it also sets BlackHole 2ch as the microphone (`audio.devices` in 2.x, `activeInputDeviceUID` in 1.x), since a preferred microphone would win over the system's default input. It then launches the app and a fresh sink, plays three warm-up segments that are not scored, and runs the block once the warm-up has finished. Each cell records the app's version, executable hash and the settings in effect (`cells.jsonl`). A cell interrupted by a driver fault resumes; finished cells are skipped, so the plan can be started again after any interruption. Rows left in a run directory are reused only for the same plan: a cell whose saved manifest, system or `settle_s` differs is refused, and so is a `plan.json` that differs in anything but `cells`. The plan exits with status 1 when a cell is left incomplete.

Before the first cell, `run_cells.py` copies the settings it is about to change into `RUN_DIR/settings-backup/`, writing each copy to a temporary file and renaming it once complete: the settings document (`settings:app`) of each Voxt 2.x database, and Voxt 1.x's defaults with `defaults export`. After the last cell, or when the run stops on an error or Ctrl-C, it quits the apps, puts the copies back and deletes them; the history the apps recorded during the run stays. A copy left by a killed run is checked and kept, and the next start restores it; a copy that does not parse stops the run. Relative `wav` paths in the plan's manifest are read against the manifest's directory.

```bash
python3 run_cells.py --plan plan.json --check
caffeinate -dimsu python3 run_cells.py --plan plan.json --root PILOT_DIR --pilot 5
caffeinate -dimsu python3 run_cells.py --plan plan.json --root RUN_DIR
python3 run_cells.py --summary --root RUN_DIR
```

`family` says how a system is configured: `2x` is Voxt 2.x, whose main language, speech model and language model clean-up the cell writes into its settings database; `1x` is Voxt 1.x, set with `defaults`; `other` is any other dictation app, set up by hand once. `--pilot N` runs the first cell of each system on its first N segments, to see that every app answers the shortcut before a full run.

## AASR-Bench sample

```bash
python3 aasr_sample.py select --benchmark benchmark.jsonl --n 150 --out sample.jsonl
python3 aasr_sample.py fetch --sample sample.jsonl --dir W/aasr --out W/manifest.jsonl
```

`select` takes each language and scene's proportional share of N, rounded by largest remainder, and within a stratum the utterances in the order of the SHA-256 of their `source_record_id`, so the sample depends on the benchmark alone. `fetch` downloads the clips at a pinned revision of the dataset, checks each against the dataset's checksums and writes the manifest. The dataset is for evaluation only; keep the clips on the measuring Mac.

## What a segment records

Every timestamp is `mach_continuous_time` in nanoseconds, shared by the driver and the sink.

1. The sink empties its field (`t_clear`). If its window is not key, the driver activates it once, and records a driver fault without pressing a key if that fails.
2. The driver presses the shortcut (`t_keydown`), plays 1.5 s of silence, the wav and 0.5 s of silence in real time (`t_audio_start`, `t_audio_end`), and releases the shortcut (`t_stop`: the key-up with `hold`, the last key-up of the stop gesture with `toggle`).
3. It observes the field until `t_stop` + 60 s + the wav's length and writes:
   - `t_first`, the first change after `t_keydown` that leaves text in the field, and `t_last`, the last change before the deadline. A system that writes while the speaker talks gets a negative `t_first − t_stop`.
   - `final_text`, the field at the deadline, which scoring uses.
   - `status`: `empty` when the field holds only whitespace, `timeout` when it still changed within 2 s of the deadline, else `complete`.
   - `focus_lost`, `n_changes_before_stop`, `n_changes_before_keydown` and `late_changes_prev` (writes that arrived after the previous segment's deadline).
4. With `--mem-app NAME`, the peak `phys_footprint` of the app's process tree between `t_keydown` and the deadline. The tree is every process whose executable lies inside `NAME.app/`, its helpers and XPC services included, plus their descendants, sampled at 10 Hz into `mem.jsonl`. `replay-driver mem --app NAME --out idle.jsonl --seconds 600` samples an idle app.

Insertion latency is `t_last − t_stop`; `score.py` reports its median over `complete` segments, and calls a segment whose last edit came before the stop key `early`. An app that finishes while the key is held makes early segments, so the report prints their count next to each median.

A write that arrives after a segment's observation ended is logged in the next segment: before its clear (`late_changes_prev`) or between its clear and key-down (`n_changes_before_keydown`). The earlier segment then misses that write, and the later one may keep it as its final text. A segment that ended as a `timeout` can still be written to after the next key-down, where neither counter sees it. `score.py` marks as `contaminated` both rows of a late write, the row after a timeout and any row that lost focus, leaves them out of error rates and waits, and prints how many there are and why.

## Error rates

Before tokenization, Traditional characters become simplified with ICU's Hant-Hans transform, on the reference and the output alike. The rest follows Voxt's eval (`internal/eval/score.go` at `voxt-v2.0.1`): full-width to half-width, lower case, Chinese numerals and English number words to digits, punctuation dropped, and the fillers 嗯 呃 额 啊 唔 um uh er erm ah hmm dropped from both sides (`--keep-fillers` keeps them). CJK characters count one token each and Latin text counts by word. `[UNK]` is removed from references. A segment with an empty reference is a silence or noise control and counts only towards the false-output rate.

## Tests

Run these on a Mac before changing the harness; CI does not run them.

```bash
python3 -m unittest discover -s tests
./build.sh && python3 tests/e2e_dryrun.py [--play-blackhole]
```

The end-to-end self-test runs the driver with `--dry-run`: no key events, the audio written to a file at the real-time pace, and a fake app typing through the sink's socket. `--play-blackhole` adds a segment played to BlackHole 2ch.
