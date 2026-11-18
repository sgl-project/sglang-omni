# Reading the history from other programs

Voxt can hand your dictation history to a program you run on this Mac: a nightly script that writes Markdown into a notes folder, a personal agent, or anything else that can run a shell command. This page is the whole interface.

## What the switch does and does not do

Settings → Privacy → Other programs → **Let programs on this Mac read the history** is off by default and asks once before it turns on. With it off the connection does not exist, and `Voxt history` fails with exit 3. With it on, each read is logged (time, entry count, program name) and a program name not seen before raises a notice. The log holds no history text.

**The switch governs this command and its socket, nothing more.** Voxt is not sandboxed and keeps its data in `~/Library/Application Support/VoxtNew`. Any non-sandboxed program running under your account can read `voxtnew.db` directly, whether this switch is on or off, and that reading is not logged. The switch makes the documented way in opt-in and visible; it does not protect your history from a program already running as you. Other macOS accounts cannot reach the socket (mode 0600, inside `~/Library`), and there is no network port, so web pages, other devices and your network cannot reach it either. Do not treat this as a security boundary against software on your own account.

Where a program sends the text afterwards, a cloud model included, is up to that program.

Program names in the log are best effort: the name a program reports with `--as` and the process it runs under can both be forged. They tell you what is using the command, not who to trust.

## The command

The executable is the app itself:

```sh
VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
$VOXT history export --day 2026-11-08 --format md --as daily-obsidian
$VOXT history export --since 2026-11-01 --until 2026-11-08 --kind transcribe
$VOXT history export --since 2026-11-08T09:00:00+08:00
$VOXT history status
```

`export` needs `--day YYYY-MM-DD`, or `--since T` with an optional `--until T` (RFC 3339, or a date meaning its local midnight). `--day` is the calendar day in this Mac's time zone, so it is 23 or 25 hours long when the clocks change. Ranges include their start and exclude their end, and rows come oldest first. Without a range it refuses; there is no "everything" default.

`--format` is `jsonl` (default, one JSON object a line, Chinese as written) or `md` (a `## YYYY-MM-DD` section a day, `- 09:14 · App: text` a row). `--kind` is `transcribe`, `translate` or `rewrite`. `--as NAME` is the name shown in Voxt's read log. Stdout carries only data; anything meant for a person goes to stderr.

`status` prints one JSON line: `{"schema":1,"keep_history":true,"keep_days":90,"entries":1234,"oldest":"2026-08-10T09:14:00+08:00"}`. `keep_days` is 0 when no clean-up by age runs. It is the way to check that the history is being kept at all.

| Exit code | Meaning |
|---|---|
| 0 | Done |
| 2 | Usage error |
| 3 | No answer: Voxt is not running, or the switch is off (it cannot tell these apart) |
| 4 | The read stopped before the end, for example the switch was turned off during it; whatever was printed is incomplete |

The command opens no window and no database, and leaves Voxt's windows alone.

## What a row contains

| Field | Meaning | Present |
|---|---|---|
| `id` | The entry's id (entries imported from Voxt 1.x have ids that are not ULIDs) | Always |
| `created_at` | RFC 3339, in this Mac's local zone | Always |
| `kind` | `transcribe`, `translate` or `rewrite` | Always |
| `app` | Name of the app in front | When known |
| `text` | The final text that was delivered | Always |
| `heard` | What speech recognition heard, before any clean-up | Always |
| `instruction` | The spoken instruction of a rewrite | When there was one |
| `target_lang` | The translation's target language | For a translation |
| `selection` | Text you selected in the app | **Only** with Settings → Privacy → Also include the text you selected |

Only entries that finished with a result are listed; failed or cancelled ones are not. Never exported, by this command or by Export text: recordings, the URL host, error messages, rule names, model names, timings. Within schema 1 fields are only added, never removed. The socket's own wire format is internal to Voxt and can change; the command's output and exit codes are the contract.

## A daily file, without opening Voxt

There is no scheduler inside Voxt. Use launchd: it runs the script below once a day (and once when the Mac wakes if it slept through 07:00). Turn on Settings → General → Launch at login so Voxt is usually running; if it is not, the script stops and tries again tomorrow.

Save this as `~/bin/voxt-daily.sh` and make it executable (`chmod +x`):

```bash
#!/bin/bash
# Writes one Markdown file a day into OUT, for the last 7 days, never today
# (today is incomplete until it ends). Finished days stay as they are.
VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
OUT="$HOME/Documents/Voxt history"
mkdir -p "$OUT"
for back in 1 2 3 4 5 6 7; do
  day=$(date -v-${back}d +%F)
  file="$OUT/$day.md"
  [ -e "$file" ] && continue
  tmp=$(mktemp "$OUT/.voxt.XXXXXX") || exit 1
  "$VOXT" history export --day "$day" --format md --as daily-obsidian >"$tmp"
  code=$?
  if [ "$code" -eq 0 ] && [ -s "$tmp" ]; then # a day with no entries leaves no file
    mv "$tmp" "$file"
  else
    rm -f "$tmp"
    [ "$code" -eq 3 ] && exit 0 # Voxt is not running or the switch is off: try again next time
  fi
done
```

Save this as `~/Library/LaunchAgents/com.example.voxt-daily.plist`, replacing `YOU` with your user name:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.example.voxt-daily</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/bin/voxt-daily.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>7</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
</dict>
</plist>
```

Then load it once with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.voxt-daily.plist`, and run it now with `launchctl kickstart gui/$(id -u)/com.example.voxt-daily`.

Limits worth knowing: the history keeps 90 days by default (Settings → Privacy → History settings), so a Mac off for longer than that loses those days. An entry retried and fixed a few days later stays on its original day, and a day already written is not rewritten; remove the `[ -e "$file" ]` line if you want every run to refresh the last week.

## Saving a file by hand

No script needed: History → History settings → **Export text…** saves Markdown or JSONL to a file you choose (the file is yours: the automatic clean-up and Clear the history do not touch it). It has the same fields and honours the same "selected text" setting, and it works whether the switch is on or off, because you are the one doing it.
