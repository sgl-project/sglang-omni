# Starting Voxt from other apps and devices

`Voxt toggle` and `Voxt cancel` do what Voxt's shortcuts do, from anything that can run a command: a launcher such as Raycast or Alfred, a key remapper such as Karabiner-Elements or BetterTouchTool, the Shortcuts app, a Stream Deck or a foot pedal. Use them to start Voxt from a key or button its own shortcuts cannot use: for example a third-party keyboard's Fn key, which never reaches macOS (only Apple keyboards report it), or a mouse's side button.

## The commands

The executable is the app itself. Voxt must be running:

```sh
VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
$VOXT toggle transcribe     # start dictating, or finish the recording that is on
$VOXT toggle translate
$VOXT toggle rewrite
$VOXT toggle note
$VOXT cancel                # what the cancel shortcut (Esc) does
```

`toggle FEATURE` does exactly what one tap of that feature's shortcut does, by the same rules:

- With nothing recording, it starts the feature in the app in front. That app gets the text, as with the shortcut.
- While a recording is on, it finishes it, whichever feature started it. With Settings → Shortcuts → Strict shortcuts on, only the feature that started the recording finishes it.
- While Voxt recognises or writes, it waits, as the shortcut does.

The command returns once Voxt has acted on it, not when the text arrives. A feature hidden in Settings → General does nothing, as its shortcuts do, and the command says so. The feature needs no shortcut of its own: the command works with its shortcut cleared.

## Examples

**Karabiner-Elements**: F13 as push-to-talk, recording while it is held. Add a rule in Complex Modifications:

```json
{
  "description": "F13 dictates with Voxt while held",
  "manipulators": [{
    "type": "basic",
    "from": { "key_code": "f13" },
    "to": [{ "shell_command": "/Applications/Voxt.app/Contents/MacOS/Voxt toggle transcribe" }],
    "to_after_key_up": [{ "shell_command": "/Applications/Voxt.app/Contents/MacOS/Voxt toggle transcribe" }]
  }]
}
```

Leave out `to_after_key_up` to tap the key once to start and once to finish.

**Raycast**: a Script Command in `silent` mode, so that Raycast's window is gone and the app you were writing in is in front again when Voxt starts:

```sh
#!/bin/bash
# @raycast.schemaVersion 1
# @raycast.title Dictate with Voxt
# @raycast.mode silent
/Applications/Voxt.app/Contents/MacOS/Voxt toggle transcribe
```

**Shortcuts**: a shortcut with one Run Shell Script action running `/Applications/Voxt.app/Contents/MacOS/Voxt toggle transcribe`. Give it a keyboard shortcut or put it in the menu bar.

**BetterTouchTool, Stream Deck and others**: run the same command as a shell script or terminal command.

Whatever runs the command, the text goes to the app in front when Voxt starts. A tool that keeps its own window in front while it runs the command (a launcher in its default mode, a terminal you type the command into) gets the text itself.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Voxt acted on the command. |
| 2 | Wrong arguments. |
| 3 | Voxt is not running. |
| 4 | Voxt did nothing: the feature is hidden in Settings → General, or Voxt did not answer within 10 seconds. stderr says which. |

## Privacy

The command sends only the action's name to the running Voxt through a socket in `~/Library/Application Support/VoxtNew` that only your account can open (`trigger.sock`, mode 0600, no network port), as `Voxt transcribe` and `Voxt dictionary import` do. No text or audio goes through it, and the overlay shows every recording a command starts, as it does for the shortcuts.
