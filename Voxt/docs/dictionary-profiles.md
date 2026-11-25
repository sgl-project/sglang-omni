# Dictionary profiles

A dictionary profile is a named set of dictionary entries. Use different profiles for different situations, such as work and home. Transcribe, Translate, and Rewrite use only the active profile's entries for speech hints, alias replacement, and the language model glossary.

Voxt moves an existing dictionary into the active Default profile. Its entries keep their terms, aliases, notes, switches, and timestamps.

The Dictionary page can create, rename, delete, and switch profiles. The active profile is selected on the page. Deleting the active profile selects a remaining one. Voxt does not delete the last profile.

A profile contains the entries on the Terms view. Imports, exports, selected-text additions, and accepted suggestions apply to the active profile. The Personal dictionary Markdown and Speech context remain shared across profiles.

## Switch from the command line

Run this command with the profile's name:

```sh
open -g 'voxtnew://dictionary/profile?name=Work'
```

URL-encode spaces and other reserved characters. For example, this command selects Research & Writing:

```sh
open -g 'voxtnew://dictionary/profile?name=Research%20%26%20Writing'
```

The command uses Voxt's existing `voxtnew://` URL scheme. It needs no new permission, and Settings → Privacy → Let programs on this Mac read the history can stay off. When Voxt is running, `-g` keeps it in the background and the switch leaves the current page unchanged. macOS opens Voxt if it is not running.

In Hammerspoon, run the same command with `hs.execute`:

```lua
hs.execute("open -g 'voxtnew://dictionary/profile?name=Work'")
```

In macOS Shortcuts, add a Run Shell Script action and use the same `open -g` command.

An unknown name leaves the active profile unchanged. The `open` command reports whether macOS accepted the URL; it does not report whether Voxt found the profile.
