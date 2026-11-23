# Importing dictionary terms from the command line

`Voxt dictionary import` adds terms to your dictionary from a file, the same way Dictionary → Import… does. Use it from a script: to keep a list of names and terms in a file you edit and import it again, or to add the corrections you find when you review your transcripts.

## The command

The executable is the app itself. Voxt must be running:

```sh
VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
$VOXT dictionary import terms.csv
$VOXT dictionary import --dry-run terms.csv
$VOXT dictionary import terms.txt --json
```

- `FILE` is read as Dictionary → Import… reads it, by its extension:
  - CSV (any other extension): the columns `term`, `aliases`, `note`, `enabled`, by a header that names them (also `词条`, `别名`, `备注`, `启用`) or in that order. Aliases are separated by commas or semicolons.
  - Text (`.txt`): a term per line, `Term = alias, alias` for aliases, `#` for a comment.
  - A Voxt 1.x dictionary export (`.json`).
- `--dry-run` says how many terms the import would add and update, and saves nothing.
- `--json` prints the result as one JSON object (`added`, `updated`, `skipped`, `dry_run`) instead of a sentence.

The import merges as Dictionary → Import… does. A spelling you do not have yet (ignoring case) is added with the file's aliases, note and switch. A spelling you have gets the aliases it lacked, and the file's note only when it had none. Nothing is removed or turned off. Rows without a term are skipped and counted. Importing the same file again changes nothing.

When the import saves terms, Voxt refreshes the Dictionary page and shows the same toast as Import…. The next dictation uses the new terms: sessions read the dictionary when they start.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Done (or counted, with `--dry-run`). |
| 2 | Wrong arguments. |
| 3 | Voxt is not running. |
| 4 | The import stopped, or its JSON result could not be written: the file is not a CSV or text list of terms or a Voxt 1.x export, the dictionary could not be saved, or stdout failed. stderr says which. |
| 5 | The file does not exist or cannot be read. |

## Privacy

The command reads the file itself and sends its name and bytes to the running Voxt through a socket in `~/Library/Application Support/VoxtNew` that only your account can open (`dictionary.sock`, mode 0600, no network port), as `Voxt transcribe` does. Voxt reads none of your folders. A successful answer contains only the counts; a failed answer returns the file name and detailed error to the command so it can explain what went wrong. The app log contains only the counts and a generic failure, never the file name, terms, or detailed error. Files over 8 MB are refused both by Dictionary → Import… and by the command.
