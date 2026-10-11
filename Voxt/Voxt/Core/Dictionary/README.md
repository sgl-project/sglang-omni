# Dictionary

Dictionary domain logic for glossary storage, matching, validation, import, transfer, and learning.

## Responsibilities

- Maintains dictionary entries, suggestions, matching rules, aliases, and validation behavior.
- Supports project scanning, one-click ingestion, export/import, and repository persistence.
- Derives automatic-learning candidates from edits to delivered text; history scanning has its own support path.

## Source boundaries

`DictionaryModels` holds domain values and serialization. `DictionaryStore` owns published state, reload generations, validation caches and writes; `DictionaryStoreQueries` contains read/query and matcher assembly methods using the same immutable dependencies.

`DictionarySuggestionModels` preserves legacy suggestion/history-snapshot Codable values and scan progress/results. `DictionaryHistoryScanPolicy` owns prompt migration, bounds and candidate filtering. `DictionarySuggestionStore` retains private scan/settings state and the legacy file reload/merge/write-back path; its defaults/file URL/reader can be isolated in tests. Both reload paths supersede older reads; asynchronous reads are awaited off the main actor, cancelled results are dropped, and read failures preserve the current snapshot and file. Explicit history scans add directly to `DictionaryStore`. Retired empty discovery, pending-suggestion mutations and their unused UI/test helpers are removed; legacy files and persisted history fields are not deleted.

Automatic learning is separated into observation/request policy (`DictionaryLearningMonitor`), prompt/candidate formatting (`DictionaryLearningPrompt`), scope/line matching (`DictionaryLearningTextScope`), and active edit comparison (`DictionaryLearningTextDiff`). The unused semantic scoring/phrase-expansion branch was removed; token/LCS deletion detection is still active and must remain covered.
