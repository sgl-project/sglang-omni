# Voxt in this repository

`Voxt/` started as a byte-for-byte import of [hehehai/voxt](https://github.com/hehehai/voxt)
at commit `baa6f316a6648cc0ef90c70d724acd10085eedaa` (tree `7cc4dd3fb6383406b7a63b911ee57f1019ee89fe`),
without `.git`, caches, models or build products. Voxt keeps its own license in `LICENSE`.

Check that the imported snapshot is unchanged:

```bash
git rev-parse 5dd24587:Voxt   # 7cc4dd3fb6383406b7a63b911ee57f1019ee89fe
```

Later commits add, without changing upstream behavior for other models:

- `backend/`: the supervisor that owns Voxt's local SGLang-Omni server, the
  model views it serves from, and the pinned Python environment.
- `Voxt/Transcription/Omni*.swift`: the client for that server, and routing of
  the migrated checkpoints to it when the Omni development build enables it.
- `Config/OmniDev.xcconfig`, `Voxt/VoxtOmniDev.entitlements`: the isolated
  development build (`backend/run_omni_dev.sh`).
