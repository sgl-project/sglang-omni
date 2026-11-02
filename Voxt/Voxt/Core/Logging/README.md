# Logging

Logging infrastructure for app diagnostics, Apple Unified Logging, local rolling files, and user-facing log export.

## Responsibilities

- Provides typed log categories and a `swift-log` backend for app modules.
- Sends records to OSLog and a local rolling diagnostic file store.
- Redacts sensitive values before logs are displayed, exported, or persisted.
