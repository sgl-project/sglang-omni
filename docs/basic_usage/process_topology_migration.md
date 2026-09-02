# Process Topology Migration

`StageConfig.process` is now the only source of process membership. The legacy
`--isolate-stage`, `--stage-process`, and `fused_stages` entries have been
removed and are not auto-migrated.

## Replacements

| Removed entry | Replacement |
| --- | --- |
| `--isolate-stage vocoder` | `--vocoder.process vocoder` |
| `--stage-process preprocessing=frontend` | `--preprocessing.process frontend` |
| `fused_stages: [[a, b]]` | Set the same `process` value on stages `a` and `b`. |

The CLI already implies the `stages.` prefix. An explicit flag such as
`--stages.vocoder.process` is rejected.

## Migrating `fused_stages`

Before:

```yaml
fused_stages:
  - [preprocessing, audio_encoder]
```

After:

```yaml
stages:
  preprocessing:
    process: frontend
  audio_encoder:
    process: frontend
```

Keep all existing factory, routing, runtime, and placement fields when updating
a complete config. Use the actual Stage Names declared by the model config;
legacy isolation-role aliases are not translated. Re-check edge transport and
memory budgets when a handoff changes from local to cross-process. Every non-TP
stage must declare `process`; equal names share one OS process and different
names isolate stages. A TP stage owns its process exclusively.

The top-level `processes` mapping only configures replicas for process names
already declared by stages. See the
[configuration reference](../developer_reference/config.md#logical-processes-and-replicas)
for validation and placement rules.
