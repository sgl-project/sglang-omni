# Root Go file migration plan

Base: `c7e31a0a2b465d21b03a4049448347a441c20c98`. At this commit, `Voxt/` has 505 root Go files: 305 production files and 200 test files. Every file declares `package main`.

## Package boundaries

Go packages are directory-scoped, and the root executable cannot import another `main` package. A file that leaves the root must join or start an importable package.

The current architecture keeps MyGo views, localized view-state builders, native controllers, shared UI, demos, and integration tests in root `package main`. Pure feature logic can move to `internal/pages/<feature>`. A page package reaches another page through its `Host`, which root `app` implements; `internal/archtest` rejects direct imports between page packages. Keep MyGo and `i18n` out of these page packages. Add every new internal import to `internal/archtest/imports.txt`.

In the table, `main` means the root package, `pages/x` means `internal/pages/x`, and `providers` means `internal/providers`. `main + pages/x` means that pure declarations move while views and app-facing adapters stay in root. Split mixed files instead of moving app state or view code into a domain package.

Patterns match root basenames. `meeting*` covers both `meeting_*.go` and `meetings_*.go`. Rewrite and Translate have no separate production files: their workflows live in `session*`, their pages live in `feature*`, and their integration tests live in `session_e2e_*_test.go` and `write_pages_test.go`.

## Target map and current blockers

The blocking PR column comes from the complete, paginated file lists of every open PR on 2026-11-19. Before each move, refresh the lists. A group can move only when its files and every required caller edit are clear of open PRs.

| Phase | Root files | Target | Files | Blocking PRs |
|---|---|---|---:|---|
| 1 | `legacy_openai.go` | `providers` | 1 | None |
| With consumers | `ui_*`, `find_marks*`, `floating*`, `helpers.go` | `main` | 16 | 3856, 3861, 4073, 4174, 4186 |
| 2 | `history*` | `main + pages/history` | 27 | 3861, 3959, 4070, 4073, 4101, 4163, 4174, 4186, 4240, 4291 |
| 2 | `meeting*`, `settings_meeting*`, except `meeting_demo*`, `meeting_fixture*` | `main + pages/meeting` | 34 | 3861, 4069, 4073, 4098, 4141, 4162, 4172, 4174, 4178 |
| 2 | `appenhance*`, `appicons*` | `main + pages/appenhance` | 18 | 3856, 3894, 4073, 4174 |
| 2 | `dictionary*`, `notes*`, `learn*` | `main + pages/dictionary` | 32 | 3856, 3975, 4047, 4073, 4098, 4162, 4175, 4191, 4241, 4291 |
| 2 | `models*`, `settings_about_models*`, `speech_page_order_test.go` | `main + pages/models` | 33 | 3856, 3861, 4056, 4069, 4070, 4073, 4099, 4162, 4241, 4281, 4287, 4288, 4291 |
| 3 | `updates*`, `update_check.go`, `release_appcast_test.go`, `upgrade_script_test.go`, `download_links_test.go`, `build_*` | `main + pages/updates`; release checks and build-tag hooks stay in `main` | 16 | 3856, 3861, 4048, 4073, 4201, 4212, 4229, 4232, 4259, 4287, 4291, 4293 |
| 3 | `remote*`, `llm*` | `main + pages/remote` | 30 | 3856, 3861, 4056, 4073, 4122, 4170, 4172, 4233, 4241, 4264, 4291 |
| 3 | `transcribe_*` | `main + pages/filetranscribe`; CLI dispatch stays in `main` | 3 | 3861, 4098, 4162 |
| 4 | `session*`, `rewrite*`, `mute*`, `stop_answer_e2e_test.go`, `copy_result_test.go`, except `session_overlay_*` | `main + pages/session` | 32 | 3861, 4046, 4047, 4049, 4056, 4070, 4073, 4101, 4163, 4164, 4167, 4175, 4191, 4197, 4201, 4203, 4217, 4233, 4243, 4244, 4265, 4267, 4291 |
| 5 | `overlay*`, `session_overlay_*`, `native_overlay_check.go`, `transcript_*` | `main + pages/session/overlay` | 37 | 3861, 4056, 4070, 4073, 4101, 4126, 4164, 4167, 4172, 4174, 4237, 4241, 4287, 4291 |
| 5 | `feature*`, `try*`, `write_pages_test.go` | `main + pages/feature` | 25 | 3856, 3861, 4046, 4056, 4070, 4073, 4101, 4174, 4175, 4241, 4243, 4287, 4291 |
| 6 | `home*` | `main + pages/home` | 12 | 3856, 3861, 4069, 4070, 4073, 4162, 4172, 4174, 4231 |
| 6 | `onboarding*`, `opening*` | `main + pages/onboarding`; keep embedded assets with their view | 20 | 3861, 4056, 4069, 4070, 4073, 4162, 4172, 4243, 4268, 4291 |
| 6 | `settings*`, `shell_settings*`, `permissions*`, `key_recorder*`, `shortcuts_snapshot_test.go`, `sounds_*`, `sync_sounds_test.go`, `dock_test.go`, except `settings_meeting*`, `settings_about_models*`, `settings_demo_host.go`, `settings_fixture_build_test.go` | `main + pages/settings` | 72 | 3856, 3861, 4049, 4056, 4070, 4073, 4101, 4163, 4172, 4174, 4175, 4178, 4186, 4238, 4240, 4241, 4254, 4275, 4283, 4287, 4291 |
| 7 | `shell*`, `announcements*`, `tray*`, `pages.go`, `window_close_test.go`, except `shell_settings*`, `shell_demo_host.go` | `main + pages/shell` | 35 | 3856, 3861, 4049, 4056, 4073, 4098, 4162, 4172, 4174, 4175, 4178, 4230, 4241, 4254, 4287, 4291 |
| 7 | `legacy*`, except `legacy_openai.go` | `main + pages/legacy` | 14 | 3861, 4056, 4122, 4170, 4241, 4279, 4291 |
| 7 | `meeting_demo*`, `meeting_fixture*`, `settings_demo_host.go`, `shell_demo_host.go`, `settings_fixture_build_test.go` | `main` | 13 | 4070, 4073, 4163, 4175, 4178, 4186 |
| 7 | `app.go`, `main.go`, `library.go`, `events.go`, `helperlink.go`, `family.go`, `family_runtime.go`, `logs_check.go`, `smoke.go`, `sysinfo.go` | `main` | 10 | 3975, 4048, 4070, 4073, 4101, 4141, 4162 |
| With its feature | `eval_*`, `mac_e2e_*`, `access_reach_test.go`, `button_heights_test.go`, `dialogs_*`, `family_runtime_test.go`, `family_ui_test.go`, `live_text_test.go`, `local_fakes_test.go`, `name_test.go`, `render_bench_test.go`, `resource_bench_test.go`, `snapshot_*`, `testutil_test.go`, `text_fit_test.go` | `main` | 25 | 3861, 3959, 4047, 4056, 4069, 4070, 4073, 4101, 4103, 4162, 4174, 4175, 4198, 4242, 4254, 4273, 4274, 4291 |

A set-partition check over these include and exclude patterns assigned all 505 files to exactly one row, with no unmatched or duplicate paths. Recheck coverage whenever a PR adds, removes, or renames a root file.

## Move order

1. Move `legacy_openai.go` into `internal/providers`. Export only `LegacyModelIs`, `LegacyOpenAIGeneration`, and `LegacyResponsesAPI`; keep `legacyOpenAIEfforts` private. The only external caller is `legacy_settings.go`, which already imports `providers`. This move adds no internal import edge and touches no file in an open PR.
2. Extract leaf logic from History, Meeting, App enhancement, Dictionary, and Models after their blockers clear. Keep each feature's views and app adapters in root. Use one package per commit.
3. Move Updates, Remote, and file transcription next. Leave release checks, startup hooks, and CLI dispatch in root.
4. Split Session workflows. Rewrite, Transcribe, and Translate stay in the Session and Feature packages rather than starting duplicate packages around the same workflow.
5. Move the Session overlay and Feature pages after the Session interfaces stabilize.
6. Move Home, Onboarding, and Settings after their Session and Feature dependencies have stable interfaces.
7. Finish Shell, the remaining legacy import code, shared demos, and app composition last. Keep integration tests in root and move a test only with the package that owns its API.

Use `git mv` for whole-file moves and export only declarations that external callers need. Do not add forwarding wrappers to bypass a blocked caller. For a same-package move, compare `declhash` output exactly. For a package extraction, use a comparison that normalizes only the new package qualifiers and exported names, then prove that the remaining source is identical. Run the Go tests, `go tool mygo vet .`, `internal/archtest`, `TestKeyParity`, and `pre-commit` after every package.
