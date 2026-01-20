# OmniEngine Implementation Progress

## Overview

Tracking progress for implementing the OmniEngine framework as designed in [omni_engine_design.md](./omni_engine_design.md).

---

## Milestones

### Milestone 1: Core Types & Protocols
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/base.py` | ✅ Exists | Already has Engine ABC + EchoEngine |
| `engines/omni/types.py` | ✅ Done | Request, RequestStatus, SchedulerOutput, RequestOutput, ModelRunnerOutput |
| `engines/omni/runtime/interfaces.py` | ✅ Done | BatchPlanner, ResourceManager, IterationController, Input/Output protocols |
| `engines/omni/__init__.py` | ✅ Done | Exports types |
| `engines/omni/runtime/__init__.py` | ✅ Done | Exports runtime components |

---

### Milestone 2: Generic Scheduler
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/scheduler.py` | ✅ Done | Generic Scheduler class |

**Test**: Unit test with mock runtime components

---

### Milestone 3: Generic ModelRunner
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/model_runner.py` | ✅ Done | Generic ModelRunner class |

**Test**: Unit test with mock components

---

### Milestone 4: OmniEngine
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/engine.py` | ✅ Done | OmniEngine class |
| `engines/omni/__init__.py` | ✅ Done | Public exports |

**Test**: Integration test with mock components

---

### Milestone 5: Encoder Support
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/runtime/encoder.py` | ✅ Done | EncoderBatchPlanner + data types + preparer + processor |
| `engines/omni/factory.py` | ✅ Done | `create_encoder_engine()` |

**Test**: Run with real BERT model

---

### Milestone 6: AR Support
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/runtime/ar.py` | ✅ Done | ARBatchPlanner + data types + preparer + processor |
| `engines/omni/factory.py` | ✅ Done | `create_ar_engine()` |

**Test**: Run with small LLaMA/GPT-2

---

## Next TODOs

- [ ] Refresh tests for runtime protocols (scheduler, runner, engine)
- [ ] Add step_id validation if/when multiple in-flight batches are enabled
- [ ] Add batched AR support and paged KV cache manager

---

## File Structure (Target)

```
sglang_omni/engines/
├── base.py                    # ✅ Engine ABC (exists)
├── __init__.py                # Update exports
│
└── omni/
    ├── __init__.py
    ├── types.py               # M1
    ├── scheduler.py           # M2
    ├── model_runner.py        # M3
    ├── engine.py              # M4
    ├── factory.py             # M5, M6
    │
    └── runtime/
        ├── __init__.py
        ├── interfaces.py      # M1
        ├── common.py          # M1
        ├── encoder.py         # M5
        └── ar.py              # M6
```

---

## Cleanup Plan

After OmniEngine is working:
- [ ] Remove `engines/encoder/` (replaced by `engines/omni/runtime/encoder.py`)
- [ ] Update `engines/__init__.py` to export OmniEngine

---

## Log

| Date | Milestone | Action |
|------|-----------|--------|
| 2026-01-19 | M1 | Core types & protocols implemented |
| 2026-01-19 | M2 | Generic Scheduler implemented |
| 2026-01-19 | M3 | Generic ModelRunner implemented |
| 2026-01-19 | M4 | OmniEngine implemented - Framework complete! |
| 2026-01-19 | M5 | Encoder support implemented |
| 2026-01-19 | M6 | AR support implemented - All milestones complete! |
