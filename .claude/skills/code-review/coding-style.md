# Coding style

This document describes the coding style for human contributors and AI agents.
Sources: the [style prompt and do/don't examples](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5651721937)
and [additional anti-pattern examples](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5650093336),
including [discarded parameters](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5673011724)
and [f-strings plus no backticks in comments](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5734141980).

## Principles

Write clean, professional, maintainable code. Match the surrounding codebase's conventions
where they exist; where they don't, follow these rules.

The overriding goal is simplicity: fewer, smaller files and fewer functions. Avoid
speculative generality.

## SIMPLICITY & NON-DUPLICATION

- Introduce a helper, wrapper, or abstraction layer only when it is used at least
  twice (≥2 call sites) now and genuinely clarifies the call site.
- Don't reinvent what stdlib or an already-imported dep provides (`itertools`, `functools`,
  `collections`, `pathlib`, `dataclasses`, `pydantic`, `torch.nn.functional`). A 3-line
  wrapper around `lru_cache` is noise.
- Don't pre-extract `_helper`/`_impl` for "cleaner main flow" unless it's reused at
  least twice (≥2 call sites) or too long to read in one screen. A 60-line
  top-to-bottom function beats three 20-line `_step_one/_two/_three` called once each.
- Don't add interfaces/base classes/registries/plugin systems before a second concrete
  implementation exists. Two cases first, then abstract.
- Reuse alone is not enough to justify a trivial helper: inline a short operation
  used only once (1 call site) when extraction adds indirection without clarity.
  Keep nested callbacks local when their scope or captured state requires it.

## LANGUAGE & COMMENTS

- English only: comments, docstrings, log strings, CLI help. (User-facing translatable
  strings go through i18n — not this rule's concern.) If the repo is non-English, match it.
- Comments sparse, one-line, only for the genuinely non-obvious. Restating code is noise.
- Comments must be self-contained and explain _why_, not _how_ or _what_. Don't narrate
  what a code block does or how it does it — the code already shows that. But only document
  the "why" when specific codes are hard to understand without the context of the comments.
  Avoid verbose "why" rationale in comments.
- Preserve license/copyright notices and concise upstream attribution.
- Sign non-obvious/note comments with the author's name: `# note (name): ...`.
- No backticks in Python comments or docstrings. Identifiers stay in plain
  text: `"""Valid frame counts after the encoder's stride-2 conv2."""`
  is correct; wrapping the name in double backticks is not.
- NO process markers: no ★, `# P1`, `# [FIX]`, `# TODO` without a ticket, `# === SECTION ===`
  banners.
- NO provenance leakage: never name other repos/upstream/"the closed source" in source.
- Docstrings: Google-style, 1-3 lines. Args/Returns only when non-obvious. One short module
  docstring per file.
- Do not add opening comment blocks or lengthy explanations after class definitions.
  Use the short module docstring and brief explanations where needed.
- A long-term workaround gets a one-line comment naming the constraint.
- `# noqa: <code>` allowed with a reason; bare `# noqa` is not.

## NAMING

- Classes PascalCase; functions/variables snake_case; constants UPPER_SNAKE.
  Preserve language-defined special methods such as `__init__`.
- **Do not prefix names with `_` by default.** A leading underscore is the
  exception, not a habit or a "this is internal" badge. It is not a marker
  for "used only in this class", "used only in this file", "set in
  `__init__`", or "not part of the HTTP API". Use `_` only when the name
  must stay invisible to every caller outside its defining class or module
  — a helper that would be a mistake to call from anywhere else.
  Attributes other methods of the same class read are public: `is_ragged`,
  never `_is_ragged`. Module-level constants are public UPPER_SNAKE:
  `FA3_PAGE_SIZE`, never `_FA3_PAGE_SIZE`. Boolean names already start
  with `is_` / `has_` / `should_` / `can_`; do not add a second underscore
  in front. Public functions, variables, and constants must not have a
  leading underscore.
- Precision beats brevity. Every parameter, local variable, function, method,
  class, and type alias names the unit it is. A reader must know which
  physical unit it is without reading the body or another file. Name that
  unit, not the mechanism: `load_checkpoint`, `speaker_embedding`.
- Write the full word. Do not clip a word, and do not use a generic role
  when the unit is known.
  Wrong: `op`, `SessionOp`, `rid`, `seq`, `cmd`, `ctx`, `req`, `fn`, `cb`,
  `tmp`, `data`, `info`, `item`, `obj`, `handler`, `manager`.
  Right: `operation`, `SessionOperation`, `request_id`, `sequence`,
  `session_operation`, `session_context`, `request`, `request_compute`,
  `session_hooks`.
- Single letters only for loop indices (`i`, `j`) or math (`x`, `y`, `t`).
  Do not append `T` to invent a type name.

## TYPING & SIGNATURES

- Full type hints on every function, method, and attribute. Annotate parameters
  and return types. One syntax only: `X | Y`, `list[int]`, `dict[str, int]`,
  `X | None`. Do not introduce `Optional` or `Union`.
- Either `requires-python >= 3.10` (native `X | Y`) or `from __future__ import annotations`.
  A quoted annotation is only a forward reference inside the same class
  (`"ModelConfig"`). Do not quote a name to avoid importing it.
- Do not use `if TYPE_CHECKING:`. It hides imports from runtime and from
  pre-commit. Import the name at module level, or write the concrete type in
  the annotation (`tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
  instead of a gated alias). If an import is circular, move the shared type
  into a third module.
- Closed value sets → `Literal[...]` or `Enum`, not bare strings in comparisons.
- No mutable function defaults: `def f(x=[])`/`= {}` are bugs. Use a `None` sentinel.
- Every annotation names a concrete type: a dataclass, TypedDict, NamedTuple,
  Enum, `Literal`, or a union of those. A reader must be able to see the
  fields, the element types, and the return type. Parameterize every
  container (`dict[str, int]`, `list[TokenId]`, `tuple[torch.Tensor, torch.Tensor]`).
  This includes model and decoder types, structured values, and resource handles.
- Vague types are banned in annotations, aliases, and casts. They disable
  checking and hide the real contract. Do not write any of these, including
  under `typing` or `collections.abc`:
  `Any`, `AnyStr`, `object`, `Callable`, `TypeVar`, `Generic`, `ParamSpec`,
  `Concatenate`, `TypeVarTuple`, `cast`, `Optional`, `Union`,
  `dict[str, Any]`, `list[Any]`, `tuple[Any, ...]`, `Sequence[Any]`,
  `Mapping[str, Any]`, `Iterable[Any]`, `Coroutine[Any, Any, T]`, `type[Any]`.
  Bare `dict`, `list`, `tuple`, `set`, `Mapping`, `Sequence`, `Iterable`,
  `Iterator`, and `Collection` are the same failure. A coroutine that does
  not yield is `Coroutine[None, None, Concrete]`, with `Concrete` named.
  Wrong: `def append(self, state: object, emit: Callable[[TimedChunk], None]) -> Any`.
  Right: `emit` is a `ChunkEmitter` Protocol whose `__call__` takes `TimedChunk`
  and returns `None`, and `append` returns `StagePayload`.
- A callback is a `Protocol` whose `__call__` names each parameter and the
  return type. Do not recover an erased signature with `*args` or `**kwargs`.
  If two wrapped functions do not share one parameter list, write each one
  separately and name its parameters. Do not use `TypeVar` to connect a
  stored value to a later parameter. If callers do not share one type, the
  code that creates the value keeps it and names that type. The shared
  signature does not accept it.
- The only `object` exception is an untrusted boundary: decoded JSON, a wire
  dict, or a raw request body. That parameter may be `object` or
  `Mapping[str, object]`. The next use narrows it with `isinstance` before
  reading a field, and does not pass the `object` onward. Do not use `cast`
  or `# type: ignore` in place of that narrowing.
- Do not accept a parameter only to immediately delete it to silence type or lint
  checks, such as starting a function with `del request_id`. Remove unnecessary
  parameters and update callers. If an established interface requires an unused
  parameter, preserve the contract and make that constraint explicit rather than
  pretending the parameter is used.

## DATA STRUCTURES

- Internal value objects (config, messages, state): `@dataclass`, prefer `kw_only=True`;
  mutable defaults via `field(default_factory=...)`.
- Cross-boundary schemas (API req/resp, untrusted input, needs validation): `pydantic.BaseModel`.
  Don't hand-roll validators pydantic gives free.
- Don't mix the two for one concept. Pick per role, not per mood.
- Remove a mapping entry with `mapping.pop(key)`. Do not write `del mapping[key]`.
  `del` also deletes names and attributes, so the statement does not say that
  an entry is leaving a mapping. `pop` names the mapping and the key.

## FILE STRUCTURE

- File > ~400 lines → extract a module. But a 30-line file holding one `_helper` called once
  is also wrong — merge it. Related functions belong together; one concern per file
  does not mean one function per file.
- One entry-point mechanism per repo (hydra/argparse/fire/`[project.scripts]`). `if __name__
== "__main__"` only in entry-point modules, never library modules.
- Module-level side effects (env mutation, global state, `setup_root`, resolver registration)
  only in entry-point modules. Library imports must be side-effect-free.
- No dead code: unreachable branches, commented-out blocks, stale TODOs,
  "kept for later" stubs.

## ERROR HANDLING & MATURITY

- `assert` for internal invariants (shapes, device, dtype, preconditions you control) — fail
  fast, loud, consistently across sibling modules.
- `raise ValueError/RuntimeError/FileNotFoundError` for user input / environment errors.
- No bare `except:`. No `except Exception as e: pass`. Catch specific exceptions; if broad,
  log with a reason or re-raise.
- Maturity progression — this matters:
  - Prototyping: try/except around the call you don't yet trust is fine.
  - Mature: remove debugging-only guards and workarounds once the contract is
    established. Keep necessary error handling, trust internal invariants, and
    let unexpected failures surface as stack traces.
  - Review: every try/except and defensive `if` must answer "what breaks if I delete this?"
    If the answer is "nothing, it was for debugging" — delete it.
- Do not wrap large blocks in `try/except` to guard against speculative, extremely
  unlikely failures. Handle errors that realistically occur on the main execution
  path; let unexpected failures surface.
- Do not turn execution failures into fabricated successful outputs. When a
  function guarantees a return type, use its result directly instead of adding
  impossible `None` checks or redundant type coercion. Validate genuinely untrusted
  results at their boundary instead of weakening an established internal contract.
- Access fields on known types directly. Do not use `getattr` defaults or `hasattr`
  probes to hide missing required attributes, or catch attribute errors for
  operations guaranteed by the interface. Missing required attributes should fail
  visibly rather than trigger a search for alternate fields or fallback defaults.

## CONTROL FLOW

- Validate inputs and preconditions before the main logic. Organize conditions
  into mutually exclusive if/elif/else branches. Keep the main execution path
  in the final branch.
- Every `if` has an `else`, or belongs to one `if`/`elif`/`else` chain that ends
  in `else`. A short `if` is where this gets skipped, and a short `if` without
  `else` is still a violation. Raising, returning, or calling one function does
  not exempt it. When the other branch does nothing, write `else: pass`. Do not
  drop the `else` to save a line.

  Wrong:

  ```python
  if session is not None:
      close_session(session)
  payload.data = {"closed": True}
  ```

  Right:

  ```python
  if session is not None:
      close_session(session)
  else:
      pass
  payload.data = {"closed": True}
  ```

## LOGGING

- This repo uses stdlib `logging`. Configure once; don't introduce a second logging library.
- One logger per module: `logging.getLogger(__name__)`.
- Keep messages terse. No manual `[Info]`, `[Warn]`, or `[step N]` prefixes;
  use the logger's level and context fields.
- `print` only for CLI output the end user reads (`--help`, visualization, `__main__` demo).
  Runtime info — even debug — goes through the logger.
- Always use f-strings, including in log calls. Do not use %-style interpolation:
  `logger.info(f"Loading {repo_id} config={config_name} split={split}")`.

## CONFIG & MAGIC VALUES

- No hardcoded URLs, absolute paths, or hostnames. Required deployment values must
  come from validated configuration, without embedded machine-specific defaults.
- Replace unexplained magic numbers with named constants at module scope.
- Empirical constants named + provenance comment: `DECODE_TOKS_PER_SEC = 6.7  # measured on H20`.
- Use one config mechanism, following the repository's existing choice (Hydra /
  argparse+yaml / dataclass / env). If none exists, pick the lightest that fits.
  Do not stack config dataclasses,
  YAML loaders, validators, and CLI overrides when argparse + dict would do.
- Do not ship a YAML file that is "documentation only" and never loaded.
- Define a constant used by only one module in that module; do not create a
  cross-file import solely for it.
- Parameters flow top-down. A default owned by pipeline or stage config
  (FactoryArgs, StageConfig, CLI) is written once at that layer and passed
  down. Do not re-declare the same value as a lower-layer module constant or
  factory default. The factory takes the knobs as required parameters and
  forwards them; it does not invent a second copy of the policy.
  Wrong: `CODE2WAV_MAX_BATCH_SIZE = 8` in stages.py plus
  `FactoryArgs(max_batch_size=8)` in config.py. Right: only the config
  FactoryArgs; `create_code2wav_executor(..., max_batch_size: int, ...)`.

## IMPORTS

- Group: stdlib / third-party / local, blank-line separated, alphabetical within group.
- Manage import paths consistently at the project level. Don’t patch sys.path ad hoc in individual files.
- Import at the top of the file. Do not lazy-import inside a function just
  to keep the factory "light" or to hide a heavy dependency. Wrong: a
  block of `from ... import ...` at the start of
  `create_sglang_talker_executor_from_config`. Right: the same names at
  module scope. Function-local imports are allowed only for optional
  dependencies, necessary initialization ordering, or a documented
  circular-dependency break. Do not use `if TYPE_CHECKING:` to keep an
  import "type-only" or to silence pre-commit.
- For repository-internal imports, import from the defining module using the full
  package path, such as `from xxx.yy.zzz import kkk`, rather than through
  `__init__.py`. Keep package re-exports minimal and define an explicit
  `__all__`; no wildcard imports. For third-party libraries, prefer their
  documented public import paths (e.g. `from pydantic import BaseModel`).
- Do not import a class or factory from a sibling model package. Shared
  runtime belongs in sglang_omni/scheduling (or another non-model module).
  Wrong: MiniCPM-o stages importing Qwen3-Omni StreamingDetokenizeScheduler.
  Right: both models import the shared scheduler and pass their own
  build_result.

## TOOLING

- This repository already configures linting, formatting, and other checks in
  [.pre-commit-config.yaml](/.pre-commit-config.yaml). Run
  `pre-commit run --all-files` before completing a change.
- Test the public contract (inputs → outputs), not internal layout.
  Do not snapshot helper tuples, page tables, prefix sums, or other
  encodings that can change without changing behavior. A representation
  refactor that preserves the caller's result should not break tests.
- Do not add a new test file for one case that belongs next to the
  existing suite. Put GPU or optional-backend cases on the same module
  with a marker; skip when that backend is absent.
- Test actual failure contracts and supported fallback paths;
  do not add tests solely to preserve speculative recovery scaffolding.
