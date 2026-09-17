# Coding style

This document describes the coding style for human contributors and AI agents.
Sources: the [style prompt and do/don't examples](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5651721937)
and [additional anti-pattern examples](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5650093336),
including [discarded parameters](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32#issuecomment-5673011724).

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
- No backticks in Python comments.
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
  Use a leading underscore only for private functions and variables. Public
  functions and variables must not have a leading underscore.
  Preserve language-defined special methods such as `__init__`.
- Names say what, not how: `load_checkpoint` not `do_thing`; `num_codebooks` not `n`.
  Single letters only for loop indices (`i`,`j`) or math (`x`,`y`,`t`).
- Interface names must identify the domain meaning, role, or unit of a value.
  For example, `speaker_embedding` is clearer than `spks`. Choose names that fit
  the actual operation; short mathematical names belong in local equations.

## TYPING & SIGNATURES

- Full type hints, modern syntax: `X | Y`, `list[...]`, `dict[str, int]`, `X | None`.
  Annotate return types.
- ONE typing style per repo. Don't mix `Optional[X]`/`Union[X,Y]` with `X | Y`. Modern
  preferred; if the repo uses `Optional`, match it.
- Either `requires-python >= 3.10` (native `X | Y`) or `from __future__ import annotations`.
  Don't use `Optional` to work around forward refs — use quoted annotations
  (`"ModelConfig"`). Type-only imports are covered in IMPORTS.
- Closed value sets → `Literal[...]` or `Enum`, not bare strings in comparisons.
- No mutable function defaults: `def f(x=[])`/`= {}` are bugs. Use a `None` sentinel.
- Use concrete types, including model and decoder types, rather than `any`/`Any`
  or bare `dict`/`list`/`tuple`. Annotate structured values and resource handles
  according to their actual contracts, including element types and optionality.
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

- Validate inputs and preconditions before the main logic. When possible, organizing
 conditions into clear, mutually exclusive if/elif/else branches. Return early for
 invalid cases, and keep the main execution path in the final branch to avoid unnecessary
 lookups, repeated checks, and deeply nested logic.

## LOGGING

- This repo uses stdlib `logging`. Configure once; don't introduce a second logging library.
- One logger per module: `logging.getLogger(__name__)`.
- Keep messages terse. No manual `[Info]`, `[Warn]`, or `[step N]` prefixes;
  use the logger's level and context fields.
- `print` only for CLI output the end user reads (`--help`, visualization, `__main__` demo).
  Runtime info — even debug — goes through the logger.
- Prefer `%-style` for hot-path log calls so formatting is skipped when the level is
  disabled: `logger.info("Loading: %s", path)`.

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

## IMPORTS

- Group: stdlib / third-party / local, blank-line separated, alphabetical within group.
- Manage import paths consistently at the project level. Don’t patch sys.path ad hoc in individual files.
- Prefer module-level imports; allow function-local imports for optional
  dependencies, necessary initialization ordering, or documented
  circular-dependency breaks. Put type-only cycle-breaking imports under
  `if TYPE_CHECKING:` with quoted annotations.
- For repository-internal imports, import from the defining module using the full
  package path, such as `from xxx.yy.zzz import kkk`, rather than through
  `__init__.py`. Keep package re-exports minimal and define an explicit
  `__all__`; no wildcard imports. For third-party libraries, prefer their
  documented public import paths (e.g. `from pydantic import BaseModel`).

## TOOLING

- This repository already configures linting, formatting, and other checks in
  [.pre-commit-config.yaml](/.pre-commit-config.yaml). Run
  `pre-commit run --all-files` before completing a change.
- Test actual failure contracts and supported fallback paths;
  do not add tests solely to preserve speculative recovery scaffolding.
