# Contributing

## Pull request checklist

Paste the block below into the pull request description when the change touches `docs/` or claims a performance difference. The documentation review tool reads it as self-reported evidence. It cannot confirm the review for you. A missing item is a maintainer follow-up, not an automatic rejection.

A pull request that does not change `docs/` and does not claim a performance change can omit the block.

Run `pre-commit run --all-files` before submitting. Re-run once if the first pass rewrites files. Record the command you actually ran and the result you actually saw.

### Fields

- Purpose: what changed and why.
- Reviewed the final diff: `yes` after you personally read the final diff.
- Checks: the command and the real result.
- Untested: what you did not run, or `none`.
- Performance: `not applicable`, or the measurements below.

### Performance measurements

Fill these in when the change claims a speed or quality difference. A documentation correction with no performance claim keeps `Performance: not applicable` and does not need a benchmark.

- Baseline commit:
- Head commit:
- Before:
- After:
- Unit:
- Command:
- Hardware:
- Software:
- Model:
- Dataset:
- Config:
- Load:
- Repeats:
- Raw results:
- Quality:
- Conditions comparable: yes
- Baseline choice:

Speech workloads use the SeedTTS baseline when it applies. If you use a different dataset, say why in Baseline choice.

Before and after values need units, a command to reproduce them, and a quality result. Record errors and quality regressions next to any speed change. Do not compare numbers from different hardware, models, datasets, or load.

### Example

```markdown
## Contributor checklist

- Purpose: Fix the install command in the cookbook.
- Reviewed the final diff: yes
- Checks: pre-commit run --all-files — passed
- Untested: none
- Performance: not applicable
```
