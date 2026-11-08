# Contribution Guide

Run the repository's pre-commit checks before opening a pull request. From the repository root:

```bash
pre-commit run --all-files
```

If a hook modifies files, review its changes and run the checks again.

## Naming and lint checks

Use public names for classes, functions, methods, and attributes defined in `sglang_omni/` and `tests/`. The leading-underscore lint hook checks every Python file under those directories, including new model packages and test suites. It flags class and function definitions, every attribute read or write, and string names passed to `getattr`. It ignores vendor copies, dunder names such as `__init__`, a lone `_`, and definitions inside functions.

A name inherited from an upstream object still fails the check. Keep that spelling, and mark the line:

```python
session_request._omni_prompt_cache_key = adapter_request._omni_prompt_cache_key  # noqa: leading-underscore
```

When a local definition must keep a name because an external interface requires it, put `# noqa: leading-underscore` on that line and explain why:

```python
class ExternalAdapter(ExternalBase):
    # ExternalBase calls this hook by its exact name.
    def _required_external_hook(self):  # noqa: leading-underscore
        return self.state
```

The exemption keeps the definition out of lint violations and automatic renames. Its references retain the same name. A comment such as `# skip leading-underscore class and function names` is not an exemption.

The pre-commit hook runs `scripts/check_leading_underscore.py` and only reports violations. It does not rewrite files. To rename violating class and function definitions and their references within the same file, run the fixer and review the diff:

```bash
python scripts/check_leading_underscore.py --fix
```

The fixer never renames attributes, and it does not update callers in other files, pytest fixture parameters, or string references such as `monkeypatch.setattr(module, "_name", ...)`. Rename those by hand, preserve third-party API names, and update those references explicitly.

Every `if` checked under `sglang_omni/` (excluding `sglang_omni/vendor/`) must have an `else`, or belong to an `if`/`elif` chain that ends in `else`. Returning, raising, or a one-line body does not exempt it. There is no `# noqa` exemption. Prefer an `else` branch that does real work. At least write `else: pass`.

The pre-commit hook runs `scripts/check_if_else.py` and only reports violations. It does not rewrite files. When the other branch truly does nothing, insert `else: pass` yourself, or run the fixer:

```bash
python scripts/check_if_else.py --fix
```

## Write documentation

See the [documentation README](https://github.com/sgl-project/sglang-omni/blob/main/docs/README.md) for writing conventions, local previews, and documentation checks.
