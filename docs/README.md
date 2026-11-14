# SGLang-Omni Documentation

Most docs live under `docs/` as Markdown cookbooks and guides. Start there if you are new to the codebase.

## Docs Workflow

### Install Dependency

```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

### Update Documentation

Edit the Markdown (or RST) under `docs/`.

- Run `pre-commit run --all-files` before opening a PR. Re-run once if the first pass auto-fixes files.
- Preview locally:

```bash
bash serve.sh

# custom port
PORT=8080 make serve
```

## Style

- Prefer **Markdown** for cookbooks and usage guides. Use notebooks only when the example must be executed in docs CI.
- Prefer relative links (`../get_started/installation.md`), not absolute docs URLs.
- Match existing launch / request examples in neighboring cookbooks.
- Put shared TTS HTTP, Python, and streaming usage in [TTS Model Usage](basic_usage/tts.md). Cookbooks should link to it and keep examples that demonstrate model-specific inputs, capabilities, or behavior.

## Documentation checks

- Use one H1 per Markdown page and headings no deeper than H3. Headings inside fenced code blocks do not count.
- Name Markdown files with lowercase ASCII letters, digits, and underscores, such as `model_v2.md`. Directory names are unrestricted. `docs/README.md` is the only filename exception.
- List each Markdown page in a Sphinx toctree, either in `index.rst` or in the relevant RST or MyST page. Use paths relative to the containing page or start with `/` to resolve from `docs/`. Toctrees accept explicit titles and glob patterns. See `TOCTREE_EXEMPTIONS` in [the checker](../scripts/check_doc_style.py) for pages exempt from this rule.
- Give every opening code fence a language or format, such as `python`, `bash`, `text`, or `mermaid`. Close every fence.
- Keep emoji out of prose, inline code, and URLs. Text presentation is allowed. Only top-level fenced code blocks indented by at most three spaces are exempt from the emoji check.

The `doc-style` hook reports violations with file names and line numbers. To run just that hook from the repository root:

```bash
pre-commit run doc-style --all-files
```
