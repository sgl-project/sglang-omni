---
name: sglang-omni-cookbook
description: Refactor or review SGLang-Omni model Cookbooks under docs/cookbook. Use for shortening the path to a first result, reusing shared API examples, relocating advanced material, and checking that documentation changes preserve serving behavior and published links.
metadata:
  version: "0.1"
---

# SGLang-Omni Cookbook

Start with the [repository documentation workflow](../../../docs/README.md). For a PR, pin the head and its merge base. Read the model page, the relevant shared guide, and any existing refactor for that page in [the Cookbook tracker](https://github.com/sgl-project/sglang-omni/issues/3926). Open [the reference cases](references/cases.md) for the closest task or a comparable editorial decision.

## Edit the path to a first result

1. Find the installed backend, checkpoint, launch configuration, readiness signal, and smallest supported request. Match the command and request to the checked-in configuration, package extras, CLI, and request implementation at the target revision. Make the advertised result observable, such as a WAV file or a transcript. If a shared example works with substitutions, link to its exact section and name those substitutions.
2. Move repeated protocol, client, and shared option explanations to the guide that already owns them. Use [TTS](../../../docs/basic_usage/tts.md), [ASR](../../../docs/basic_usage/asr.md), [Omni](../../../docs/basic_usage/omni.md), or [Full-Duplex](../../../docs/developer_reference/full_duplex_guide.md) according to the actual route and backend. Before removing a duplicate, check that the linked example supports the same fields and result. Retain unique facts from both copies.
3. Keep readers' model choices, required inputs, model-specific defaults, supported modes, platform differences, and limits with the model. Use an attributed architecture figure with meaningful alt text after the introduction when it helps explain a choice or data flow. Put long tuning, platform, or benchmark material behind specific links when it interrupts the first-use path. Keep short pages self-contained where that is easier to navigate.
4. Link advanced subpages from the model page's **Advanced Guides** section or from the relevant short summaries. Keep the model page as the sole left-sidebar entry for that Cookbook. Follow the Qwen3-TTS pattern by putting `orphan: true` in each advanced subpage's frontmatter, keeping it outside the Sphinx toctrees, and recording the path and reason in `TOCTREE_EXEMPTIONS` in `scripts/check_doc_style.py`.

## Verify the diff

- Compare launch and request commands before and after. For unchanged behavior, preserve configuration flags, working-directory assumptions, checkpoint or served model IDs, ports, shell continuations, and request fields. For a wheel-based path, verify the extra and required files against the published wheel. Point readers to the matching checkout or config download when a later command needs a repository-only file.
- Run `pre-commit run --all-files` and build the Sphinx HTML with `make -C docs html`. Compare warnings against the merge base and inspect the rendered paths through the changed sections. Verify incoming and outgoing links, old anchors, and the owning toctree. Follow the advanced links from the rendered model page and verify that the sidebar lists only the model page for that Cookbook, including when viewing an advanced subpage. Check that each preserved ID occurs once in the rendered page.
- For a changed installation or serving path, use an available authorized backend to check readiness and complete the first request. Record the revision, backend, checkpoint, configuration, and result. Record any path checked only through source or rendering.

For a refactor, summarize what moved, what model behavior the page still explains, and which checks ran. For a PR review, report findings with a file and line, the reader impact, and a concrete fix. Separate confirmed failures from unverified runtime claims.
