"""Write a rendered diff of two Sphinx HTML builds for pull request review.

Docs Check builds the documentation twice, once from the pull request's base
commit and once from the pull request, and runs this script on the two output
directories. For every page whose article text differs, the script writes a
side-by-side diff with the changed lines highlighted under OUT_DIR/pages/,
plus an index of the changed, added and removed pages at OUT_DIR/index.html.
Docs Preview publishes the result next to the preview, so a reviewer reads
only what changed instead of the whole page.

Only the Python standard library is used.

Usage: python .github/scripts/docs_diff.py BASE_HTML PR_HTML OUT_DIR
"""

from __future__ import annotations

import difflib
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# Pages that Sphinx generates from the others; their changes are noise.
SKIP_PAGES = {"genindex.html", "search.html", "py-modindex.html"}

BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
    "section",
    "article",
    "table",
    "dt",
    "dd",
    "br",
    "figcaption",
}
TEXT_FREE_TAGS = {"script", "style", "noscript"}
# Elements without an end tag; they must not change the nesting depth.
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
# The sphinx_book_theme wraps the page body in this element.
ARTICLE_TAG, ARTICLE_CLASS = "article", "bd-article"
# Sphinx adds a "#" link after every heading; it is not content.
HEADER_LINK_CLASS = "headerlink"


class ArticleText(HTMLParser):
    """Collect the visible text of the article as one line per block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._current: list[str] = []
        self._depth = 0  # nesting depth inside the article, 0 when outside
        self._skip = 0  # nesting depth inside script/style/headerlink

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if self._depth == 0:
            if tag == ARTICLE_TAG and ARTICLE_CLASS in classes:
                self._depth = 1
            return
        if tag in VOID_TAGS:
            if tag == "br":
                self._flush()
            elif tag == "img" and not self._skip:
                self._current.append(
                    f"[image: {attributes.get('alt') or attributes.get('src') or ''}]"
                )
            return
        self._depth += 1
        if tag in TEXT_FREE_TAGS or HEADER_LINK_CLASS in classes:
            self._skip = self._depth
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A self-closing tag is a void element here, never a container.
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._depth == 0 or tag in VOID_TAGS:
            return
        if tag in BLOCK_TAGS:
            self._flush()
        if self._skip and self._depth <= self._skip:
            self._skip = 0
        self._depth -= 1
        if self._depth == 0:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._depth and not self._skip:
            self._current.append(data)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._current)).strip()
        if text:
            self.lines.append(text)
        self._current = []

    @classmethod
    def extract(cls, page: Path) -> list[str]:
        parser = cls()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        parser.close()
        parser._flush()
        return parser.lines


def page_title(lines: list[str], rel: str) -> str:
    return lines[0] if lines else rel


def list_pages(root: Path) -> dict[str, Path]:
    pages = {}
    for path in sorted(root.rglob("*.html")):
        rel = path.relative_to(root).as_posix()
        if rel in SKIP_PAGES or rel.startswith(("_static/", "_sources/", "_diff/")):
            continue
        pages[rel] = path
    return pages


STYLE = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1f2328; }
a { color: #0969da; }
table.diff { width: 100%; border-collapse: collapse; font-size: 0.9rem; table-layout: fixed; }
table.diff th { background: #f6f8fa; padding: 0.4rem; text-align: left; }
table.diff td { vertical-align: top; padding: 0.25rem 0.4rem; border-top: 1px solid #d0d7de; white-space: pre-wrap; word-break: break-word; }
table.diff td.diff_header, table.diff td.diff_next { width: 2.5rem; color: #6e7781; background: #f6f8fa; white-space: nowrap; }
.diff_add { background: #dafbe1; }
.diff_chg { background: #fff8c5; }
.diff_sub { background: #ffebe9; }
.summary li { margin: 0.25rem 0; }
.tag { font-size: 0.75rem; padding: 0 0.4rem; border-radius: 0.6rem; background: #ddf4ff; color: #0550ae; margin-left: 0.4rem; }
"""


def write_page_diff(
    rel: str, before: list[str], after: list[str], out: Path, preview_href: str
) -> None:
    table = difflib.HtmlDiff(wrapcolumn=90).make_table(
        before,
        after,
        fromdesc="Base",
        todesc="This pull request",
        context=True,
        numlines=2,
    )
    # difflib emits legacy attributes and a nowrap that breaks the layout.
    table = table.replace(' nowrap="nowrap"', "").replace(
        ' cellspacing="0" cellpadding="0" rules="groups"', ""
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Diff: {html.escape(rel)}</title><style>{STYLE}</style></head><body>"
        f'<p><a href="{html.escape(index_href(rel))}">All changed pages</a> &middot; '
        f'<a href="{html.escape(preview_href)}">Open the rendered page</a></p>'
        f"<h1>{html.escape(rel)}</h1>"
        "<p>Visible text of the page, one line per paragraph, list item or table row. "
        "Yellow marks a changed line, green an added one and red a removed one.</p>"
        f"{table}</body></html>",
        encoding="utf-8",
    )


def index_href(rel: str) -> str:
    # Diff pages live under OUT_DIR/pages/, one level below the index.
    return "../" * (rel.count("/") + 1) + "index.html"


def preview_href(rel: str) -> str:
    # From OUT_DIR/pages/<rel> up to the site root, then down to the page.
    return "../" * (rel.count("/") + 2) + rel


def main(base_dir: str, pr_dir: str, out_dir: str) -> int:
    base_root, pr_root, out_root = Path(base_dir), Path(pr_dir), Path(out_dir)
    base_pages, pr_pages = list_pages(base_root), list_pages(pr_root)

    changed: list[dict] = []
    added = sorted(set(pr_pages) - set(base_pages))
    removed = sorted(set(base_pages) - set(pr_pages))

    for rel in sorted(set(base_pages) & set(pr_pages)):
        before, after = ArticleText.extract(base_pages[rel]), ArticleText.extract(
            pr_pages[rel]
        )
        if before == after:
            continue
        ratio = difflib.SequenceMatcher(None, before, after, autojunk=False)
        plus = minus = 0
        for op, i1, i2, j1, j2 in ratio.get_opcodes():
            if op != "equal":
                minus += i2 - i1
                plus += j2 - j1
        # The diff page mirrors the preview page's path under pages/.
        write_page_diff(rel, before, after, out_root / "pages" / rel, preview_href(rel))
        changed.append(
            {
                "page": rel,
                "title": page_title(after, rel),
                "added": plus,
                "removed": minus,
            }
        )

    for rel in added:
        after = ArticleText.extract(pr_pages[rel])
        write_page_diff(rel, [], after, out_root / "pages" / rel, preview_href(rel))

    out_root.mkdir(parents=True, exist_ok=True)
    items = []
    for entry in changed:
        rel = entry["page"]
        items.append(
            f"<li><a href=\"pages/{html.escape(rel)}\">{html.escape(entry['title'])}</a> "
            f"<code>{html.escape(rel)}</code> <span class=\"tag\">+{entry['added']} &minus;{entry['removed']} lines</span> "
            f'&middot; <a href="../{html.escape(rel)}">rendered</a></li>'
        )
    for rel in added:
        items.append(
            f'<li><a href="pages/{html.escape(rel)}">{html.escape(page_title(ArticleText.extract(pr_pages[rel]), rel))}</a> '
            f'<code>{html.escape(rel)}</code> <span class="tag">new page</span> '
            f'&middot; <a href="../{html.escape(rel)}">rendered</a></li>'
        )
    for rel in removed:
        items.append(
            f'<li><code>{html.escape(rel)}</code> <span class="tag">removed</span></li>'
        )
    body = (
        '<ul class="summary">' + "".join(items) + "</ul>"
        if items
        else "<p>No rendered page changed.</p>"
    )
    (out_root / "index.html").write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Changed pages</title>'
        f"<style>{STYLE}</style></head><body><h1>Changed pages</h1>"
        "<p>Pages whose visible text differs from the pull request's base. "
        "Each link opens a side-by-side diff; <em>rendered</em> opens the page in the preview.</p>"
        f'{body}<p><a href="../index.html">Back to the preview</a></p></body></html>',
        encoding="utf-8",
    )
    (out_root / "pages.json").write_text(
        json.dumps({"changed": changed, "added": added, "removed": removed}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(changed)} changed, {len(added)} added, {len(removed)} removed page(s); written to {out_root}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
