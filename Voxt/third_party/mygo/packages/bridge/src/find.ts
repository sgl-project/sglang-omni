// Find in page (Window.FindInPage): matches are marked with the CSS Custom
// Highlight API, which leaves the page's DOM and selection alone; engines
// without it select the active match instead.

export interface FindOptions {
  matchCase?: boolean;
  backward?: boolean;
  next?: boolean;
}

export interface FindResult {
  matches: number;
  active: number;
}

let state: { text: string; matchCase: boolean; ranges: Range[]; active: number } | null = null;
let sheet: CSSStyleSheet | null = null;

const highlights = () =>
  typeof CSS !== "undefined" && "highlights" in CSS && typeof Highlight !== "undefined";

/** Ranges of the visible text matching text, in document order. */
function collect(text: string, matchCase: boolean): Range[] {
  const ranges: Range[] = [];
  if (!text || !document.body) return ranges;
  const needle = matchCase ? text : text.toLowerCase();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const el = node.parentElement;
      if (!el || /^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|TEXTAREA)$/.test(el.tagName)) return NodeFilter.FILTER_REJECT;
      return el.getClientRects().length ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const hay = matchCase ? node.nodeValue! : node.nodeValue!.toLowerCase();
    for (let i = hay.indexOf(needle); i >= 0; i = hay.indexOf(needle, i + needle.length)) {
      const r = document.createRange();
      r.setStart(node, i);
      r.setEnd(node, i + needle.length);
      ranges.push(r);
    }
  }
  return ranges;
}

function style() {
  if (sheet || !("adoptedStyleSheets" in document)) return;
  // A constructed style sheet, which a Content Security Policy allows.
  sheet = new CSSStyleSheet();
  sheet.replaceSync(
    "::highlight(mygo-find){background-color:#fdeb62;color:#000}" +
      "::highlight(mygo-find-active){background-color:#ff9632;color:#000}",
  );
  document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
}

export function find(text: string, opts: FindOptions = {}): FindResult {
  const matchCase = !!opts.matchCase;
  if (!opts.next || !state || state.text !== text || state.matchCase !== matchCase) {
    state = { text, matchCase, ranges: collect(text, matchCase), active: -1 };
  }
  const n = state.ranges.length;
  if (n === 0) {
    stopFind();
    return { matches: 0, active: 0 };
  }
  state.active =
    state.active < 0 ? (opts.backward ? n - 1 : 0) : (state.active + (opts.backward ? n - 1 : 1)) % n;
  const range = state.ranges[state.active]!;
  if (highlights()) {
    style();
    CSS.highlights.set("mygo-find", new Highlight(...state.ranges));
    CSS.highlights.set("mygo-find-active", new Highlight(range));
  } else {
    const sel = getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
  }
  range.startContainer.parentElement?.scrollIntoView({ block: "center", inline: "nearest" });
  return { matches: n, active: state.active + 1 };
}

export function stopFind() {
  if (state && !highlights()) getSelection()?.removeAllRanges();
  state = null;
  if (highlights()) {
    CSS.highlights.delete("mygo-find");
    CSS.highlights.delete("mygo-find-active");
  }
}
