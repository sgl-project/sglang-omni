(() => {
  const { html, version } = JSON.parse(document.getElementById("transcript-data").textContent);
  const transcript = document.getElementById("transcript");
  const template = document.createElement("template");
  template.innerHTML = html;

  // Keep the Markdown writer's elements and attributes. Template contents are
  // inert until appended, so HTML from a transcript cannot load resources.
  const allowed = new Set([
    "A", "B", "BLOCKQUOTE", "BR", "CODE", "DEL", "DIV", "EM",
    "H1", "H2", "H3", "H4", "H5", "H6", "HR", "I", "INPUT",
    "LI", "OL", "P", "PRE", "S", "SECTION", "SPAN", "STRONG", "SUB", "SUP",
    "TABLE", "TBODY", "TD", "TH", "THEAD", "TR", "UL",
  ]);
  for (const node of template.content.querySelectorAll("*")) {
    if (node.tagName === "IMG") {
      node.replaceWith(node.getAttribute("alt") || "");
      continue;
    }
    if (!allowed.has(node.tagName) || node.namespaceURI !== "http://www.w3.org/1999/xhtml") {
      node.replaceWith(node.outerHTML);
      continue;
    }

    const href = node.getAttribute("href")?.trim();
    const className = node.getAttribute("class");
    const id = node.getAttribute("id");
    const align = node.getAttribute("align");
    const start = node.getAttribute("start");
    const checked = node.hasAttribute("checked");
    const checkbox = node.getAttribute("type") === "checkbox";
    for (const { name } of Array.from(node.attributes)) node.removeAttribute(name);
    if (id && /^fn(?:ref)?-\d+$/.test(id)) node.id = id;
    if (className && /^(?:math (?:inline|display)|footnotes|language-[\w+.#-]+)$/.test(className)) {
      node.className = className;
    }
    if (node.tagName === "A" && href) {
      if (/^#fn(?:ref)?-\d+$/.test(href)) {
        node.href = href;
      } else if (/^(?:https?:\/\/|mailto:)/i.test(href)) {
        node.href = href;
        node.target = "_blank";
        node.rel = "noopener noreferrer";
      }
    }
    if ((node.tagName === "TH" || node.tagName === "TD") && /^(left|center|right)$/.test(align)) {
      node.style.textAlign = align;
    }
    if (node.tagName === "OL" && /^\d+$/.test(start)) node.start = Number(start);
    if (node.tagName === "INPUT") {
      if (!checkbox) {
        node.remove();
        continue;
      }
      node.type = "checkbox";
      node.disabled = true;
      node.checked = checked;
      node.parentElement?.classList.add("task");
    }
  }

  for (const math of template.content.querySelectorAll(".math")) {
    const tex = math.textContent;
    const displayMode = math.classList.contains("display");
    try {
      window.katex.render(tex, math, {
        displayMode, output: "mathml", trust: false, strict: "ignore",
        throwOnError: true, maxExpand: 500, maxSize: 15,
      });
    } catch {
      math.textContent = displayMode ? "$$" + tex + "$$" : "$" + tex + "$";
      math.classList.add("invalid");
    }
  }
  for (const table of template.content.querySelectorAll("table")) {
    const viewport = document.createElement("div");
    viewport.className = "table-viewport";
    table.replaceWith(viewport);
    viewport.append(table);
  }
  transcript.append(template.content);

  let lastHeight = 0;
  const reportSize = () => {
    const { width, height } = transcript.getBoundingClientRect();
    const pixels = Math.ceil(height);
    if (width > 0 && pixels > 0 && pixels !== lastHeight) {
      lastHeight = pixels;
      window.mygo?.call("TranscriptPreview.Resize", version, pixels).catch(() => {});
    }
  };
  new ResizeObserver(reportSize).observe(transcript);
})();
