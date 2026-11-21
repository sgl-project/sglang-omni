(() => {
  const transcript = document.getElementById("transcript");
  let version = 0, streaming = false;
  const render = (data) => {
    if (data.version < version) return;
    const follow = (streaming || data.streaming) && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4;
    version = data.version;
    streaming = data.streaming;
    // While the answer streams the page follows its end, and each step
    // there flashed the scroll bar: it shows again once the answer is done.
    document.documentElement.classList.toggle("streaming", streaming);
    const template = document.createElement("template");
    template.innerHTML = data.html;

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
      if (className && /^(?:math (?:inline|display)|footnotes|token-(?:keyword|string|comment|number|function|invalid)|language-[\w+.#-]+)$/.test(className)) {
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
    // Keep unchanged blocks (including their selection and horizontal scroll).
    // Only the paragraph or code block currently being written is replaced.
    const before = Array.from(transcript.childNodes);
    const after = Array.from(template.content.childNodes);
    for (let i = 0; i < Math.max(before.length, after.length); i++) {
      if (!after[i]) { before[i].remove(); continue; }
      if (!before[i]) { transcript.append(after[i]); continue; }
      if (before[i].isEqualNode(after[i])) continue;
      const scroll = before[i].nodeType === Node.ELEMENT_NODE ? before[i].scrollLeft : 0;
      before[i].replaceWith(after[i]);
      if (scroll) after[i].scrollLeft = scroll;
    }
    if (follow) window.scrollTo(0, document.documentElement.scrollHeight);
    reportSize();
  };

  let lastSize = "";
  const reportSize = () => {
    const { width, height } = transcript.getBoundingClientRect();
    const pixels = Math.max(1, Math.ceil(height));
    const size = `${version}/${pixels}`;
    if (width > 0 && size !== lastSize) {
      lastSize = size;
      window.mygo?.call("TranscriptPreview.Resize", version, pixels).catch(() => {});
    }
  };
  document.addEventListener("copy", (event) => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !event.clipboardData ||
        !transcript.contains(selection.anchorNode) || !transcript.contains(selection.focusNode)) return;
    event.clipboardData.clearData();
    event.clipboardData.setData("text/plain", selection.toString());
    event.preventDefault();
  });
  window.mygo?.on("transcript-preview", render);
  render(JSON.parse(document.getElementById("transcript-data").textContent));
  new ResizeObserver(reportSize).observe(transcript);
})();
