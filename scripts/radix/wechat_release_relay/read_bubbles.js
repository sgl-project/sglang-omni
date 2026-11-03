// SPDX-License-Identifier: Apache-2.0
// note (luojiaxuan): prints {window title: [[text, height], ...] | null} for the chat bubbles currently loaded in each named WeChat window; null means the window cannot be read right now.
function run(argv) {
  const proc = Application("System Events").processes.byName("WeChat");
  const out = {};
  for (const title of argv) {
    out[title] = readWindow(proc, title);
  }
  return JSON.stringify(out);
}

function readWindow(proc, title) {
  const wins = proc.windows.whose({ name: title })();
  if (wins.length === 0) {
    return null;
  }
  let cells;
  try {
    cells = wins[0].splitterGroups[0].lists[0].uiElements;
    var ids = cells.attributes.byName("AXIdentifier").value();
    var texts = cells.attributes.byName("AXTitle").value();
    var sizes = cells.size();
  } catch (e) {
    // note (luojiaxuan): a minimized window viewed from another Space is listed but its contents are not reachable (-1728); report it as unreadable like a missing window.
    if (e.errorNumber === -1728) {
      return null;
    }
    throw e;
  }
  return ids
    .map((id, i) => [id, texts[i] || "", sizes[i][1]])
    .filter((c) => c[0] === "chat_bubble_item_view")
    .map((c) => [c[1], c[2]]);
}
