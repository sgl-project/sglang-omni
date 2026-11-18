package transcriptpreview

import (
	"context"
	"encoding/json"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestPageKeepsPreviewDataInJSON(t *testing.T) {
	html := Page(Update{HTML: `<p><script>attack()</script></p>`, Version: 7}, 13, "#123456", "#123456", "#fafafa", false)
	if strings.Contains(html, "<script>attack()") || !strings.Contains(html, `\u003cscript\u003eattack()`) {
		t.Fatal("transcript markup escaped the JSON data script")
	}
	if !strings.Contains(html, `"version":7`) || !strings.Contains(html, `default-src 'none'`) {
		t.Fatal("preview is missing its generation or offline content policy")
	}
}

// TestPageSanitizesTranscriptInBrowser exercises the rendered document,
// including markup that would execute code or hide the transcript.
func TestPageSanitizesTranscriptInBrowser(t *testing.T) {
	var browser string
	for _, name := range []string{
		"google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
		"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
	} {
		if path, err := exec.LookPath(name); err == nil {
			browser = path
			break
		}
	}
	if browser == "" {
		t.Skip("requires Chrome or Chromium for the rendered preview test")
	}
	markup := `<p>Visible start</p>
<script nonce="preview">window.previewAttack = true</script>
<style>main{display:none}</style>
<img src=x onerror="window.previewAttack = true">
<p><img src=x alt="Image description" onerror="window.previewAttack = true"></p>
<p><a href="javascript:window.previewAttack = true">unsafe link</a></p>
<article><strong>nested allowed</strong></article>
<p><strong>safe bold</strong> <a href="https://example.com">safe link</a></p>
<p>Visible end</p>`
	probe := `<script nonce="preview">
(() => {
  const root = document.getElementById("transcript");
  const text = root.innerText;
  const links = Array.from(root.querySelectorAll("a"));
  const unsafeLink = links.find(node => node.textContent === "unsafe link");
  const safeLink = links.find(node => node.textContent === "safe link");
  unsafeLink?.click();
  const safeBold = Array.from(root.querySelectorAll("strong")).find(node => node.textContent === "safe bold");
  const selection = getSelection();
  const range = document.createRange();
  range.selectNodeContents(safeBold);
  selection.removeAllRanges();
  selection.addRange(range);
  const clipboard = new DataTransfer();
  clipboard.setData("text/html", '<span style="background-color:#f8edf4">safe bold</span>');
  const backgroundBeforeCopy = getComputedStyle(document.body).backgroundColor;
  const copy = new ClipboardEvent("copy", {bubbles: true, cancelable: true, clipboardData: clipboard});
  safeBold.dispatchEvent(copy);
  const checks = {
    transcriptVisible: getComputedStyle(root).display !== "none" &&
      root.getBoundingClientRect().height > 0 &&
      text.includes("Visible start") && text.includes("Visible end"),
    scriptIsLiteral: text.includes('<script nonce="preview">window.previewAttack = true'),
    styleIsLiteral: text.includes("<style>main{display:none}</style>"),
    nestedMarkupIsLiteral: text.includes("<article><strong>nested allowed</strong></article>") &&
      !Array.from(root.querySelectorAll("strong")).some(node => node.textContent === "nested allowed"),
    imageDescriptionVisible: text.includes("Image description"),
    noBlockedElements: root.querySelector("script, style, img, article") === null,
    noCodeExecuted: window.previewAttack !== true,
    copyIsPlainText: copy.defaultPrevented && clipboard.getData("text/plain") === "safe bold" &&
      clipboard.getData("text/html") === "" && Array.from(clipboard.types).join() === "text/plain",
    copyKeepsBackground: getComputedStyle(document.body).backgroundColor === backgroundBeforeCopy,
    backgroundMatches: getComputedStyle(document.body).backgroundColor === "rgb(248, 237, 244)",
    unsafeLinkDisabled: !!unsafeLink && !unsafeLink.hasAttribute("href"),
    safeMarkupPreserved: !!safeLink && safeLink.href === "https://example.com/" &&
      Array.from(root.querySelectorAll("strong")).some(node => node.textContent === "safe bold"),
    noEventHandlers: Array.from(root.querySelectorAll("*")).every(node =>
      Array.from(node.attributes).every(attribute => !attribute.name.startsWith("on"))),
  };
  document.documentElement.setAttribute("data-preview-test", encodeURIComponent(JSON.stringify({checks, text})));
})();
</script>`
	page := Page(Update{HTML: markup, Version: 1}, 13, "#111111", "#0000ee", "#f8edf4", false) + probe
	dir := t.TempDir()
	path := filepath.Join(dir, "preview.html")
	if err := os.WriteFile(path, []byte(page), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, browser, "--headless", "--disable-gpu", "--no-first-run",
		"--no-default-browser-check", "--user-data-dir="+filepath.Join(dir, "profile"),
		"--dump-dom", (&url.URL{Scheme: "file", Path: path}).String())
	dom, err := command.Output()
	if err != nil {
		t.Fatalf("render preview: %v", err)
	}
	match := regexp.MustCompile(`data-preview-test="([^"]+)"`).FindSubmatch(dom)
	if match == nil {
		t.Fatal("browser did not finish the preview assertions")
	}
	decoded, err := url.PathUnescape(string(match[1]))
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Checks map[string]bool
		Text   string
	}
	if err := json.Unmarshal([]byte(decoded), &result); err != nil {
		t.Fatal(err)
	}
	for _, check := range []string{
		"transcriptVisible", "scriptIsLiteral", "styleIsLiteral", "nestedMarkupIsLiteral",
		"imageDescriptionVisible", "noBlockedElements", "noCodeExecuted", "unsafeLinkDisabled",
		"safeMarkupPreserved", "noEventHandlers", "backgroundMatches", "copyIsPlainText", "copyKeepsBackground",
	} {
		if !result.Checks[check] {
			t.Errorf("%s failed; visible transcript: %q", check, result.Text)
		}
	}
}
