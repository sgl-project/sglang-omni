// Package transcriptpreview builds offline documents for formatted transcription results.
package transcriptpreview

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

//go:embed katex.min.js
var katex string

//go:embed preview.js
var script string

//go:embed preview.css
var css string

// Page embeds the renderer and the JSON-escaped markup in a self-contained page.
func Page(markup string, version uint64, size float32, textColor, linkColor string, dark bool) string {
	data, _ := json.Marshal(struct {
		HTML    string `json:"html"`
		Version uint64 `json:"version"`
	}{markup, version})
	scheme := "light"
	if dark {
		scheme = "dark"
	}
	math := ""
	if strings.Contains(markup, `class="math `) {
		math = `<script nonce="preview">` + katex + `</script>`
	}
	return fmt.Sprintf(`<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-preview'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<style>:root{color-scheme:%s;color:%s;--link:%s;font-size:%gpx}%s</style>
<main id="transcript"></main>
<script type="application/json" id="transcript-data">%s</script>
%s<script nonce="preview">%s</script>`, scheme, textColor, linkColor, size, css, data, math, script)
}
