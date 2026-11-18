// Package transcriptpreview builds offline documents for formatted results.
package transcriptpreview

import (
	_ "embed"
	"encoding/json"
	"fmt"
)

//go:embed katex.min.js
var katex string

//go:embed preview.js
var script string

//go:embed preview.css
var css string

// Update is sent to an existing page as the model writes its answer.
type Update struct {
	HTML      string `json:"html"`
	Version   uint64 `json:"version"`
	Streaming bool   `json:"streaming"`
}

// Page embeds the renderer and the JSON-escaped markup in a self-contained page.
func Page(update Update, size, rightPadding float32, textColor, linkColor, backgroundColor string, dark bool) string {
	data, _ := json.Marshal(update)
	scheme := "light"
	if dark {
		scheme = "dark"
	}
	return fmt.Sprintf(`<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-preview'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<style>:root{color-scheme:%s;color:%s;--link:%s;--background:%s;font-size:%gpx;--padding-right:%gpx}%s</style>
<main id="transcript"></main>
<script type="application/json" id="transcript-data">%s</script>
<script nonce="preview">%s</script><script nonce="preview">%s</script>`, scheme, textColor, linkColor, backgroundColor, size, rightPadding, css, data, katex, script)
}
