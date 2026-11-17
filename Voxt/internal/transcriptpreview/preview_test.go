package transcriptpreview

import (
	"strings"
	"testing"
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
