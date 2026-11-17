package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestMarkdownPreview(t *testing.T) {
	tests := []struct {
		name, source string
		want         []string
	}{
		{"plain", "第一行\n第二行\n\n另起一段", nil},
		{"prices", "Cost: $10–$25; add \\$30.", nil},
		{"prices with formatting", "**Budget**: $10, $25, and $30.", []string{"<strong>Budget</strong>: $10, $25, and $30."}},
		{"escaped dollars", `Price: \$10$ and $20\$.`, nil},
		{"unfinished inline", `这是 \(x^2`, nil},
		{"unfinished display", "Equation:\n\n$$\nx^2", nil},
		{"markdown", "# Results\n\n1. **One**\n2. *Two*\n\n| x | y |\n|---|---|\n| 1 | 2 |", []string{"<h1>Results</h1>", "<ol>", "<strong>One</strong>", "<em>Two</em>", "<table>"}},
		{"emphasis and math", `**中文** *italic* ***both*** and \(x\).`, []string{"<strong>中文</strong>", "<em>italic</em>", "<em><strong>both</strong></em>", `<span class="math inline">x</span>`}},
		{"number-led math", `Costs $20 or $35. Solve $2x + 1$ using $1/2$.`, []string{"Costs $20 or $35.", `<span class="math inline">2x + 1</span>`, `<span class="math inline">1/2</span>`}},
		{"inline math", `值是 \(\frac{a}{b}\)，即 $x^2$。`, []string{`<span class="math inline">\frac{a}{b}</span>`, `<span class="math inline">x^2</span>`}},
		{"display math", "## Formula\n\n\\[\n\\begin{matrix}a&b\\\\c&d\\end{matrix}\n\\]\n\n$$x^2$$", []string{`<div class="math display">\begin{matrix}a&amp;b\\c&amp;d\end{matrix}</div>`, `<div class="math display">x^2</div>`}},
		{"code is literal", "`$2x$`\n\n```sh\necho \"$HOME\"\n```", []string{"<code>$2x$</code>", `<code class="language-sh">echo &quot;$HOME&quot;`}},
		{"unfinished after markdown", "## Formula\n\n\\[\nx^2", []string{"<h2>Formula</h2>", `<p>\[`}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := transcriptMarkdown(tt.source)
			if tt.want == nil && got != "" {
				t.Fatalf("wanted native text, got %s", got)
			}
			for _, want := range tt.want {
				if !strings.Contains(got, want) {
					t.Errorf("missing %q in %s", want, got)
				}
			}
			if (tt.name == "code is literal" || tt.name == "unfinished after markdown") && strings.Contains(got, `class="math`) {
				t.Errorf("literal text became math: %s", got)
			}
		})
	}
}

func TestCopyFormattedTranscriptFromHistory(t *testing.T) {
	a, _, tt := newTestApp(t)
	source := `**Energy**: \(E=mc^2\) and *momentum*.`
	e := store.Entry{ID: "formatted", Kind: store.KindTranscribe, Status: store.StatusDone, CreatedAt: time.Now(), Result: source}
	if err := a.lib.store.History().Put(context.Background(), e); err != nil {
		t.Fatal(err)
	}
	a.navigate("/history")
	frames(tt)
	a.hist.sel = e.ID
	frames(tt)
	click(t, tt, "Copy result")
	if got := tt.Clipboard(); got != source {
		t.Errorf("copied %q, want %q", got, source)
	}
}
