package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
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
		{"number-led subscripts", `$2_x + 3_y$ and $2\times 3$`, []string{`<span class="math inline">2_x + 3_y</span>`, `<span class="math inline">2\times 3</span>`}},
		{"inline math", `值是 \(\frac{a}{b}\)，即 $x^2$。`, []string{`<span class="math inline">\frac{a}{b}</span>`, `<span class="math inline">x^2</span>`}},
		{"display math", "## Formula\n\n\\[\n\\begin{matrix}a&b\\\\c&d\\end{matrix}\n\\]\n\n$$x^2$$", []string{`<div class="math display">\begin{matrix}a&amp;b\\c&amp;d\end{matrix}</div>`, `<div class="math display">x^2</div>`}},
		{"code is literal", "`$2x$`\n\n```sh\necho \"$HOME\"\n```", []string{"<code>$2x$</code>", `<code class="language-sh">`, `$HOME`}},
		{"unfinished inline after markdown", "## Formula\n\nvalue \\(x^2", []string{"<h2>Formula</h2>", `<p>value \(x^2</p>`}},
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
	for _, kind := range []store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite} {
		t.Run(string(kind), func(t *testing.T) {
			a, _, tt := newTestApp(t)
			source := `**Energy**: \(E=mc^2\) and *momentum*.`
			e := store.Entry{ID: "formatted", Kind: kind, Status: store.StatusDone, CreatedAt: time.Now(), Result: source}
			if err := a.lib.store.History().Put(context.Background(), e); err != nil {
				t.Fatal(err)
			}
			a.hist.tab = len(store.Kinds)
			a.navigate("/history")
			frames(tt)
			a.hist.sel = e.ID
			frames(tt)
			click(t, tt, "Copy result")
			if got := tt.Clipboard(); got != source {
				t.Errorf("copied %q, want %q", got, source)
			}
		})
	}
}

func TestStreamingTranscript(t *testing.T) {
	for _, tc := range []struct{ source, want string }{
		{"**still writing", "<strong>still writing</strong>"},
		{`value \(x^2`, `<span class="math inline">x^2</span>`},
		{"$$\nx^2", `<div class="math display">x^2</div>`},
		{"```go\nfunc main() {", `class="token-keyword">func</span>`},
	} {
		if got := transcriptMarkup(tc.source, true); !strings.Contains(got, tc.want) {
			t.Errorf("stream %q: got %s, want %s", tc.source, got, tc.want)
		}
	}
}

func TestStreamingTranscriptEveryPrefix(t *testing.T) {
	source := "**Energy**: \\(E=mc^2\\), $2x+1$.\n\n```sh\necho '$literal$'\n```\n\n$$\n\\frac{1}{2}\n$$"
	for i := 0; i <= len(source); i++ {
		_ = transcriptMarkup(source[:i], true)
	}
	for _, source := range []string{"`literal \\(x^2", "```sh\necho '\\(x^2'", `literal \\(x^2`} {
		if got := transcriptMarkup(source, true); strings.Contains(got, `class="math`) {
			t.Errorf("literal became math: %s", got)
		}
	}
}

func TestFormattedAnswerCopyAndInsert(t *testing.T) {
	for _, kind := range []store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite} {
		t.Run(string(kind), func(t *testing.T) {
			a, helper, _ := newTestApp(t)
			new(fakeTimers).install(a)
			source := `**Energy**: \(E=mc^2\) and *momentum*.`
			var pasted string
			helper.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
				return proto.FocusResult{PID: 812, Role: "AXStaticText"}, nil
			}
			helper.handlers[proto.MethodTextPaste] = func(raw json.RawMessage) (any, error) {
				var p proto.PasteParams
				if err := json.Unmarshal(raw, &p); err != nil {
					return nil, err
				}
				pasted = p.Text
				return proto.PasteResult{Posted: true}, nil
			}
			snap(a, session.Snapshot{ID: "formatted", Kind: kind, State: session.Done, Result: source, Card: true, Delivery: deliver.ReasonNotEditable})
			tt := cardTester(t, a)
			frames(tt)
			click(t, tt, "Copy")
			if got := tt.Clipboard(); got != source {
				t.Errorf("copied %q", got)
			}
			click(t, tt, "Insert")
			if pasted != source {
				t.Errorf("inserted %q", pasted)
			}
		})
	}
}
