package compose

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// joinPieces is the pieces with what followed them: the input again.
func joinPieces(ps []Piece) string {
	var b strings.Builder
	for _, p := range ps {
		b.WriteString(p.Text + p.After)
	}
	return b.String()
}

func runes(s string) int { return utf8.RuneCountInString(s) }

// TestPieces: short text is one piece; long text is cut at sentence
// ends into pieces of at most PieceChars, about as long as each other,
// that give the text back; a paragraph ends a piece early; a sentence
// longer than a piece is cut at clauses, then spaces, then anywhere.
func TestPieces(t *testing.T) {
	zh := "嗯，那个，我们下周二，呃不对，下周三下午三点开个会，讨论一下预算。" // 33 characters
	en := "Let's ship the release on Friday, no wait, on Thursday morning. "
	for _, c := range []struct {
		name    string
		text    string
		n       int // pieces
		longest int // the longest piece's characters at most
		ends    string
	}{
		{"short", strings.Repeat(zh, 9), 1, 297, "。"},
		{"just over", strings.Repeat(zh, 11), 2, 198, "。"},
		{"chinese", strings.Repeat(zh, 20), 3, PieceChars, "。"},
		{"english", strings.Repeat(en, 10), 3, PieceChars, "."},
		{"mixed", strings.Repeat(en+zh, 6), 2, PieceChars, "。."},
		{"no stops", strings.Repeat("我们下周三下午三点开个会，", 50), 3, PieceChars, "，"},
		{"no punctuation", strings.Repeat("abcdefghi ", 50), 2, PieceChars, "i"},
		{"one long word", strings.Repeat("会", 700), 3, PieceChars, "会"},
	} {
		t.Run(c.name, func(t *testing.T) {
			ps := Pieces(c.text)
			if len(ps) != c.n {
				t.Fatalf("%d pieces, want %d: %q", len(ps), c.n, ps)
			}
			if got := joinPieces(ps); got != strings.TrimSpace(c.text) {
				t.Errorf("joined %q", got)
			}
			if ps[len(ps)-1].After != "" {
				t.Errorf("the last piece is followed by %q", ps[len(ps)-1].After)
			}
			shortest, longest := runes(c.text), 0
			for _, p := range ps {
				n := runes(p.Text)
				shortest, longest = min(shortest, n), max(longest, n)
				if p.Text != strings.TrimSpace(p.Text) {
					t.Errorf("piece %q not trimmed", p.Text)
				}
				last, _ := utf8.DecodeLastRuneInString(p.Text)
				if !strings.ContainsRune(c.ends, last) {
					t.Errorf("piece ends in %q: %q", last, p.Text)
				}
			}
			if longest > c.longest {
				t.Errorf("a piece of %d characters", longest)
			}
			if c.n > 1 && shortest < longest/2 {
				t.Errorf("pieces of %d to %d characters", shortest, longest)
			}
		})
	}
	para := strings.Repeat(zh, 5) + "\n\n" + strings.Repeat(zh, 8)
	ps := Pieces(para)
	if len(ps) != 3 || ps[0].After != "\n\n" || runes(ps[0].Text) != 5*runes(zh) || joinPieces(ps) != para {
		t.Errorf("paragraph: %q", ps)
	}
}

// TestLanguagePieces: speech is cut where whole sentences change
// language, at any length; a sentence with words of the other script, or
// with no letters, stays with its run; one language is one piece, as
// Pieces cuts it.
func TestLanguagePieces(t *testing.T) {
	zh := "嗯，那个，我们下周二，呃不对，下周三下午三点开个会，讨论一下预算。"
	for _, c := range []struct {
		name, text string
		want       []string
	}{
		{"alternating", "So um, update the docs. 嗯，我们下周三开会。Let's ship on Thursday. OK.",
			[]string{"So um, update the docs.", "嗯，我们下周三开会。", "Let's ship on Thursday. OK."}},
		{"words of the other script", "我们的服务用 Kubernetes 部署，CPU 占用太高了，需要 scale out。今天下午我们 review 一下这个 PR。",
			[]string{"我们的服务用 Kubernetes 部署，CPU 占用太高了，需要 scale out。今天下午我们 review 一下这个 PR。"}},
		{"no letters", "123. 我们开会。Let's go! 456。", []string{"123. 我们开会。", "Let's go! 456。"}},
		{"one language", zh + zh, []string{zh + zh}},
	} {
		ps := LanguagePieces(c.text)
		var got []string
		for _, p := range ps {
			got = append(got, p.Text)
		}
		if strings.Join(got, "|") != strings.Join(c.want, "|") || joinPieces(ps) != c.text {
			t.Errorf("%s: %q", c.name, ps)
		}
	}
	// A long run is cut by length too.
	long := strings.Repeat(zh, 20) + "Let's ship on Thursday."
	if ps := LanguagePieces(long); len(ps) != 4 || ps[3].Text != "Let's ship on Thursday." || joinPieces(ps) != long {
		t.Errorf("long run: %d pieces %q", len(ps), ps)
	}
}

// TestBuildParts: Transcribe of long or mixed speech gets a part per
// piece, every one with the same system prompt and example turn;
// Translate, Rewrite and short speech get one.
func TestBuildParts(t *testing.T) {
	zh := "嗯，那个，我们下周二，呃不对，下周三下午三点开个会，讨论一下预算。"
	long := strings.Repeat(zh, 14) + "Auralink 的发布计划。"
	glossary := []Term{{Term: "Auralink"}}
	for i := range 40 {
		glossary = append(glossary, Term{Term: "term" + strings.Repeat("x", i)})
	}
	in := Input{Task: Transcribe, Cleanup: true, MainLanguage: "zh", Glossary: glossary, Text: long}
	parts, err := BuildParts(in)
	if err != nil {
		t.Fatal(err)
	}
	whole, _ := Build(in)
	if len(parts) != 2 {
		t.Fatalf("%d parts", len(parts))
	}
	for i, p := range parts {
		// The glossary of a large dictionary is the whole text's, in each
		// user message: Auralink is in the last piece only.
		if p.System != whole.System || len(p.History) != 2 || p.User != whole.Glossary+"\n\n"+UserMessage(p.Text) ||
			!strings.Contains(p.User, "- Auralink") || strings.Contains(p.System, "Auralink") {
			t.Errorf("part %d: %+v", i, p)
		}
	}
	for _, in := range []Input{
		{Task: Translate, Cleanup: true, TargetLanguage: "en", Text: long},
		{Task: Translate, Cleanup: true, TargetLanguage: "en", Text: "So um, update the docs. 嗯，我们下周三开会。"},
		{Task: Translate, Selected: true, TargetLanguage: "en", Text: long},
		{Task: Rewrite, Instruction: "shorter", Text: long},
		{Task: Transcribe, Text: zh},
	} {
		parts, err := BuildParts(in)
		if err != nil || len(parts) != 1 || parts[0].Text != in.Text {
			t.Errorf("%s: %d parts, %v", in.Task, len(parts), err)
		}
	}
	if parts, _ := BuildParts(Input{Task: Transcribe, Text: "So um, update the docs. 嗯，我们下周三开会。"}); len(parts) != 2 {
		t.Errorf("mixed Transcribe: %d parts", len(parts))
	}
}

// TestJoinParts: answers joined with the input's breaks, a space after
// Latin script, nothing after Chinese; empty answers skipped.
func TestJoinParts(t *testing.T) {
	parts := []Part{{Piece: Piece{After: " "}}, {Piece: Piece{After: ""}}, {Piece: Piece{After: "\n\n"}}, {}}
	for _, c := range []struct {
		answers []string
		want    string
	}{
		{[]string{"我们开会。", "Let's go.", "好的。", "Bye."}, "我们开会。Let's go. 好的。\n\nBye."},
		{[]string{"One.", "", "Two."}, "One. Two."},
		{[]string{"- a\n- b", "Then."}, "- a\n- b\nThen."},
		{[]string{" 开会。 "}, "开会。"},
		{nil, ""},
	} {
		if got := JoinParts(parts, c.answers); got != c.want {
			t.Errorf("%q: %q, want %q", c.answers, got, c.want)
		}
	}
}
