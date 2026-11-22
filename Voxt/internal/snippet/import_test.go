package snippet

import (
	"strconv"
	"strings"
	"testing"
)

func TestImport(t *testing.T) {
	for _, tc := range []struct {
		name, data string
		want       []Snippet
		skipped    int
		err        bool
	}{
		{"Wispr Flow", `[{"name":"my email address","text":"ruiling@example.com"},{"name":"sig","text":"Best,\nR"}]`,
			[]Snippet{{Trigger: "my email address", Text: "ruiling@example.com"}, {Trigger: "sig", Text: "Best,\nR"}}, 0, false},
		{"Raycast: the keyword is the trigger", `[{"name":"Email","keyword":"my email","text":"a@b.c"}]`,
			[]Snippet{{Trigger: "my email", Text: "a@b.c"}}, 0, false},
		{"Voxt", `{"snippets":[{"id":"x","trigger":"我的邮箱","text":"a@b.c"}]}`,
			[]Snippet{{Trigger: "我的邮箱", Text: "a@b.c"}}, 0, false},
		{"incomplete items", `[{"name":"only a trigger"},{"text":"only text"},{"name":"...","text":"x"},{"name":"ok","text":"fine"}]`,
			[]Snippet{{Trigger: "ok", Text: "fine"}}, 3, false},
		{"not snippets", `{"terms":[]}`, nil, 0, true},
		{"not JSON", `trigger,text`, nil, 0, true},
	} {
		got, skipped, err := Import([]byte(tc.data))
		if (err != nil) != tc.err || skipped != tc.skipped || len(got) != len(tc.want) {
			t.Errorf("%s: %+v, %d skipped, %v", tc.name, got, skipped, err)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%s: [%d] = %+v, want %+v", tc.name, i, got[i], tc.want[i])
			}
		}
	}
	var many strings.Builder
	many.WriteString("[")
	for i := range MaxImportItems + 2 {
		if i > 0 {
			many.WriteString(",")
		}
		many.WriteString(`{"name":"t` + strconv.Itoa(i) + `","text":"x"}`)
	}
	many.WriteString("]")
	if got, skipped, err := Import([]byte(many.String())); err != nil || len(got) != MaxImportItems || skipped != 2 {
		t.Errorf("over the limit: %d, %d skipped, %v", len(got), skipped, err)
	}
}

func TestMerge(t *testing.T) {
	l := List{Snippets: []Snippet{{ID: "1", Trigger: "my email", Text: "old@example.com"}, {ID: "2", Trigger: "sig", Text: "R"}}}
	n := 0
	out, added, updated := Merge(l, []Snippet{
		{Trigger: "My Email!", Text: "new@example.com"}, // the same trigger: its text changes
		{Trigger: "sig", Text: "R"},                     // nothing new
		{Trigger: "meeting link", Text: "https://meet.example.com/x"},
	}, func() string { n++; return "new" + strconv.Itoa(n) })
	if added != 1 || updated != 1 || len(out.Snippets) != 3 {
		t.Fatalf("added %d, updated %d: %+v", added, updated, out.Snippets)
	}
	if out.Snippets[0].Text != "new@example.com" || out.Snippets[0].Trigger != "my email" || out.Snippets[2].ID != "new1" {
		t.Errorf("merged %+v", out.Snippets)
	}
	if l.Snippets[0].Text != "old@example.com" {
		t.Error("Merge changed its input")
	}
}
