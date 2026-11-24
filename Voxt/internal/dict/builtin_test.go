package dict

import (
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestReplaceWithBuiltin: with no dictionary the built-in aliases become
// their terms and a spelled name gets its case back, unless it is one
// word all in one case (a package, an address or a variable) or also a
// word (cursor). The
// user's terms win: their alias or word stays, a term they turned off is
// left alone, and a term they have keeps their spelling with the
// built-in aliases.
func TestReplaceWithBuiltin(t *testing.T) {
	saved := builtin
	t.Cleanup(func() { builtin = saved })
	builtin = []builtinTerm{
		{term: "SGLang", aliases: []string{"SG狼", "SG Lang"}, spelled: true},
		{term: "Voxt", aliases: []string{"Vox", "Voxed"}, spelled: true},
		{term: "GitHub", spelled: true},
		{term: "Hugging Face", spelled: true},
		{term: "Cursor"},
	}
	for _, c := range []struct {
		name  string
		terms []store.Term
		in    string
		want  string
	}{
		{"no dictionary", nil, "用SG狼部署, Vox and Voxed on Github in cursor", "用SGLang部署, Voxt and Voxt on GitHub in cursor"},
		{"no dictionary, as spelled", nil, "SGLang, Voxt, GitHub, Cursor", "SGLang, Voxt, GitHub, Cursor"},
		{"one case stays", nil, "pip install sglang from github.com with GITHUB_TOKEN, GitHub", "pip install sglang from github.com with GITHUB_TOKEN, GitHub"},
		{"two words in one case", nil, "on sg lang and SG LANG, on hugging face", "on SGLang and SGLang, on Hugging Face"},
		{"whole words only", nil, "Voxel and Voxes stay", "Voxel and Voxes stay"},
		{"the user's word", []store.Term{{Term: "Vox", Enabled: true}}, "Vox and Voxed", "Vox and Voxt"},
		{"the user's alias", []store.Term{{Term: "Vox Media", Aliases: []string{"vox"}, Enabled: true}}, "vox", "Vox Media"},
		{"the user's word, turned off", []store.Term{{Term: "Vox", Enabled: false}}, "Vox and Voxed", "Vox and Voxt"},
		{"a term the user turned off", []store.Term{{Term: "voxt", Enabled: false}}, "Vox and VOXT", "Vox and VOXT"},
		{"a term the user has", []store.Term{{Term: "SGlang", Aliases: []string{"S G Long"}, Enabled: true}}, "SG狼, S G Long, SGLANG", "SGlang, SGlang, SGLANG"},
		{"the user's alias first", []store.Term{{Term: "SG-Lang", Aliases: []string{"SG Lang"}, Enabled: true}}, "SG Lang", "SG-Lang"},
		{"empty", nil, "", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := ReplaceWithBuiltin(c.in, c.terms)
			if got != c.want {
				t.Errorf("%q: %q, want %q", c.in, got, c.want)
			}
			if again := ReplaceWithBuiltin(got, c.terms); again != got {
				t.Errorf("%q twice: %q", c.in, again)
			}
			if plain := Replace(c.in, c.terms); c.terms == nil && plain != c.in {
				t.Errorf("Replace without terms changed %q to %q", c.in, plain)
			}
		})
	}
}

// TestBuiltinHints: the built-in hints come after the user's, without
// a term the user has or turned off (ignoring case).
func TestBuiltinHints(t *testing.T) {
	saved := builtin
	t.Cleanup(func() { builtin = saved })
	builtin = []builtinTerm{{term: "SGLang", hint: true}, {term: "Voxt", hint: true}, {term: "DeepSeek", hint: true}, {term: "GitHub"}}
	user := []store.Term{{Term: "voxt", Enabled: true}, {Term: "DeepSeek", Enabled: false}}
	if got := strings.Join(BuiltinHints(user), ","); got != "SGLang" {
		t.Errorf("hints %q, want SGLang", got)
	}
	if got := strings.Join(BuiltinHints(nil), ","); got != "SGLang,Voxt,DeepSeek" {
		t.Errorf("hints %q", got)
	}
}

// TestBuiltinList: the lists are short; every built-in term is one a
// user could keep (Check), spelled once; an alias is not its term in
// another case, not another term's alias, and not a lone Chinese,
// Japanese or Korean character, which Replace would find inside other
// words.
func TestBuiltinList(t *testing.T) {
	if hints := BuiltinHints(nil); len(builtin) == 0 || len(builtin) > 100 || len(hints) > 30 {
		t.Fatalf("%d built-in terms, %d hints: keep the lists short (long hints make the model write terms never said)", len(builtin), len(hints))
	}
	seen := map[string]string{}
	for _, b := range builtin {
		if err := Check(store.Term{Term: b.term}); err != nil || b.term != oneLine(b.term) {
			t.Errorf("term %q: %v", b.term, err)
		}
		if prev, dup := seen[strings.ToLower(b.term)]; dup {
			t.Errorf("term %q is also %q", b.term, prev)
		}
		seen[strings.ToLower(b.term)] = b.term
	}
	for _, b := range builtin {
		if len(b.aliases) > 2*MaxAliases {
			t.Errorf("%q has %d aliases", b.term, len(b.aliases))
		}
		for _, a := range b.aliases {
			key := strings.ToLower(a)
			if prev, dup := seen[key]; dup {
				t.Errorf("alias %q of %q is also %q", a, b.term, prev)
			}
			seen[key] = b.term
			if a != oneLine(a) || a == "" {
				t.Errorf("alias %q of %q is not one trimmed line", a, b.term)
			}
			if r, _ := utf8.DecodeRuneInString(a); utf8.RuneCountInString(a) == 1 && unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) {
				t.Errorf("alias %q of %q is one character", a, b.term)
			}
		}
	}
}
