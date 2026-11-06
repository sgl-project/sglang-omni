package compose

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// golden compares got with testdata/golden/name, or writes it with
// VOXTNEW_UPDATE_GOLDEN=1. Review the diff before committing.
func golden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", "golden", name)
	if os.Getenv("VOXTNEW_UPDATE_GOLDEN") == "1" {
		os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (VOXTNEW_UPDATE_GOLDEN=1 creates it)", err)
	}
	if string(want) != got {
		t.Errorf("%s differs (VOXTNEW_UPDATE_GOLDEN=1 rewrites it)\n--- got\n%s\n--- want\n%s", path, got, want)
	}
}

var sampleTerms = []Term{
	{Term: "Kubernetes"},
	{Term: "VoxtNew", Aliases: []string{"aura link", "奥拉链接"}},
	{Term: "DeepSeek", Aliases: []string{"deep seek"}},
}

// TestGoldenTranscribe fixes the prompt of every combination of the
// Transcribe layers, in both template sets.
func TestGoldenTranscribe(t *testing.T) {
	texts := map[string]string{
		English: "um so the aura link release is like seventy percent done comma no wait eighty percent",
		Chinese: "嗯那个 aura link 的发布大概完成了百分之七十，不对，百分之八十",
	}
	mains := map[string]string{English: "en", Chinese: "zh"}
	others := map[string][]string{English: {"zh"}, Chinese: {"en", "ja"}}
	n := 0
	for _, set := range []string{English, Chinese} {
		for mask := range 32 {
			in := Input{
				Task:           Transcribe,
				Cleanup:        mask&1 != 0,
				Structure:      mask&2 != 0,
				MainLanguage:   mains[set],
				OtherLanguages: others[set],
				Text:           texts[set],
			}
			name := "transcribe-" + set
			if in.Cleanup {
				name += "-cleanup"
			}
			if in.Structure {
				name += "-structure"
			}
			if mask&4 != 0 {
				in.TaskPrompt = "Write {{USER_MAIN_LANGUAGE}} with full-width punctuation. Keep {{UNKNOWN}} as is."
				name += "-custom"
			}
			if mask&8 != 0 {
				in.Scene = &Scene{AppName: "Slack", Prompt: "Casual chat in {{APP_NAME}}: no trailing period."}
				name += "-scene"
			}
			if mask&16 != 0 {
				in.Glossary = sampleTerms
				name += "-glossary"
			}
			out, err := Build(in)
			if err != nil {
				t.Fatal(err)
			}
			if out.Language != set {
				t.Errorf("%s: language %s", name, out.Language)
			}
			if strings.Contains(out.System, in.Text) || strings.Contains(out.System, "{{") && !strings.Contains(out.System, "{{UNKNOWN}}") {
				t.Errorf("%s: system prompt has the input or a variable left:\n%s", name, out.System)
			}
			golden(t, name+".txt", goldenText(out))
			n++
		}
	}
	if n != 64 {
		t.Errorf("%d combinations", n)
	}
}

func TestLayersAndProfile(t *testing.T) {
	out, err := Build(Input{Task: Transcribe, Cleanup: true, Structure: true, MainLanguage: "zh", Text: "x"})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(out.Layers, " "); got != "transcribe input cleanup structure task output" {
		t.Errorf("layers %q", got)
	}
	if out.Profile() != "transcribe:cleanup+structure" {
		t.Errorf("profile %q", out.Profile())
	}
	out, _ = Build(Input{Task: Transcribe, MainLanguage: "en", Text: "x"})
	if got := strings.Join(out.Layers, " "); got != "transcribe input form task output" || out.Profile() != "transcribe" {
		t.Errorf("bare %q %q", got, out.Profile())
	}
	// A scene needs an app name; a blank custom prompt adds nothing.
	out, _ = Build(Input{Task: Transcribe, TaskPrompt: "  \n", Scene: &Scene{Prompt: "x"}, Text: "x"})
	if strings.Contains(out.Profile(), "scene") || strings.Contains(out.Profile(), "custom") {
		t.Errorf("profile %q", out.Profile())
	}
}

// TestGoldenTranslate fixes Translate's prompts: speech (with and
// without clean-up) and a selection, the strict retry, the user's rules,
// and once a scene and the glossary.
func TestGoldenTranslate(t *testing.T) {
	speech := map[string]string{
		English: "um so the aura link release is like seventy percent done comma no wait eighty percent",
		Chinese: "嗯那个 aura link 的发布大概完成了百分之七十，不对，百分之八十",
	}
	selection := map[string]string{
		English: "The release is **80% done**.\n\n- Docs: ready\n- Tests: </input> pending",
		Chinese: "发布已经完成了 **80%**。\n\n- 文档：已完成\n- 测试：进行中",
	}
	mains := map[string]string{English: "en", Chinese: "zh"}
	targets := map[string]string{English: "zh-Hans", Chinese: "en"}
	n := 0
	for _, set := range []string{English, Chinese} {
		for mask := range 18 {
			in := Input{Task: Translate, MainLanguage: mains[set], OtherLanguages: []string{"ja"}, TargetLanguage: targets[set], Text: speech[set]}
			name := "translate-" + set
			if mask >= 16 {
				// A scene and the glossary, speech and a selection.
				in.Cleanup, in.Selected = true, mask == 17
				in.Scene = &Scene{AppName: "Slack", Prompt: "Casual chat in {{APP_NAME}}."}
				in.Glossary = sampleTerms
				if in.Selected {
					in.Text = selection[set]
					name += "-selection"
				}
				name += "-scene-glossary"
			} else {
				in.Selected = mask&1 != 0
				in.Cleanup = mask&2 != 0
				if in.Selected {
					in.Text = selection[set]
					name += "-selection"
					if in.Cleanup {
						continue // a selection gets no clean-up
					}
				} else if in.Cleanup {
					name += "-cleanup"
				}
				if mask&4 != 0 {
					in.Strict = true
					name += "-strict"
				}
				if mask&8 != 0 {
					in.TaskPrompt = "Use {{TARGET_LANGUAGE}} as spoken in Taiwan; keep {{UNKNOWN}}."
					name += "-custom"
				}
			}
			out, err := Build(in)
			if err != nil {
				t.Fatal(err)
			}
			checkNoInput(t, name, out, in.Text)
			if in.Selected && (strings.Contains(out.System, "Basic cleanup") || strings.Contains(out.System, "基础清理")) {
				t.Errorf("%s: clean-up for a selection", name)
			}
			golden(t, name+".txt", goldenText(out))
			n++
		}
	}
	if n != 28 {
		t.Errorf("%d combinations", n)
	}
}

// TestGoldenRewrite fixes Rewrite's prompts: a selection or none (the
// JSON answer), the instruction's clean-up, the user's rules, a
// follow-up, and once a scene and the glossary.
func TestGoldenRewrite(t *testing.T) {
	instruction := map[string]string{
		English: "um make it shorter no wait more formal and as a list",
		Chinese: "嗯帮我精简一下，不对，改得正式一点，整理成列表",
	}
	selection := map[string]string{
		English: "hey team, aura link 0.1 is basically done, docs r ready, tests still running, ship friday?",
		Chinese: "大家好，aura link 0.1 基本做完了，文档好了，测试还在跑，周五发？",
	}
	generate := map[string]string{
		English: "write a two sentence update for the team saying aura link ships friday",
		Chinese: "写一段两句话的进展，告诉团队 aura link 周五发布",
	}
	mains := map[string]string{English: "en", Chinese: "zh"}
	n := 0
	for _, set := range []string{English, Chinese} {
		for mask := range 10 {
			in := Input{Task: Rewrite, MainLanguage: mains[set], OtherLanguages: []string{"en", "zh"}}
			name := "rewrite-" + set
			switch {
			case mask < 8:
				if mask&1 != 0 {
					in.Text, in.Instruction = selection[set], instruction[set]
					name += "-selection"
				} else {
					in.Instruction, in.Answer = generate[set], true
					name += "-answer"
				}
				if mask&2 != 0 {
					in.Cleanup = true
					name += "-cleanup"
				}
				if mask&4 != 0 {
					in.TaskPrompt = "Write like {{APP_NAME}} people do; no emoji."
					name += "-custom"
				}
			case mask == 8:
				in.Instruction, in.Answer, in.Cleanup = instruction[set], true, true
				in.Previous = []Turn{{Instruction: generate[set], Answer: Answer{Title: "Update", Content: "VoxtNew ships on Friday."}}}
				name += "-answer-cleanup-followup"
			default:
				in.Text, in.Instruction, in.Cleanup = selection[set], instruction[set], true
				in.Scene = &Scene{AppName: "Mail", Prompt: "Formal mail."}
				in.Glossary = sampleTerms
				name += "-selection-cleanup-scene-glossary"
			}
			out, err := Build(in)
			if err != nil {
				t.Fatal(err)
			}
			checkNoInput(t, name, out, in.Instruction)
			checkNoInput(t, name, out, in.Text)
			golden(t, name+".txt", goldenText(out))
			n++
		}
	}
	if n != 20 {
		t.Errorf("%d combinations", n)
	}
}

func checkNoInput(t *testing.T, name string, out Output, text string) {
	t.Helper()
	if text != "" && strings.Contains(out.System, text) || strings.Contains(out.System, "{{") && !strings.Contains(out.System, "{{UNKNOWN}}") {
		t.Errorf("%s: system prompt has the input or a variable left:\n%s", name, out.System)
	}
}

func goldenText(out Output) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# profile: %s\n# layers: %s\n\n%s\n", out.Profile(), strings.Join(out.Layers, " "), out.System)
	for _, m := range out.History {
		fmt.Fprintf(&b, "\n---- %s ----\n%s\n", m.Role, m.Content)
	}
	fmt.Fprintf(&b, "\n---- user ----\n%s\n", out.User)
	return b.String()
}

func TestBuildErrors(t *testing.T) {
	for _, in := range []Input{
		{Task: "summarize", Text: "x"},
		{Task: Translate, Text: "x"},
		{Task: Rewrite, Text: "x", Instruction: "  "},
	} {
		if _, err := Build(in); err == nil {
			t.Errorf("%+v: no error", in)
		}
	}
	if _, err := Build(Input{Task: "summarize"}); !errors.Is(err, ErrTask) {
		t.Errorf("task: %v", err)
	}
	if _, err := Build(Input{Task: Translate, Text: "x"}); !errors.Is(err, ErrInput) {
		t.Errorf("target: %v", err)
	}
}

func TestLayersOfTheTasks(t *testing.T) {
	for _, tc := range []struct {
		in             Input
		layers, profle string
	}{
		{Input{Task: Translate, Cleanup: true, TargetLanguage: "en", Text: "x"}, "translate input cleanup form task output", "translate:cleanup"},
		{Input{Task: Translate, Cleanup: true, Selected: true, Strict: true, TargetLanguage: "en", Text: "x"}, "translate selection task strict output", "translate:selection+strict"},
		{Input{Task: Rewrite, Cleanup: true, Instruction: "y", Text: "x"}, "rewrite selection cleanup task output", "rewrite:selection+cleanup"},
		{Input{Task: Rewrite, Instruction: "y", Answer: true}, "rewrite input task answer", "rewrite:answer"},
		{Input{Task: Rewrite, Instruction: "y", Answer: true, Previous: []Turn{{Instruction: "z"}}}, "rewrite input task followup answer", "rewrite:followup+answer"},
	} {
		out, err := Build(tc.in)
		if err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(out.Layers, " "); got != tc.layers || out.Profile() != tc.profle {
			t.Errorf("%+v: %q %q", tc.in, got, out.Profile())
		}
	}
	out, _ := Build(Input{Task: Rewrite, Instruction: "y", Answer: true, Previous: []Turn{{Instruction: "z", Selection: "s", Answer: Answer{Title: "T", Content: "c"}}}})
	msgs := out.Messages()
	if len(msgs) != 4 || msgs[1].Role != "user" || msgs[2].Content != `{"title":"T","content":"c"}` || msgs[3].Content != "<instruction>\ny\n</instruction>" ||
		msgs[1].Content != "<instruction>\nz\n</instruction>\n\n<selected_text>\ns\n</selected_text>" {
		t.Errorf("messages %+v", msgs)
	}
}

func TestLanguageName(t *testing.T) {
	for _, tc := range [][3]string{
		{"zh-Hans", English, "Simplified Chinese"}, {"zh-Hant", Chinese, "繁体中文"}, {"ja", English, "Japanese"},
		{"ja", Chinese, "日语"}, {"xx", English, "xx"}, {"en", Chinese, "英语"},
	} {
		if got := LanguageName(tc[0], tc[1]); got != tc[2] {
			t.Errorf("%s in %s: %q", tc[0], tc[1], got)
		}
	}
	for _, code := range TargetLanguages {
		if LanguageName(code, English) == code {
			t.Errorf("%s has no name", code)
		}
	}
}

func TestUntranslated(t *testing.T) {
	for _, tc := range []struct {
		src, res, target string
		want             bool
	}{
		{"Hello, world!", "hello world", "zh-Hans", true},          // the source again
		{"Hello, world!", "Hello world, friends", "zh-Hans", true}, // Latin back for Chinese
		{"Hello, world!", "你好，世界！", "zh-Hans", false},
		{"Deploy Kubernetes today", "今天部署 Kubernetes", "zh-Hans", false}, // a name stays Latin
		{"我们周五发布", "我们周五发布。", "en", true},
		{"我们周五发布", "我们在周五发版", "en", true}, // still Chinese
		{"我们周五发布", "We ship on Friday.", "en", false},
		{"我们周五发布", "我们周五发布", "zh-Hans", false},       // already Chinese: nothing to redo
		{"Good morning", "Good morning", "en", true}, // Latin into Latin: only the same text
		{"Good morning", "Bonjour", "fr", false},
		{"Привет", "Привет", "ru", false},
		{"Hello", "Привет", "ru", false},
		{"Hello", "Hello", "ru", true},
		{"", "x", "en", false},
		{"x", "", "en", false},
		{"12:30", "12:30", "en", true},
	} {
		if got := Untranslated(tc.src, tc.res, tc.target); got != tc.want {
			t.Errorf("Untranslated(%q, %q, %s) = %v", tc.src, tc.res, tc.target, got)
		}
	}
}

func TestParseAnswer(t *testing.T) {
	for _, tc := range []struct {
		text string
		want Answer
		ok   bool
	}{
		{`{"title":"Update","content":"Ships Friday."}`, Answer{"Update", "Ships Friday."}, true},
		{"```json\n{\"title\": \" **Update** \", \"content\": \"Line 1\\nLine 2\"}\n```", Answer{"Update", "Line 1\nLine 2"}, true},
		{`Sure! {"title":"T","content":"x {y} z"} hope it helps`, Answer{"T", "x {y} z"}, true},
		{`{"title":"T","content":"  "}`, Answer{Title: "T"}, true},
		{`{"title":"T","text":"x"}`, Answer{}, false},
		{`{"title":"T"`, Answer{}, false},
		{"just text", Answer{}, false},
		{`{bad} {"content":"second"}`, Answer{Content: "second"}, true},
	} {
		got, ok := ParseAnswer(tc.text)
		if ok != tc.ok || got != tc.want {
			t.Errorf("ParseAnswer(%q) = %+v %v", tc.text, got, ok)
		}
	}
}

func TestPartialAnswer(t *testing.T) {
	full := `{"title": "周五发布", "content": "大家好：\n- VoxtNew \"0.1\" 周五发布 \ud83d\ude80 \u4f60"}`
	var last Answer
	for i := 1; i <= len(full); i++ {
		a := PartialAnswer(full[:i])
		if !strings.HasPrefix(full, full[:i]) || len(a.Content) < len(last.Content) && !strings.HasPrefix(last.Content, a.Content) {
			t.Fatalf("at %d: %+v after %+v", i, a, last)
		}
		if strings.ContainsRune(a.Content, '\ufffd') {
			t.Fatalf("at %d: a broken rune in %q", i, a.Content)
		}
		last = a
	}
	if last.Title != "周五发布" || last.Content != "大家好：\n- VoxtNew \"0.1\" 周五发布 🚀 你" {
		t.Errorf("final %+v", last)
	}
	for text, want := range map[string]Answer{
		"":                               {},
		"Plain text answer":              {Content: "Plain text answer"},
		"```":                            {},
		"```json\n{\"title\":\"T\",\"co": {Title: "T"},
		`{"content":"a\`:                 {Content: "a"},
		`{"content": "x", "title": "`:    {Content: "x"},
	} {
		if got := PartialAnswer(text); got != want {
			t.Errorf("PartialAnswer(%q) = %+v", text, got)
		}
	}
}

func TestTemplateSet(t *testing.T) {
	for code, want := range map[string]string{"zh": Chinese, "zh-Hans": Chinese, "yue": Chinese, "cmn": Chinese, "en": English, "ja": English, "": English, "xx": English} {
		if got := TemplateSet(code); got != want {
			t.Errorf("%q → %s", code, got)
		}
	}
}

func TestVariables(t *testing.T) {
	v := Variables(Input{MainLanguage: "zh", OtherLanguages: []string{"en", "zh", " ", "ja"}, TargetLanguage: "en", Scene: &Scene{AppName: " Slack "}}, Chinese)
	if v["USER_OTHER_LANGUAGES"] == "" || strings.Contains(v["USER_OTHER_LANGUAGES"], v["USER_MAIN_LANGUAGE"]) || !strings.Contains(v["USER_OTHER_LANGUAGES"], "、") {
		t.Errorf("others %q (main %q)", v["USER_OTHER_LANGUAGES"], v["USER_MAIN_LANGUAGE"])
	}
	if v["APP_NAME"] != "Slack" || v["TARGET_LANGUAGE"] == "" {
		t.Errorf("vars %v", v)
	}
	e := Variables(Input{}, English)
	if e["USER_MAIN_LANGUAGE"] != "English" || e["USER_OTHER_LANGUAGES"] != "" || e["APP_NAME"] != "" {
		t.Errorf("empty %v", e)
	}
}

func TestRender(t *testing.T) {
	vars := map[string]string{"A": "x", "B": "", "C": "{{A}}"}
	for tpl, want := range map[string]string{
		"{{A}}-{{B}}-{{C}}":               "x--{{A}}", // a value is not expanded again
		"{{#if A}}yes {{A}}{{/if}}":       "yes x",
		"{{#if B}}yes{{/if}}no":           "no",
		"{{#if Z}}z{{/if}}{{Z}}":          "{{Z}}",
		"a{{#if A}}\nmulti\nline{{/if}}b": "a\nmulti\nlineb",
	} {
		if got := Render(tpl, vars, nil); got != want {
			t.Errorf("%q → %q, want %q", tpl, got, want)
		}
	}
	if got := Render("{{A}} {{T}}", vars, map[string]string{"T": "{{B}}", "A": "over"}); got != "over {{B}}" {
		t.Errorf("extra: %q", got)
	}
}

func TestGlossary(t *testing.T) {
	terms := []Term{{Term: "Zeta"}, {Term: "VoxtNew", Aliases: []string{"aura link", "voxtnew"}}, {Term: " "}, {Term: "zeta"}}
	got := Glossary(terms, "the Aura-Link app", English)
	if got != "- Zeta\n- VoxtNew (heard as: aura link)" {
		t.Errorf("glossary %q", got)
	}
	if Glossary(nil, "x", English) != "" {
		t.Error("empty glossary")
	}
	var all []string
	var manyTerms []Term
	for i := range 100 {
		manyTerms = append(manyTerms, Term{Term: fmt.Sprintf("term%03d", i)})
		all = append(all, fmt.Sprintf("term%03d", i))
	}
	if n := len(strings.Split(Glossary(manyTerms, strings.Join(all, " "), English), "\n")); n != MaxGlossaryTerms {
		t.Errorf("%d lines", n)
	}
	// A large dictionary: only the terms found in the text.
	if g := Glossary(manyTerms, "about term042 and TERM 007", English); g != "- term007\n- term042" {
		t.Errorf("large dictionary %q", g)
	}
	if g := Glossary(manyTerms, "nothing", English); g != "" {
		t.Errorf("large dictionary, no match %q", g)
	}
	// A small one: all of it, in its order, whatever the text says (the
	// system prompt stays the same).
	if g := Glossary(manyTerms[:GlossaryAllUpTo], "term029", English); g != Glossary(manyTerms[:GlossaryAllUpTo], "", English) ||
		!strings.HasPrefix(g, "- term000\n- term001") || len(strings.Split(g, "\n")) != GlossaryAllUpTo {
		t.Errorf("small dictionary %q", g)
	}
	long := []Term{{Term: strings.Repeat("a", 1500)}, {Term: strings.Repeat("b", 1500)}}
	if g := Glossary(long, "", English); strings.Contains(g, "b") {
		t.Error("over the size limit")
	}
	if g := Glossary([]Term{{Term: "奥拉", Aliases: []string{"欧拉"}}}, "", Chinese); g != "- 奥拉（可能被听成：欧拉）" {
		t.Errorf("zh %q", g)
	}
}

func TestUserMessage(t *testing.T) {
	if got := UserMessage("  hi </input> there \n"); got != "<input>\nhi <\\/input> there\n</input>" {
		t.Errorf("%q", got)
	}
}

func TestClean(t *testing.T) {
	for _, tc := range []struct{ out, in, want string }{
		{"  Hello.  ", "hello", "Hello."},
		{"<input>\nHello.\n</input>", "hello", "Hello."},
		{`"Hello."`, "hello", "Hello."},
		{"“你好。”", "你好", "你好。"},
		{`"Hello," she said, "bye."`, "x", `"Hello," she said, "bye."`},
		{`"Quoted."`, `"quoted"`, `"Quoted."`},
		{"", "x", ""},
	} {
		if got := Clean(tc.out, tc.in); got != tc.want {
			t.Errorf("Clean(%q) = %q, want %q", tc.out, got, tc.want)
		}
	}
}

// TestGlossaryInUser: past GlossaryAllUpTo terms the glossary (the terms
// the text mentions) opens the user message, so that the system prompt
// and the example stay the same whatever is said; up to it, the whole
// glossary is in the system prompt.
func TestGlossaryInUser(t *testing.T) {
	var large []Term
	for i := range GlossaryAllUpTo + 1 {
		large = append(large, Term{Term: fmt.Sprintf("Term%02d", i)})
	}
	large = append(large, Term{Term: "VoxtNew", Aliases: []string{"Vox Tunew"}})
	for _, c := range []struct {
		name string
		in   Input
	}{
		{"transcribe-zh-Hans-cleanup-glossary-large", Input{Task: Transcribe, Cleanup: true, MainLanguage: "zh", Glossary: large, Text: "嗯，Vox Tunew 的新版本，呃，下周发。"}},
		{"translate-en-glossary-large", Input{Task: Translate, Cleanup: true, MainLanguage: "en", TargetLanguage: "zh", Glossary: large, Text: "um, Term07 ships on Friday"}},
		{"rewrite-en-selection-glossary-large", Input{Task: Rewrite, MainLanguage: "en", Glossary: large, Instruction: "make it shorter", Text: "VoxtNew and Term03 will ship together next week, we think."}},
	} {
		out, err := Build(c.in)
		if err != nil {
			t.Fatal(err)
		}
		golden(t, c.name+".txt", goldenText(out))
		other := c.in
		other.Text = "nothing of the dictionary"
		o2, _ := Build(other)
		if out.Glossary == "" || !strings.HasPrefix(out.User, out.Glossary+"\n\n") || o2.System != out.System || strings.Contains(out.System, "Term") ||
			!strings.Contains(out.Profile(), "glossary") || o2.Glossary != "" || !strings.Contains(out.System, "<glossary>") {
			t.Errorf("%s: glossary %q, user %q, profiles %s / %s", c.name, out.Glossary, out.User, out.Profile(), o2.Profile())
		}
	}
	small := Input{Task: Transcribe, MainLanguage: "en", Glossary: large[:GlossaryAllUpTo], Text: "Term01"}
	if out, _ := Build(small); out.Glossary != "" || !strings.Contains(out.System, "- Term29") || out.User != UserMessage("Term01") {
		t.Errorf("a small glossary: %+v", out)
	}
}
