package snippet

import (
	"strings"
	"testing"
	"unicode/utf8"
)

var list = []Snippet{
	{ID: "1", Trigger: "my email address", Text: "ruiling@example.com"},
	{ID: "2", Trigger: "my LinkedIn", Text: "https://www.linkedin.com/in/john-doe/"},
	{ID: "3", Trigger: "我的邮箱", Text: "ruiling@example.com"},
	{ID: "4", Trigger: "my email", Text: "short@example.com"},
	{ID: "5", Trigger: "sig", Text: "— Ruilin"},
}

func TestWhole(t *testing.T) {
	for _, tc := range []struct {
		text, want string
	}{
		{"My email address.", "1"},
		{"my linked in", "2"},
		{"  我的邮箱。", "3"},
		{"My email address is here", ""},
		{"", ""},
		{"...", ""},
	} {
		got, ok := Whole(tc.text, list)
		if ok != (tc.want != "") || ok && got.ID != tc.want {
			t.Errorf("Whole(%q) = %q, %v; want %q", tc.text, got.ID, ok, tc.want)
		}
	}
}

func TestExpand(t *testing.T) {
	for _, tc := range []struct {
		text, want string
	}{
		{"Send it to my email address, thanks.", "Send it to ruiling@example.com, thanks."},
		{"Here is My Linked-In: take a look", "Here is https://www.linkedin.com/in/john-doe/: take a look"},
		{"请发到我的邮箱，谢谢", "请发到ruiling@example.com，谢谢"},
		// The longest trigger at a place wins over a shorter one it holds.
		{"my email address and my email", "ruiling@example.com and short@example.com"},
		// A Latin trigger does not cut a word.
		{"design signals", "design signals"},
		{"end with sig", "end with — Ruilin"},
		{"nothing here", "nothing here"},
	} {
		if got := Expand(tc.text, list); got != tc.want {
			t.Errorf("Expand(%q) = %q; want %q", tc.text, got, tc.want)
		}
	}
	if got := Expand("my email", nil); got != "my email" {
		t.Errorf("no snippets: %q", got)
	}
}

func TestCheck(t *testing.T) {
	l := List{Snippets: list}
	for _, tc := range []struct {
		s    Snippet
		want string
	}{
		{Snippet{Trigger: "my phone", Text: "+1 555"}, ""},
		{Snippet{Trigger: " ,. ", Text: "x"}, ProblemNoTrigger},
		{Snippet{Trigger: "my phone", Text: "  "}, ProblemNoText},
		{Snippet{Trigger: "My Email Address!", Text: "x"}, ProblemTaken},
		{Snippet{ID: "1", Trigger: "my email address", Text: "new@example.com"}, ""}, // itself
	} {
		if got := Check(l, tc.s); got != tc.want {
			t.Errorf("Check(%+v) = %q; want %q", tc.s, got, tc.want)
		}
	}
}

func TestStoredForm(t *testing.T) {
	data, err := List{Snippets: list[:1]}.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	back, err := Unmarshal(data)
	if err != nil || len(back.Snippets) != 1 || back.Snippets[0] != list[0] {
		t.Fatalf("%s → %+v, %v", data, back, err)
	}
}

func TestMark(t *testing.T) {
	got, marked := Mark("send it to my email address, or my email. design sig", list)
	if want := "send it to ⟦my email address⟧, or ⟦my email⟧. design ⟦sig⟧"; got != want {
		t.Errorf("Mark = %q, want %q", got, want)
	}
	if len(marked) != 3 || marked[0].ID != "1" || marked[1].ID != "4" || marked[2].ID != "5" {
		t.Errorf("marked %+v", marked)
	}
	if got, marked := Mark("nothing here", list); got != "nothing here" || len(marked) != 0 {
		t.Errorf("Mark without a trigger = %q, %+v", got, marked)
	}
}

// TestRestore: what a language model may do to the marks.
func TestRestore(t *testing.T) {
	_, marked := Mark("请发到我的邮箱，或者 my linked in", list)
	for _, tc := range []struct {
		name, cleaned, want string
	}{
		{"kept", "请发到⟦我的邮箱⟧，或者 ⟦my linked in⟧。", "请发到ruiling@example.com，或者 https://www.linkedin.com/in/john-doe/。"},
		{"kept, written differently", "请发到⟦我的邮箱⟧，或者 ⟦My LinkedIn⟧。", "请发到ruiling@example.com，或者 https://www.linkedin.com/in/john-doe/。"},
		// The model changed what the marks hold: their place tells.
		{"changed inside", "请发到⟦我的电子邮箱⟧，或者 ⟦领英⟧。", "请发到ruiling@example.com，或者 https://www.linkedin.com/in/john-doe/。"},
		// The model dropped the marks but kept the words.
		{"unmarked", "请发到我的邮箱，或者 my LinkedIn。", "请发到ruiling@example.com，或者 https://www.linkedin.com/in/john-doe/。"},
		// One mark lost and one changed: the changed one cannot be placed.
		{"lost one", "请发到我的电子邮箱，或者 ⟦领英⟧。", "请发到我的电子邮箱，或者 领英。"},
		{"stray mark", "请发到⟦我的邮箱，或者 my LinkedIn。", "请发到ruiling@example.com，或者 https://www.linkedin.com/in/john-doe/。"},
	} {
		if got := Restore(tc.cleaned, marked, list); got != tc.want {
			t.Errorf("%s: Restore(%q) = %q, want %q", tc.name, tc.cleaned, got, tc.want)
		}
	}
	// A snippet's text is not searched for triggers again.
	own := []Snippet{{ID: "a", Trigger: "my sign off", Text: "Thanks, sig"}, {ID: "b", Trigger: "sig", Text: "— R"}}
	_, m := Mark("my sign off", own)
	if got := Restore("⟦My sign-off⟧", m, own); got != "Thanks, sig" {
		t.Errorf("Restore inside a snippet = %q", got)
	}
}

// TestFit: a snippet ending in punctuation before more loses its own,
// marked or not; elsewhere it keeps it.
func TestFit(t *testing.T) {
	intro := []Snippet{{ID: "i", Trigger: "插入自我介绍", Text: "我是睿临，在做语音输入。"}, {ID: "s", Trigger: "sign off", Text: "Best,"}}
	for _, tc := range []struct{ text, want string }{
		{"先插入自我介绍，然后聊项目", "先我是睿临，在做语音输入，然后聊项目"},
		{"先插入自我介绍。", "先我是睿临，在做语音输入。"},
		{"插入自我介绍 然后聊项目", "我是睿临，在做语音输入。 然后聊项目"},
		{"插入自我介绍", "我是睿临，在做语音输入。"},
		{"thanks, sign off.", "thanks, Best,."}, // Latin punctuation stays,
	} {
		if got := Expand(tc.text, intro); got != tc.want {
			t.Errorf("Expand(%q) = %q; want %q", tc.text, got, tc.want)
		}
	}
	marked, m := Mark("先插入自我介绍然后聊项目", intro)
	if got := Restore(marked, m, intro); got != "先我是睿临，在做语音输入。然后聊项目" {
		t.Errorf("Restore, no punctuation after: %q", got)
	}
	if got := Restore("先⟦插入自我介绍⟧，然后聊项目。", m, intro); got != "先我是睿临，在做语音输入，然后聊项目。" {
		t.Errorf("Restore before a comma: %q", got)
	}
	if got := Restore("先⟦插入的自我介绍⟧，然后聊项目。", m, intro); got != "先我是睿临，在做语音输入，然后聊项目。" {
		t.Errorf("Restore by place, before a comma: %q", got)
	}
}

// FuzzRestore: whatever the model answers, Restore does not panic and
// leaves no mark behind.
func FuzzRestore(f *testing.F) {
	for _, s := range [][2]string{
		{"请发到我的邮箱，或者 my linked in", "请发到⟦我的邮箱⟧，或者 ⟦领英⟧。"},
		{"my email address and sig", "⟦⟦my email⟧ address⟧ and ⟧sig⟦"},
		{"sig", "⟦"},
	} {
		f.Add(s[0], s[1])
	}
	f.Fuzz(func(t *testing.T, spoken, answer string) {
		if !utf8.ValidString(spoken) || !utf8.ValidString(answer) {
			return // a model's answer is text
		}
		marked, m := Mark(spoken, list)
		if Restore(marked, m, list) != Expand(spoken, list) {
			t.Errorf("Restore(Mark(%q)) is not Expand", spoken)
		}
		if got := Restore(answer, m, list); strings.Contains(got, Open) || strings.Contains(got, Close) {
			t.Errorf("Restore(%q) = %q", answer, got)
		}
	})
}
