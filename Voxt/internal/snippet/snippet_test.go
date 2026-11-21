package snippet

import "testing"

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
