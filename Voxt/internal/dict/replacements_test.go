package dict

import "testing"

func TestTextReplacementIsDirectionalAndSinglePass(t *testing.T) {
	rules, badLine := ParseReplacements("Alice → Client A\nAlice Chen -> Client X\n上海项目 → 项目乙\nClient A → VIP")
	if badLine != 0 {
		t.Fatalf("invalid line %d", badLine)
	}
	in := "ALICE CHEN met Alice and Malice. 上海项目下周启动。 Client A called."
	want := "Client X met Client A and Malice. 项目乙下周启动。 VIP called."
	if got := rules.Replace(in); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestReplacementRulesRejectMalformedLines(t *testing.T) {
	for _, line := range []string{"Alice", "→ Client A", "Alice -> "} {
		if _, invalid := ParseReplacements("OK -> valid\n\n" + line); invalid != 3 {
			t.Errorf("%q: invalid line = %d, want 3", line, invalid)
		}
	}
}
