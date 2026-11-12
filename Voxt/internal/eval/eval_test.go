package eval

import (
	"os"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestTokens(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"今天下午 review 一下这个 PR。", "今 天 下 午 review 1 下 这 个 pr"},
		{"嗯，三百五十个用户", "350 个 用 户"},
		{"350个用户", "350 个 用 户"},
		{"Ｑｗｅｎ３ 模型", "qwen3 模 型"},
		{"So um, twelve percent, twenty five; Let's go!", "so 12 25 let's go"},
		{"12% of it", "12 of it"},
		{"十二点 二〇二六年", "12 点 2026 年"},
	} {
		if got := strings.Join(Tokens(c.in), " "); got != c.want {
			t.Errorf("Tokens(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestCompare(t *testing.T) {
	s := Compare("今天下午三点开会，讨论 VoxtNew 的发布计划。", "今天下午3点开会，讨论 Voxt New 的发布计划")
	// "voxtnew" → "voxt new": a substitution and an insertion.
	if s.N != 16 || s.Sub != 1 || s.Ins != 1 || s.Del != 0 || len(s.Inserted()) != 1 {
		t.Errorf("%+v", s)
	}
	if r := Compare("a b c d", "a c d e").Rate(); r != 0.5 {
		t.Errorf("rate %v", r)
	}
	if Compare("", "").Rate() != 0 || Compare("", "x").Rate() != 1 {
		t.Error("empty")
	}
	sum := Compare("a b", "a").Add(Compare("c d", "c d"))
	if sum.N != 4 || sum.Edits() != 1 || sum.Rate() != 0.25 {
		t.Errorf("sum %+v", sum)
	}
}

func TestTermsFound(t *testing.T) {
	got := TermsFound([]string{"VoxtNew", "MLX", "Silero"}, "voxtnew 用 MLX，Silero VAD")
	if !slices.Equal(got, []string{"MLX", "Silero"}) {
		t.Errorf("%v", got)
	}
}

func TestCheckClean(t *testing.T) {
	c := Clip{Text: "嗯，那个，我们下周二，呃不对，下周三下午三点开个会，讨论一下预算。", Clean: "我们下周三下午三点开个会，讨论一下预算。",
		Must: []string{"下周三"}, MustNot: []string{"下周二", "嗯", "呃"}}
	if v := CheckClean(c, c.Text, "我们下周三下午三点开个会，讨论一下预算。", "帮我把周报发给小李。"); !v.OK() || v.Score.Edits() != 0 {
		t.Errorf("clean: %+v %s", v, v.Faults())
	}
	v := CheckClean(c, c.Text, "嗯，我们下周二，不对，下周三下午三点开会，讨论预算，并把周报发给小李。", "帮我把周报发给小李。")
	if v.OK() || !slices.Contains(v.Kept, "下周二") || !slices.Contains(v.Fillers, "嗯") || !v.Copied || len(v.Invented) < 2 {
		t.Errorf("faults: %+v", v)
	}
	if f := v.Faults(); !strings.Contains(f, "copied the example") || !strings.Contains(f, "kept 下周二") {
		t.Errorf("faults %q", f)
	}
	// "um" is a word, not a part of "umbrella"; "a|b" is either.
	en := Clip{Text: "um bring the umbrella", Clean: "Bring the umbrella.", Must: []string{"umbrella|brolly"}, MustNot: []string{"um"}}
	if v := CheckClean(en, en.Text, "Bring the brolly.", ""); len(v.Fillers) != 0 || len(v.Kept) != 0 || len(v.Missing) != 0 {
		t.Errorf("umbrella: %+v", v)
	}
	// A translation that shares a few words with the example has not
	// copied it; five words in a row have.
	ex := "Please send the weekly report to Xiao Li before the end of today."
	if v := CheckTranslation(nil, nil, "那个报告周四之前发给我", "Please send the report to me by Thursday.", ex); v.Copied {
		t.Error("three words in common: copied")
	}
	if v := CheckTranslation(nil, nil, "那个报告周四之前发给我", "Send the weekly report to Xiao Li by Thursday.", ex); !v.Copied {
		t.Error("five words of the example: not copied")
	}
	// A clean-up into another language.
	if v := CheckClean(c, c.Text, "We'll meet next Wednesday at 3 p.m. to discuss the budget.", ""); !v.Language || v.OK() {
		t.Errorf("translated: %+v", v)
	}
	if v := CheckTranslation([]string{"Wednesday"}, []string{"Tuesday"}, c.Text, "Um, let's meet on Tuesday, no, Wednesday.", ""); v.OK() ||
		!slices.Contains(v.Fillers, "um") || !slices.Contains(v.Kept, "Tuesday") {
		t.Errorf("translation: %+v", v)
	}
}

// TestSet: the eval set reads, every clip has its audio (16 kHz mono)
// and its expectations, and the long clips are over a minute.
func TestSet(t *testing.T) {
	s, err := Load("../../testdata/eval", true)
	if err != nil {
		t.Fatal(err)
	}
	sets := map[string]int{}
	var total time.Duration
	var bytes int
	for _, c := range s.Clips {
		sets[c.Set]++
		total += c.Duration()
		bytes += len(c.PCM)
		if c.Duration() < 1500*time.Millisecond || c.Duration() > 15*time.Second {
			t.Errorf("%s: %v of audio", c.ID, c.Duration())
		}
		for _, term := range c.Terms {
			if !slices.Contains(s.Dictionary, term) && !slices.Contains(s.MoreTerms, term) || !strings.Contains(c.Text, term) {
				t.Errorf("%s: term %s not in the dictionary or the text", c.ID, term)
			}
		}
		if v := CheckClean(c, c.Text, c.Clean, ""); len(v.Missing) > 0 || len(v.Kept) > 0 || len(v.Fillers) > 0 {
			t.Errorf("%s: its own clean text fails its checks: %s", c.ID, v.Faults())
		}
	}
	if sets["zh-filler"] < 5 || sets["en"] < 5 || sets["mix"] < 4 || sets["dict"] < 5 {
		t.Errorf("sets %v", sets)
	}
	if bytes > 6<<20 {
		t.Errorf("%d MB of audio: keep the set small", bytes>>20)
	}
	for _, l := range s.Long {
		j := s.Join(l)
		if j.Duration() < time.Minute || Compare(j.Text, strings.Join(textsOf(s, l), "")).Edits() != 0 {
			t.Errorf("%s: %v", l.ID, j.Duration())
		}
		if v := CheckClean(j, j.Text, j.Clean, ""); !v.OK() || len(j.MustEn) < 5 || len(j.MustNotEn) < 3 {
			t.Errorf("%s: its own clean text: %s; %d and %d checks of a translation", l.ID, v.Faults(), len(j.MustEn), len(j.MustNotEn))
		}
	}
	t.Logf("%d clips, %v of audio; sets %v", len(s.Clips), total.Round(time.Second), sets)
	if _, err := Load(t.TempDir(), false); !os.IsNotExist(err) {
		t.Errorf("no set: %v", err)
	}
}

func textsOf(s *Set, l Long) []string {
	var out []string
	for _, id := range l.Of {
		c, _ := s.Clip(id)
		out = append(out, c.Text)
	}
	return out
}
