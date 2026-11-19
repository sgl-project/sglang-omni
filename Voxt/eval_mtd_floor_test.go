package main

import (
	"fmt"
	"testing"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/eval"
)

// mtdEnglishCeiling is the most error the English clips may have as a
// group before the eval fails. The last good runtime measured 14.0% on
// them, the one with #4211's fix 18.6% (voxt-sync studio/mtd-bisect.md);
// #4188 measured 69.8% and #4189 123.3% while the eval, which had no
// limit, still passed.
const mtdEnglishCeiling = 0.25

// mtdFloor fails a config that answers in the wrong language or whose
// English clips' group error rate passes mtdEnglishCeiling: #4188
// returned Vietnamese for an English clip and the unchecked eval still
// passed.
func mtdFloor(t *testing.T, config string, runs []*mtdRun) {
	t.Helper()
	var english eval.Score
	for _, r := range runs {
		for _, text := range r.texts {
			if wrong := mtdWrongScript(r.clip, text); wrong != "" {
				t.Errorf("%s %s: %s: %q", config, r.clip.ID, wrong, mdText(text))
			}
		}
		if r.clip.Set == "en" {
			for _, s := range r.scores {
				english = english.Add(s)
			}
		}
	}
	if english.N > 0 && english.Rate() > mtdEnglishCeiling {
		t.Errorf("%s: the English clips' error rate %s passes the %s ceiling (the last good runtime's 14.0%% with headroom)", config, pct(english.Rate()), pct(mtdEnglishCeiling))
	}
}

// mtdWrongScript is how a transcript's script differs from the clip's
// language, "" when it matches. An English transcript's letters are all
// ASCII (Vietnamese, Spanish and Russian are not); a Chinese one's are
// Han or ASCII Latin, and not far less Han than the clip's own text (the
// 30 points CheckClean allows a clean-up, clean.go).
func mtdWrongScript(clip eval.Clip, text string) string {
	for _, r := range text {
		if !unicode.IsLetter(r) || r < 0x80 {
			continue
		}
		if clip.Lang == "en" {
			return fmt.Sprintf("not English script (%c)", r)
		}
		if !unicode.Is(unicode.Han, r) {
			return fmt.Sprintf("not Chinese or English script (%c)", r)
		}
	}
	if clip.Lang == "en" {
		return ""
	}
	if share, ref := hanShare(text), hanShare(clip.Text); share < ref-0.3 {
		return fmt.Sprintf("Chinese is %.0f%% of the letters, the clip's is %.0f%%", share*100, ref*100)
	}
	return ""
}

// hanShare is the share of Han in text's letters.
func hanShare(text string) float64 {
	var han, n int
	for _, r := range text {
		if unicode.IsLetter(r) {
			n++
		}
		if unicode.Is(unicode.Han, r) {
			han++
		}
	}
	if n == 0 {
		return 0
	}
	return float64(han) / float64(n)
}

// TestMTDWrongScript: the floor's script check takes the good runtimes'
// transcripts and rejects the Vietnamese #4188 answered an English clip
// in (voxt-sync studio/mtd-bisect.md); the other wrong-language cases
// (Spanish, Russian, Chinese, Japanese, a full translation) are
// synthetic boundaries, not observed outputs.
func TestMTDWrongScript(t *testing.T) {
	en := eval.Clip{Lang: "en", Text: "So um, the first thing is we need to update the docs, and uh, then we should fix the login bug."}
	zh := eval.Clip{Lang: "zh", Text: "今天的天气很好，我们下午去公园散步，晚上一起吃饭吧。"}
	mix := eval.Clip{Lang: "zh", Text: "今天下午我们 review 一下这个 PR，然后把那个 bug 修掉。"}
	cases := []struct {
		clip eval.Clip
		text string
		want bool // the script is wrong
	}{
		{en, "[S01] So,[S01] um,[S01] the first thing is we need to update the docs and uh then we should fix the login bug.", false},
		{en, "[2.62]Please remind me to call the dentist tomorrow at 9:00 in the morning.[6.62]", false},
		{en, "[0.26]Điều đầu tiên là chúng ta cần cập nhật các tài liệu[5.46]và chúng ta nên sửa lỗi đăng nhập.", true},
		{en, "[2.62]Por favor me recuerde llamar al dentista mañana a las nueve de la mañana.[6.62]", true},
		{en, "Пожалуйста, напомни мне позвонить дантисту завтра утром.", true},
		{en, "请提醒我明天早上给牙医打电话。", true},
		{zh, "今天的天气很好，我们下午去公园散步，晚上一起吃饭吧。", false},
		{zh, "今日の天気はとてもいいですね。午後は公園を散歩しましょう。", true},
		{zh, "The weather is nice today; we will walk in the park this afternoon.", true},
		{mix, "今天下午我们 review 一下这个 PR，然后把那个 bug 修掉。", false},
	}
	for _, c := range cases {
		if got := mtdWrongScript(c.clip, c.text); (got != "") != c.want {
			t.Errorf("mtdWrongScript(%q) = %q, want wrong=%v", c.text, got, c.want)
		}
	}
}
