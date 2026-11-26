package main

// TestEvalMTD measures MOSS-Transcribe-Diarize (MTD) against Qwen3-ASR on
// this Mac, through VoxtNew's own runtime path (internal/omni: the same
// servers, flags and request policies the app uses), for the decision
// whether MTD can be the default speech model (docs/MTDEvaluation.zh-CN.md):
//
//	VOXTNEW_EVAL=1 \
//	  VOXTNEW_OMNI_RUNTIME=/Applications/Voxt.app/Contents/Resources/omni \
//	  VOXTNEW_MTD_DIR=<MTD snapshot> VOXTNEW_MTD_DRAFT_DIR=<dflash snapshot> \
//	  VOXTNEW_EVAL_OUT=build/eval go test -count=1 -timeout 2h -run TestEvalMTD -v .
//
// Qwen3-ASR 1.7B and 0.6B come from the model library (models.Installed),
// 1.7B with its DFlash draft as the app runs it; a model that is not
// installed is skipped with a log line. Each timing is the median of
// mtdRounds repeats, with the load average noted beside it; the peak
// memory is the server process's largest footprint over the run
// (watchFootprint). Only one speech server runs at a time (the runtime
// retires the others).

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/eval"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
)

// mtdRounds is how often every timing repeats (the median is reported).
const mtdRounds = 3

// mtdConfig is one measured row: a speech model, with its draft or
// without.
type mtdConfig struct {
	name  string // the tables'
	id    string // the catalogue's
	dir   string
	draft string
}

// mtdRun is one clip's repeats under one config.
type mtdRun struct {
	clip   eval.Clip
	texts  []string
	times  []time.Duration
	scores []eval.Score
}

// mtdConfigs are the rows to measure: MTD from VOXTNEW_MTD_DIR (with its
// DFlash draft when VOXTNEW_MTD_DRAFT_DIR is set), and the catalogue's
// Qwen3-ASR models that the model library has.
func mtdConfigs(t *testing.T, e *evalRun) []mtdConfig {
	t.Helper()
	var out []mtdConfig
	if dir := os.Getenv("VOXTNEW_MTD_DIR"); dir != "" {
		out = append(out, mtdConfig{"MTD", models.ASRMoss, dir, ""})
		if d := os.Getenv("VOXTNEW_MTD_DRAFT_DIR"); d != "" {
			out = append(out, mtdConfig{"MTD+dflash", models.ASRMoss, dir, d})
		} else {
			t.Log("VOXTNEW_MTD_DRAFT_DIR not set: MTD's draft row is skipped")
		}
	} else {
		t.Log("VOXTNEW_MTD_DIR not set: the MTD rows are skipped")
	}
	if m, _ := models.Lookup(models.ASRAccurate); models.Installed(e.hub, m) != "" {
		out = append(out, mtdConfig{"qwen3-1.7B-8bit+dflash", models.ASRAccurate, models.Installed(e.hub, m), e.draft(models.ASRAccurate)})
	} else {
		t.Log("asr.accurate is not installed: its row is skipped")
	}
	if m, _ := models.Lookup(models.ASRFast); models.Installed(e.hub, m) != "" {
		out = append(out, mtdConfig{"qwen3-0.6B-4bit", models.ASRFast, models.Installed(e.hub, m), ""})
	} else {
		t.Log("asr.fast is not installed: its row is skipped")
	}
	return out
}

func TestEvalMTD(t *testing.T) {
	e := evalSetup(t)
	if out, err := exec.Command("pgrep", "-fl", "moss_transcribe_diarize_server").Output(); err == nil && len(out) > 0 {
		t.Logf("CONTENTION: a MOSS server was running before the eval:\n%s", out)
	}
	exe, _ := os.Executable()
	server, err := omni.Locate(exe)
	if err != nil {
		t.Fatal(err)
	}
	rt := omni.NewRuntime(omni.Config{Binary: server, Log: func(level, msg string) {
		if level != "info" && level != "debug" {
			t.Logf("omni %s: %s", level, msg)
		}
	}})
	defer rt.Close()
	ctx := context.Background()
	clips := e.clips()
	configs := mtdConfigs(t, e)
	if len(configs) == 0 {
		t.Fatal("nothing to measure: set VOXTNEW_MTD_DIR or install a Qwen3-ASR model")
	}
	var md strings.Builder
	fmt.Fprintf(&md, "# MTD against Qwen3-ASR (%d clips, %d rounds each)\n\n%s\n", len(clips), mtdRounds, mtdMachineNote())

	texts := map[string]map[string][]string{} // config -> clip -> the rounds' transcripts
	var times table
	times.head = []string{"config", "first Final (cold)", "warm Final median", "warm Final max", "long Final", "real-time factor", "peak footprint", "load average"}
	for _, cfg := range configs {
		a, err := omni.NewASR(rt, cfg.id, cfg.dir)
		if err != nil {
			t.Fatal(err)
		}
		if cfg.draft != "" {
			t.Logf("%s with its draft %s", cfg.name, cfg.draft)
			a = a.WithDraft(cfg.draft)
		}
		t0 := time.Now()
		if _, err := a.Transcribe(ctx, asr.FileRequest{Audio: clips[0].PCM, Format: "pcm", Lang: clips[0].Lang, Hints: e.set.Dictionary}); err != nil {
			t.Fatalf("%s: %v", cfg.name, err)
		}
		cold := time.Since(t0)
		var peak int64
		stop := watchFootprint(rt, cfg.id, &peak)
		var loads []string
		var runs []*mtdRun
		for round := 0; round < mtdRounds; round++ {
			for i, c := range clips {
				if round == 0 {
					runs = append(runs, &mtdRun{clip: c})
				}
				r := runs[i]
				loads = append(loads, loadAvg())
				res, err := a.Transcribe(ctx, asr.FileRequest{Audio: c.PCM, Format: "pcm", Lang: c.Lang, Hints: e.set.Dictionary})
				if err != nil {
					t.Fatalf("%s %s: %v", cfg.name, c.ID, err)
				}
				r.texts = append(r.texts, res.Text)
				r.times = append(r.times, res.Latency)
				r.scores = append(r.scores, eval.Compare(c.Text, res.Text))
			}
		}
		stop()
		for _, failure := range mtdFloor(cfg.name, runs) {
			t.Error(failure)
		}
		texts[cfg.name] = map[string][]string{}
		for _, r := range runs {
			texts[cfg.name][r.clip.ID] = r.texts
		}
		md.WriteString(mtdAccuracy(cfg.name, runs))
		md.WriteString(mtdTerms(t, ctx, a, cfg, e, runs))
		var warm, long []time.Duration
		var audio time.Duration
		for _, r := range runs {
			if r.clip.Set == "long" {
				long = append(long, median(r.times))
				continue
			}
			warm = append(warm, median(r.times))
			audio += r.clip.Duration()
		}
		row := []any{cfg.name, ms(cold), fmt.Sprintf("%s of %d clips (%.1f s audio on average)", ms(median(warm)), len(warm), audio.Seconds()/float64(max(len(warm), 1))), ms(maxOf(warm))}
		if len(long) > 0 {
			row = append(row, ms(maxOf(long)))
		} else {
			row = append(row, "–")
		}
		row = append(row, fmt.Sprintf("%.3f", float64(median(warm))/float64(audio/time.Duration(max(len(warm), 1)))), fmt.Sprintf("%.2f GB", float64(peak)/(1<<30)), loadRange(loads))
		times.add(row...)
		rt.Unload(cfg.id)
	}
	fmt.Fprintf(&md, "\n## Times\n\nEach request's time is the median of %d repeats; the load average is the range of the 1-minute average read before every request (sysctl vm.loadavg). The peak footprint is the server process's largest physical footprint over the whole run (poll every 0.5 s), the weights included.\n\n%s", mtdRounds, times.String())
	md.WriteString(mtdDraftIdentity(t, texts))
	e.report(t, "mtd.md", md.String())
}

// TestEvalMTDExpectedLanguages checks the two short English clips whose
// translations into other languages prompted the native-runtime regression.
func TestEvalMTDExpectedLanguages(t *testing.T) {
	e := evalSetup(t)
	if os.Getenv("VOXTNEW_MTD_DIR") == "" {
		t.Skip("set VOXTNEW_MTD_DIR to the MTD snapshot")
	}
	exe, _ := os.Executable()
	server, err := omni.Locate(exe)
	if err != nil {
		t.Fatal(err)
	}
	rt := omni.NewRuntime(omni.Config{Binary: server, Log: func(level, msg string) {
		if level != "info" && level != "debug" {
			t.Logf("omni %s: %s", level, msg)
		}
	}})
	defer rt.Close()
	ctx := context.Background()
	for _, cfg := range mtdConfigs(t, e) {
		if cfg.id != models.ASRMoss {
			continue
		}
		a, err := omni.NewASR(rt, cfg.id, cfg.dir)
		if err != nil {
			t.Fatal(err)
		}
		if cfg.draft != "" {
			a = a.WithDraft(cfg.draft)
		}
		required := map[string][]string{
			"en-1": {"first thing is we need to update the docs", "login bug"},
			"en-3": {"remind me to call the dentist"},
		}
		seen := map[string]int{}
		failures := 0
		for _, c := range e.clips() {
			wants, ok := required[c.ID]
			if !ok {
				continue
			}
			seen[c.ID]++
			res, err := a.Transcribe(ctx, asr.FileRequest{Audio: c.PCM, Format: "pcm", Lang: c.Lang, Hints: e.set.Dictionary})
			if err != nil {
				t.Fatalf("%s %s: %v", cfg.name, c.ID, err)
			}
			for _, want := range wants {
				if !strings.Contains(res.Text, want) {
					failures++
					t.Errorf("%s %s: expected English %q in %q", cfg.name, c.ID, want, res.Text)
				}
			}
		}
		for _, id := range []string{"en-1", "en-3"} {
			if seen[id] != 1 {
				t.Fatalf("%s: required en-1 and en-3 exactly once, got %s=%d", cfg.name, id, seen[id])
			}
		}
		if failures > 0 {
			t.Errorf("%s: %d expected-language assertions failed", cfg.name, failures)
		}
		rt.Unload(cfg.id)
	}
}

// mtdAccuracy is one config's error-rate table by set, over all rounds.
func mtdAccuracy(name string, runs []*mtdRun) string {
	var b strings.Builder
	fmt.Fprintf(&b, "\n## %s\n\nError rate (Chinese characters, English words; fillers and punctuation aside), every round scored:\n\n", name)
	by := map[string][]*mtdRun{}
	var sets []string
	for _, r := range runs {
		if len(by[r.clip.Set]) == 0 {
			sets = append(sets, r.clip.Set)
		}
		by[r.clip.Set] = append(by[r.clip.Set], r)
	}
	tb := table{head: []string{"clips", "n", "tokens", "error rate"}}
	for _, s := range append(sets, "all") {
		rs := by[s]
		if s == "all" {
			rs = runs
		}
		var sum eval.Score
		for _, r := range rs {
			for _, sc := range r.scores {
				sum = sum.Add(sc)
			}
		}
		tb.add(s, len(rs), sum.N, pct(sum.Rate()))
	}
	b.WriteString(tb.String())
	b.WriteString("\nBy clip, the error rate, the Final's median and the first round's transcript (long ones: their first 60 characters):\n\n")
	ct := table{head: []string{"clip", "error", "Final", "transcript"}}
	for _, r := range runs {
		var sum eval.Score
		for _, sc := range r.scores {
			sum = sum.Add(sc)
		}
		text := []rune(r.texts[0])
		if r.clip.Set == "long" && len(text) > 60 {
			text = append(text[:60], '…')
		}
		ct.add(r.clip.ID, pct(sum.Rate()), ms(median(r.times)), mdText(string(text)))
	}
	b.WriteString(ct.String() + "\n")
	return b.String()
}

// mtdTerms counts the dictionary's terms each config wrote exactly on the
// clips that say them (dict and dict2), once with the whole dictionary as
// the context bias or MOSS hotwords (the set's nine terms and the user's own,
// as the app sends them) and once without.
func mtdTerms(t *testing.T, ctx context.Context, a *omni.ASR, cfg mtdConfig, e *evalRun, runs []*mtdRun) string {
	t.Helper()
	full := append(slices.Clone(e.set.Dictionary), e.set.MoreTerms...)
	var b strings.Builder
	total, with, without := 0, 0, 0
	rows := table{head: []string{"clip", "terms", "with the dictionary", "without"}}
	for _, r := range runs {
		if !strings.HasPrefix(r.clip.Set, "dict") {
			continue
		}
		res, err := a.Transcribe(ctx, asr.FileRequest{Audio: r.clip.PCM, Format: "pcm", Lang: r.clip.Lang, Hints: full})
		if err != nil {
			t.Fatalf("%s %s: %v", cfg.name, r.clip.ID, err)
		}
		withHotwords := res.Text
		res, err = a.Transcribe(ctx, asr.FileRequest{Audio: r.clip.PCM, Format: "pcm", Lang: r.clip.Lang})
		if err != nil {
			t.Fatalf("%s %s without the dictionary: %v", cfg.name, r.clip.ID, err)
		}
		plain := res.Text
		w, wo := len(eval.TermsFound(r.clip.Terms, withHotwords)), len(eval.TermsFound(r.clip.Terms, plain))
		total, with, without = total+len(r.clip.Terms), with+w, without+wo
		rows.add(r.clip.ID, strings.Join(r.clip.Terms, ", "), mdText(withHotwords), mdText(plain))
	}
	fmt.Fprintf(&b, "\nDictionary terms written exactly: %d of %d with the dictionary, %d without.\n\n%s", with, total, without, rows.String())
	return b.String()
}

// mtdDraftIdentity checks MTD with its DFlash draft against without, item
// by item: the draft proposes, the model verifies, so the texts should be
// the same.
func mtdDraftIdentity(t *testing.T, texts map[string]map[string][]string) string {
	t.Helper()
	off, on := texts["MTD"], texts["MTD+dflash"]
	if off == nil || on == nil {
		return "\n## The draft's texts\n\nNot measured (one of the MTD rows was skipped).\n"
	}
	same, total := 0, 0
	var diff []string
	for id, want := range off {
		for i, w := range want {
			total++
			if i < len(on[id]) && on[id][i] == w {
				same++
				continue
			}
			diff = append(diff, fmt.Sprintf("%s round %d", id, i+1))
			if i < len(on[id]) {
				at := firstDiff(w, on[id][i])
				t.Logf("draft differs on %s round %d at character %d: %q against %q", id, i+1, at, around(w, at), around(on[id][i], at))
			}
		}
	}
	return fmt.Sprintf("\n## The draft's texts\n\nIdentical without and with the draft in %d of %d requests.%s\n", same, total, map[bool]string{true: "", false: " Different: " + strings.Join(diff, ", ") + "."}[len(diff) == 0])
}

// firstDiff is where two texts first differ, in characters.
func firstDiff(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	for i := 0; i < min(len(ra), len(rb)); i++ {
		if ra[i] != rb[i] {
			return i
		}
	}
	return min(len(ra), len(rb))
}

// around is the 20 characters of s before and after character at.
func around(s string, at int) string {
	rs := []rune(s)
	return string(rs[max(0, at-20):min(len(rs), at+20)])
}

// mtdMachineNote is the measured Mac, the runtime and the date.
func mtdMachineNote() string {
	read := func(name string, args ...string) string {
		out, err := exec.Command(name, args...).Output()
		if err != nil {
			return "?"
		}
		return strings.TrimSpace(string(out))
	}
	mem, _ := strconv.ParseInt(read("sysctl", "-n", "hw.memsize"), 10, 64)
	return fmt.Sprintf("Machine: %s (%s, %d GB, macOS %s), %s. Runtime: %s.",
		read("sysctl", "-n", "hw.model"), read("sysctl", "-n", "machdep.cpu.brand_string"), mem/(1<<30),
		read("sw_vers", "-productVersion"), time.Now().Format("2006-01-02"), os.Getenv("VOXTNEW_OMNI_RUNTIME"))
}

// loadAvg is the Mac's 1-minute load average, read beside every timing.
func loadAvg() string {
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		return "?"
	}
	f := strings.Fields(strings.Trim(string(out), "{ }\n"))
	if len(f) == 0 {
		return "?"
	}
	return f[0]
}

// loadRange is the smallest and largest load average seen, as "a–b".
func loadRange(loads []string) string {
	lo, hi := 1e9, -1.0
	for _, l := range loads {
		v, err := strconv.ParseFloat(l, 64)
		if err != nil {
			continue
		}
		lo, hi = min(lo, v), max(hi, v)
	}
	if hi < 0 {
		return "?"
	}
	return fmt.Sprintf("%.2f–%.2f", lo, hi)
}

// TestEvalMTDSpeakers runs MTD's own diarization task (WithInstruction
// "": the server's built-in prompt) over 2-, 3- and 4-speaker clips
// joined from the eval set's voices, and scores the speaker labels
// against the known layout.
func TestEvalMTDLanguages(t *testing.T) {
	e := evalSetup(t)
	dir := os.Getenv("VOXTNEW_MTD_DIR")
	if dir == "" {
		t.Skip("set VOXTNEW_MTD_DIR to the MTD snapshot")
	}
	exe, _ := os.Executable()
	server, err := omni.Locate(exe)
	if err != nil {
		t.Fatal(err)
	}
	rt := omni.NewRuntime(omni.Config{Binary: server, Log: func(level, msg string) {
		if level != "info" && level != "debug" {
			t.Logf("omni %s: %s", level, msg)
		}
	}})
	defer rt.Close()
	ctx := context.Background()
	probes := []struct{ lang, voice, text string }{
		{"ja", "Kyoko", "今日の天気はとてもいいですね。午後は公園を散歩しましょう。"},
		{"ko", "Yuna", "오늘은 날씨가 정말 좋네요. 오후에 공원을 산책해요."},
		{"yue", "Sinji", "今日天氣好好，我哋下晝去公園散步啦。"},
		{"es", "Mónica", "Hace muy buen tiempo hoy. Vamos a pasear por el parque esta tarde."},
	}
	var md strings.Builder
	fmt.Fprintf(&md, "# MTD on other languages (macOS say clips)\n\n%s\n\n", mtdMachineNote())
	tb := table{head: []string{"language", "said", "MTD", "Qwen3-ASR 0.6B"}}
	for _, p := range probes {
		pcm := sayPCM(t, p.voice, p.text)
		row := []any{p.lang, mdText(p.text)}
		for _, id := range []string{models.ASRMoss, models.ASRFast} {
			d := dir
			if id == models.ASRFast {
				m, _ := models.Lookup(id)
				if d = models.Installed(e.hub, m); d == "" {
					row = append(row, "(not installed)")
					continue
				}
			}
			a, err := omni.NewASR(rt, id, d)
			if err != nil {
				t.Fatal(err)
			}
			res, err := a.Transcribe(ctx, asr.FileRequest{Audio: pcm, Format: "pcm"})
			if err != nil {
				t.Fatalf("%s %s: %v", id, p.lang, err)
			}
			row = append(row, mdText(res.Text))
			rt.Unload(id)
		}
		tb.add(row...)
	}
	md.WriteString(tb.String())
	md.WriteString("\nThe clips are macOS say's (Kyoko, Yuna, Sinji, Mónica), made at run time: a probe of what the model does with a language, not an accuracy score.\n")
	e.report(t, "mtd-languages.md", md.String())
}

// sayPCM synthesises text with macOS say's voice as PCM16 mono at 16 kHz.
func sayPCM(t *testing.T, voice, text string) []byte {
	t.Helper()
	out, err := exec.Command("say", "-v", "?").Output()
	if err != nil || !strings.Contains(string(out), voice+" ") {
		t.Skipf("macOS say has no voice %s", voice)
	}
	dir := t.TempDir()
	aiff, wav := filepath.Join(dir, "clip.aiff"), filepath.Join(dir, "clip.wav")
	if out, err := exec.Command("say", "-v", voice, "-o", aiff, text).CombinedOutput(); err != nil {
		t.Skipf("say: %v (%s)", err, out)
	}
	if out, err := exec.Command("afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav).CombinedOutput(); err != nil {
		t.Skipf("afconvert: %v (%s)", err, out)
	}
	b, err := os.ReadFile(wav)
	if err != nil {
		t.Fatal(err)
	}
	pcm, rate, ch, err := asr.ReadWAV(b)
	if err != nil || rate != asr.SampleRate || ch != 1 {
		t.Fatalf("say's clip: %d Hz × %d, %v", rate, ch, err)
	}
	return pcm
}

// TestSpeakerAccuracy: the one-to-one assignment scores an over-split
// voice's extra labels wrong, and an under-merged pair wrong too.
func mtdFloor(config string, runs []*mtdRun) []string {
	var failures []string
	var english eval.Score
	for _, r := range runs {
		for _, text := range r.texts {
			if wrong := mtdWrongScript(r.clip, text); wrong != "" {
				failures = append(failures, fmt.Sprintf("%s %s: %s: %q", config, r.clip.ID, wrong, mdText(text)))
			}
		}
		if r.clip.Set == "en" {
			for _, s := range r.scores {
				english = english.Add(s)
			}
		}
	}
	if english.N > 0 && english.Rate() > mtdEnglishCeiling {
		failures = append(failures, fmt.Sprintf("%s: the English clips' error rate %s passes the %s ceiling (the last good runtime's 14.0%% with headroom)", config, pct(english.Rate()), pct(mtdEnglishCeiling)))
	}
	return failures
}

// mtdAccentedCeiling is the most of an English transcript's letters that
// may carry diacritics: names and loanwords ("Zoë at the café") stay
// near 10%, Vietnamese is about 30%.
const mtdAccentedCeiling = 0.2

// mtdWrongScript is how a transcript's script differs from the clip's
// language, "" when it matches. An English transcript's letters are
// Latin, few of them accented (a Spanish translation passes here and
// fails the error ceiling); a Chinese one's are Han or ASCII Latin, and
// not far less Han than the clip's own text (the 30 points CheckClean
// allows a clean-up, clean.go).
func mtdWrongScript(clip eval.Clip, text string) string {
	var letters, accented int
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		letters++
		if r < 0x80 {
			continue
		}
		if clip.Lang == "en" {
			if !unicode.Is(unicode.Latin, r) {
				return fmt.Sprintf("not English script (%c)", r)
			}
			accented++
			continue
		}
		if !unicode.Is(unicode.Han, r) {
			return fmt.Sprintf("not Chinese or English script (%c)", r)
		}
	}
	if clip.Lang == "en" {
		if share := float64(accented) / float64(max(letters, 1)); share > mtdAccentedCeiling {
			return fmt.Sprintf("accented letters are %.0f%% of the letters", share*100)
		}
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
// transcripts and accented English names, and rejects the Vietnamese
// #4188 answered an English clip in; the other cases are synthetic
// boundaries, not observed outputs.
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
		{en, "Zoë met me at the café near the naïve résumé desk.", false},
		{en, "[2.62]Por favor me recuerde llamar al dentista mañana a las nueve de la mañana.[6.62]", false},
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

// TestMTDFloor: over the English ceiling or in another script fails;
// accented English under it passes.
func TestMTDFloor(t *testing.T) {
	en := eval.Clip{ID: "en-1", Set: "en", Lang: "en", Text: "Ask Zoe to book the cafe for Friday."}
	zh := eval.Clip{ID: "zh-1", Set: "zh", Lang: "zh", Text: "今天的天气很好。"}
	run := func(clip eval.Clip, text string, edits int) *mtdRun {
		return &mtdRun{clip: clip, texts: []string{text}, scores: []eval.Score{{N: 100, Sub: edits}}}
	}
	cases := []struct {
		name string
		runs []*mtdRun
		want int // failures
	}{
		{"under the ceiling", []*mtdRun{run(en, "Ask Zoë to book the café for Friday.", 20)}, 0},
		{"over the ceiling", []*mtdRun{run(en, "Ask Zoe to look at the cafe on Friday.", 30)}, 1},
		{"Chinese errors do not count", []*mtdRun{run(en, "Ask Zoe to book the cafe for Friday.", 20), run(zh, "今天的天气很好。", 90)}, 0},
		{"English in another script", []*mtdRun{run(en, "请让佐伊周五订咖啡馆。", 20)}, 1},
		{"Spanish fails on the ceiling", []*mtdRun{run(en, "Pídale a Zoe que reserve el café para el viernes.", 100)}, 1},
	}
	for _, c := range cases {
		if got := mtdFloor("mtd", c.runs); len(got) != c.want {
			t.Errorf("%s: mtdFloor = %q, want %d failures", c.name, got, c.want)
		}
	}
}
