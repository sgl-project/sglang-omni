package main

// The quality evaluation of the curated models on a Mac (docs/eval.md):
// the eval set (testdata/eval, internal/eval) through the real runtimes,
// with the catalogue's models from the model library ($HUB: $HF_HUB_CACHE,
// else ~/.cache/huggingface/hub, the snapshots of their pinned commits):
//
//	VOXTNEW_EVAL=1 \
//	  VOXTNEW_OMNI_RUNTIME=~/voxtnew-runtime/omni VOXTNEW_LLM=~/voxtnew-runtime/llm/voxtnew-llm \
//	  go test -count=1 -timeout 1h -run 'TestEval' -v .
//
// VOXTNEW_EVAL_QUICK=1 runs a clip of each kind and checks the scores
// against floors (scripts/mac-regress.sh); VOXTNEW_EVAL_OUT=DIR writes the
// tables as Markdown (asr.md, llm.md); VOXTNEW_EVAL_ONLY=ID,… runs those
// clips only; VOXTNEW_EVAL_LLM=ID,… (or all) the language models
// (evalRun.llms; VoxtNew's own three by default). The tests log whether another
// voxtnew-llm or qwen3_asr_server was running, since it skews the times.

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
)

type evalRun struct {
	set   *eval.Set
	hub   string // the model library
	quick bool
	out   string
}

func evalSetup(t *testing.T) *evalRun {
	t.Helper()
	if os.Getenv("VOXTNEW_EVAL") != "1" {
		t.Skip("set VOXTNEW_EVAL=1 on a Mac with the models (eval_test.go)")
	}
	if runtime.GOOS != "darwin" {
		t.Skip("macOS only")
	}
	set, err := eval.Load(filepath.Join("testdata", "eval"), true)
	if err != nil {
		t.Fatal(err)
	}
	e := &evalRun{set: set, hub: realHub, quick: os.Getenv("VOXTNEW_EVAL_QUICK") == "1", out: os.Getenv("VOXTNEW_EVAL_OUT")}
	if out, err := exec.Command("pgrep", "-fl", "voxtnew-llm|qwen3_asr_server").Output(); err == nil && len(out) > 0 {
		t.Logf("CONTENTION: other model processes were running before the eval:\n%s", out)
	}
	return e
}

// dir is model id's snapshot in the model library.
func (e *evalRun) dir(t *testing.T, id string) string {
	t.Helper()
	m, _ := models.Lookup(id)
	d := models.Installed(e.hub, m)
	if d == "" {
		t.Fatalf("%s is not installed in %s (download it in VoxtNew, or internal/models' TestRealMoveToHub)", id, e.hub)
	}
	return d
}

// llms are the language models to evaluate, in the catalogue's order:
// VOXTNEW_EVAL_LLM's ids, "all" for every catalogue language model in
// the model library (the borrowed ones when another ActNow app installed
// them), else VoxtNew's own three. general leaves out the
// translation-only model.
func (e *evalRun) llms(t *testing.T, general bool) []string {
	t.Helper()
	env := strings.TrimSpace(os.Getenv("VOXTNEW_EVAL_LLM"))
	var out []string
	for _, m := range models.Catalog() {
		if m.Kind != models.LLM || general && len(m.Features) == 1 && m.Features[0] == "translate" {
			continue
		}
		switch {
		case env == "all":
			if m.Borrowed && models.Installed(e.hub, m) == "" {
				t.Logf("%s is not in %s: left out", m.ID, e.hub)
				continue
			}
		case env != "":
			if !slices.Contains(strings.Split(env, ","), m.ID) {
				continue
			}
		case m.Borrowed:
			continue
		}
		out = append(out, m.ID)
	}
	return out
}

// clips are the clips to run: all of them and the long ones, or in quick
// mode the first of each kind.
func (e *evalRun) clips() []eval.Clip {
	if only := os.Getenv("VOXTNEW_EVAL_ONLY"); only != "" {
		var out []eval.Clip
		for _, c := range e.set.All() {
			if slices.Contains(strings.Split(only, ","), c.ID) {
				out = append(out, c)
			}
		}
		return out
	}
	if !e.quick {
		return e.set.All()
	}
	var out []eval.Clip
	seen := map[string]bool{}
	for _, c := range e.set.Clips {
		if !seen[c.Set] {
			seen[c.Set] = true
			out = append(out, c)
		}
	}
	return out
}

// report logs a table and writes it to the output folder.
func (e *evalRun) report(t *testing.T, name, md string) {
	t.Helper()
	t.Logf("\n%s", md)
	if e.out == "" {
		return
	}
	os.MkdirAll(e.out, 0o755)
	if err := os.WriteFile(filepath.Join(e.out, name), []byte(md), 0o644); err != nil {
		t.Error(err)
	}
}

// table is a Markdown table.
type table struct {
	head []string
	rows [][]string
}

func (tb *table) add(cells ...any) {
	row := make([]string, len(cells))
	for i, c := range cells {
		row[i] = fmt.Sprint(c)
	}
	tb.rows = append(tb.rows, row)
}

func (tb *table) String() string {
	var b strings.Builder
	line := func(cells []string) { b.WriteString("| " + strings.Join(cells, " | ") + " |\n") }
	line(tb.head)
	seps := make([]string, len(tb.head))
	for i := range seps {
		seps[i] = "---"
	}
	line(seps)
	for _, r := range tb.rows {
		line(r)
	}
	return b.String()
}

func pct(v float64) string { return fmt.Sprintf("%.1f%%", 100*v) }

func ms(d time.Duration) string { return fmt.Sprintf("%d ms", d.Milliseconds()) }

// median of ds (0 for none).
func median(ds []time.Duration) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	s := slices.Clone(ds)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[len(s)/2]
}

func maxOf(ds []time.Duration) time.Duration {
	var m time.Duration
	for _, d := range ds {
		m = max(m, d)
	}
	return m
}

// mdText is text for a table cell.
func mdText(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(s), "|", "\\|"), "\n", " ⏎ ")
}
