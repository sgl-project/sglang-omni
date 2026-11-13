package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
	_ "time/tzdata"

	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func cli(args []string, socket string) (int, string, string) {
	var out, errOut bytes.Buffer
	code := historyCLI(args, &out, &errOut, time.UTC, socket)
	return code, out.String(), errOut.String()
}

// TestHistoryCLIUsage: anything that is not a clear request exits 2
// before it goes near the app, which here does not exist.
func TestHistoryCLIUsage(t *testing.T) {
	for _, args := range [][]string{
		{}, {"bogus"}, {"status", "extra"}, {"export"},
		{"export", "--day", "yesterday"},
		{"export", "--day", "2026-11-08", "--since", "2026-11-01"},
		{"export", "--until", "2026-11-01"},
		{"export", "--since", "soon"},
		{"export", "--day", "2026-11-08", "--kind", "nope"},
		{"export", "--day", "2026-11-08", "--format", "csv"},
		{"export", "--day", "2026-11-08", "stray"},
	} {
		code, out, errOut := cli(args, "/nowhere/history.sock")
		if code != exitUsage || out != "" || errOut == "" {
			t.Errorf("%v: code %d, out %q, err %q", args, code, out, errOut)
		}
	}
}

// TestHistoryCLIUnavailable: nothing to reach is exit 3 with one
// sentence naming both causes; stdout stays empty. A HOME that does not
// exist shows the command does not need the app's data.
func TestHistoryCLIUnavailable(t *testing.T) {
	t.Setenv("HOME", filepath.Join(t.TempDir(), "no", "such", "home"))
	t.Setenv("XDG_CONFIG_HOME", "")
	for _, args := range [][]string{{"export", "--day", "2026-11-08"}, {"status"}} {
		var out, errOut bytes.Buffer
		code := historyCLI(args, &out, &errOut, time.UTC, historySocketPath())
		if code != exitUnavailable || out.Len() != 0 {
			t.Errorf("%v: code %d, out %q", args, code, out.String())
		}
		for _, s := range []string{"not running", "Settings → Privacy"} {
			if !strings.Contains(errOut.String(), s) {
				t.Errorf("%v: stderr %q has no %q", args, errOut.String(), s)
			}
		}
	}
}

// TestHistoryDayRange: a day is the calendar's in the Mac's zone, 25
// hours long on the autumn change in New York.
func TestHistoryDayRange(t *testing.T) {
	ny, err := time.LoadLocation("America/New_York")
	if err != nil {
		t.Fatal(err)
	}
	from, to, err := historyRange("2026-11-01", "", "", ny)
	if err != nil {
		t.Fatal(err)
	}
	if d := to.Sub(from); d != 25*time.Hour {
		t.Errorf("the day is %v", d)
	}
	if from.Format(time.RFC3339) != "2026-11-01T00:00:00-04:00" {
		t.Errorf("from %v", from)
	}
	f2, t2, err := historyRange("", "2026-11-01", "2026-11-02T12:00:00Z", time.UTC)
	if err != nil || !f2.Equal(time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)) || !t2.Equal(time.Date(2026, 11, 2, 12, 0, 0, 0, time.UTC)) {
		t.Errorf("%v %v %v", f2, t2, err)
	}
}

// TestHistoryCLIExport drives the command against a real socket: JSONL
// by default, Markdown on request, status as one line; stdout only data.
func TestHistoryCLIExport(t *testing.T) {
	a, _, _ := newTestApp(t)
	at := time.Date(2026, 11, 8, 9, 14, 0, 0, time.UTC)
	if err := a.lib.store.History().Put(context.Background(), store.Entry{ID: "c1", Kind: store.KindTranscribe, CreatedAt: at, Status: store.StatusDone, AppName: "Notes", RawText: "嗯 你好", Result: "你好", SourceText: "selected-secret"}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), historyapi.SocketName)
	ln, err := historyapi.Listen(path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		(&historyapi.Server{History: a.lib.store.History(), Policy: func() historyapi.Policy { return historyapi.Policy{KeepHistory: true, KeepDays: 90} }}).Serve(ctx, ln)
	}()
	defer func() { cancel(); <-done }()

	code, out, errOut := cli([]string{"export", "--day", "2026-11-08", "--as", "daily"}, path)
	if code != exitOK || errOut != "" {
		t.Fatalf("code %d, stderr %q", code, errOut)
	}
	var row map[string]any
	if err := json.Unmarshal([]byte(out), &row); err != nil {
		t.Fatalf("%q: %v", out, err)
	}
	if row["text"] != "你好" || row["heard"] != "嗯 你好" || row["selection"] != nil || !strings.Contains(out, "你好") {
		t.Errorf("row %s", out)
	}
	code, out, _ = cli([]string{"export", "--day", "2026-11-08", "--format", "md"}, path)
	if code != exitOK || !strings.Contains(out, "## 2026-11-08") || !strings.Contains(out, "Notes: 你好") {
		t.Errorf("md %d %q", code, out)
	}
	// The next day is empty and a success.
	if code, out, _ := cli([]string{"export", "--day", "2026-11-09"}, path); code != exitOK || out != "" {
		t.Errorf("empty day %d %q", code, out)
	}
	code, out, _ = cli([]string{"status"}, path)
	var st map[string]any
	if err := json.Unmarshal([]byte(out), &st); code != exitOK || err != nil || st["entries"] != float64(1) || st["keep_history"] != true || st["schema"] != float64(1) {
		t.Errorf("status %d %q %v", code, out, err)
	}
	if strings.Count(strings.TrimSpace(out), "\n") != 0 {
		t.Errorf("status is not one line: %q", out)
	}
}

// TestDailyScriptLeavesNoEmptyFile runs the launchd script from
// docs/history-access.md with a fake Voxt and date: a day with no entries
// leaves no file, a day with entries one, and exit 3 stops the run.
func TestDailyScriptLeavesNoEmptyFile(t *testing.T) {
	doc, err := os.ReadFile("docs/history-access.md")
	if err != nil {
		t.Fatal(err)
	}
	_, rest, ok := strings.Cut(string(doc), "```bash\n")
	script, _, closed := strings.Cut(rest, "```")
	if !ok || !closed {
		t.Fatal("no script in docs/history-access.md")
	}
	dir := t.TempDir()
	fakes := map[string]string{
		"date": "#!/bin/bash\nback=${1#-v-}\nprintf '2026-11-0%s\\n' \"${back%d}\"\n",
		"voxt": `#!/bin/bash
case "$4" in
2026-11-02|2026-11-07) echo '# a day' ;;
2026-11-05) exit 3 ;;
esac
`,
	}
	for name, body := range fakes {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	const real = "VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt"
	if !strings.Contains(script, real) {
		t.Fatalf("the script does not set %s", real)
	}
	voxt := filepath.Join(dir, "voxt")
	if runtime.GOOS == "windows" {
		voxt = filepath.ToSlash(voxt)
	}
	cmd := exec.Command("bash", "-c", strings.Replace(script, real, "VOXT="+voxt, 1))
	cmd.Env = append(os.Environ(), "HOME="+dir, "PATH="+bashPathWithFakeBin(dir))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	out := filepath.Join(dir, "Documents", "Voxt history")
	files, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range files {
		names = append(names, f.Name())
	}
	if len(names) != 1 || names[0] != "2026-11-02.md" {
		t.Fatalf("files %q, want only 2026-11-02.md", names)
	}
	if b, err := os.ReadFile(filepath.Join(out, names[0])); err != nil || len(b) == 0 {
		t.Errorf("the day with entries: %q %v", b, err)
	}
}
