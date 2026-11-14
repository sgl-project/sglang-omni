package historyapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestRenderJSONL(t *testing.T) {
	var b bytes.Buffer
	rd := NewRenderer(&b, FormatJSONL)
	for _, x := range []Row{{ID: "1", Text: "你好 <world> & co", CreatedAt: testStart, Kind: store.KindTranscribe}, {ID: "2", Text: "two\nlines", CreatedAt: testStart}} {
		if err := rd.Row(x); err != nil {
			t.Fatal(err)
		}
	}
	lines := strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("%d lines: %q", len(lines), b.String())
	}
	var r Row
	if err := json.Unmarshal([]byte(lines[0]), &r); err != nil || r.ID != "1" {
		t.Fatalf("%v %+v", err, r)
	}
	if !strings.Contains(lines[0], "你好") || !strings.Contains(lines[0], "<world>") || !strings.Contains(lines[0], "& co") {
		t.Errorf("escaped: %s", lines[0])
	}
}

// TestRenderMarkdown: days from the row's own clock, a row's app, a
// missing app, and a text of two lines staying in its bullet.
func TestRenderMarkdown(t *testing.T) {
	loc := time.FixedZone("Z8", 8*3600)
	var b bytes.Buffer
	rd := NewRenderer(&b, FormatMarkdown)
	for _, x := range []Row{
		{Text: "晚上", App: "Obsidian", CreatedAt: time.Date(2026, 11, 7, 23, 59, 0, 0, loc)},
		{Text: "早上\nsecond", CreatedAt: time.Date(2026, 11, 8, 0, 1, 0, 0, loc)},
	} {
		if err := rd.Row(x); err != nil {
			t.Fatal(err)
		}
	}
	want := "## 2026-11-07\n\n- 23:59 · Obsidian: 晚上\n\n## 2026-11-08\n\n- 00:01 · 早上\n  second\n"
	if b.String() != want {
		t.Errorf("got\n%s\nwant\n%s", b.String(), want)
	}
}

// TestWriteFile: the rows, mode 0600, no temporary file left.
func TestWriteFile(t *testing.T) {
	f := &fakeHistory{list: []store.Entry{secretEntry()}}
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.jsonl")
	n, err := WriteFile(context.Background(), f, store.HistoryQuery{}, false, dst, FormatJSONL)
	if err != nil || n != 1 {
		t.Fatalf("%d %v", n, err)
	}
	fi, err := os.Stat(dst)
	if err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("%v %v", fi, err)
	}
	data, _ := os.ReadFile(dst)
	if s := string(data); !strings.Contains(s, "final text") || strings.Contains(s, "selected-secret") {
		t.Errorf("wrote %s", s)
	}
	if m, _ := filepath.Glob(filepath.Join(dir, ".voxt-export-*")); len(m) != 0 {
		t.Errorf("left behind %v", m)
	}
}

// TestWriteFileFailure leaves no partial file where one was asked for.
func TestWriteFileFailure(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.md")
	if err := os.WriteFile(dst, []byte("earlier export"), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := WriteFile(ctx, &fakeHistory{list: []store.Entry{secretEntry()}}, store.HistoryQuery{}, false, dst, FormatMarkdown); !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	if data, _ := os.ReadFile(dst); string(data) != "earlier export" {
		t.Errorf("file now %q", data)
	}
	if m, _ := filepath.Glob(filepath.Join(dir, ".voxt-export-*")); len(m) != 0 {
		t.Errorf("left behind %v", m)
	}
}
