package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func seedExport(t *testing.T, a *app) {
	t.Helper()
	h := a.lib.store.History()
	base := time.Date(2026, 11, 8, 9, 0, 0, 0, time.Local)
	for i, e := range []store.Entry{
		{ID: "a", Kind: store.KindTranscribe, Status: store.StatusDone, Result: "第一条", RawText: "第一条", AppName: "Notes"},
		{ID: "b", Kind: store.KindTranslate, Status: store.StatusDone, Result: "second", RawText: "第二条", SourceText: "selected-secret", TargetLang: "en"},
		{ID: "c", Kind: store.KindTranscribe, Status: store.StatusFailed, Error: "boom", Result: ""},
	} {
		e.CreatedAt = base.Add(time.Duration(i) * time.Minute)
		if err := h.Put(context.Background(), e); err != nil {
			t.Fatal(err)
		}
	}
}

// TestHistoryExportText: the file at the user's path, 0600, without
// the selected text or failures; the shown scope follows the page; the
// selection switch applies here too.
func TestHistoryExportText(t *testing.T) {
	a, _, _ := newTestApp(t)
	seedExport(t, a)
	dir := t.TempDir()
	read := func(name string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		fi, _ := os.Stat(filepath.Join(dir, name))
		if runtime.GOOS == "windows" {
			t.Log("skipping the export mode assertion: Windows does not expose Unix permission bits")
		} else if fi.Mode().Perm() != 0o600 {
			t.Errorf("mode %o", fi.Mode().Perm())
		}
		return string(data)
	}
	a.exportHistoryText(filepath.Join(dir, "a.jsonl"), historyapi.FormatJSONL, false)
	s := read("a.jsonl")
	for _, want := range []string{"第一条", "second", `"target_lang":"en"`} {
		if !strings.Contains(s, want) {
			t.Errorf("no %q in %s", want, s)
		}
	}
	for _, not := range []string{"selected-secret", "boom", "fail"} {
		if strings.Contains(s, not) {
			t.Errorf("%q in %s", not, s)
		}
	}
	if len(a.toasts) == 0 || !strings.Contains(a.toasts[len(a.toasts)-1], "Exported 2 entries") {
		t.Errorf("toast %q", a.toasts)
	}
	// Markdown, shown scope (the Transcribe tab).
	a.hist.tab = 0
	a.exportHistoryText(filepath.Join(dir, "b.md"), historyapi.FormatMarkdown, true)
	md := read("b.md")
	if !strings.Contains(md, "Notes: 第一条") || strings.Contains(md, "second") {
		t.Errorf("shown md: %s", md)
	}
	// The selection switch reaches the manual export.
	a.prefs.Privacy.HistorySelection = true
	a.exportHistoryText(filepath.Join(dir, "c.jsonl"), historyapi.FormatJSONL, false)
	if s := read("c.jsonl"); !strings.Contains(s, "selected-secret") {
		t.Errorf("selection missing: %s", s)
	}
}

// TestHistoryExportDialog: the row is under Export recordings, and its
// button goes to the save panel with the format's name.
func TestHistoryExportDialog(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedExport(t, a)
	dir := t.TempDir()
	orig := chooseHistoryFile
	t.Cleanup(func() { chooseHistoryFile = orig })
	var saved string
	chooseHistoryFile = func(a *app, name string, then func(string)) {
		saved = name
		then(filepath.Join(dir, name))
	}
	openHistSettings(t, a, tt)
	if !has(tt, "Export text") || !has(tt, "Markdown") {
		t.Fatalf("no row: %q", firstTexts(tt))
	}
	a.histSet.textFormat = "JSONL"
	frames(tt)
	click(t, tt, "Export text…")
	if saved != "voxt-history.jsonl" {
		t.Errorf("name %q", saved)
	}
	if _, err := os.Stat(filepath.Join(dir, "voxt-history.jsonl")); err != nil {
		t.Errorf("no file: %v", err)
	}
}
