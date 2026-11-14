package main

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

// importStore is a real store in a temporary folder with one term, llm.
func importStore(t *testing.T) store.Dictionary {
	t.Helper()
	ctx := context.Background()
	st, err := sqlitestore.Open(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	repo := st.Dictionary()
	if err := repo.Put(ctx, store.Term{ID: core.NewID(time.Now()), Term: "llm", Note: "kept", Enabled: true}); err != nil {
		t.Fatal(err)
	}
	return repo
}

func termsByName(t *testing.T, repo store.Dictionary) map[string]store.Term {
	t.Helper()
	list, err := repo.List(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]store.Term{}
	for _, tm := range list {
		out[tm.Term] = tm
	}
	return out
}

func TestImportDataMergesAsThePageDoes(t *testing.T) {
	repo := importStore(t)
	ctx := context.Background()
	csv := "term,aliases,note,enabled\nSol,,model name,true\nLLM,G L,new note,true\n,orphan,,\n"

	dry, err := importData(ctx, repo, "batch.csv", []byte(csv), true)
	if err != nil || dry != (dictapi.Result{Added: 1, Updated: 1, Skipped: 1}) {
		t.Fatalf("dry run %+v %v", dry, err)
	}
	if got := termsByName(t, repo); len(got) != 1 {
		t.Fatalf("a dry run saved terms: %v", got)
	}

	res, err := importData(ctx, repo, "batch.csv", []byte(csv), false)
	if err != nil || res != (dictapi.Result{Added: 1, Updated: 1, Skipped: 1}) {
		t.Fatalf("import %+v %v", res, err)
	}
	got := termsByName(t, repo)
	if sol := got["Sol"]; sol.ID == "" || sol.Note != "model name" || !sol.Enabled {
		t.Errorf("Sol %+v", sol)
	}
	// A known spelling (ignoring case) gains the aliases it lacked and
	// keeps its note.
	if llm := got["llm"]; !slices.Equal(llm.Aliases, []string{"G L"}) || llm.Note != "kept" {
		t.Errorf("llm %+v", llm)
	}

	// The same file again changes nothing.
	again, err := importData(ctx, repo, "batch.csv", []byte(csv), false)
	if err != nil || again.Added+again.Updated != 0 {
		t.Errorf("second import %+v %v", again, err)
	}
}

func TestImportDataFormats(t *testing.T) {
	repo := importStore(t)
	ctx := context.Background()
	res, err := importData(ctx, repo, "terms.TXT", []byte("# comment\nwiki = V K\n"), false)
	if err != nil || res.Added != 1 {
		t.Fatalf("text %+v %v", res, err)
	}
	if wiki := termsByName(t, repo)["wiki"]; !slices.Equal(wiki.Aliases, []string{"V K"}) {
		t.Errorf("wiki %+v", wiki)
	}
	if _, err := importData(ctx, repo, "old.json", []byte("{"), false); !errors.Is(err, errImportFormat) {
		t.Errorf("bad json: %v, want errImportFormat", err)
	}
	if _, err := importData(ctx, repo, "bad.csv", []byte{0xff, 0xfe}, true); !errors.Is(err, errImportFormat) {
		t.Errorf("not UTF-8: %v, want errImportFormat", err)
	}
}

// Imports at once (the page's and the command's) do not add a spelling
// twice.
func TestImportDataConcurrent(t *testing.T) {
	repo := importStore(t)
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := importData(context.Background(), repo, "same.csv", []byte("carbon\nSol\n"), false)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := termsByName(t, repo); len(got) != 3 {
		t.Fatalf("terms %v, want llm, carbon and Sol once each", got)
	}
}

func TestCommandImportedRefreshesOnlyWhenItSaved(t *testing.T) {
	cases := []struct {
		res   dictapi.Result
		err   error
		toast bool
	}{
		{dictapi.Result{Added: 1}, nil, true},
		{dictapi.Result{Added: 1, DryRun: true}, nil, false},
		{dictapi.Result{}, nil, false},
		{dictapi.Result{}, errors.New("broken"), false},
	}
	for _, c := range cases {
		a, _, _ := newTestApp(t)
		a.toasts = nil
		a.commandImported(c.res, c.err)
		if got := len(a.toasts) > 0; got != c.toast {
			t.Errorf("%+v %v: toast %v, want %v", c.res, c.err, got, c.toast)
		}
	}
}

// A command's import while the Dictionary page shows refreshes it at
// once; a dry run leaves it as it is.
func TestCommandImportRefreshesTheOpenPage(t *testing.T) {
	a, tt := dictTester(t)
	seedTerms(t, a)
	a.loadDict()
	frames(tt)
	if !has(tt, "5 terms") {
		t.Fatalf("before: %q", firstTexts(tt))
	}
	ctx := context.Background()
	file := []byte("term,aliases\nSol,Soul\n")
	if _, err := a.importForCommand(ctx, "cli.csv", file, true); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if !has(tt, "5 terms") || has(tt, "Sol") {
		t.Errorf("a dry run changed the page: %q", firstTexts(tt))
	}
	a.toasts = nil
	if _, err := a.importForCommand(ctx, "cli.csv", file, false); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if !has(tt, "6 terms") || !has(tt, "Sol") {
		t.Errorf("after the import: %q", firstTexts(tt))
	}
	if !strings.Contains(strings.Join(a.toasts, "\n"), "Imported: 1 new, 0 updated.") && !has(tt, "Imported: 1 new, 0 updated.") {
		t.Errorf("toasts %q", a.toasts)
	}
}
