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
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
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

func TestPageImportRefusesOversizeFile(t *testing.T) {
	a, tt := dictTester(t)
	suffix := "\nat-limit\n"
	atLimit := "#" + strings.Repeat("x", dictapi.MaxBytes-len("#")-len(suffix)) + suffix
	a.importFile(writeTermsFile(t, "at-limit.txt", atLimit))
	frames(tt)

	path := writeTermsFile(t, "oversize.txt", "#"+strings.Repeat("x", dictapi.MaxBytes)+"\nnot-imported\n")
	a.importFile(path)
	frames(tt)
	if got := strings.Join(a.toasts, "\n"); !strings.Contains(got, "larger than 8 MB") && !has(tt, "larger than 8 MB") {
		t.Fatalf("oversize import toast %q, want the size limit", got)
	}
	got := terms(t, a)
	if len(got) != 1 || got[0].Term != "at-limit" {
		t.Fatalf("imports at and over the limit saved %v, want only at-limit", got)
	}

	i18n.SetLocale("zh-Hans")
	a.toasts = nil
	a.importFile(path)
	frames(tt)
	if got := strings.Join(a.toasts, "\n"); !strings.Contains(got, "无法导入：文件超过 8 MB") && !has(tt, "无法导入：文件超过 8 MB") {
		t.Fatalf("Chinese oversize import toast %q, want a translated size limit", got)
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

func TestImportDataSerializesConcurrentImports(t *testing.T) {
	_, repo, _ := importRace(t)
	firstDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "same.csv", []byte("term,aliases\nselected,import alias\n"), false)
		firstDone <- err
	}()
	<-repo.putStarted
	secondDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "same.csv", []byte("term,aliases\nselected,import alias\n"), false)
		secondDone <- err
	}()
	locked := writerIsLocked(repo, repo.otherPut)
	finishImportRace(t, repo, firstDone, repo.secondList, locked)
	if err := <-secondDone; err != nil {
		t.Fatal(err)
	}
	if got := termsByName(t, repo)["selected"].Aliases; !slices.Contains(got, "import alias") {
		t.Fatalf("concurrent imports lost the alias: %v", got)
	}
}

type importRaceStore struct {
	libraryStore
	dictionary store.Dictionary
}

func (s importRaceStore) Dictionary() store.Dictionary { return s.dictionary }

type importRaceRepo struct {
	store.Dictionary
	mu                   sync.Mutex
	metaMu               sync.Mutex
	lockCount            int
	listCount            int
	firstPut             sync.Once
	putStarted           chan struct{}
	allowPut             chan struct{}
	otherPut             chan struct{}
	deleteStarted        chan struct{}
	activateStarted      chan struct{}
	deleteProfileStarted chan struct{}
	deleteDone           chan struct{}
	writeDone            chan struct{}
	activateDone         chan struct{}
	deleteProfileDone    chan struct{}
	secondLock           chan struct{}
	secondList           chan struct{}
}

func (r *importRaceRepo) List(ctx context.Context, query string) ([]store.Term, error) {
	terms, err := r.Dictionary.List(ctx, query)
	r.metaMu.Lock()
	r.listCount++
	if r.listCount == 2 {
		close(r.secondList)
	}
	r.metaMu.Unlock()
	return terms, err
}

func (r *importRaceRepo) Lock() {
	r.metaMu.Lock()
	r.lockCount++
	if r.lockCount == 2 {
		close(r.secondLock)
	}
	r.metaMu.Unlock()
	r.mu.Lock()
}
func (r *importRaceRepo) Unlock() { r.mu.Unlock() }
func (r *importRaceRepo) Put(ctx context.Context, term store.Term) error {
	first := false
	r.firstPut.Do(func() {
		first = true
		close(r.putStarted)
	})
	if first {
		<-r.allowPut
	}
	if !first {
		close(r.otherPut)
	}
	err := r.Dictionary.Put(ctx, term)
	if !first {
		close(r.writeDone)
	}
	return err
}

func (r *importRaceRepo) Delete(ctx context.Context, id, profileID string) error {
	close(r.deleteStarted)
	err := r.Dictionary.Delete(ctx, id, profileID)
	close(r.deleteDone)
	return err
}

func (r *importRaceRepo) ActivateProfile(ctx context.Context, id string) error {
	close(r.activateStarted)
	err := r.Dictionary.ActivateProfile(ctx, id)
	close(r.activateDone)
	return err
}

func (r *importRaceRepo) DeleteProfile(ctx context.Context, id string) error {
	close(r.deleteProfileStarted)
	err := r.Dictionary.DeleteProfile(ctx, id)
	close(r.deleteProfileDone)
	return err
}

func importRace(t *testing.T) (*app, *importRaceRepo, store.Term) {
	t.Helper()
	a, _, _ := newTestApp(t)
	term := store.Term{ID: "race", Term: "llm", Note: "kept", Enabled: true}
	if err := a.lib.store.Dictionary().Put(context.Background(), term); err != nil {
		t.Fatal(err)
	}
	if err := a.lib.store.Dictionary().CreateProfile(context.Background(), store.DictionaryProfile{ID: "work", Name: "Work"}); err != nil {
		t.Fatal(err)
	}
	repo := &importRaceRepo{
		Dictionary: a.lib.store.Dictionary(), putStarted: make(chan struct{}), allowPut: make(chan struct{}),
		otherPut: make(chan struct{}), deleteStarted: make(chan struct{}), activateStarted: make(chan struct{}), deleteProfileStarted: make(chan struct{}),
		deleteDone: make(chan struct{}), writeDone: make(chan struct{}), activateDone: make(chan struct{}), deleteProfileDone: make(chan struct{}), secondLock: make(chan struct{}), secondList: make(chan struct{}),
	}
	a.lib.store = importRaceStore{libraryStore: a.lib.store, dictionary: repo}
	return a, repo, term
}

func waitImportRace(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("dictionary write did not finish")
	}
}

func finishImportRace(t *testing.T, repo *importRaceRepo, importDone <-chan error, otherDone <-chan struct{}, otherLocked bool) {
	t.Helper()
	if otherLocked {
		close(repo.allowPut)
		waitImportRace(t, otherDone)
	} else {
		waitImportRace(t, otherDone)
		close(repo.allowPut)
	}
	if err := <-importDone; err != nil {
		t.Fatal(err)
	}
}

func writerIsLocked(repo *importRaceRepo, started <-chan struct{}) bool {
	select {
	case <-repo.secondLock:
		return true
	case <-started:
		return false
	}
}

func TestImportDataDoesNotRecreateADeletedTerm(t *testing.T) {
	a, repo, term := importRace(t)
	importDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "race.csv", []byte("term,aliases\nllm,import alias\n"), false)
		importDone <- err
	}()
	<-repo.putStarted
	controllerDone := make(chan struct{})
	go func() {
		a.deleteTerm(term)
		close(controllerDone)
	}()
	deleteLocked := writerIsLocked(repo, repo.deleteStarted)
	finishImportRace(t, repo, importDone, repo.deleteDone, deleteLocked)
	waitImportRace(t, controllerDone)
	if _, ok := termsByName(t, repo)["llm"]; ok {
		t.Fatal("the import recreated the term deleted while it was running")
	}
}

func TestImportDataDoesNotOverwriteAPageEdit(t *testing.T) {
	a, repo, term := importRace(t)
	importDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "race.csv", []byte("term,aliases\nllm,import alias\n"), false)
		importDone <- err
	}()
	<-repo.putStarted
	term.Note = "edited by the user"
	controllerDone := make(chan struct{})
	go func() {
		a.saveTerm(term, false)
		close(controllerDone)
	}()
	saveLocked := writerIsLocked(repo, repo.otherPut)
	finishImportRace(t, repo, importDone, repo.writeDone, saveLocked)
	waitImportRace(t, controllerDone)
	if got := termsByName(t, repo)["llm"].Note; got != "edited by the user" {
		t.Fatalf("the import replaced the page edit with %q", got)
	}
}

func TestImportDataDoesNotOverwriteASelectedTerm(t *testing.T) {
	a, repo, _ := importRace(t)
	importDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "race.csv", []byte("term,aliases\nselected,import alias\n"), false)
		importDone <- err
	}()
	<-repo.putStarted
	selectionDone := make(chan struct{})
	go func() {
		a.addSelectedTerm("selected")
		close(selectionDone)
	}()
	selectedLocked := writerIsLocked(repo, repo.otherPut)
	if !selectedLocked {
		waitImportRace(t, repo.writeDone)
		waitImportRace(t, selectionDone)
	}
	finishImportRace(t, repo, importDone, repo.secondList, selectedLocked)
	waitImportRace(t, selectionDone)
	if got := termsByName(t, repo)["selected"].Aliases; !slices.Contains(got, "import alias") {
		t.Fatalf("the selection and import lost the alias: %v", got)
	}
}

func TestImportDataDoesNotOverwriteAnAcceptedSuggestion(t *testing.T) {
	a, repo, _ := importRace(t)
	book := learn.NewBook(a.lib.store.Documents(), repo)
	importDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "race.csv", []byte("term,aliases\nllm,import alias\n"), false)
		importDone <- err
	}()
	<-repo.putStarted
	acceptDone := make(chan error, 1)
	go func() {
		_, err := book.Accept(context.Background(), "suggestion", store.Term{Term: "llm", Aliases: []string{"accepted alias"}})
		acceptDone <- err
	}()
	acceptLocked := writerIsLocked(repo, repo.otherPut)
	finishImportRace(t, repo, importDone, repo.writeDone, acceptLocked)
	if err := <-acceptDone; err != nil {
		t.Fatal(err)
	}
	if got := termsByName(t, repo)["llm"].Aliases; !slices.Contains(got, "accepted alias") {
		t.Fatalf("the import dropped the accepted suggestion's aliases: %v", got)
	}
}

func TestImportDataStaysInItsProfileWhenThePageSwitches(t *testing.T) {
	a, repo, _ := importRace(t)
	importDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "race.csv", []byte("term,aliases\nllm,import alias\n"), false)
		importDone <- err
	}()
	<-repo.putStarted
	controllerDone := make(chan struct{})
	go func() {
		a.switchDictionaryProfile("work")
		close(controllerDone)
	}()
	switchLocked := writerIsLocked(repo, repo.activateStarted)
	finishImportRace(t, repo, importDone, repo.activateDone, switchLocked)
	waitImportRace(t, controllerDone)
	if got := termsByName(t, repo); len(got) != 0 {
		t.Fatalf("the import followed the profile switch: %v", got)
	}
	if err := repo.Dictionary.ActivateProfile(context.Background(), "default"); err != nil {
		t.Fatal(err)
	}
	if got := termsByName(t, repo)["llm"].Aliases; !slices.Contains(got, "import alias") {
		t.Fatalf("the default profile lost the import: %v", got)
	}
}

func TestImportDataStaysInItsProfileWhenThePageDeletesIt(t *testing.T) {
	a, repo, _ := importRace(t)
	importDone := make(chan error, 1)
	go func() {
		_, err := importData(context.Background(), repo, "race.csv", []byte("term,aliases\nllm,import alias\n"), false)
		importDone <- err
	}()
	<-repo.putStarted
	controllerDone := make(chan struct{})
	go func() {
		a.deleteDictionaryProfile("default")
		close(controllerDone)
	}()
	deleteLocked := writerIsLocked(repo, repo.deleteProfileStarted)
	finishImportRace(t, repo, importDone, repo.deleteProfileDone, deleteLocked)
	waitImportRace(t, controllerDone)
	if got := termsByName(t, repo); len(got) != 0 {
		t.Fatalf("the import wrote into the next active profile: %v", got)
	}
}

func TestCommandImportedRefreshesOnlyWhenItSaved(t *testing.T) {
	cases := []struct {
		res     dictapi.Result
		err     error
		toast   bool
		refresh bool
	}{
		{dictapi.Result{Added: 1}, nil, true, true},
		{dictapi.Result{Added: 1, DryRun: true}, nil, false, false},
		{dictapi.Result{}, nil, false, false},
		{dictapi.Result{}, errors.New("broken"), false, true},
	}
	for _, c := range cases {
		a, _, _ := newTestApp(t)
		if err := a.lib.store.Dictionary().Put(context.Background(), store.Term{ID: "saved", Term: "Saved", Enabled: true}); err != nil {
			t.Fatal(err)
		}
		a.toasts = nil
		a.commandImported(c.res, c.err)
		if got := len(a.toasts) > 0; got != c.toast {
			t.Errorf("%+v %v: toast %v, want %v", c.res, c.err, got, c.toast)
		}
		if got := len(a.dv.list) == 1; got != c.refresh {
			t.Errorf("%+v %v: refreshed %v, want %v", c.res, c.err, got, c.refresh)
		}
	}
}

func TestCommandImportedFailureDoesNotLogTheInput(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.commandImported(dictapi.Result{}, errors.New("private-file.csv: term PrivateSpelling"))
	for _, entry := range a.logs.Tail(10, dictionarySource) {
		if strings.Contains(entry.Message, "private-file") || strings.Contains(entry.Message, "PrivateSpelling") {
			t.Fatalf("the log exposes the import error: %q", entry.Message)
		}
	}
}

func TestDictionaryLogSourceRegistered(t *testing.T) {
	if !slices.Contains(logSources, dictionarySource) {
		t.Fatalf("log sources %v do not include %q", logSources, dictionarySource)
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
