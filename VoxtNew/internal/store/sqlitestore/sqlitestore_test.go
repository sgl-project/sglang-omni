package sqlitestore

// These tests need the plugin's native SQLite library. Outside a packaged
// app it is found through MYGO_SQLITE_LIBRARY or downloaded once (SHA-256
// checked) into os.UserCacheDir()/mygo/natives; see the README's Tests.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

var ctx = context.Background()

func openTest(t *testing.T, dir string) *Store {
	t.Helper()
	s, err := Open(ctx, dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestSplitSQL(t *testing.T) {
	got := splitSQL(`-- a comment; with a semicolon;
CREATE TABLE a (x);

CREATE TRIGGER t AFTER INSERT ON a BEGIN
  INSERT INTO b VALUES (1);
  INSERT INTO b VALUES (2);
END;
-- trailing;
CREATE INDEX i ON a(x);
`)
	if len(got) != 3 || !strings.HasPrefix(got[1], "CREATE TRIGGER") || !strings.HasSuffix(got[1], "END;") {
		t.Fatalf("%q", got)
	}
	if n := len(migrations[0].statements); n < 9 {
		t.Errorf("migration 1 has %d statements", n)
	}
}

func TestOpenMigratesAndReopens(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir)
	if v, _ := userVersion(ctx, s.db); v != Version() {
		t.Fatalf("user_version %d", v)
	}
	if fi, err := os.Stat(s.History().AudioDir()); err != nil || !fi.IsDir() || filepath.Dir(s.History().AudioDir()) != dir {
		t.Errorf("audio dir: %v", err)
	}
	for _, table := range []string{"documents", "history", "history_fts", "dictionary_terms"} {
		rows, err := s.db.Query(ctx, `SELECT name FROM sqlite_master WHERE name = ?`, table)
		if err != nil || len(rows.Values) != 1 {
			t.Errorf("table %s: %v", table, err)
		}
	}
	s.Close()
	s = openTest(t, dir)
	if matches, _ := filepath.Glob(filepath.Join(dir, "*.bak-*")); len(matches) != 0 {
		t.Errorf("backups without a migration: %v", matches)
	}
}

func TestBackupBeforeMigrating(t *testing.T) {
	dir := t.TempDir()
	saved := migrations
	t.Cleanup(func() { migrations = saved })

	s := openTest(t, dir)
	if err := s.Documents().Put(ctx, "k", []byte(`{"v":1}`)); err != nil {
		t.Fatal(err)
	}
	s.Close()

	// One more migration: the file is copied first.
	cur := len(saved)
	migrations = append(slices.Clone(saved), migration{version: cur + 1, statements: []string{"CREATE TABLE extra (x)"}})
	s = openTest(t, dir)
	if v, _ := userVersion(ctx, s.db); v != cur+1 {
		t.Fatalf("user_version %d", v)
	}
	bak := filepath.Join(dir, FileName+".bak-"+strconv.Itoa(cur))
	if _, err := os.Stat(bak); err != nil {
		t.Fatal(err)
	}
	s.Close()

	// The backup is the database as it was, with its data.
	migrations = saved
	old := t.TempDir()
	if err := copyFile(bak, filepath.Join(old, FileName)); err != nil {
		t.Fatal(err)
	}
	b := openTest(t, old)
	if v, _ := userVersion(ctx, b.db); v != cur {
		t.Errorf("backup version %d", v)
	}
	if doc, ok, _ := b.Documents().Get(ctx, "k"); !ok || string(doc) != `{"v":1}` {
		t.Errorf("backup data %s %v", doc, ok)
	}

	// A database newer than the app is refused, untouched.
	if _, err := Open(ctx, dir); !errors.Is(err, core.ErrUnsupported) {
		t.Fatalf("newer database: %v", err)
	}

	// A failing migration rolls back and leaves the version as it was.
	migrations = append(slices.Clone(saved), migration{version: cur + 1, statements: []string{"CREATE TABLE ok (x)", "NOT SQL"}})
	fresh := t.TempDir()
	if _, err := Open(ctx, fresh); err == nil {
		t.Fatal("bad migration succeeded")
	}
	migrations = saved
	s = openTest(t, fresh)
	if rows, _ := s.db.Query(ctx, "SELECT name FROM sqlite_master WHERE name = 'ok'"); len(rows.Values) != 0 {
		t.Error("partial migration kept")
	}
	if v, _ := userVersion(ctx, s.db); v != cur {
		t.Errorf("version after failed migration %d", v)
	}
}

func TestDocuments(t *testing.T) {
	d := openTest(t, t.TempDir()).Documents()
	if _, ok, err := d.Get(ctx, "settings:app"); ok || err != nil {
		t.Fatalf("missing: %v %v", ok, err)
	}
	if err := d.Put(ctx, "settings:app", []byte(`{"version":1}`)); err != nil {
		t.Fatal(err)
	}
	d.Put(ctx, "settings:app", []byte(`{"version":2}`))
	if doc, ok, _ := d.Get(ctx, "settings:app"); !ok || string(doc) != `{"version":2}` {
		t.Errorf("got %s", doc)
	}
	if err := d.Put(ctx, "x", []byte(`not json`)); !errors.Is(err, core.ErrInvalid) {
		t.Errorf("bad JSON: %v", err)
	}
	d.Delete(ctx, "settings:app")
	d.Delete(ctx, "settings:app")
	if _, ok, _ := d.Get(ctx, "settings:app"); ok {
		t.Error("not deleted")
	}
}

func entry(id string, kind store.Kind, at time.Time, raw, result string) store.Entry {
	return store.Entry{ID: id, Kind: kind, CreatedAt: at, RawText: raw, Result: result, AppBundle: "com.tinyspeck.slackmacgap",
		AppName: "Slack", RuleID: "g1", RuleName: "工作沟通", RuleSource: "app", ASRProvider: "doubao", ASRModel: "volc.seedasr.sauc.duration",
		LLMProvider: "deepseek", LLMModel: "deepseek-chat", AudioMS: 4200, ASRMS: 310, LLMMS: 820, TotalMS: 5400, Delivery: "insert"}
}

func TestHistory(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	t0 := time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)
	es := []store.Entry{
		entry("a", store.KindTranscribe, t0, "嗯 明天上午十点开会", "明天上午十点开会。"),
		entry("b", store.KindTranslate, t0.Add(time.Minute), "send the draft to Anna", "把草稿发给 Anna"),
		entry("c", store.KindRewrite, t0.Add(2*time.Minute), "", "Dear team, the release moves to Friday."),
	}
	es[2].SourceText, es[2].PromptText = "release friday", "改正式一点"
	es[0].Audio = "a.wav"
	es[0].SentMS, es[0].DSP, es[0].GainDB = 3100, "highpass+level", 7.5
	if err := os.WriteFile(filepath.Join(h.AudioDir(), "a.wav"), []byte("RIFF"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, e := range es {
		if err := h.Put(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	got, err := h.Get(ctx, "a")
	if err != nil {
		t.Fatal(err)
	}
	want := es[0]
	want.Status = store.StatusDone
	if !got.CreatedAt.Equal(want.CreatedAt) {
		t.Errorf("time %v", got.CreatedAt)
	}
	got.CreatedAt = want.CreatedAt
	if got != want {
		t.Errorf("round trip:\n%+v\n%+v", got, want)
	}
	ids := func(q store.HistoryQuery) string {
		list, err := h.List(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, e := range list {
			out = append(out, e.ID)
		}
		return strings.Join(out, ",")
	}
	for q, want := range map[store.HistoryQuery]string{
		{}:                                        "c,b,a",
		{Kind: store.KindTranslate}:               "b",
		{Search: "十点开会"}:                          "a", // trigram, CJK
		{Search: "开会"}:                            "a", // under 3 characters: LIKE
		{Search: "Anna"}:                          "b",
		{Search: "改正式"}:                           "c", // the spoken instruction
		{Search: "release", Limit: 1}:             "c",
		{Limit: 2, Offset: 1}:                     "b,a",
		{Search: "nothing like this"}:             "",
		{Kind: store.KindRewrite, Search: "Anna"}: "",
	} {
		if got := ids(q); got != want {
			t.Errorf("%+v: %q, want %q", q, got, want)
		}
	}
	// Put replaces, and the index follows.
	es[1].Result = "把草稿交给 Anna"
	h.Put(ctx, es[1])
	if got := ids(store.HistoryQuery{Search: "交给 Anna"}); got != "b" {
		t.Errorf("after update: %q", got)
	}
	if err := h.Put(ctx, store.Entry{ID: "x", Kind: "meeting"}); !errors.Is(err, core.ErrInvalid) {
		t.Errorf("bad kind: %v", err)
	}
	if err := h.Put(ctx, store.Entry{ID: "../x", Kind: store.KindTranscribe}); !errors.Is(err, core.ErrInvalid) {
		t.Errorf("bad id: %v", err)
	}
	// Delete takes the audio with it.
	if err := h.Delete(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(h.AudioDir(), "a.wav")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("audio kept: %v", err)
	}
	if err := h.Delete(ctx, "a"); !errors.Is(err, core.ErrNotFound) {
		t.Errorf("deleted twice: %v", err)
	}
	if _, err := h.Get(ctx, "a"); !errors.Is(err, core.ErrNotFound) {
		t.Errorf("get deleted: %v", err)
	}
	if got := ids(store.HistoryQuery{Search: "十点开会"}); got != "" {
		t.Errorf("index kept a deleted row: %q", got)
	}
	if n, err := h.DeleteAll(ctx, store.KindRewrite); n != 1 || err != nil {
		t.Errorf("delete rewrite: %d %v", n, err)
	}
	if st, _ := s.Stats(ctx); st.History != 1 {
		t.Errorf("stats %+v", st)
	}
	if n, err := h.DeleteAll(ctx, ""); n != 1 || err != nil {
		t.Errorf("delete all: %d %v", n, err)
	}
}

func TestDictionary(t *testing.T) {
	s := openTest(t, t.TempDir())
	d := s.Dictionary()
	t0 := time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)
	for _, term := range []store.Term{
		{ID: "1", Term: " VoxtNew ", Aliases: []string{"vox tunew", "vox tunew", " "}, Enabled: true, CreatedAt: t0},
		{ID: "2", Term: "Kubernetes", Aliases: []string{"k8s"}, Note: "work", Enabled: true},
		{ID: "3", Term: "豆包", Enabled: false},
	} {
		if err := d.Put(ctx, term); err != nil {
			t.Fatal(err)
		}
	}
	got, err := d.Get(ctx, "1")
	if err != nil || got.Term != "VoxtNew" || !slices.Equal(got.Aliases, []string{"vox tunew"}) || !got.Enabled ||
		!got.CreatedAt.Equal(t0) || !got.UpdatedAt.Equal(t0) {
		t.Fatalf("%+v %v", got, err)
	}
	if got, _ := d.Get(ctx, "3"); got.Enabled || got.Aliases != nil {
		t.Errorf("disabled term %+v", got)
	}
	// Unique ignoring case.
	if err := d.Put(ctx, store.Term{ID: "4", Term: "voxtnew"}); !errors.Is(err, core.ErrExists) {
		t.Errorf("duplicate: %v", err)
	}
	if err := d.Put(ctx, store.Term{ID: "5", Term: "  "}); !errors.Is(err, core.ErrInvalid) {
		t.Errorf("empty: %v", err)
	}
	names := func(search string) string {
		list, err := d.List(ctx, search)
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, t := range list {
			out = append(out, t.Term)
		}
		return strings.Join(out, ",")
	}
	if got := names(""); got != "Kubernetes,VoxtNew,豆包" {
		t.Errorf("all: %q", got)
	}
	if got := names("k8s"); got != "Kubernetes" {
		t.Errorf("by alias: %q", got)
	}
	if got := names("VOX"); got != "VoxtNew" {
		t.Errorf("ignoring case: %q", got)
	}
	if err := d.Delete(ctx, "2"); err != nil {
		t.Fatal(err)
	}
	if err := d.Delete(ctx, "2"); !errors.Is(err, core.ErrNotFound) {
		t.Errorf("deleted twice: %v", err)
	}
	if st, _ := s.Stats(ctx); st.Dictionary != 2 {
		t.Errorf("stats %+v", st)
	}
}

func TestCheckAndRewritePaths(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir)
	e := entry("a", store.KindTranscribe, time.Now(), "x", "y")
	e.Error = "could not read " + dir + "/audio/a.wav"
	s.History().Put(ctx, e)
	s.Documents().Put(ctx, "k", []byte(`{"path":"`+dir+`/audio/a.wav"}`))
	s.Close()
	db := filepath.Join(dir, FileName)
	n, err := RewritePaths(ctx, db, dir, "/Volumes/Data/VoxtNew")
	if err != nil || n == 0 {
		t.Fatalf("rewrite: %d %v", n, err)
	}
	if v, err := CheckFile(ctx, db); err != nil || v != Version() {
		t.Fatalf("check: %d %v", v, err)
	}
	s = openTest(t, dir)
	got, _ := s.History().Get(ctx, "a")
	if got.Error != "could not read /Volumes/Data/VoxtNew/audio/a.wav" {
		t.Errorf("history: %q", got.Error)
	}
	if doc, _, _ := s.Documents().Get(ctx, "k"); string(doc) != `{"path":"/Volumes/Data/VoxtNew/audio/a.wav"}` {
		t.Errorf("document: %s", doc)
	}
}

// TestHistoryInsertedInto: the app a result was inserted into, when the
// user moved on, rides in the delivery column (no migration) and comes
// back as it went in, the entry's app staying the start app; entries
// without it (older rows: a plain "insert") read as before.
func TestHistoryInsertedInto(t *testing.T) {
	s := openTest(t, t.TempDir())
	h := s.History()
	t0 := time.Date(2026, 10, 18, 9, 0, 0, 0, time.UTC)
	moved := entry("m", store.KindTranscribe, t0, "hi", "Hi.")
	moved.InsertedBundle, moved.InsertedName = "com.apple.TextEdit", "Text|Edit" // a "|" in a name
	plain := entry("p", store.KindTranscribe, t0.Add(time.Minute), "hi", "Hi.")
	card := entry("c", store.KindTranscribe, t0.Add(2*time.Minute), "hi", "Hi.")
	card.Delivery = "card"
	for _, e := range []store.Entry{moved, plain, card} {
		if err := h.Put(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	for _, want := range []store.Entry{moved, plain, card} {
		got, err := h.Get(ctx, want.ID)
		if err != nil {
			t.Fatal(err)
		}
		if got.Delivery != want.Delivery || got.InsertedBundle != want.InsertedBundle || got.InsertedName != want.InsertedName || got.AppBundle != "com.tinyspeck.slackmacgap" || got.AppName != "Slack" {
			t.Errorf("%s: %q %q %q, app %q %q", want.ID, got.Delivery, got.InsertedBundle, got.InsertedName, got.AppBundle, got.AppName)
		}
	}
	for id, want := range map[string]string{"p": "insert", "c": "card", "m": "insert|com.apple.TextEdit|Text|Edit"} {
		rows, err := s.db.Query(ctx, `SELECT delivery FROM history WHERE id = ?`, id)
		if err != nil || len(rows.Values) != 1 || asString(rows.Values[0][0]) != want {
			t.Errorf("%s: the column %v %v", id, rows, err)
		}
	}
}
