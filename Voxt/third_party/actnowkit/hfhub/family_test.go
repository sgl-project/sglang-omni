package hfhub

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
)

// Two family apps on one Mac: the same hub, the same family directory,
// each with its own Client and Store (in one test process: flock locks
// belong to open files, so they exclude each other as two processes do).

const (
	bidDicta = "dev.hehehai.dicta"
	bidAudio = "dev.hehehai.audiosmith"
)

type famEnv struct {
	fam family.Dir
	hub string
}

func newFamEnv(t *testing.T) *famEnv {
	t.Helper()
	root := t.TempDir()
	e := &famEnv{fam: family.At(filepath.Join(root, "ActNow")), hub: filepath.Join(root, "hub")}
	if _, err := e.fam.Ensure("test"); err != nil {
		t.Fatal(err)
	}
	return e
}

// app is a family app's Client and Store, downloading from srv.
func (e *famEnv) app(t *testing.T, srv string, bid string, mod func(*Options)) (*Client, *models.Store) {
	t.Helper()
	a, _, err := family.ParseBundleID(bid)
	if err != nil {
		t.Fatal(err)
	}
	s := &models.Store{Family: e.fam, Hub: e.hub, BundleID: bid, AppID: a.ID}
	o := Options{Store: s, Endpoints: []string{srv}, Official: srv, ChunkSize: 1024, ReserveBytes: -1}
	if mod != nil {
		mod(&o)
	}
	c := New(o)
	c.progressEvery, c.waitPoll = 0, 5*time.Millisecond
	return c, s
}

func pinned(h *fakeHub) Entry {
	e := h.entry()
	e.Revision = testCommit
	return e
}

func claimOn(t *testing.T, s *models.Store, ref models.Ref) (models.Claim, bool) {
	t.Helper()
	cl, err := s.Claims()
	if err != nil {
		t.Fatal(err)
	}
	return cl.Find(ref)
}

func wait(t *testing.T, what string, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(10 * time.Second):
		panic("timed out waiting for " + what)
	}
}

func result(t *testing.T, what string, ch <-chan error) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(30 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return nil
	}
}

// Two apps download the same model at once: each blob comes from the
// network once (the second app waits for the first one's blob locks and
// then finds the blobs complete), both end with the model and claim it.
func TestFamilyConcurrentDownload(t *testing.T) {
	h := newFakeHub(t)
	env := newFamEnv(t)
	ca, sa := env.app(t, h.srv.URL, bidDicta, nil)
	cb, sb := env.app(t, h.srv.URL, bidAudio, nil)
	e := pinned(h)
	reached := h.holdAt(t, "model.safetensors", 512)
	ctx := context.Background()

	errA := make(chan error, 1)
	go func() { errA <- ca.Download(ctx, e, nil) }()
	wait(t, "Dicta's download to reach the big file", reached)

	var mu sync.Mutex
	var waitingFor []string
	sawDicta := make(chan struct{}, 1)
	var once sync.Once
	errB := make(chan error, 1)
	go func() {
		errB <- cb.Download(ctx, e, func(p Progress) {
			if p.WaitingFor == "" {
				return
			}
			mu.Lock()
			waitingFor = append(waitingFor, p.WaitingFor)
			mu.Unlock()
			if p.WaitingFor == bidDicta {
				once.Do(func() { sawDicta <- struct{}{} })
			}
		})
	}()
	wait(t, "AudioSmith to wait for Dicta", sawDicta)
	h.letGo()
	if err := result(t, "Dicta", errA); err != nil {
		t.Fatalf("Dicta: %v", err)
	}
	if err := result(t, "AudioSmith", errB); err != nil {
		t.Fatalf("AudioSmith: %v", err)
	}
	mu.Lock()
	for _, w := range waitingFor {
		if w != bidDicta {
			t.Errorf("waiting for %q", w)
		}
	}
	mu.Unlock()

	if got, want := h.count("served"), sumSizes(h.files); got != want {
		t.Errorf("served %d bytes, want each blob once: %d", got, want)
	}
	checkSnapshot(t, ca, h)
	checkSnapshot(t, cb, h)
	ref := models.Ref{Repo: testRepo, Revision: testCommit}
	ac, ok := claimOn(t, sa, ref)
	if !ok || ac.Origin != models.OriginDownloaded || ac.Aux || ac.Model != "tiny" || ac.Role != models.RoleTTS || ac.Bytes != uniqueSizes(h.files) {
		t.Errorf("Dicta's claim %+v %v", ac, ok)
	}
	if bc, ok := claimOn(t, sb, ref); !ok || bc.Model != "tiny" {
		t.Errorf("AudioSmith's claim %+v %v", bc, ok)
	}
	// No partial snapshot either.
	snaps, _ := os.ReadDir(filepath.Join(RepoDir(env.hub, testRepo), "snapshots"))
	if len(snaps) != 1 || snaps[0].Name() != testCommit {
		t.Errorf("snapshots %v", snaps)
	}
}

// A download cancelled half way leaves its partial files; another app
// (with another chunk size) resumes them instead of starting again.
func TestFamilyResumeAcrossApps(t *testing.T) {
	h := newFakeHub(t)
	env := newFamEnv(t)
	ca, sa := env.app(t, h.srv.URL, bidDicta, func(o *Options) { o.Parallel = 1 })
	cb, sb := env.app(t, h.srv.URL, bidAudio, func(o *Options) { o.ChunkSize = 2048; o.Parallel = 2 })
	e := pinned(h)
	ctx := context.Background()
	cctx, cancel := context.WithCancel(ctx)
	err := ca.Download(cctx, e, func(p Progress) {
		if p.Bytes > p.TotalBytes/2 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) || Code(err) != CodeCancelled {
		t.Fatalf("cancelled download: %v", err)
	}
	// The big file was half done: its parts file says which ranges.
	big := filepath.Join(RepoDir(env.hub, testRepo), "blobs", sha(h.files["model.safetensors"]))
	data, err := os.ReadFile(big + ".incomplete.parts")
	if err != nil {
		t.Fatal(err)
	}
	var pf struct {
		Size  int64  `json:"size"`
		Chunk int64  `json:"chunk"`
		Done  []bool `json:"done"`
	}
	if err := json.Unmarshal(data, &pf); err != nil || pf.Size != int64(len(h.files["model.safetensors"])) || pf.Chunk != 1024 ||
		len(pf.Done) != 10 || !slices.Contains(pf.Done, true) {
		t.Fatalf("parts file %s: %v", data, err)
	}
	if l, _ := cb.Status(ctx, e); l.State != Partial || l.Bytes == 0 {
		t.Fatalf("after cancel: %+v", l)
	}
	p, err := cb.Plan(ctx, e)
	if err != nil || p.Bytes >= p.TotalBytes || p.Bytes == 0 {
		t.Fatalf("plan after cancel: %d of %d to go, %v", p.Bytes, p.TotalBytes, err)
	}
	served := h.count("served")
	if err := cb.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, cb, h)
	if again := h.count("served") - served; again > p.Bytes+1024 {
		t.Errorf("resumed download fetched %d bytes, plan said %d", again, p.Bytes)
	}
	ref := models.Ref{Repo: testRepo, Revision: testCommit}
	if _, ok := claimOn(t, sa, ref); ok {
		t.Error("the cancelled app claims the model")
	}
	if c, ok := claimOn(t, sb, ref); !ok || c.Origin != models.OriginDownloaded {
		t.Errorf("claim %+v %v", c, ok)
	}
}

// While an app downloads a model, another cannot delete it: it is told
// at once who is downloading.
func TestFamilyDeleteDuringDownload(t *testing.T) {
	h := newFakeHub(t)
	env := newFamEnv(t)
	ca, sa := env.app(t, h.srv.URL, bidDicta, nil)
	cb, sb := env.app(t, h.srv.URL, bidAudio, nil)
	e := pinned(h)
	ref := models.Ref{Repo: testRepo, Revision: testCommit}
	ctx := context.Background()
	reached := h.holdAt(t, "model.safetensors", 512)
	errA := make(chan error, 1)
	go func() { errA <- ca.Download(ctx, e, nil) }()
	wait(t, "the download", reached)

	if o, ok := models.Activity(env.hub, testRepo, bidAudio); !ok || o.BundleID != bidDicta {
		t.Errorf("activity %+v %v", o, ok)
	}
	_, err := sb.Delete(ctx, ref, models.AllApps)
	var be *models.BusyError
	if !errors.As(err, &be) || be.Owner == nil || be.Owner.BundleID != bidDicta {
		t.Fatalf("Store.Delete during a download: %v", err)
	}
	_, err = cb.Delete(ctx, e, models.AllApps)
	if !errors.As(err, &be) || be.Owner == nil || be.Owner.BundleID != bidDicta || Code(err) != CodeBusy {
		t.Fatalf("Client.Delete during a download: %v", err)
	}
	// A Client without a Store respects the lock too.
	plain := New(Options{CacheDir: env.hub, BundleID: "org.example.tool", Endpoints: []string{h.srv.URL}, Official: h.srv.URL})
	if _, err := plain.Delete(ctx, e, models.ThisApp); !errors.As(err, &be) {
		t.Fatalf("plain delete during a download: %v", err)
	}
	h.letGo()
	if err := result(t, "the download", errA); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, ca, h)
	if _, ok := claimOn(t, sa, ref); !ok {
		t.Error("no claim after the download")
	}
	if _, ok := claimOn(t, sb, ref); ok {
		t.Error("the app that failed to delete claims it")
	}
}

// "Remove from this app" keeps files another app claims; the last app's
// removal deletes them. Auxiliary repos are judged on their own.
func TestFamilyDeleteKeepsOthersFiles(t *testing.T) {
	h := newFakeHub(t)
	g2p := newFakeHub(t)
	g2p.repo = "acme/g2p"
	g2p.files = map[string][]byte{"config.json": []byte(`{"g2p":1}`), "g2p.safetensors": bytes.Repeat([]byte{3}, 3000)}
	g2p.lfs = map[string]bool{"g2p.safetensors": true}
	both := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/acme/g2p/") {
			g2p.serve(w, r)
		} else {
			h.serve(w, r)
		}
	}))
	defer both.Close()
	env := newFamEnv(t)
	ca, sa := env.app(t, both.URL, bidDicta, nil)
	cb, sb := env.app(t, both.URL, bidAudio, nil)
	ctx := context.Background()
	e := pinned(h)
	e.Aux = []Aux{{Repo: "acme/g2p", Revision: testCommit}}
	main := models.Ref{Repo: testRepo, Revision: testCommit}
	aux := models.Ref{Repo: "acme/g2p", Revision: testCommit}

	if err := ca.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	if c, ok := claimOn(t, sa, aux); !ok || !c.Aux || c.Model != "tiny" {
		t.Errorf("aux claim %+v %v", c, ok)
	}
	if c, ok := claimOn(t, sa, main); !ok || c.Aux {
		t.Errorf("main claim %+v %v", c, ok)
	}
	// AudioSmith uses the model too: nothing to download, it reuses it.
	served := h.count("served") + g2p.count("served")
	if err := cb.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	if h.count("served")+g2p.count("served") != served {
		t.Error("the second app downloaded again")
	}
	if c, ok := claimOn(t, sb, main); !ok || c.Origin != models.OriginReused {
		t.Errorf("reuse claim %+v %v", c, ok)
	}

	// Dicta removes it: AudioSmith still uses both repos, nothing goes.
	res, err := ca.Delete(ctx, e, models.ThisApp)
	if err != nil || len(res) != 2 || res[0].Deleted || res[1].Deleted ||
		len(res[0].Decision.Users) != 1 || res[0].Decision.Users[0].BundleID != bidAudio || res[1].Decision.Ref != aux {
		t.Fatalf("delete from Dicta: %+v %v", res, err)
	}
	if l, _ := cb.Status(ctx, e); l.State != Downloaded {
		t.Errorf("status after Dicta removed it: %+v", l)
	}
	if _, ok := claimOn(t, sa, main); ok {
		t.Error("Dicta still claims it")
	}

	// AudioSmith keeps only the G2P model (another entry uses it), then
	// removes the main model: the main repo goes, the aux one stays.
	e2 := e
	e2.Aux = nil
	res, err = cb.Delete(ctx, e2, models.ThisApp)
	if err != nil || len(res) != 1 || !res[0].Deleted || !res[0].Decision.LastSnapshot {
		t.Fatalf("delete from AudioSmith: %+v %v", res, err)
	}
	if _, err := os.Stat(RepoDir(env.hub, testRepo)); !os.IsNotExist(err) {
		t.Errorf("main repo left: %v", err)
	}
	if l := statusOne(env.hub, Entry{Repo: "acme/g2p", Revision: testCommit}, true); l.State != Downloaded {
		t.Errorf("aux repo: %+v", l)
	}
	if _, ok := claimOn(t, sb, aux); !ok {
		t.Error("AudioSmith's aux claim went")
	}
	// Removing the aux repo itself (nobody else claims it) deletes it.
	res, err = cb.Delete(ctx, Entry{Repo: "acme/g2p", Revision: testCommit}, models.ThisApp)
	if err != nil || len(res) != 1 || !res[0].Deleted {
		t.Fatalf("delete aux: %+v %v", res, err)
	}
	if _, err := os.Stat(RepoDir(env.hub, "acme/g2p")); !os.IsNotExist(err) {
		t.Errorf("aux repo left: %v", err)
	}
}

// Verify removes a corrupt blob (not while someone holds its lock) and
// tells the family; any app's next download repairs it.
func TestFamilyVerifyRepair(t *testing.T) {
	h := newFakeHub(t)
	env := newFamEnv(t)
	ca, _ := env.app(t, h.srv.URL, bidDicta, nil)
	cb, sb := env.app(t, h.srv.URL, bidAudio, nil)
	e := pinned(h)
	ctx := context.Background()
	if err := ca.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	w, err := env.fam.Watch(family.TopicModels)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	l, _ := ca.Status(ctx, e)
	blob, _ := models.ResolveBlob(RepoDir(env.hub, testRepo), filepath.Join(l.Snapshot, "model.safetensors"))
	data, _ := os.ReadFile(blob)
	data[100] ^= 1
	os.WriteFile(blob, data, 0o644)

	// Someone holds the blob's lock (repairing it): reported, not removed.
	held, err := models.TryLockBlob(env.hub, testRepo, filepath.Base(blob), models.Owner{BundleID: bidAudio, Op: models.OpVerify})
	if err != nil {
		t.Fatal(err)
	}
	bad, err := ca.Verify(ctx, e)
	if err != nil || !slices.Equal(bad, []string{"model.safetensors"}) {
		t.Fatalf("verify: %v %v", bad, err)
	}
	if _, err := os.Stat(blob); err != nil {
		t.Error("removed a blob whose lock is held")
	}
	held.Release()

	bad, err = ca.Verify(ctx, e)
	if err != nil || !slices.Equal(bad, []string{"model.safetensors"}) {
		t.Fatalf("verify: %v %v", bad, err)
	}
	if _, err := os.Stat(blob); !os.IsNotExist(err) {
		t.Error("the bad blob stays")
	}
	select {
	case topic := <-w.C:
		if topic != family.TopicModels {
			t.Errorf("topic %q", topic)
		}
	case <-time.After(5 * time.Second):
		t.Error("no models notification after removing a blob")
	}
	if l, _ := cb.Status(ctx, e); l.State != Partial || !slices.Contains(l.Missing, "model.safetensors") {
		t.Errorf("status after verify: %+v", l)
	}
	served := h.count("served")
	if err := cb.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	if got := h.count("served") - served; got != int64(len(h.files["model.safetensors"])) {
		t.Errorf("repair fetched %d bytes", got)
	}
	checkSnapshot(t, ca, h)
	if bad, err := cb.Verify(ctx, e); err != nil || len(bad) != 0 {
		t.Errorf("verify after repair: %v %v", bad, err)
	}
	if _, ok := claimOn(t, sb, models.Ref{Repo: testRepo, Revision: testCommit}); !ok {
		t.Error("the repairing app does not claim it")
	}
}

// A parts file that does not fit its blob is not trusted: the blob
// downloads again from the start.
func TestStalePartsDiscarded(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, nil)
	big := h.files["model.safetensors"]
	blobs := filepath.Join(RepoDir(c.opts.CacheDir, testRepo), "blobs")
	os.MkdirAll(blobs, 0o755)
	name := filepath.Join(blobs, sha(big))
	os.WriteFile(name+".incomplete", bytes.Repeat([]byte{9}, len(big)), 0o644)
	os.WriteFile(name+".incomplete.parts", []byte(`{"size":5,"chunk":1024,"done":[true]}`), 0o644)
	p, _ := c.Plan(context.Background(), h.entry())
	if p.Bytes != p.TotalBytes {
		t.Errorf("plan counts a stale partial: %d of %d", p.Bytes, p.TotalBytes)
	}
	if err := c.Download(context.Background(), h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
}

// huggingface_hub locks the same blob lock file but writes no owner: the
// download waits for it all the same, as for an unknown owner.
func TestWaitForPythonTool(t *testing.T) {
	h := newFakeHub(t)
	env := newFamEnv(t)
	c, _ := env.app(t, h.srv.URL, bidDicta, nil)
	blob := sha(h.files["model.safetensors"])
	lockFile := models.BlobLockFile(env.hub, testRepo, blob)
	os.MkdirAll(filepath.Dir(lockFile), 0o755)
	tool, err := family.TryLock(lockFile, family.Exclusive, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	waiting := make(chan struct{})
	var once sync.Once
	done := make(chan error, 1)
	go func() {
		done <- c.Download(context.Background(), pinned(h), func(p Progress) {
			if p.WaitingFor == UnknownOwner {
				once.Do(func() { close(waiting) })
			}
		})
	}()
	wait(t, "the download to wait", waiting)
	// The tool finishes the blob, then lets go.
	os.WriteFile(filepath.Join(RepoDir(env.hub, testRepo), "blobs", blob), h.files["model.safetensors"], 0o644)
	tool.Unlock()
	if err := result(t, "the download", done); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
	if got, want := h.count("served"), sumSizes(h.files)-int64(len(h.files["model.safetensors"])); got != want {
		t.Errorf("served %d, want %d (the tool's blob is not fetched)", got, want)
	}
}

// Two entries of one app share repos (here the auxiliary one, and even
// the main one): deleting one entry keeps what the other still uses.
func TestFamilySharedWithinApp(t *testing.T) {
	h := newFakeHub(t)
	g2p := newFakeHub(t)
	g2p.repo = "acme/g2p"
	g2p.files = map[string][]byte{"config.json": []byte(`{"g2p":1}`), "g2p.safetensors": bytes.Repeat([]byte{3}, 3000)}
	g2p.lfs = map[string]bool{"g2p.safetensors": true}
	both := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/acme/g2p/") {
			g2p.serve(w, r)
		} else {
			h.serve(w, r)
		}
	}))
	defer both.Close()
	env := newFamEnv(t)
	c, s := env.app(t, both.URL, bidDicta, nil)
	ctx := context.Background()
	a := pinned(h)
	a.Aux = []Aux{{Repo: "acme/g2p", Revision: testCommit}}
	b := a
	b.ID = "tiny-b"
	aux := models.Ref{Repo: "acme/g2p", Revision: testCommit}
	for _, e := range []Entry{a, b} {
		if err := c.Download(ctx, e, nil); err != nil {
			t.Fatal(err)
		}
	}
	if cl, ok := claimOn(t, s, aux); !ok || !slices.Equal(cl.Users(), []string{"tiny", "tiny-b"}) {
		t.Fatalf("aux claim %+v", cl)
	}
	res, err := c.Delete(ctx, a, models.ThisApp)
	if err != nil || len(res) != 2 {
		t.Fatalf("%+v %v", res, err)
	}
	for _, r := range res {
		if r.Deleted || !slices.Equal(r.OwnModels, []string{"tiny-b"}) {
			t.Errorf("deleting tiny: %+v", r)
		}
	}
	if cl, ok := claimOn(t, s, aux); !ok || !slices.Equal(cl.Users(), []string{"tiny-b"}) {
		t.Errorf("aux claim after deleting tiny: %+v %v", cl, ok)
	}
	if l, _ := c.Status(ctx, b); l.State != Downloaded {
		t.Errorf("tiny-b broken: %+v", l)
	}
	res, err = c.Delete(ctx, b, models.ThisApp)
	if err != nil || len(res) != 2 || !res[0].Deleted || !res[1].Deleted {
		t.Fatalf("deleting tiny-b: %+v %v", res, err)
	}
	if _, ok := claimOn(t, s, aux); ok {
		t.Error("aux claim left")
	}
}
