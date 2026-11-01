package models

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/family"
	fammodels "github.com/hehehai/actnowkit/models"
)

// The Hub for real (VOXTNEW_MODELS_REAL=1, on a Mac with a little room),
// into a temporary $FAMILY and $HUB: the VAD downloaded, a part of the
// fast speech model downloaded, stopped, resumed and deleted. The real
// family directory and Hugging Face cache are never touched.
//
//	VOXTNEW_MODELS_REAL=1 go test -v -run TestReal ./internal/models

func realOnly(t *testing.T) *Manager {
	if os.Getenv("VOXTNEW_MODELS_REAL") != "1" {
		t.Skip("set VOXTNEW_MODELS_REAL=1")
	}
	fx := conformance.New(t)
	m := Open(Options{Store: fx.Peer("dev.hehehai.voxtnew.test")})
	t.Cleanup(m.Close)
	return m
}

func TestRealDownloadVAD(t *testing.T) {
	m := realOnly(t)
	start := time.Now()
	if err := m.Download(VADSilero); err != nil {
		t.Fatal(err)
	}
	s := waitReal(t, m, VADSilero, 2*time.Minute)
	if !s.Installed || s.Err != nil {
		t.Fatalf("%+v", s)
	}
	t.Logf("silero-vad: %d bytes in %v, at %s", s.Total, time.Since(start).Round(time.Millisecond), m.Dir(VADSilero))
	if !Open(Options{Store: m.Store()}).Installed(VADSilero) {
		t.Error("not found again")
	}
	if res, err := m.Delete(context.Background(), VADSilero, fammodels.ThisApp); err != nil || len(res) != 1 || !res[0].Deleted {
		t.Fatalf("%+v %v", res, err)
	}
}

func TestRealCancelResume(t *testing.T) {
	m := realOnly(t)
	part := func(minimum int64) Status {
		t.Helper()
		m.Download(ASRFast)
		deadline := time.Now().Add(3 * time.Minute)
		for m.Status(ASRFast).Done < minimum && time.Now().Before(deadline) {
			if s := m.Status(ASRFast); !s.Downloading {
				t.Fatalf("ended: %+v", s)
			}
			time.Sleep(100 * time.Millisecond)
		}
		m.Cancel(ASRFast)
		m.Refresh()
		return m.Status(ASRFast)
	}
	first := part(40 << 20)
	if first.Done < 40<<20 || first.Installed {
		t.Fatalf("first part: %+v", first)
	}
	again := Open(Options{Store: m.Store()})
	if s := again.Status(ASRFast); s.Done < first.Done {
		t.Errorf("a new Manager sees %d bytes of %d", s.Done, first.Done)
	}
	again.Close()
	second := part(first.Done + 40<<20)
	t.Logf("kept %d MB, then %d MB", first.Done>>20, second.Done>>20)
	if _, err := m.Delete(context.Background(), ASRFast, fammodels.ThisApp); err != nil {
		t.Fatal(err)
	}
	if m.Refresh(); m.Status(ASRFast).Done != 0 {
		t.Errorf("left %d bytes", m.Status(ASRFast).Done)
	}
}

func waitReal(t *testing.T, m *Manager, id string, limit time.Duration) Status {
	deadline := time.Now().Add(limit)
	for time.Now().Before(deadline) {
		if s := m.Status(id); !s.Downloading {
			return s
		}
		time.Sleep(100 * time.Millisecond)
	}
	m.Cancel(id)
	t.Fatalf("%s: not done in %v: %+v", id, limit, m.Status(id))
	return Status{}
}

// TestRealMoveToHub moves models installed the old way (a folder per
// model, $VOXTNEW_MOVE_FROM/<id>/, VoxtNew's models before the family's
// library) into $HUB, with hard links (no room taken): each file,
// checked against the catalogue's SHA-256, becomes the blob the Hub names
// it by (its SHA-256 when in LFS, else its git blob id), the snapshot is
// linked beside its place and renamed in holding the repo lock, as hfhub
// finishes one, and VoxtNew claims it (origin migrated) as $VOXTNEW_MOVE_AS
// (default dev.hehehai.voxtnew). A model already in $HUB is only claimed.
//
//	VOXTNEW_MOVE_FROM=$HOME/voxtnew-models go test -count=1 -v -run TestRealMoveToHub ./internal/models
func TestRealMoveToHub(t *testing.T) {
	from := os.Getenv("VOXTNEW_MOVE_FROM")
	if from == "" {
		t.Skip("set VOXTNEW_MOVE_FROM to the folder of <id>/ folders")
	}
	as := os.Getenv("VOXTNEW_MOVE_AS")
	if as == "" {
		as = "dev.hehehai.voxtnew"
	}
	fam := family.Default()
	if _, err := fam.Ensure(as); err != nil {
		t.Fatal(err)
	}
	hub := Hub()
	store := &fammodels.Store{Family: fam, Hub: hub, BundleID: as, AppID: "voxtnew"}
	ctx := context.Background()
	for _, m := range Catalog() {
		if m.Borrowed {
			continue
		}
		if Installed(hub, m) == "" {
			if err := moveIn(ctx, hub, filepath.Join(from, m.ID), m, as); err != nil {
				t.Fatalf("%s: %v", m.ID, err)
			}
			t.Logf("%s: moved into %s", m.ID, SnapshotDir(hub, m, ""))
		} else {
			t.Logf("%s: in the hub already", m.ID)
		}
		if err := store.Claim(ctx, m.Claim("", fammodels.OriginMigrated)); err != nil {
			t.Fatalf("%s: %v", m.ID, err)
		}
	}
}

func moveIn(ctx context.Context, hub, dir string, m Model, as string) error {
	repo := fammodels.RepoDir(hub, m.Repo)
	lock, err := fammodels.LockRepo(ctx, hub, m.Repo, family.Exclusive, fammodels.Owner{BundleID: as, Op: fammodels.OpFinish})
	if err != nil {
		return err
	}
	defer lock.Release()
	tmp := filepath.Join(repo, "snapshots", m.Revision+".incomplete")
	if err := os.RemoveAll(tmp); err != nil {
		return err
	}
	for _, f := range m.Files {
		src := filepath.Join(dir, filepath.FromSlash(f.Path))
		sum, git, err := hashes(src)
		if err != nil {
			return err
		}
		if sum != f.SHA256 {
			return fmt.Errorf("%s: sha256 %s, the catalogue %s", f.Path, sum, f.SHA256)
		}
		blob := git
		if f.LFS {
			blob = sum
		}
		dst := filepath.Join(repo, "blobs", blob)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if _, err := os.Stat(dst); os.IsNotExist(err) {
			if err := os.Link(src, dst); err != nil {
				return err
			}
		}
		link := filepath.Join(tmp, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return err
		}
		if err := os.Symlink(strings.Repeat("../", strings.Count(f.Path, "/")+2)+"blobs/"+blob, link); err != nil {
			return err
		}
	}
	if err := os.Rename(tmp, filepath.Join(repo, "snapshots", m.Revision)); err != nil {
		return err
	}
	ref := filepath.Join(repo, "refs", "main")
	if _, err := os.Stat(ref); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(ref), 0o755); err != nil {
			return err
		}
		return os.WriteFile(ref, []byte(m.Revision), 0o644)
	}
	return nil
}

// hashes are a file's SHA-256 and git blob id.
func hashes(path string) (sum, git string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return "", "", err
	}
	s, g := sha256.New(), sha1.New()
	fmt.Fprintf(g, "blob %d\x00", fi.Size())
	if _, err := io.Copy(io.MultiWriter(s, g), f); err != nil {
		return "", "", err
	}
	return hex.EncodeToString(s.Sum(nil)), hex.EncodeToString(g.Sum(nil)), nil
}

// TestRealSnapshots prints "<id>=<snapshot folder>" for each catalogue
// model installed in $HUB, a line each (MISSING <id> for the others):
// scripts/mac-regress.sh reads the models' folders from it.
//
//	VOXTNEW_SNAPSHOTS_OUT=list.txt go test -count=1 -run TestRealSnapshots ./internal/models
func TestRealSnapshots(t *testing.T) {
	out := os.Getenv("VOXTNEW_SNAPSHOTS_OUT")
	if out == "" {
		t.Skip("set VOXTNEW_SNAPSHOTS_OUT to the file to write")
	}
	var b strings.Builder
	for _, m := range Catalog() {
		if d := Installed(Hub(), m); d != "" {
			fmt.Fprintf(&b, "%s=%s\n", m.ID, d)
		} else {
			fmt.Fprintf(&b, "MISSING %s\n", m.ID)
		}
	}
	if err := os.WriteFile(out, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
