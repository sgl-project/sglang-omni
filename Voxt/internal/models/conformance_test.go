package models

// actnowkit's conformance scenarios (family spec §16.1) against VoxtNew's
// Manager: reuse without downloading, deletion protection (claims, locks,
// loaded models), deleting for all apps, concurrent and resumed downloads
// against another app, cleaning up after an uninstalled one. The adapter
// turns the scenarios' repos into catalogue models (whole snapshots) and
// drives the Manager as the app's pages do.

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
)

func TestFamilyConformance(t *testing.T) {
	skipWindowsModelLibrary(t)
	newApp := func(t testing.TB, fx *conformance.Fixture, bundleID string) any {
		return &adapter{fx: fx, bundleID: bundleID}
	}
	for _, scenario := range conformance.Scenarios {
		if scenario.Name == "ResumeOtherAppsDownload" {
			scenario.Run = resumeOtherAppsDownload
		}
		t.Run(scenario.Name, func(t *testing.T) {
			fx := conformance.New(t)
			scenario.Run(t, fx, newApp)
		})
	}
}

var repoResume = conformance.TestRepo("mlx-community/test-resume", "4444444444444444444444444444444444444444", 3<<20)

// resumeOtherAppsDownload cancels after the peer writes a partial blob. A
// separate fake Hub keeps late peer traffic out of the app's byte count.
func resumeOtherAppsDownload(t testing.TB, fx *conformance.Fixture, newApp conformance.Factory) {
	app := newApp(t, fx, conformance.AppUnderTest).(conformance.Downloader)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	t.Cleanup(cancel)
	peerHub := conformance.NewFakeHub(repoResume)
	defer peerHub.Close()
	const cut = 2 << 20
	reached, release := peerHub.Pause(repoResume.Ref.Repo, "model.safetensors", cut)
	peerCtx, cancelPeer := context.WithCancel(ctx)
	peer := hfhub.New(hfhub.Options{Store: fx.Peer(conformance.PeerSmith), Endpoints: []string{peerHub.URL}, ReserveBytes: -1})
	entry := hfhub.Entry{ID: "whisper", Repo: repoResume.Ref.Repo, Revision: repoResume.Ref.Revision, Files: []hfhub.File{{Path: "model.safetensors"}}}
	peerErr := make(chan error, 1)
	go func() {
		peerErr <- peer.Download(peerCtx, entry, nil)
	}()
	select {
	case <-reached:
	case <-ctx.Done():
		t.Fatal("the peer never reached the pause")
	}
	plan, err := peer.Plan(ctx, entry)
	if err != nil {
		t.Fatalf("plan the remaining download: %v", err)
	}
	for plan.Bytes != plan.TotalBytes-cut {
		if plan.Bytes < plan.TotalBytes-cut {
			t.Fatalf("the peer wrote past the pause: %d of %d bytes left", plan.Bytes, plan.TotalBytes)
		}
		plan, err = peer.Plan(ctx, entry)
		if err != nil {
			t.Fatalf("watch the partial download: %v", err)
		}
	}
	cancelPeer()
	<-peerErr
	release()
	if plan.Bytes == 0 || plan.Bytes >= plan.TotalBytes {
		t.Fatalf("the peer left %d of %d bytes to download", plan.Bytes, plan.TotalBytes)
	}
	appHub := conformance.NewFakeHub(repoResume)
	defer appHub.Close()
	if err := app.Download(ctx, appHub.URL, repoResume.Ref); err != nil {
		t.Fatalf("app download: %v", err)
	}
	if !fx.Installed(repoResume.Ref) {
		t.Fatal("not installed")
	}
	again := appHub.ServedOf(repoResume.Ref.Repo, "model.safetensors")
	if again != plan.Bytes {
		t.Errorf("the app fetched %d bytes after cancellation left %d bytes to download: it did not resume the partial blob (.incomplete, §7.1)", again, plan.Bytes)
	}
	if left := fx.Incomplete(repoResume.Ref.Repo); len(left) > 0 {
		t.Errorf("partial files left: %v", left)
	}
}

// adapter is VoxtNew's Manager as conformance's ModelApp, Downloader and
// OrphanCleaner.
type adapter struct {
	fx       *conformance.Fixture
	bundleID string
}

func (a *adapter) BundleID() string { return a.bundleID }

// manager is a Manager whose catalogue holds refs, downloading from
// endpoint ("" none).
func (a *adapter) manager(endpoint string, refs ...fammodels.Ref) (*Manager, []string) {
	var cat []Model
	var ids []string
	for i, r := range refs {
		id := fmt.Sprintf("m%d", i)
		cat = append(cat, Model{ID: id, Kind: LLM, Runtime: MLX, Repo: r.Repo, Revision: r.Revision, Family: "test"})
		ids = append(ids, id)
	}
	opt := Options{Store: a.fx.Peer(a.bundleID), Catalog: cat, ReserveBytes: -1}
	if endpoint != "" {
		opt.Network.Endpoints = []string{endpoint}
	}
	return Open(opt), ids
}

func (a *adapter) Reuse(ctx context.Context, refs ...fammodels.Ref) error {
	m, ids := a.manager("", refs...)
	defer m.Close()
	return m.Use(ctx, ids...)
}

func (a *adapter) Remove(ctx context.Context, ref fammodels.Ref, allApps bool) error {
	m, ids := a.manager("", ref)
	defer m.Close()
	scope := fammodels.ThisApp
	if allApps {
		scope = fammodels.AllApps
	}
	_, err := m.Delete(ctx, ids[0], scope)
	return err
}

func (a *adapter) Download(ctx context.Context, endpoint string, ref fammodels.Ref) error {
	m, ids := a.manager(endpoint, ref)
	defer m.Close()
	if err := m.Download(ids[0]); err != nil {
		return err
	}
	for m.Status(ids[0]).Downloading {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(10 * time.Millisecond):
		}
	}
	return m.Status(ids[0]).Err
}

func (a *adapter) CleanOrphans(ctx context.Context, bundleID string) error {
	_, err := a.fx.Peer(a.bundleID).CleanOrphans(ctx, bundleID)
	return err
}
