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
	fammodels "github.com/hehehai/actnowkit/models"
)

func TestFamilyConformance(t *testing.T) {
	conformance.Run(t, func(t testing.TB, fx *conformance.Fixture, bundleID string) any {
		return &adapter{fx: fx, bundleID: bundleID}
	})
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
