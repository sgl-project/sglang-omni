package conformance

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/hfhub"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/settings"
	"github.com/hehehai/actnowkit/shortcuts"
)

// KitApp is the app under test played by actnowkit: the reference
// adapter, and an example of writing one. It implements every interface
// of the suite.
type KitApp struct {
	fx       *Fixture
	bundleID string
	app      family.App
	store    *models.Store

	mu     sync.Mutex
	follow settings.Follow
	own    map[string]string
	sc     []shortcuts.Shortcut
}

// Kit is a Factory making KitApps.
func Kit(t testing.TB, fx *Fixture, bundleID string) any {
	a, _, _ := family.ParseBundleID(bundleID)
	k := &KitApp{fx: fx, bundleID: bundleID, app: a, follow: settings.Follow{}, own: map[string]string{}}
	k.store = &models.Store{Family: fx.Family, Hub: fx.Hub, BundleID: bundleID, AppID: a.ID, Apps: k.discover}
	return k
}

// discover is the app's view of the family: manifests checked against
// the disk (no LaunchServices in a fixture).
func (k *KitApp) discover() map[string]models.AppState {
	found, _ := registry.Discover(k.fx.Family, FixtureSystem{})
	return registry.States(found)
}

func (k *KitApp) BundleID() string { return k.bundleID }

func (k *KitApp) Reuse(ctx context.Context, refs ...models.Ref) error {
	var errs []error
	for _, r := range refs {
		errs = append(errs, k.store.Claim(ctx, models.Claim{Repo: r.Repo, Revision: r.Revision, Origin: models.OriginReused}))
	}
	return errors.Join(errs...)
}

func (k *KitApp) Remove(ctx context.Context, ref models.Ref, allApps bool) error {
	scope := models.ThisApp
	if allApps {
		scope = models.AllApps
	}
	_, err := k.store.Delete(ctx, ref, scope)
	return err
}

func (k *KitApp) UseModel(ctx context.Context, model string, refs ...models.Ref) error {
	var errs []error
	for _, r := range refs {
		errs = append(errs, k.store.Claim(ctx, models.Claim{Repo: r.Repo, Revision: r.Revision, Model: model, Origin: models.OriginReused}))
	}
	return errors.Join(errs...)
}

func (k *KitApp) RemoveModel(ctx context.Context, model string, refs ...models.Ref) error {
	var errs []error
	for _, r := range refs {
		_, err := k.store.DeleteModel(ctx, r, model, models.ThisApp)
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func (k *KitApp) Download(ctx context.Context, endpoint string, ref models.Ref) error {
	c := hfhub.New(hfhub.Options{Store: k.store, Endpoints: []string{endpoint}, ReserveBytes: -1})
	return c.Download(ctx, hfhub.Entry{ID: filepath.Base(ref.Repo), Repo: ref.Repo, Revision: ref.Revision}, nil)
}

func (k *KitApp) CleanOrphans(ctx context.Context, bundleID string) error {
	_, err := k.store.CleanOrphans(ctx, bundleID)
	return err
}

func (k *KitApp) Set(ctx context.Context, key, value string) error {
	k.mu.Lock()
	follow := k.follow.Of(key)
	if !follow {
		k.own[key] = value
	}
	k.mu.Unlock()
	if !follow {
		return nil
	}
	_, err := settings.Set(ctx, k.fx.Family, k.bundleID, key, value)
	return err
}

func (k *KitApp) StopFollowing(ctx context.Context, key string) error {
	fam, _, err := settings.Read(k.fx.Family)
	if err != nil {
		return err
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	// Keep using the value in effect, now as the app's own.
	k.own[key] = settings.Effective(fam, key, k.follow.Of(key), k.own[key], "")
	k.follow[key] = false
	return nil
}

func (k *KitApp) Effective(ctx context.Context, key string) (string, error) {
	fam, _, err := settings.Read(k.fx.Family)
	if err != nil {
		return "", err
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	return settings.Effective(fam, key, k.follow.Of(key), k.own[key], ""), nil
}

func (k *KitApp) Declare(ctx context.Context, sc []shortcuts.Shortcut) error {
	k.mu.Lock()
	k.sc = sc
	k.mu.Unlock()
	return registry.Write(k.fx.Family, registry.Manifest{
		App: k.app.ID, BundleID: k.bundleID, Name: k.app.Name, Version: "1.0.0",
		Path: filepath.Join(filepath.Dir(k.fx.Family.Root), "Applications", k.bundleID, k.app.Name+".app"), Shortcuts: sc,
	})
}

func (k *KitApp) Conflicts(ctx context.Context) ([]string, error) {
	found, _ := registry.Discover(k.fx.Family, FixtureSystem{})
	me, err := registry.Read(k.fx.Family, k.bundleID)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, c := range shortcuts.FindConflicts(registry.HolderOf(me), registry.Holders(found)) {
		out = append(out, c.Other.BundleID)
	}
	return out, nil
}
