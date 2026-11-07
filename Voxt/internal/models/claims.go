package models

// What VoxtNew tells the family about its models: claims (it uses a
// model: keep it), and deletions that follow the family's rules (family
// spec §7.3, §7.6).

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/hehehai/actnowkit/family"
	fammodels "github.com/hehehai/actnowkit/models"
)

// Decide is what deleting model id would do: who else uses it (Users,
// LoadedBy: the dialog's "Remove from Voxt" or "Delete for all
// apps…"), what it frees. It reads the family's files: off the main
// thread.
func (m *Manager) Decide(id string) (fammodels.Decision, error) {
	mod, ok := m.Lookup(id)
	if !ok {
		return fammodels.Decision{}, fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	return m.store.Decide(mod.Ref(m.Status(id).Revision))
}

// Delete stops model id's download and removes it from VoxtNew (scope
// fammodels.ThisApp: its claim, and the files only when no other app
// claims or loads them; fammodels.AllApps: the files whoever uses them,
// with a tombstone for the others). A partial download goes the same way.
// A repo lock held elsewhere (another app downloads it) fails at once
// with a *fammodels.BusyError. It blocks on the disk: off the main
// thread.
func (m *Manager) Delete(ctx context.Context, id string, scope fammodels.Scope) ([]fammodels.Result, error) {
	mod, ok := m.Lookup(id)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	m.Cancel(id)
	m.mu.Lock()
	client, rev := m.client, m.status[id].Revision
	s := m.status[id]
	s.Err = nil
	m.status[id] = s
	m.mu.Unlock()
	res, err := client.Delete(ctx, mod.Entry(rev), scope)
	m.Refresh()
	return res, err
}

// Use claims the installed models ids for VoxtNew (origin "reused"):
// "Use them" on the setup guide's reuse card, "Use it" for an
// alternative, and a session opening a model nobody claimed for VoxtNew
// yet. A model not on disk is skipped. Disk and family files: off the
// main thread.
func (m *Manager) Use(ctx context.Context, ids ...string) error {
	return m.claim(ctx, fammodels.OriginReused, ids)
}

// Migrate claims, with origin "migrated", the catalogue's models that are
// on disk and no other app claims (not the borrowed or retired ones): the
// first start with the family's protocol (spec §15.2 step 3). Another app's
// models are VoxtNew's when the user uses them (the setup guide's reuse
// card). Disk only, no network; idempotent.
func (m *Manager) Migrate(ctx context.Context) error {
	var ids []string
	for _, mod := range m.catalog {
		if s := m.Status(mod.ID); !mod.Borrowed && !mod.Retired && len(s.Users) == 0 {
			ids = append(ids, mod.ID)
		}
	}
	return m.claim(ctx, fammodels.OriginMigrated, ids)
}

// Import brings the catalogue's models (at the pinned and accepted
// commits) over into the library, once: from the family's default
// library, where 2.0.0 kept them (ActNow/models/hub, now that VoxtNew has
// its own folder, HubIn), and from the Hugging Face caches (family spec
// v1.0.9). Hard links, nothing downloaded, the sources untouched. VoxtNew's previous claims are made
// again there; the others are left to Migrate and the reuse card. A
// retired model is left where it is: a new user is never shown a model
// Voxt no longer uses. It returns how many models it brought and their
// bytes.
func (m *Manager) Import(ctx context.Context, backup bool) (n int, bytes int64, err error) {
	from := fammodels.LegacyHubs(m.store.Hub)
	if def := fammodels.DefaultHubIn(m.store.Family); !fammodels.SameHub(def, m.store.Hub) {
		from = append([]string{def}, from...) // 2.0.0's library (HubIn)
	}
	res, err := m.store.Import(ctx, from, backup, m.wanted()...)
	for _, r := range res {
		if r.Action == fammodels.ImportImported {
			n++
			bytes += r.Linked + r.Copied
		}
	}
	m.Refresh()
	return n, bytes, err
}

// wanted are the catalogue's models Import looks for (not the retired).
func (m *Manager) wanted() []fammodels.Claim {
	var want []fammodels.Claim
	for _, mod := range m.catalog {
		if mod.Retired {
			continue
		}
		for _, rev := range append([]string{mod.Revision}, mod.Accept...) {
			if fammodels.IsCommit(rev) {
				want = append(want, mod.Claim(rev, fammodels.OriginImported))
			}
		}
	}
	return want
}

// MoveTo makes hub, the folder the user chose (Settings → Storage), the
// library of every app in the family (family spec §7.1: the models in one
// place). It brings the models there, so that no app downloads one again
// (same disk: hard links, no space used; another disk: copies): the
// catalogue's that are in VoxtNew's library, VoxtNew's claims (claimed
// again there), and the models the other apps claim, from the library
// each keeps them in (an empty root is the family's default library,
// ActNow/models/hub). Then commit writes the family's models.root.
// Nothing in the old libraries changes. When a model could not be
// brought (busy, no space, an error) or commit fails, VoxtNew's claims
// stay with this library and the error names the models; what was
// brought stays in hub.
func (m *Manager) MoveTo(ctx context.Context, hub string, backup bool, commit func() error) error {
	from := m.store.Hub
	srcs, want := m.others(hub, []string{from})
	want = append(want, slices.DeleteFunc(m.wanted(), func(w fammodels.Claim) bool { return fammodels.Check(from, w).State != fammodels.Installed })...)
	mine, err := m.store.Claims()
	if err != nil {
		return err
	}
	// Import skips the models it handled for hub before: an earlier
	// attempt's, whose files may be gone. A move looks at every one.
	if err := forgetImported(m.store.Family, m.store.BundleID, hub); err != nil {
		return err
	}
	dst := &fammodels.Store{Family: m.store.Family, Hub: hub, BundleID: m.store.BundleID, AppID: m.store.AppID, Apps: m.store.Apps, Now: m.store.Now}
	_, err = dst.Import(ctx, srcs, backup, want...)
	if err == nil {
		err = notBrought(srcs, hub, want)
	}
	if err == nil {
		err = commit()
	}
	if err != nil && len(mine.Claims) > 0 {
		// The import claimed them in hub: claim them here again.
		err = errors.Join(err, m.store.Claim(context.Background(), mine.Claims...))
	}
	return err
}

// Gather makes VoxtNew's library (SGLang Omni/models, HubIn) the library
// of every app in the family, when the family has none (models.root
// unset): the models in one place. It brings the models the other apps
// claim there from the library each keeps them in (an empty root is the
// family's default library, ActNow/models/hub, where they were before
// 2.0.1): hard links on the same disk, nothing downloaded, the old
// libraries untouched. Then commit writes the family's models.root, so
// that the apps that follow it look for their models there. When a model
// could not be brought (busy, no space, an error), commit is not called:
// the next start tries again.
func (m *Manager) Gather(ctx context.Context, backup bool, commit func() error) error {
	hub := m.store.Hub
	srcs, want := m.others(hub, nil)
	if len(want) > 0 {
		if _, err := m.store.Import(ctx, srcs, backup, want...); err != nil {
			return err
		}
		if err := notBrought(srcs, hub, want); err != nil {
			return err
		}
		m.Refresh()
	}
	return commit()
}

// others are the models the other apps claim outside hub, and srcs with
// the libraries they keep them in (an empty root is the family's default
// library).
func (m *Manager) others(hub string, srcs []string) ([]string, []fammodels.Claim) {
	var want []fammodels.Claim
	all, _ := fammodels.ReadAllClaims(m.store.Family)
	for id, cs := range all {
		root := cmp.Or(cs.Root, fammodels.DefaultHubIn(m.store.Family))
		if id == m.store.BundleID || fammodels.SameHub(root, hub) {
			continue
		}
		if !slices.ContainsFunc(srcs, func(s string) bool { return fammodels.SameHub(s, root) }) {
			srcs = append(srcs, root)
		}
		want = append(want, cs.Claims...)
	}
	return srcs, want
}

// notBrought names the models of want that one of srcs has and hub does
// not.
func notBrought(srcs []string, hub string, want []fammodels.Claim) error {
	var missing []string
	for _, w := range want {
		there := slices.ContainsFunc(srcs, func(s string) bool { return fammodels.Check(s, w).State == fammodels.Installed })
		if there && fammodels.Check(hub, w).State != fammodels.Installed && !slices.Contains(missing, w.Repo) {
			missing = append(missing, w.Repo)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("not brought over: %s", strings.Join(missing, ", "))
	}
	return nil
}

// forgetImported drops hub from the app's record of imports.
func forgetImported(d family.Dir, bundleID, hub string) error {
	path := fammodels.ImportedFile(d, bundleID)
	var rec fammodels.Imported
	if err := family.Load(path, fammodels.KindImported, 1, &rec); err != nil {
		if family.IsNotExist(err) {
			return nil
		}
		return err
	}
	n := len(rec.Hubs)
	rec.Hubs = slices.DeleteFunc(rec.Hubs, func(h fammodels.ImportedHub) bool { return fammodels.SameHub(h.Root, hub) })
	if len(rec.Hubs) == n {
		return nil
	}
	return family.WriteJSON(path, rec)
}

func (m *Manager) claim(ctx context.Context, origin string, ids []string) error {
	var claims []fammodels.Claim
	for _, id := range ids {
		mod, ok := m.Lookup(id)
		if !ok {
			continue
		}
		if rev, _ := find(m.store.Hub, mod); rev != "" {
			claims = append(claims, mod.Claim(rev, origin))
		}
	}
	var err error
	if len(claims) > 0 {
		err = m.store.Claim(ctx, claims...)
	}
	m.Refresh()
	return err
}

// EnsureClaimed claims model id when it is installed and VoxtNew does
// not claim it yet (a session is about to open it): cheap when it does.
func (m *Manager) EnsureClaimed(ctx context.Context, id string) error {
	s := m.Status(id)
	if !s.OnDisk || s.Mine {
		return nil
	}
	return m.Use(ctx, id)
}

// Found is a model of the catalogue already on this Mac, for the setup
// guide's reuse card: who installed it (other apps' names, "" none).
type Found struct {
	ID    string
	Bytes int64
	By    []string
}

// Survey lists which of ids are on disk, and which other apps claim them
// (family spec §7.10.1). Off the main thread.
func (m *Manager) Survey(ids ...string) []Found {
	var want []fammodels.Claim
	byRef := map[fammodels.Ref]string{}
	for _, id := range ids {
		mod, ok := m.Lookup(id)
		if !ok {
			continue
		}
		if rev, _ := find(m.store.Hub, mod); rev != "" {
			c := mod.Claim(rev, "")
			want = append(want, c)
			byRef[c.Ref()] = id
		}
	}
	var out []Found
	for _, f := range m.store.Survey(want...) {
		x := Found{ID: byRef[f.Want.Ref()], Bytes: f.Status.Bytes}
		for _, u := range f.By {
			x.By = append(x.By, u.Name)
		}
		out = append(out, x)
	}
	return out
}

// Usage is the disk space of VoxtNew's claimed models: only VoxtNew's,
// and shared with other ActNow apps (blobs counted once).
func (m *Manager) Usage() (fammodels.Usage, error) { return m.store.Usage() }

// Busy reports a delete's or a claim's failure because another app holds
// the model's lock (it downloads the model): its bundle id.
func Busy(err error) (bundleID string, ok bool) {
	var b *fammodels.BusyError
	if !errors.As(err, &b) {
		return "", false
	}
	if b.Owner == nil {
		return "", true
	}
	return b.Owner.BundleID, true
}
