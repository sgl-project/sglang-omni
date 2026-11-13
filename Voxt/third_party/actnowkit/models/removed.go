package models

import (
	"context"
	"errors"
	"slices"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// KindRemoved is the schema kind of models/removed.json.
const KindRemoved = "models.removed"

// TombstoneTTL is how long a "deleted for all apps" record is kept
// (spec §7.6).
const TombstoneTTL = 90 * 24 * time.Hour

// Tombstone records that a model was deleted for all apps: the other apps
// drop their claims on it and show "Deleted in Dicta · Download again".
type Tombstone struct {
	Repo      string    `json:"repo"`
	Revision  string    `json:"revision"`
	RemovedBy string    `json:"removedBy"` // bundle id
	At        time.Time `json:"at"`
	// Root is the $HUB the files were deleted from; claims of another
	// root are unaffected. "" (older writers) = any root.
	Root  string       `json:"root,omitempty"`
	Extra family.Extra `json:"-"`
}

func (t *Tombstone) UnmarshalJSON(b []byte) error {
	type plain Tombstone
	return family.UnmarshalKnown(b, (*plain)(t), &t.Extra)
}

func (t Tombstone) MarshalJSON() ([]byte, error) {
	type plain Tombstone
	return family.MarshalKnown(plain(t), t.Extra)
}

// Ref is the tombstone's model.
func (t Tombstone) Ref() Ref { return Ref{Repo: t.Repo, Revision: t.Revision} }

// Removed is models/removed.json (multi-writer, locks/removed.lock).
type Removed struct {
	Schema  string       `json:"schema"`
	Removed []Tombstone  `json:"removed"`
	Extra   family.Extra `json:"-"`
}

func (r *Removed) UnmarshalJSON(b []byte) error {
	type plain Removed
	return family.UnmarshalKnown(b, (*plain)(r), &r.Extra)
}

func (r Removed) MarshalJSON() ([]byte, error) {
	type plain Removed
	return family.MarshalKnown(plain(r), r.Extra)
}

// ReadTombstones reads models/removed.json (none when missing).
func ReadTombstones(d family.Dir) ([]Tombstone, error) {
	var r Removed
	err := family.Load(d.RemovedFile(), KindRemoved, 1, &r)
	if family.IsNotExist(err) || errors.Is(err, family.ErrNewerSchema) {
		err = nil
	}
	return r.Removed, err
}

func updateRemoved(ctx context.Context, d family.Dir, fn func(r *Removed) bool) error {
	var r Removed
	err := d.Update(ctx, family.LockRemoved, d.RemovedFile(), KindRemoved, 1, &r, func(bool) error {
		changed := fn(&r)
		// Expired records go whenever the file is written.
		n := len(r.Removed)
		r.Removed = slices.DeleteFunc(r.Removed, func(t Tombstone) bool { return time.Since(t.At) > TombstoneTTL })
		if !changed && n == len(r.Removed) {
			return family.ErrNoChange
		}
		r.Schema = family.SchemaOf(KindRemoved, 1)
		if r.Removed == nil {
			r.Removed = []Tombstone{}
		}
		return nil
	})
	if err == nil {
		_ = d.Post(family.TopicModels)
	}
	return err
}

// AddTombstone records that ref was deleted for all apps from hub.
func AddTombstone(ctx context.Context, d family.Dir, hub string, ref Ref, by string, at time.Time) error {
	return updateRemoved(ctx, d, func(r *Removed) bool {
		r.Removed = slices.DeleteFunc(r.Removed, func(t Tombstone) bool { return t.Ref() == ref && tombstoneRoot(t, hub) })
		r.Removed = append(r.Removed, Tombstone{Repo: ref.Repo, Revision: ref.Revision, RemovedBy: by, At: at, Root: hub})
		return true
	})
}

// ClearTombstone forgets that ref was deleted from hub: it is on disk
// again (downloaded or claimed).
func ClearTombstone(ctx context.Context, d family.Dir, hub string, ref Ref) error {
	ts, err := ReadTombstones(d)
	if err != nil || !slices.ContainsFunc(ts, func(t Tombstone) bool { return t.Ref() == ref && tombstoneRoot(t, hub) }) {
		return err // cheap path: nothing to clear, no lock taken
	}
	return updateRemoved(ctx, d, func(r *Removed) bool {
		n := len(r.Removed)
		r.Removed = slices.DeleteFunc(r.Removed, func(t Tombstone) bool { return t.Ref() == ref && tombstoneRoot(t, hub) })
		return n != len(r.Removed)
	})
}

func tombstoneRoot(t Tombstone, hub string) bool { return t.Root == "" || SameHub(t.Root, hub) }

// ApplyTombstones drops this app's claims on models another app deleted
// for all apps after this app claimed them, and returns those tombstones
// (for "Deleted in Dicta · Download again", TextDeletedIn). Call it at
// start-up and on TopicModels.
func (s *Store) ApplyTombstones() ([]Tombstone, error) {
	ts, err := ReadTombstones(s.Family)
	if err != nil || len(ts) == 0 {
		return nil, err
	}
	var hit []Tombstone
	err = s.update(func(c *Claims) bool {
		n := len(c.Claims)
		c.Claims = slices.DeleteFunc(c.Claims, func(cl Claim) bool {
			for _, t := range ts {
				if t.Ref() == cl.Ref() && tombstoneRoot(t, s.Hub) && t.RemovedBy != s.BundleID && !cl.Since.After(t.At) {
					hit = append(hit, t)
					return true
				}
			}
			return false
		})
		return len(c.Claims) != n
	})
	return hit, err
}
