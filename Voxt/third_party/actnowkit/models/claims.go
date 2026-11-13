package models

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// KindClaims is the schema kind of models/claims/<bundle id>.json.
const KindClaims = "models.claims"

// Origins of a claim.
const (
	OriginDownloaded = "downloaded" // this app downloaded it
	OriginReused     = "reused"     // it was there (another app's), this app uses it
	OriginMigrated   = "migrated"   // registered at the first start with the spec (§15.2)
)

// Roles of a claimed model.
const (
	RoleASR     = "asr"
	RoleTTS     = "tts"
	RoleLLM     = "llm"
	RoleVAD     = "vad"
	RoleAlign   = "align"
	RoleDiarize = "diarize"
	RoleEnhance = "enhance"
	RoleEmbed   = "embed" // embedding models (search by meaning)
	RoleVLM     = "vlm"   // vision language models used only for images
	RoleOther   = "other"
)

// Claim is "this app uses repo@revision, keep it" (spec §7.3).
type Claim struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Model    string `json:"model,omitempty"` // the app's catalogue id (the first model that claimed it)
	// Models are every catalogue entry of the app that uses the repo:
	// an auxiliary repo (a tokenizer) shared by two models of one app is
	// one claim with both. Removing one model releases only its name
	// (Store.Release, Store.DeleteModel); the claim goes with the last.
	// Absent in files of older writers: Model alone counts.
	Models []string `json:"models,omitempty"`
	Family string   `json:"family,omitempty"` // model series: "qwen3-asr"
	Role   string   `json:"role,omitempty"`
	Bytes  int64    `json:"bytes,omitempty"`
	// Files lists the files used when only part of the repo is (globs
	// allowed: "v3/*"); nil = the whole snapshot.
	Files  []string  `json:"files"`
	Aux    bool      `json:"aux"`
	Since  time.Time `json:"since"`
	Origin string    `json:"origin,omitempty"`
	// Sizes are the expected sizes of files (path → bytes), from the
	// app's catalogue, for Check; not written to the claims file.
	Sizes map[string]int64 `json:"-"`
	Extra family.Extra     `json:"-"`
}

func (c *Claim) UnmarshalJSON(b []byte) error {
	type plain Claim
	return family.UnmarshalKnown(b, (*plain)(c), &c.Extra)
}

func (c Claim) MarshalJSON() ([]byte, error) {
	type plain Claim
	return family.MarshalKnown(plain(c), c.Extra)
}

// Ref is the claim's identity.
func (c Claim) Ref() Ref { return Ref{Repo: c.Repo, Revision: c.Revision} }

// Users are the app's models using the claimed repo: Models and Model,
// each once, sorted; empty when the claim names none.
func (c Claim) Users() []string {
	out := slices.Clone(c.Models)
	if c.Model != "" {
		out = append(out, c.Model)
	}
	slices.Sort(out)
	return slices.Compact(out)
}

// merge folds a new claim of the same ref into an existing one: models
// and files are united (no file list = the whole snapshot), the first
// model, since and origin are kept.
func merge(old, c Claim) Claim {
	users := append(old.Users(), c.Users()...)
	slices.Sort(users)
	users = slices.Compact(users)
	if old.Model != "" {
		c.Model = old.Model
	}
	c.Models = nil
	if len(users) > 0 {
		c.Models = users
	}
	if old.Files == nil || c.Files == nil {
		c.Files = nil
	} else {
		f := append(slices.Clone(old.Files), c.Files...)
		slices.Sort(f)
		c.Files = slices.Compact(f)
	}
	if c.Family == "" {
		c.Family = old.Family
	}
	if c.Role == "" {
		c.Role = old.Role
	}
	c.Aux = c.Aux && old.Aux
	c.Since, c.Extra = old.Since, mergeExtra(old.Extra, c.Extra)
	if old.Origin != "" {
		c.Origin = old.Origin // keep how it first came
	}
	return c
}

// Claims is one app's claims file, models/claims/<bundle id>.json. Only
// that app writes it.
type Claims struct {
	Schema   string       `json:"schema"`
	App      string       `json:"app"`
	BundleID string       `json:"bundleId"`
	Updated  time.Time    `json:"updated"`
	Root     string       `json:"root"` // the $HUB these claims protect
	Claims   []Claim      `json:"claims"`
	Extra    family.Extra `json:"-"`
}

func (c *Claims) UnmarshalJSON(b []byte) error {
	type plain Claims
	return family.UnmarshalKnown(b, (*plain)(c), &c.Extra)
}

func (c Claims) MarshalJSON() ([]byte, error) {
	type plain Claims
	return family.MarshalKnown(plain(c), c.Extra)
}

// Find returns the claim on ref.
func (c Claims) Find(ref Ref) (Claim, bool) {
	for _, x := range c.Claims {
		if x.Ref() == ref {
			return x, true
		}
	}
	return Claim{}, false
}

// ReadClaims reads one app's claims; a missing file is no claims.
func ReadClaims(d family.Dir, bundleID string) (Claims, error) {
	var c Claims
	err := family.Load(d.ClaimsFile(bundleID), KindClaims, 1, &c)
	if family.IsNotExist(err) {
		return Claims{BundleID: bundleID}, nil
	}
	if errors.Is(err, family.ErrNewerSchema) {
		err = nil // what is understood still protects
	}
	if c.BundleID == "" {
		c.BundleID = bundleID
	}
	return c, err
}

// ReadAllClaims reads every app's claims file, keyed by the bundle id of
// the file name. Files that cannot be read are reported in errs and left
// out; callers that delete must treat errs as "someone may use anything"
// (Decide does).
func ReadAllClaims(d family.Dir) (all map[string]Claims, errs []error) {
	all = map[string]Claims{}
	entries, err := os.ReadDir(d.ClaimsDir())
	if err != nil {
		if !os.IsNotExist(err) {
			errs = append(errs, err)
		}
		return all, errs
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".json") {
			continue
		}
		bid := strings.TrimSuffix(name, ".json")
		c, err := ReadClaims(d, bid)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		all[bid] = c
	}
	return all, errs
}

// AppState is what the registry knows of an app (registry.States).
type AppState struct {
	BundleID  string
	AppID     string
	Name      string
	Installed bool
	Running   bool
	// ModelsRoot is the $HUB the app's manifest names (manifest
	// models.root): the app keeps models there. An installed app with a
	// ModelsRoot equal to this app's $HUB but no claims file makes
	// deletions unsure (spec §7.6, v1.0.9: the claims may have been lost).
	ModelsRoot string
}

// Store is one app's view of the model library: its claims, and the
// decisions and deletions that respect the other apps' claims.
type Store struct {
	Family   family.Dir
	Hub      string // $HUB in use (ResolveHub)
	BundleID string // this app's bundle id
	AppID    string // this app's id: "dicta"
	// Apps reports the family apps by bundle id (registry.States). An app
	// with claims but missing from the map counts as installed: when in
	// doubt, keep (spec §3.4). nil = every app counts as installed.
	Apps func() map[string]AppState
	// Now is the clock (tests); nil = time.Now.
	Now func() time.Time

	mu sync.Mutex // serialises writes of the own claims file
}

func (s *Store) now() time.Time {
	if s.Now != nil {
		return s.Now().UTC().Truncate(time.Second)
	}
	return family.Now()
}

func (s *Store) who(op string) Owner {
	return Owner{BundleID: s.BundleID, Op: op}
}

// Claims reads this app's claims.
func (s *Store) Claims() (Claims, error) { return ReadClaims(s.Family, s.BundleID) }

// update rewrites this app's claims file.
func (s *Store) update(fn func(c *Claims) (changed bool)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	path := s.Family.ClaimsFile(s.BundleID)
	var c Claims
	err := family.Load(path, KindClaims, 1, &c)
	var ce *family.CorruptError
	switch {
	case err == nil, family.IsNotExist(err):
	case errors.As(err, &ce):
		if err := family.Quarantine(path); err != nil {
			return err
		}
		c = Claims{}
	default:
		return err // includes ErrNewerSchema: never write back
	}
	if c.Root != "" && !SameHub(c.Root, s.Hub) {
		// Claims of another root protect nothing here; Rehome decides
		// what to carry over. Writing new claims starts the new root.
		c.Claims = nil
	}
	if !fn(&c) {
		return nil
	}
	c.Schema = family.SchemaOf(KindClaims, 1)
	c.App, c.BundleID, c.Root, c.Updated = s.AppID, s.BundleID, s.Hub, s.now()
	if c.Claims == nil {
		c.Claims = []Claim{}
	}
	if err := family.WriteJSON(path, c); err != nil {
		return err
	}
	_ = s.Family.Post(family.TopicModels)
	return nil
}

// ErrNotInstalled: a claim was asked for a model that is not on disk.
var ErrNotInstalled = errors.New("model is not installed")

// Claim adds (or refreshes) claims (spec §7.3): after a download, when
// reusing a model another app installed, or at migration. Each is added
// holding the repo's shared lock, and checked to be installed afterwards
// (a racing delete either finished before, and the check fails, or waits
// for the lock and then sees the claim). A model that is not installed is
// not claimed: ErrNotInstalled. Tombstones of claimed models are cleared.
func (s *Store) Claim(ctx context.Context, claims ...Claim) error {
	var errs []error
	for _, c := range claims {
		if err := s.claimOne(ctx, c); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", c.Ref(), err))
		}
	}
	return errors.Join(errs...)
}

func (s *Store) claimOne(ctx context.Context, c Claim) error {
	if !ValidRepo(c.Repo) || c.Revision == "" {
		return errBadRepo
	}
	lock, err := LockRepo(ctx, s.Hub, c.Repo, family.Shared, s.who(OpClaim))
	if err != nil {
		return err
	}
	defer lock.Release()
	return s.ClaimLocked(ctx, c)
}

// ClaimLocked is Claim for a caller that already holds c's repo lock
// (shared or exclusive), such as a downloader finishing a snapshot:
// taking it again from the same process would wait for itself.
func (s *Store) ClaimLocked(ctx context.Context, c Claim) error {
	if !ValidRepo(c.Repo) || c.Revision == "" {
		return errBadRepo
	}
	if c.Since.IsZero() {
		c.Since = s.now()
	}
	if c.Origin == "" {
		c.Origin = OriginReused
	}
	st := Check(s.Hub, c)
	if st.State != Installed {
		return ErrNotInstalled
	}
	if c.Bytes == 0 {
		c.Bytes = st.Bytes
	}
	if users := c.Users(); len(users) > 0 {
		c.Models = users
	}
	if err := s.update(func(f *Claims) bool {
		for i, x := range f.Claims {
			if x.Ref() == c.Ref() {
				f.Claims[i] = merge(x, c)
				return true
			}
		}
		f.Claims = append(f.Claims, c)
		return true
	}); err != nil {
		return err
	}
	if Check(s.Hub, c).State != Installed {
		_ = s.Unclaim(c.Ref())
		return ErrNotInstalled
	}
	_ = ClearTombstone(ctx, s.Family, s.Hub, c.Ref())
	return nil
}

func mergeExtra(old, new family.Extra) family.Extra {
	if len(old) == 0 {
		return new
	}
	out := family.Extra{}
	for k, v := range old {
		out[k] = v
	}
	for k, v := range new {
		out[k] = v
	}
	return out
}

// Unclaim removes this app's claims on refs (the user removed the model
// from this app, or the catalogue dropped it). Files are not touched: see
// Delete.
func (s *Store) Unclaim(refs ...Ref) error {
	return s.update(func(f *Claims) bool {
		n := len(f.Claims)
		f.Claims = slices.DeleteFunc(f.Claims, func(c Claim) bool { return slices.Contains(refs, c.Ref()) })
		return len(f.Claims) != n
	})
}

// Release removes one model (a catalogue id) from this app's claim on
// ref: the claim stays while other models of the app still use the repo
// (remaining lists them) and goes with the last one. A claim naming no
// model at all, or model "", goes at once. Files are not touched: see
// DeleteModel.
func (s *Store) Release(ref Ref, model string) (remaining []string, err error) {
	err = s.update(func(f *Claims) bool {
		for i, x := range f.Claims {
			if x.Ref() != ref {
				continue
			}
			users := slices.DeleteFunc(x.Users(), func(u string) bool { return u == model })
			if model == "" || len(users) == 0 {
				f.Claims = slices.Delete(f.Claims, i, i+1)
				return true
			}
			remaining = users
			if len(users) == len(x.Users()) {
				return false // model did not use it
			}
			x.Models = users
			if x.Model == model {
				x.Model = users[0]
			}
			f.Claims[i] = x
			return true
		}
		return false
	})
	return remaining, err
}

// Rehome moves this app's claims to the hub s.Hub after the app's model
// folder changed (spec §15.2 step 5): claims whose model is installed
// under the new hub are kept, the others dropped.
func (s *Store) Rehome() error {
	s.mu.Lock()
	var c Claims
	err := family.Load(s.Family.ClaimsFile(s.BundleID), KindClaims, 1, &c)
	s.mu.Unlock()
	if err != nil && !family.IsNotExist(err) {
		return err
	}
	var keep []Claim
	for _, x := range c.Claims {
		if Check(s.Hub, x).State == Installed {
			keep = append(keep, x)
		}
	}
	return s.update(func(f *Claims) bool {
		f.Claims = keep
		return true
	})
}

// Heal recreates this app's claims file when it is missing (spec §7.6,
// v1.0.9): the family folder or its models/claims were deleted (a cleaner,
// a migration script, the user) and with them the protection of this
// app's models. installed are the app's catalogue models that are on
// disk; each that is installed is claimed again (OriginMigrated), and an
// empty file is written when none is, so that the others see this app
// keeps no models rather than "claims lost". Call it at launch and when
// the app becomes active; with the file present it only stats it. It
// reports whether it wrote the file.
func (s *Store) Heal(ctx context.Context, installed ...Claim) (bool, error) {
	if _, err := os.Stat(s.Family.ClaimsFile(s.BundleID)); err == nil || !os.IsNotExist(err) {
		return false, nil
	}
	var errs []error
	for _, c := range installed {
		if c.Origin == "" {
			c.Origin = OriginMigrated
		}
		if err := s.Claim(ctx, c); err != nil && !errors.Is(err, ErrNotInstalled) {
			errs = append(errs, err)
		}
	}
	if _, err := os.Stat(s.Family.ClaimsFile(s.BundleID)); os.IsNotExist(err) {
		if err := s.update(func(*Claims) bool { return true }); err != nil {
			errs = append(errs, err)
		}
	}
	return true, errors.Join(errs...)
}

// Remove deletes this app's claims file (e.g. the app wipes its data
// and its models).
func (s *Store) Remove() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	err := os.Remove(s.Family.ClaimsFile(s.BundleID))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	_ = s.Family.Post(family.TopicModels)
	return nil
}
