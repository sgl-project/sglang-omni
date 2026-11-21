package models

import (
	"context"
	"errors"
	"slices"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// KindLicences is the schema kind of models/licences.json.
const KindLicences = "models.licences"

// Acceptance is the user accepting a model's licence in one app
// (spec §7.9).
type Acceptance struct {
	Repo    string       `json:"repo"`
	Licence string       `json:"licence"` // SPDX-like: "CC-BY-NC-4.0"
	By      string       `json:"by"`      // bundle id
	At      time.Time    `json:"at"`
	Extra   family.Extra `json:"-"`
}

func (a *Acceptance) UnmarshalJSON(b []byte) error {
	type plain Acceptance
	return family.UnmarshalKnown(b, (*plain)(a), &a.Extra)
}

func (a Acceptance) MarshalJSON() ([]byte, error) {
	type plain Acceptance
	return family.MarshalKnown(plain(a), a.Extra)
}

// Licences is models/licences.json (multi-writer, locks/licences.lock).
type Licences struct {
	Schema   string       `json:"schema"`
	Accepted []Acceptance `json:"accepted"`
	Extra    family.Extra `json:"-"`
}

func (l *Licences) UnmarshalJSON(b []byte) error {
	type plain Licences
	return family.UnmarshalKnown(b, (*plain)(l), &l.Extra)
}

func (l Licences) MarshalJSON() ([]byte, error) {
	type plain Licences
	return family.MarshalKnown(plain(l), l.Extra)
}

// ReadLicences reads every acceptance.
func ReadLicences(d family.Dir) ([]Acceptance, error) {
	var l Licences
	err := family.Load(d.LicencesFile(), KindLicences, 1, &l)
	if family.IsNotExist(err) || errors.Is(err, family.ErrNewerSchema) {
		err = nil
	}
	return l.Accepted, err
}

// LicenceAccepted finds an acceptance of repo's licence (any app). The
// app still shows the licence, worded TextLicenceAcceptedIn, with a
// "Continue" button.
func LicenceAccepted(d family.Dir, repo, licence string) (Acceptance, bool) {
	all, _ := ReadLicences(d)
	for i := len(all) - 1; i >= 0; i-- {
		a := all[i]
		if a.Repo == repo && strings.EqualFold(a.Licence, licence) {
			return a, true
		}
	}
	return Acceptance{}, false
}

// AcceptLicence records that the user accepted repo's licence in the app
// bundleID. An acceptance of the same repo and licence is replaced.
func AcceptLicence(ctx context.Context, d family.Dir, repo, licence, bundleID string) error {
	var l Licences
	err := d.Update(ctx, family.LockLicences, d.LicencesFile(), KindLicences, 1, &l, func(bool) error {
		l.Schema = family.SchemaOf(KindLicences, 1)
		l.Accepted = slices.DeleteFunc(l.Accepted, func(a Acceptance) bool {
			return a.Repo == repo && strings.EqualFold(a.Licence, licence)
		})
		l.Accepted = append(l.Accepted, Acceptance{Repo: repo, Licence: licence, By: bundleID, At: family.Now()})
		return nil
	})
	if err == nil {
		_ = d.Post(family.TopicModels)
	}
	return err
}
