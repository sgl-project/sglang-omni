package credentials

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/hehehai/actnowkit/family"
)

// Kind is the schema kind of credentials.json.
const Kind = "credentials"

// Entry is a shared vendor credential's metadata in credentials.json.
// Never a secret: hint and fingerprint only.
type Entry struct {
	Account     string       `json:"account"` // "vendor.deepseek"
	Auth        string       `json:"auth"`
	AppID       string       `json:"appId,omitempty"` // Doubao speech's App ID
	Hint        string       `json:"hint"`
	Fingerprint string       `json:"fingerprint"`
	AddedBy     string       `json:"addedBy"` // bundle id
	Updated     time.Time    `json:"updated"`
	Users       []string     `json:"users"` // bundle ids that agreed to use it
	Extra       family.Extra `json:"-"`
}

func (e *Entry) UnmarshalJSON(b []byte) error {
	type plain Entry
	return family.UnmarshalKnown(b, (*plain)(e), &e.Extra)
}

func (e Entry) MarshalJSON() ([]byte, error) {
	type plain Entry
	return family.MarshalKnown(plain(e), e.Extra)
}

// File is credentials.json (multi-writer, locks/credentials.lock).
type File struct {
	Schema  string           `json:"schema"`
	Rev     int              `json:"rev"`
	Vendors map[string]Entry `json:"vendors"`
	Extra   family.Extra     `json:"-"`
}

func (f *File) UnmarshalJSON(b []byte) error {
	type plain File
	return family.UnmarshalKnown(b, (*plain)(f), &f.Extra)
}

func (f File) MarshalJSON() ([]byte, error) {
	type plain File
	return family.MarshalKnown(plain(f), f.Extra)
}

// Read reads credentials.json; readOnly when a newer major wrote it.
func Read(d family.Dir) (f File, readOnly bool, err error) {
	err = family.Load(d.CredentialsFile(), Kind, 1, &f)
	switch {
	case family.IsNotExist(err):
		return File{}, false, nil
	case errors.Is(err, family.ErrNewerSchema):
		return f, true, nil
	}
	return f, false, err
}

// Fingerprint is "sha256:" and the first 12 hex digits of the secret's
// SHA-256: enough to tell whether two apps hold the same key, nothing
// more (spec §8.2).
func Fingerprint(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return "sha256:" + hex.EncodeToString(sum[:])[:12]
}

// Hint shows a secret as the UI may: its prefix ("sk-", "hf_") and last 4
// characters, "sk-…a1b2"; short secrets show nothing.
func Hint(secret string) string {
	secret = strings.TrimSpace(secret)
	n := utf8.RuneCountInString(secret)
	if n < 12 {
		return "…"
	}
	r := []rune(secret)
	prefix := ""
	for i := 0; i < len(r) && i < 4; i++ {
		if r[i] == '-' || r[i] == '_' {
			prefix = string(r[:i+1])
			break
		}
	}
	return prefix + "…" + string(r[n-4:])
}

// Ref is the credentialRef an app keeps in its provider settings for a
// shared credential: "actnow:vendor.deepseek".
func Ref(vendor string) string { return "actnow:" + Account(vendor) }

// ParseRef returns the vendor of a credentialRef; ok false when it is not
// a shared one.
func ParseRef(ref string) (vendor string, ok bool) {
	return strings.CutPrefix(ref, "actnow:vendor.")
}

// Errors of the Store.
var (
	ErrUnknownVendor = errors.New("unknown vendor")
	// ErrNotConsented: this app is not among the vendor's users; ask the
	// user first (TextUseKeyFrom) and call Use.
	ErrNotConsented = errors.New("the user has not agreed to use this shared key in this app")
	// ErrMissing: the metadata names a key whose Keychain item is gone
	// (deleted in Keychain Access): ask for the key again.
	ErrMissing = errors.New("the shared key is no longer in the Keychain")
)

// Store is one app's access to the shared credentials.
type Store struct {
	Family   family.Dir
	BundleID string
	Keychain Keychain // Service "" = "ActNow"
}

func (s *Store) update(ctx context.Context, fn func(f *File) error) error {
	var f File
	err := s.Family.Update(ctx, family.LockCredentials, s.Family.CredentialsFile(), Kind, 1, &f, func(bool) error {
		if err := fn(&f); err != nil {
			return err
		}
		f.Schema = family.SchemaOf(Kind, 1)
		f.Rev++
		return nil
	})
	if err == nil {
		_ = s.Family.Post(family.TopicCredentials)
	}
	return err
}

// Offer returns the shared credential of a vendor, if any, for the
// consumer's prompt "Use the DeepSeek key from Auralink (sk-…a1b2)". It
// never reads the secret.
func (s *Store) Offer(vendor string) (Entry, bool) {
	f, _, err := Read(s.Family)
	if err != nil {
		return Entry{}, false
	}
	e, ok := f.Vendors[vendor]
	return e, ok
}

// Using reports whether this app agreed to use the vendor's shared key.
func (s *Store) Using(vendor string) bool {
	e, ok := s.Offer(vendor)
	return ok && slices.Contains(e.Users, s.BundleID)
}

// SaveOptions are extra fields of Save.
type SaveOptions struct {
	// AppID is Doubao speech's App ID (Auth AuthAppIDAccessToken).
	AppID string
	// Auth overrides the vendor's default auth kind.
	Auth string
}

// Save stores a key the user entered in this app with "Let other ActNow
// apps on this Mac use this key" checked (spec §8.3): the Keychain item
// "ActNow"/"vendor.<id>" is written (replacing another app's key for
// the same vendor: TextAlsoUpdates) and this app joins its users.
func (s *Store) Save(ctx context.Context, vendor, secret string, opts SaveOptions) error {
	v, ok := VendorByID(vendor)
	if !ok {
		return fmt.Errorf("%q: %w", vendor, ErrUnknownVendor)
	}
	secret = strings.TrimSpace(secret)
	if err := s.Keychain.Set(Account(vendor), v.Label(), secret); err != nil {
		return err
	}
	return s.update(ctx, func(f *File) error {
		if f.Vendors == nil {
			f.Vendors = map[string]Entry{}
		}
		e := f.Vendors[vendor]
		if e.AddedBy == "" || e.Fingerprint != Fingerprint(secret) {
			e.AddedBy = s.BundleID
		}
		e.Account, e.Auth = Account(vendor), v.Auth
		if opts.Auth != "" {
			e.Auth = opts.Auth
		}
		if opts.AppID != "" {
			e.AppID = opts.AppID
		}
		e.Hint, e.Fingerprint, e.Updated = Hint(secret), Fingerprint(secret), family.Now()
		if !slices.Contains(e.Users, s.BundleID) {
			e.Users = append(e.Users, s.BundleID)
		}
		f.Vendors[vendor] = e
		return nil
	})
}

// Use is the user clicking "Use this key": this app joins the vendor's
// users, then reads the secret. ErrMissing when the item is gone (the
// metadata is then dropped).
func (s *Store) Use(ctx context.Context, vendor string) (string, error) {
	if _, ok := s.Offer(vendor); !ok {
		return "", ErrMissing
	}
	if err := s.update(ctx, func(f *File) error {
		e, ok := f.Vendors[vendor]
		if !ok {
			return ErrMissing
		}
		if slices.Contains(e.Users, s.BundleID) {
			return family.ErrNoChange
		}
		e.Users = append(e.Users, s.BundleID)
		f.Vendors[vendor] = e
		return nil
	}); err != nil {
		return "", err
	}
	return s.Secret(ctx, vendor)
}

// Secret reads the vendor's shared key for an app that agreed to use it
// (ErrNotConsented otherwise: never read it before the user agreed, not
// even for a connection test). A missing item drops the metadata and
// returns ErrMissing.
func (s *Store) Secret(ctx context.Context, vendor string) (string, error) {
	if !s.Using(vendor) {
		return "", ErrNotConsented
	}
	secret, err := s.Keychain.Get(Account(vendor))
	if errors.Is(err, ErrNotFound) {
		_ = s.Forget(ctx, vendor)
		return "", ErrMissing
	}
	return secret, err
}

// StopUsing is "Stop using it here": this app leaves the users; the key
// stays for the others.
func (s *Store) StopUsing(ctx context.Context, vendor string) error {
	return s.update(ctx, func(f *File) error {
		e, ok := f.Vendors[vendor]
		if !ok || !slices.Contains(e.Users, s.BundleID) {
			return family.ErrNoChange
		}
		e.Users = slices.DeleteFunc(e.Users, func(u string) bool { return u == s.BundleID })
		f.Vendors[vendor] = e
		return nil
	})
}

// OtherUsers are the vendor's users besides this app, for "Auralink also
// uses this key" before deleting.
func (s *Store) OtherUsers(vendor string) []string {
	e, _ := s.Offer(vendor)
	return slices.DeleteFunc(slices.Clone(e.Users), func(u string) bool { return u == s.BundleID })
}

// DeleteForAll removes the Keychain item and its metadata.
func (s *Store) DeleteForAll(ctx context.Context, vendor string) error {
	if err := s.Keychain.Delete(Account(vendor)); err != nil {
		return err
	}
	return s.Forget(ctx, vendor)
}

// Forget drops a vendor's metadata (its item is gone).
func (s *Store) Forget(ctx context.Context, vendor string) error {
	return s.update(ctx, func(f *File) error {
		if _, ok := f.Vendors[vendor]; !ok {
			return family.ErrNoChange
		}
		delete(f.Vendors, vendor)
		return nil
	})
}

// Compare tells whether an app's own key for a vendor is the shared one
// (spec §15.2 step 6: different keys are never merged; the consumer shows
// TextDifferentKey and lets the user choose). shared is false when the
// vendor has no shared key.
func (s *Store) Compare(vendor, secret string) (shared, same bool) {
	e, ok := s.Offer(vendor)
	if !ok {
		return false, false
	}
	return true, e.Fingerprint == Fingerprint(strings.TrimSpace(secret))
}

// MigrateLegacy copies an app's old item (e.g. "AudioSmith"/"llm.deepseek")
// to the shared item after the user agreed to share it, reads it back to
// check, and joins the users. The old item is kept until the app's next
// major version (spec §15.2 step 6).
func (s *Store) MigrateLegacy(ctx context.Context, vendor string, old Legacy, opts SaveOptions) error {
	legacy := Keychain{Service: old.Service, File: s.Keychain.File, Run: s.Keychain.Run}
	secret, err := legacy.Get(old.Account)
	if err != nil {
		return err
	}
	if err := s.Save(ctx, vendor, secret, opts); err != nil {
		return err
	}
	got, err := s.Keychain.Get(Account(vendor))
	if err != nil {
		return err
	}
	if got != strings.TrimSpace(secret) {
		return errors.New("credentials: the shared item does not read back as written")
	}
	return nil
}

// Reference texts (spec §8.3).
var (
	TextShareCheckbox = family.Text{EN: "Let other ActNow apps on this Mac use this key", ZH: "允许这台 Mac 上的其他 ActNow app 使用此密钥"}
	TextUseKeyFrom    = family.Text{EN: "Use the {vendor} key from {app} ({hint})", ZH: "使用 {app} 中的 {vendor} 密钥（{hint}）"}
	TextUseThisKey    = family.Text{EN: "Use this key", ZH: "使用此密钥"}
	TextEnterAnother  = family.Text{EN: "Enter another key", ZH: "输入其他密钥"}
	TextFoundKeys     = family.Text{
		EN: "Found API keys in other ActNow apps: {vendors}. Use them in {app}?",
		ZH: "在其他 ActNow app 中找到 API 密钥：{vendors}。要在 {app} 中使用吗？",
	}
	TextUseThem      = family.Text{EN: "Use them", ZH: "使用"}
	TextNotNow       = family.Text{EN: "Not now", ZH: "暂不"}
	TextAlsoUpdates  = family.Text{EN: "Also updates it in {apps}.", ZH: "{apps} 中也会一并更新。"}
	TextAlsoUsesKey  = family.Text{EN: "{apps} also uses this key.", ZH: "{apps} 也在使用此密钥。"}
	TextStopHere     = family.Text{EN: "Stop using it here", ZH: "仅在此处停用"}
	TextDeleteAll    = family.Text{EN: "Delete for all apps", ZH: "为所有 app 删除"}
	TextDifferentKey = family.Text{EN: "{app} has a different {vendor} key.", ZH: "{app} 中的 {vendor} 密钥与此不同。"}
	TextSecurityHelp = family.Text{
		EN: "Shared keys are stored in your login keychain. Like each app's own keys, other programs you run can read them with the security tool.",
		ZH: "共享密钥保存在登录钥匙串中。与各 app 自己保存的密钥一样，你运行的其他程序可以通过 security 工具读取它们。",
	}
)
