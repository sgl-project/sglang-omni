package family

import (
	"errors"
	"strings"
)

// App is a known family app (spec §6.1). The table lets an app find the
// others before they ever ran (installed, no manifest yet).
type App struct {
	ID         string // "audiosmith"
	Name       string // "AudioSmith"
	BundleID   string // release bundle id: "dev.hehehai.audiosmith"
	URLScheme  string // "audiosmith"
	ProductURL string // https://actnow.dev/<id>; "" while unreleased
	// TeamID is the Developer ID team the app is signed with (spec §17
	// Q1). Phase 3 features (Keychain access groups, App Groups, a shared
	// daemon) need one team and exclude apps of another.
	TeamID string
}

// Teams of the family (spec §17 Q1).
const (
	TeamActNow  = "779M533DL5" // AudioSmith, Auralink, Dicta, Pik, HereNow, CaptionCraft, Paralign, Act
	TeamVoxtNew = "RY2TQTYSKQ" // VoxtNew
)

// BundlePrefix starts every family bundle id. Bundle ids never change:
// macOS keeps privacy permissions (microphone, accessibility) by team and
// bundle id.
const BundlePrefix = "dev.hehehai."

// Apps is the built-in table, as the apps' mygo.json say (identifier,
// URL scheme).
var Apps = []App{
	{ID: "audiosmith", Name: "AudioSmith", BundleID: "dev.hehehai.audiosmith", URLScheme: "audiosmith", ProductURL: "https://actnow.dev/audiosmith", TeamID: TeamActNow},
	{ID: "dicta", Name: "Dicta", BundleID: "dev.hehehai.dicta", URLScheme: "dicta", ProductURL: "https://actnow.dev/dicta", TeamID: TeamActNow},
	{ID: "auralink", Name: "Auralink", BundleID: "dev.hehehai.auralink", URLScheme: "auralink", ProductURL: "https://actnow.dev/auralink", TeamID: TeamActNow},
	{ID: "voxtnew", Name: "VoxtNew", BundleID: "dev.hehehai.voxtnew", URLScheme: "voxtnew", ProductURL: "", TeamID: TeamVoxtNew},
	{ID: "pik", Name: "Pik", BundleID: "dev.hehehai.pik", URLScheme: "pik", ProductURL: "https://actnow.dev/pik", TeamID: TeamActNow},
	{ID: "herenow", Name: "HereNow", BundleID: "dev.hehehai.herenow", URLScheme: "herenow", ProductURL: "https://actnow.dev/herenow", TeamID: TeamActNow},
	{ID: "captioncraft", Name: "CaptionCraft", BundleID: "dev.hehehai.captioncraft", URLScheme: "captioncraft", ProductURL: "https://actnow.dev/captioncraft", TeamID: TeamActNow},
	{ID: "paralign", Name: "Paralign", BundleID: "dev.hehehai.paralign", URLScheme: "paralign", ProductURL: "https://actnow.dev/paralign", TeamID: TeamActNow},
	{ID: "act", Name: "Act", BundleID: "dev.hehehai.act", URLScheme: "act", ProductURL: "https://actnow.dev/act", TeamID: TeamActNow},
}

// AppByID finds an app of the table by id ("dicta").
func AppByID(id string) (App, bool) {
	for _, a := range Apps {
		if a.ID == id {
			return a, true
		}
	}
	return App{}, false
}

// Variant is which build of an app a bundle id is (spec §1, §6.1).
type Variant string

const (
	Release Variant = "release" // the shipped app (the beta channel too)
	Dev     Variant = "dev"     // `mygo dev`: <release id>.dev
	Test    Variant = "test"    // any other suffix: .updtest, .betatest, .demo, .m7…
)

// Errors of ParseBundleID.
var (
	ErrNotFamily = errors.New("not a family bundle id")
	// ErrComponent: a helper or engine of an app (….helper, ….engine);
	// they are not apps and write no manifest.
	ErrComponent = errors.New("an app's helper or engine, not an app")
)

// componentSuffixes end the bundle ids of apps' subprocesses.
var componentSuffixes = []string{".helper", ".engine"}

// ParseBundleID finds the family app and variant of a bundle id:
// "dev.hehehai.dicta" → dicta, Release; "dev.hehehai.dicta.dev" → Dev;
// "dev.hehehai.pik.m7" → Test. Bundle ids are compared case-insensitively,
// as LaunchServices does.
func ParseBundleID(bundleID string) (App, Variant, error) {
	lower := strings.ToLower(bundleID)
	for _, a := range Apps {
		rel := strings.ToLower(a.BundleID)
		if lower == rel {
			return a, Release, nil
		}
		rest, ok := strings.CutPrefix(lower, rel+".")
		if !ok || rest == "" {
			continue
		}
		for _, s := range componentSuffixes {
			if strings.HasSuffix("."+rest, s) {
				return a, "", ErrComponent
			}
		}
		if rest == "dev" {
			return a, Dev, nil
		}
		return a, Test, nil
	}
	return App{}, "", ErrNotFamily
}
