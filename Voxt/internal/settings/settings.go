// Package settings is the user's preferences: one typed, versioned
// document (Settings), stored as JSON in the store's documents table under
// Key, read from memory by any goroutine, changed through Service.Update
// and announced with Changed on the core.Bus.
//
// Version 1 is VoxtNew's first: it owes nothing to Voxt's UserDefaults
// keys and has no migration code yet. A field added later reads as its
// zero value in an older document until Migrate fills it in, so zero
// values must be valid.
//
// Not settings: the installed models (internal/local); app enhancement
// rules are a document of their own ("appctx:rules");
// window geometry stays with MyGo (WindowOptions.StateKey).
//
// Imports: nothing internal (Changed satisfies core.Event structurally).
package settings

import (
	"context"
	"encoding/json"
	"strings"
	"time"
)

// Version is the schema version Defaults writes. Bump it when a field's
// meaning changes, and migrate older documents in Migrate.
const Version = 4

// Settings is the whole document.
type Settings struct {
	Version        int            `json:"version"`
	General        General        `json:"general"`
	Appearance     Appearance     `json:"appearance"`
	Privacy        Privacy        `json:"privacy"`
	Audio          Audio          `json:"audio"`
	Overlay        Overlay        `json:"overlay"`
	Features       Features       `json:"features"`       // features.go
	AppEnhancement AppEnhancement `json:"appEnhancement"` // features.go
	Dictionary     Dictionary     `json:"dictionary"`
	Setup          Setup          `json:"setup"`
	Updates        Updates        `json:"updates,omitzero"`
	Network        Network        `json:"network,omitzero"`
	Family         Family         `json:"family,omitzero"`
	Models         Models         `json:"models,omitzero"`    // models.go
	History        History        `json:"history,omitzero"`   // history.go
	Providers      Providers      `json:"providers,omitzero"` // providers.go
}

// General: languages and how the app stays around.
type General struct {
	// Language is the UI language as a BCP-47 tag ("en", "zh-Hans"), or ""
	// to follow the system (its first preferred language, systemLanguage);
	// see i18n.Match.
	Language string `json:"language"`
	// MainLanguage is the language the user mostly speaks and writes (a
	// lang code, "zh"); OtherLanguages the ones they mix in. The prompt
	// composer fills {{USER_MAIN_LANGUAGE}} and {{USER_OTHER_LANGUAGES}}
	// with them, and translation defaults its target from them.
	// "" = from the UI language.
	MainLanguage   string   `json:"mainLanguage,omitempty"`
	OtherLanguages []string `json:"otherLanguages,omitempty"`
	// LaunchAtLogin opens VoxtNew when the user logs in.
	LaunchAtLogin bool `json:"launchAtLogin,omitempty"`
	// MenuBar shows the menu bar icon; CloseToTray keeps VoxtNew running
	// there when its window closes, so the shortcuts keep working.
	MenuBar     bool `json:"menuBar"`
	CloseToTray bool `json:"closeToTray"`
	// HideDock leaves VoxtNew out of the Dock (Voxt 1.x's "Show in Dock"
	// off); the menu bar icon is then on, so that VoxtNew can be reached.
	HideDock bool `json:"hideDock,omitempty"`
	// CopyResult leaves each finished result on the clipboard: a paste
	// does not restore what was there, and a result the answer card shows
	// is copied too (Voxt 1.x's "Also copy result to clipboard").
	CopyResult bool `json:"copyResult,omitempty"`
}

// Appearance: theme, scale and the sidebar.
type Appearance struct {
	Theme        string  `json:"theme"`        // "system" | "light" | "dark"
	Scale        float64 `json:"scale"`        // UI scale, 0.9–1.25 (1 = 100 %)
	ReduceMotion bool    `json:"reduceMotion"` // in addition to the system setting
	// Rail shows the sidebar as icons only. Without it the sidebar is
	// automatic (icons only in a narrow window) unless RailChosen: the
	// user chose names and icons. SidebarWidth is its width in DIPs at
	// 100 % (0 = the default); ClosedSections the sidebar sections the
	// user closed.
	Rail           bool     `json:"rail,omitempty"`
	RailChosen     bool     `json:"railChosen,omitempty"`
	SidebarWidth   float64  `json:"sidebarWidth,omitempty"`
	ClosedSections []string `json:"closedSections,omitempty"`
}

// Privacy: what VoxtNew keeps on this Mac and what it logs.
type Privacy struct {
	// KeepHistory saves each result in the history; KeepAudio also saves
	// its recording.
	KeepHistory bool `json:"keepHistory"`
	KeepAudio   bool `json:"keepAudio"`
	// LogText writes recognised and generated text to the log; off, the
	// log has only lengths and timings.
	LogText bool `json:"logText,omitempty"`
}

// Audio: the microphone and the sounds around a recording.
type Audio struct {
	// Devices are capture device ids in the user's order of preference;
	// the first one present records, else the system's default input.
	// DeviceNames are their names when they were last seen, to show the
	// ones that are not connected.
	Devices     []string          `json:"devices,omitempty"`
	DeviceNames map[string]string `json:"deviceNames,omitempty"`
	// MuteOutput mutes the default output device while recording (after
	// the start cue, until the recording stops), unmuting only what
	// VoxtNew muted (internal/outmute, Auralink's, 10-17). Nil until the
	// user chooses: on (MuteOn), for older documents too, without a
	// migration (no version wrote it: the P1 placeholder's false was
	// omitted). Sounds plays a cue when recording starts and another when
	// it stops. StartSound and StopSound are those cues, as choices of
	// actnowkit's sounds package: "" the default (one of the ActNow
	// family's sounds), "none", "system:<name>" for a macOS system sound,
	// or a family sound's id (10-20). SoundVolume is their volume in
	// percent (0: DefaultSoundVolume; negative: muted). SoundPreset was the
	// pair before, by Voxt's preset IDs (10-17): Migrate turns it into StartSound and
	// StopSound and never writes it again.
	MuteOutput  *bool  `json:"muteOutput,omitempty"`
	Sounds      bool   `json:"sounds"`
	StartSound  string `json:"startSound,omitempty"`
	StopSound   string `json:"stopSound,omitempty"`
	SoundVolume int    `json:"soundVolume,omitempty"`
	SoundPreset string `json:"soundPreset,omitempty"`
	// VAD is the local voice activity detection a recording goes
	// through before a speech model hears it (10-15).
	VAD VAD `json:"vad,omitzero"`
	// Clean is the light sound processing of the speech model's copy
	// of a recording (10-16).
	Clean Clean `json:"clean,omitzero"`
}

// MuteOn reports whether recordings mute the system's output: the user's
// choice, else on.
func (a Audio) MuteOn() bool { return a.MuteOutput == nil || *a.MuteOutput }

// Clean tunes the sound processing of what a speech model gets of a
// recording (internal/dsp, pure DSP, no model): a high-pass filter (80
// Hz) takes out hum and rumble, and the level stage raises a quiet voice
// (measured on the speech only), lowers a very loud one a little, and
// limits the peaks so nothing clips. The recording kept in the history is
// the original. On unless Off, so older documents read as on without a
// migration. Zero fields take dsp.Defaults; there is no UI for them but
// Off (Settings → Audio → Optimize sound).
type Clean struct {
	// Off sends the recording as it was captured.
	Off bool `json:"off,omitempty"`
	// HighPassOff and LevelOff turn one stage off (troubleshooting).
	HighPassOff bool `json:"highPassOff,omitempty"`
	LevelOff    bool `json:"levelOff,omitempty"`
	// HighPassHz is the filter's corner.
	HighPassHz float64 `json:"highPassHz,omitempty"`
	// TargetDB is the speech level (dBFS) a quiet voice is raised to, by
	// at most MaxGainDB; speech over LoudDB is lowered to it, by at most
	// MaxCutDB; the noise floor is never raised over NoiseCeilDB; the
	// limiter keeps every sample under CeilingDB.
	TargetDB    float64 `json:"targetDB,omitempty"`
	MaxGainDB   float64 `json:"maxGainDB,omitempty"`
	LoudDB      float64 `json:"loudDB,omitempty"`
	MaxCutDB    float64 `json:"maxCutDB,omitempty"`
	NoiseCeilDB float64 `json:"noiseCeilDB,omitempty"`
	CeilingDB   float64 `json:"ceilingDB,omitempty"`
}

// VAD tunes the local voice activity detection (internal/vad, energy
// over an adaptive noise floor, no model): a whole recording gets to the
// speech model without its leading and trailing silence and with long
// pauses shortened, a live stream gets nothing before the speech
// starts, and a recording without speech is not sent at all. Zero fields
// take vad.Defaults (conservative: rather send a little silence than cut
// a word); there is no UI for them but Off (Settings → Audio).
type VAD struct {
	// Off sends every recording whole.
	Off bool `json:"off,omitempty"`
	// Thresholds above the noise floor (dB) to start and to stay in
	// speech, and the level (dBFS) under which nothing is speech.
	OnsetDB    float64 `json:"onsetDB,omitempty"`
	OffsetDB   float64 `json:"offsetDB,omitempty"`
	MinLevelDB float64 `json:"minLevelDB,omitempty"`
	// HangoverMS of quiet end speech; LeadMS and TailMS are kept around
	// it; a pause over LongGapMS is shortened to GapKeepMS.
	HangoverMS int `json:"hangoverMS,omitempty"`
	LeadMS     int `json:"leadMS,omitempty"`
	TailMS     int `json:"tailMS,omitempty"`
	LongGapMS  int `json:"longGapMS,omitempty"`
	GapKeepMS  int `json:"gapKeepMS,omitempty"`
	// A stream gets PreRollMS from before the speech, and starts anyway
	// after MaxHoldMS without it (a stream is never held back for long).
	PreRollMS int `json:"preRollMS,omitempty"`
	MaxHoldMS int `json:"maxHoldMS,omitempty"`
}

// Overlay: the recording overlay's place and look.
type Overlay struct {
	Position string  `json:"position"` // "bottom" | "top"
	Opacity  float64 `json:"opacity"`  // 0.5–1
	Radius   float64 `json:"radius"`   // DIPs
	Margin   float64 `json:"margin"`   // DIPs from the screen's edge
	// ShowTime shows the recording's time in the pill (off: Voxt's pill,
	// an icon and the waveform).
	ShowTime bool `json:"showTime,omitempty"`
	// HideLiveText leaves the words out of the pill: what was heard while
	// recording, and the answer as the model writes it (Voxt 1.x's
	// "Show Realtime Text" off).
	HideLiveText bool `json:"hideLiveText,omitempty"`
}

// DefaultOpacity and DefaultRadius are the overlay face's: translucent
// over the blurred material behind it, its corners round (version 3,
// near Voxt's 82 % and 24; version 2's were 0.72 and 16, version 1's 1
// and 16).
const (
	DefaultOpacity = 0.8
	DefaultRadius  = 24
)

// Hotkey is one binding of a feature: the keys ("RightOption",
// "RightOption+Shift", "Control+Option+V") and how they trigger (internal/hotkey owns the
// grammar).
type Hotkey struct {
	Keys    string `json:"keys"`
	Trigger string `json:"trigger,omitempty"` // TriggerTap (default) | TriggerDouble | TriggerLong
}

// Hotkey triggers.
const (
	TriggerTap    = "tap"
	TriggerDouble = "double"
	TriggerLong   = "long"
)

// Dictionary: the user's terms. The terms are a table of the store
// (dictionary_terms); these say where they go.
type Dictionary struct {
	Enabled bool `json:"enabled"`
	// ASRHints sends the terms to the speech model as hints; Glossary
	// adds them to the LLM prompt.
	ASRHints bool `json:"asrHints"`
	Glossary bool `json:"glossary"`
	// SpeechContext is free text about the speech (who speaks, about
	// what) that the speech model gets before the hints, with the
	// dictionary on or off; a service that takes no prompt ignores it.
	SpeechContext string `json:"speechContext,omitempty"`
	// Learn watches the field a Transcribe result was pasted into for
	// 30 s and proposes a term when the user corrects a word of it
	// (internal/learn). Off by default.
	Learn bool `json:"learn,omitempty"`
}

// Setup: the first-run guide.
type Setup struct {
	Done bool `json:"done,omitempty"`
	// Tried: a session (a try or a real one, any feature) has given a
	// result once, or the history had one: Home stops offering its Try
	// area. It stays when the history is cleared.
	Tried bool `json:"tried,omitempty"`
}

// Updates: what the updater offers (Settings → About → Updates). Its
// other preferences (automatic checks and installs) are the updater
// plugin's, in its own updater.json.
type Updates struct {
	// Beta offers beta versions (the feed's beta channel) too. Nil until the
	// user chooses: then on while the running version is a beta (BetaOn),
	// so that a beta's users get the next one. No migration: older
	// documents have none.
	Beta *bool `json:"beta,omitempty"`
}

// BetaOn reports whether updates include betas for the app of version
// running: the user's choice, else whether version is a pre-release
// (0.1.0-beta.1).
func (u Updates) BetaOn(running string) bool {
	if u.Beta != nil {
		return *u.Beta
	}
	return strings.Contains(running, "-")
}

// The default shortcuts: the ActNow family's assignment to VoxtNew
// (actnowkit shortcuts.DefaultsOf("voxtnew"), family spec §10.2), so
// that VoxtNew and Auralink (fn) can run together. Right Option tapped
// alone transcribes, with Shift translates, with Control rewrites;
// Control+Option+V pastes the last result.
const (
	DefaultTranscribe = "RightOption"
	DefaultTranslate  = "RightOption+Shift"
	DefaultRewrite    = "Control+RightOption"
	DefaultPasteLast  = "Control+Option+V"
	// DefaultCancel is the cancel shortcut's default, in one place: it
	// only acts while a session records or the answer card shows.
	DefaultCancel = "Escape"
	// DefaultNote is the note shortcut's default, Voxt 1.x's: right
	// Command tapped alone, which no ActNow app has.
	DefaultNote = "RightCommand"
)

// Defaults returns the settings of a fresh install.
func Defaults() Settings {
	return Settings{
		Version:    Version,
		General:    General{MenuBar: true, CloseToTray: true},
		Appearance: Appearance{Theme: "system", Scale: 1},
		Privacy:    Privacy{KeepHistory: true},
		Audio:      Audio{Sounds: true},
		Overlay:    Overlay{Position: "bottom", Opacity: DefaultOpacity, Radius: DefaultRadius, Margin: 96},
		Features: Features{
			Transcribe: Transcribe{Feature: Feature{Hotkeys: []Hotkey{{Keys: DefaultTranscribe, Trigger: TriggerTap}}}, Enhance: true, Cleanup: true},
			Translate:  Translate{Feature: Feature{Hotkeys: []Hotkey{{Keys: DefaultTranslate, Trigger: TriggerTap}}}, Cleanup: true},
			Rewrite:    Rewrite{Feature: Feature{Hotkeys: []Hotkey{{Keys: DefaultRewrite, Trigger: TriggerTap}}}, Cleanup: true},
			PasteLast:  Hotkey{Keys: DefaultPasteLast, Trigger: TriggerTap},
			Cancel:     Hotkey{Keys: DefaultCancel, Trigger: TriggerTap},
			Note:       Hotkey{Keys: DefaultNote, Trigger: TriggerTap},
		},
		AppEnhancement: AppEnhancement{Enabled: true},
		Dictionary:     Dictionary{Enabled: true, ASRHints: true, Glossary: true},
	}
}

// Migrate reads a stored document, fills what it lacks with the defaults
// and brings it to Version. Unknown fields are dropped.
func Migrate(data []byte) (Settings, error) {
	s := Defaults()
	if len(data) == 0 {
		return s, nil
	}
	if err := json.Unmarshal(data, &s); err != nil {
		return Defaults(), err
	}
	// An older version's default, which nobody could tell from a choice,
	// becomes this one's; a value the user chose stays.
	if s.Version < 2 && s.Overlay.Opacity == 1 || s.Version < 3 && s.Overlay.Opacity == 0.72 {
		// Version 2: the overlay is translucent over a blur (1 was version
		// 1's default); version 3: 0.8 (version 2's was 0.72).
		s.Overlay.Opacity = DefaultOpacity
	}
	if s.Version < 3 && s.Overlay.Radius == 16 {
		s.Overlay.Radius = DefaultRadius // version 3: rounder, as Voxt's
	}
	if s.Version < 4 {
		migrateShortcuts(&s.Features)
	}
	migrateNoteKeys(data, &s.Features)
	migrateShowCard(data, &s.Features)
	migrateSoundPreset(&s.Audio)
	s.Version = Version
	return normalize(s), nil
}

// DefaultSoundVolume is the cues' volume until the user sets one.
const DefaultSoundVolume = 70

// SoundVolumeOrDefault is the cues' volume in percent: SoundVolume, or
// DefaultSoundVolume when unset (0); a negative one is muted.
func (a Audio) SoundVolumeOrDefault() int {
	switch {
	case a.SoundVolume == 0:
		return DefaultSoundVolume
	case a.SoundVolume < 0:
		return 0
	}
	return min(a.SoundVolume, 100)
}

// SetSoundVolume keeps v (percent) as the cues' volume; 0 is stored as
// muted (-1), as 0 is unset.
func (a *Audio) SetSoundVolume(v int) {
	if v <= 0 {
		a.SoundVolume = -1
		return
	}
	a.SoundVolume = min(v, 100)
}

// presetSounds are the system sounds of Voxt's presets (macx.Presets
// before 10-20), start then stop: what migrateSoundPreset keeps of a
// chosen preset. Soft (Pop, Tink), the default, becomes the family's
// default sounds.
var presetSounds = map[string][2]string{
	"glass": {"Ping", "Ping"}, "funk": {"Morse", "Morse"}, "submarine": {"Submarine", "Submarine"},
	"basso": {"Basso", "Basso"}, "bottle": {"Bottle", "Bottle"}, "frog": {"Frog", "Frog"},
	"hero": {"Hero", "Hero"}, "purr": {"Purr", "Purr"}, "sosumi": {"Sosumi", "Sosumi"},
}

// migrateSoundPreset turns a Voxt preset into the start and stop cues it
// played, as macOS system sounds; Soft and unknown IDs into the defaults.
func migrateSoundPreset(a *Audio) {
	if p, ok := presetSounds[a.SoundPreset]; ok && a.StartSound == "" && a.StopSound == "" {
		a.StartSound, a.StopSound = "system:"+p[0], "system:"+p[1]
	}
	a.SoundPreset = ""
}

// migrateShortcuts moves version 3's default shortcuts, Auralink's (fn,
// fn+Shift, fn+Control, Control+Command+V), to the family's (version 4):
// each one that is still exactly its default; a shortcut the user
// changed, added or removed stays. It is history, kept apart from
// ApplyPreset: a preset added later must not take a version 3 user's keys.
func migrateShortcuts(f *Features) {
	old := func(hs []Hotkey, keys string) bool {
		return len(hs) == 1 && oldDefault(hs[0], keys)
	}
	if old(f.Transcribe.Hotkeys, "Fn") {
		f.Transcribe.Hotkeys = []Hotkey{{Keys: DefaultTranscribe, Trigger: TriggerTap}}
	}
	if old(f.Translate.Hotkeys, "Fn+Shift") {
		f.Translate.Hotkeys = []Hotkey{{Keys: DefaultTranslate, Trigger: TriggerTap}}
	}
	if old(f.Rewrite.Hotkeys, "Fn+Control") {
		f.Rewrite.Hotkeys = []Hotkey{{Keys: DefaultRewrite, Trigger: TriggerTap}}
	}
	if oldDefault(f.PasteLast, "Control+Command+V") {
		f.PasteLast = Hotkey{Keys: DefaultPasteLast, Trigger: TriggerTap}
	}
}

// oldDefault reports whether h is keys tapped, as version 3 wrote it.
func oldDefault(h Hotkey, keys string) bool {
	return h.Keys == keys && (h.Trigger == "" || h.Trigger == TriggerTap)
}

// migrateNoteKeys gives a document from before the note shortcut the
// default, unless another shortcut already taps those keys: the note then
// stays off, and the Notes page asks for a shortcut. Such a document has
// no note key; one that has it is the user's.
func migrateNoteKeys(data []byte, f *Features) {
	var doc struct {
		Features struct {
			Note json.RawMessage `json:"note"`
		} `json:"features"`
	}
	if json.Unmarshal(data, &doc) != nil || doc.Features.Note != nil {
		return
	}
	if tapsKeys(*f, DefaultNote) {
		f.Note = Hotkey{}
	}
}

// tapsKeys reports whether a shortcut in f taps keys by itself, as
// hotkey.Check counts a clash.
func tapsKeys(f Features, keys string) bool {
	all := append([]Hotkey{f.PasteLast, f.Cancel}, f.Transcribe.Hotkeys...)
	all = append(append(all, f.Translate.Hotkeys...), f.Rewrite.Hotkeys...)
	for _, h := range all {
		if strings.TrimSpace(h.Keys) == keys && (h.Trigger == "" || h.Trigger == TriggerTap) {
			return true
		}
	}
	return false
}

// Changed is published after an update is saved. Topic "settings.changed".
type Changed struct{ Old, New Settings }

func (Changed) Topic() string { return "settings.changed" }

// Service holds the current settings. Goroutine-safe.
type Service interface {
	// Get returns the current settings: a copy, from memory (no I/O), so
	// views may call it while building.
	Get() Settings
	// Update applies fn to a copy, saves it, swaps it in and publishes
	// Changed; a no-op change publishes nothing. fn must not block.
	Update(ctx context.Context, fn func(*Settings)) error
}

// Network: how model downloads reach the Hub, VoxtNew's own values (used
// while it does not follow the ActNow family's, Family.Follow). Proxy is
// "" (automatic: the environment's, else the Mac's), "direct" or a proxy
// URL; HFEndpoints "" or "auto" (the Hub and hf-mirror.com raced),
// "official", "mirror", or URLs (family spec §9.1).
type Network struct {
	Proxy       string `json:"proxy,omitempty"`
	HFEndpoints string `json:"hfEndpoints,omitempty"`
	// ProxyUser is a custom proxy's user name and ProxyAuthHost the proxy
	// host it and its password (the Keychain's "proxy.<host>") are for:
	// Voxt's only, never the family's (models_proxy_auth.go).
	ProxyUser     string `json:"proxyUser,omitempty"`
	ProxyAuthHost string `json:"proxyAuthHost,omitempty"`
}

// Family is VoxtNew's part in the ActNow family (internal/actnow).
type Family struct {
	// Follow says, by shared key ("general.language", "network.proxy",
	// "network.hfEndpoints", "models.root"), whether VoxtNew uses the
	// family's value; a key that is missing follows (family spec §9.3).
	Follow map[string]bool `json:"follow,omitempty"`
	// Migrated: the first start with the family's protocol is done (the
	// models on disk claimed, the shared settings merged).
	Migrated bool `json:"migrated,omitempty"`
	// Conflicts are the shared keys the family had another value of at the
	// migration: their row says so once, until the user acts on it.
	Conflicts []string `json:"conflicts,omitempty"`
	// Prompted are the shortcut conflicts whose prompt was shown
	// (actnowkit shortcuts.Conflict.Key): each pair once.
	Prompted []string `json:"prompted,omitempty"`
	// KeysSince is when each declared shortcut got its keys (by the
	// manifest's shortcut id): the later of two conflicting apps prompts.
	KeysSince map[string]KeysSince `json:"keysSince,omitempty"`
}

// KeysSince is a shortcut's keys and when they were set.
type KeysSince struct {
	Keys  string    `json:"keys"`
	Since time.Time `json:"since"`
}

// Follows reports whether key follows the family.
func (f Family) Follows(key string) bool {
	v, ok := f.Follow[key]
	return !ok || v
}
