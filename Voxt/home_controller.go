package main

import (
	"context"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// homeRecent is how many recent results Home lists.
const homeRecent = 5

type homeState struct {
	hm struct {
		// read: the history was read once; loading a read is on its way
		// (seq numbers them: only the last lands); stale an entry was kept
		// since.
		read, loading, stale bool
		seq                  int
		today                dayStats
		recent               []store.Entry
		err                  string
		// gen numbers the landed reads (today/recent/err change); view is
		// the last built state, viewKey its inputs (homePageModel).
		gen     int
		view    homeViewState
		viewKey homeViewCacheKey
	}
	homeUI homeLocalUI
}

// homeViewCacheKey is every input of a homeViewState: when it matches the
// last build's, homePageModel reuses the state instead of rebuilding it
// every frame.
type homeViewCacheKey struct {
	settings     uint64
	gen          int
	read         bool
	loading      bool
	err          string
	readiness    homeReadinessKey
	retryID      string
	retryStage   string
	showTry      bool
	tryRecording bool
	minute       int64
	location     string
	zoneOffset   int
	locale       string
}

// homeReadinessKey is the comparable part of homeReadinessInput: the
// derived scalars buildHomeReadinessList reads, not the values they came
// from.
type homeReadinessKey struct {
	speechID, speechName            string
	speechReady, speechExplicit     bool
	speechDownloading, speechOnDisk bool
	speechDone, speechTotal         int64
	speechDownloadLeft              string
	languageName                    string
	languageReady                   bool
	permissionsChecked              bool
	microphone                      permission.State
	accessibility                   bool
	hotkeysInstalled                bool
	helperState                     helper.State
	helperReason                    helper.FailReason
}

func homeReadinessKeyFor(input homeReadinessInput) homeReadinessKey {
	return homeReadinessKey{
		speechID: input.SpeechModelID, speechName: input.SpeechModelName,
		speechReady: input.SpeechReady, speechExplicit: input.SpeechExplicit,
		speechDownloading: input.SpeechStatus.Downloading, speechOnDisk: input.SpeechStatus.OnDisk,
		speechDone: input.SpeechStatus.Done, speechTotal: input.SpeechStatus.Total,
		speechDownloadLeft: input.SpeechDownloadLeft,
		languageName:       input.LanguageModelName, languageReady: input.LanguageModelReady,
		permissionsChecked: input.PermissionsChecked, microphone: input.Microphone,
		accessibility: input.Accessibility, hotkeysInstalled: input.HotkeysInstalled,
		helperState: input.Helper.State, helperReason: input.Helper.Reason,
	}
}

// readHome reads the recent results and today's numbers (since midnight
// at now) from the history: entries come newest first, so it reads pages
// until one is older than today.
func readHome(ctx context.Context, h store.History, now time.Time) (dayStats, []store.Entry, error) {
	y, m, d := now.Date()
	midnight := time.Date(y, m, d, 0, 0, 0, 0, now.Location())
	st := dayStats{byKind: map[store.Kind]int{}}
	var recent []store.Entry
	const page = 200
	for off := 0; ; off += page {
		es, err := h.List(ctx, store.HistoryQuery{Limit: page, Offset: off})
		if err != nil {
			return st, recent, err
		}
		older := false
		for _, e := range es {
			done := e.Status == store.StatusDone && strings.TrimSpace(e.Result) != ""
			if done && len(recent) < homeRecent {
				recent = append(recent, e)
			}
			if e.CreatedAt.Before(midnight) {
				older = true
				continue
			}
			if done {
				st.count++
				st.chars += countChars(e.Result)
				st.byKind[e.Kind]++
			}
		}
		if len(es) < page || (older && len(recent) >= homeRecent) {
			return st, recent, nil
		}
	}
}

// loadHome reads Home's numbers and recent results in the background.
func (a *app) loadHome() {
	hm := &a.hm
	hm.seq++
	hm.loading, hm.stale = true, false
	seq := hm.seq
	h := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		st, recent, err := readHome(ctx, h, time.Now())
		a.post(func() {
			if seq != a.hm.seq {
				return
			}
			hm := &a.hm
			hm.gen++
			hm.loading, hm.read = false, true
			if err != nil {
				hm.err = err.Error()
				return
			}
			hm.today, hm.recent, hm.err = st, recent, ""
			if len(recent) > 0 {
				a.markTried() // a history from before the flag
			}
		})
	})
}

// homeChanged runs when the history changed: Home reads it again now
// when it shows, else when it shows next.
func (a *app) homeChanged() {
	if a.currentPage() == "home" && !a.settingsOpen {
		a.loadHome()
		return
	}
	a.hm.stale = true
}

func (a *app) prepareHomePage() {
	if !a.hm.loading && (!a.hm.read || a.hm.stale) {
		a.loadHome()
	}
	if !a.perms.checked {
		a.recheckPermissions()
	}
}

func (a *app) homePageModel(now time.Time) homeViewState {
	showTry := a.homeShowsTry(a.homeTrySession())
	tryRecording := a.sess.Try && a.sess.State == session.Recording
	readiness := a.homeReadinessInput()
	_, zoneOffset := now.Zone()
	key := homeViewCacheKey{
		settings: a.settingsRevision, gen: a.hm.gen, read: a.hm.read, loading: a.hm.loading, err: a.hm.err,
		readiness: homeReadinessKeyFor(readiness), retryID: a.hist.retry.id, retryStage: string(a.hist.retry.stage),
		showTry: showTry, tryRecording: tryRecording,
		minute: now.Unix() / 60, location: now.Location().String(), zoneOffset: zoneOffset, locale: i18n.Locale(),
	}
	if key != a.hm.viewKey {
		a.hm.view = buildHomeViewState(i18n.CurrentTranslator(), homeViewStateInput{
			Read: a.hm.read, Loading: a.hm.loading, Today: a.hm.today, Recent: a.hm.recent, Error: a.hm.err,
			KeepHistory: a.prefs.Privacy.KeepHistory, Features: a.prefs.Features,
			Readiness: buildHomeReadinessList(i18n.CurrentTranslator(), a.prefs.Features, readiness),
			ShowTry:   showTry, TryRecording: tryRecording, Retry: a.hist.retry, Now: now,
		})
		a.hm.viewKey = key
	}
	return a.hm.view
}

func (a *app) homeTrySession() bool {
	return a.sess.Try && a.sess.ID != "" && !a.obOwns(a.sess.ID)
}

func (a *app) readinessList() []homeReadinessViewState {
	return buildHomeReadinessList(i18n.CurrentTranslator(), a.prefs.Features, a.homeReadinessInput())
}

func (a *app) homeReadinessInput() homeReadinessInput {
	speech, speechReady := a.featureModel(local.ASR, hotkey.Transcribe)
	language, languageReady := a.featureModel(local.LLM, hotkey.Transcribe)
	return homeReadinessInput{
		SpeechModelID: speech.ID, SpeechModelName: speech.Name, SpeechReady: speechReady,
		SpeechExplicit: featureOf(a.prefs, hotkey.Transcribe).ASR == dictation.ModelID,
		SpeechStatus:   a.modelStatus[a.obSpeech()], SpeechDownloadLeft: a.obDownloadLeft(),
		LanguageModelName: language.Name, LanguageModelReady: languageReady,
		PermissionsChecked: a.perms.checked, Microphone: a.perms.mic, Accessibility: a.perms.access,
		HotkeysInstalled: a.hotkeys.Installed, Helper: a.helperStatus(),
	}
}

func (a *app) openHomeGuide()   { a.openOnboarding() }
func (a *app) openHomeHistory() { a.navigate("/history") }
func (a *app) addHomeShortcut() { a.navigate("/transcribe") }
func (a *app) openHomeRecent(id string) {
	if entry, ok := a.historyEntryByID(id); ok {
		a.openHistoryEntry(entry)
	}
}

// copyHomeRecent keeps the inline Copy behavior: the original result and a
// success toast, unlike the checked, trimmed menu Copy in History.
func (a *app) copyHomeRecent(id string) {
	entry, ok := a.historyEntryByID(id)
	if !ok {
		return
	}
	if a.copyText != nil {
		_ = a.copyText(entry.Result)
	} else {
		_ = mygo.Clipboard.Write(transfer.TextData(entry.Result))
	}
	a.toast(i18n.T("Copied"))
}

func (a *app) fixHomeReadiness(fix string) {
	switch fix {
	case homeFixSpeechDownload:
		a.downloadModel(a.obSpeech())
	case homeFixSpeechModels:
		a.navigate(modelsPath)
	case homeFixLanguageModels:
		a.navigate(remotePath)
	case homeFixMicrophone:
		a.askPermission(proto.PermissionMicrophone, nil)
	case homeFixPermissions:
		a.navigate("/settings/permissions")
	case homeFixTranscribe:
		a.navigate("/transcribe")
	case homeFixRestartHelper:
		a.restartHelper()
	case homeFixRelaunch:
		a.relaunch()
	}
}
