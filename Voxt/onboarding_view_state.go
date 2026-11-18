package main

import (
	"time"

	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type onboardingNoticeViewState struct {
	Text        string `json:"text"`
	More        string `json:"more"`
	Action      string `json:"action"`
	ActionLabel string `json:"action_label"`
}

type onboardingLegacyTodoViewState struct {
	Text        string `json:"text"`
	Kind        string `json:"kind"`
	ID          string `json:"id"`
	ServiceName string `json:"service_name"`
}

type onboardingLegacyViewState struct {
	Done     bool                            `json:"done"`
	Failed   string                          `json:"failed"`
	Imported []string                        `json:"imported"`
	Skipped  []string                        `json:"skipped"`
	Todos    []onboardingLegacyTodoViewState `json:"todos"`
}

type onboardingWelcomeViewState struct {
	LegacyAvailable bool                      `json:"legacy_available"`
	Legacy          onboardingLegacyViewState `json:"legacy"`
}

type onboardingPermissionRowViewState struct {
	Code       string `json:"code"`
	Text       string `json:"text"`
	OK         bool   `json:"ok"`
	CanAsk     bool   `json:"can_ask"`
	CanOpen    bool   `json:"can_open"`
	AskAction  string `json:"ask_action"`
	AskTooltip string `json:"ask_tooltip"`
}

type onboardingPermissionsViewState struct {
	AskRequired bool                             `json:"ask_required"`
	Microphone  onboardingPermissionRowViewState `json:"microphone"`
	Access      onboardingPermissionRowViewState `json:"access"`
	Speech      onboardingPermissionRowViewState `json:"speech"`
}

type onboardingAlternativeViewState struct {
	WantID        string `json:"want_id"`
	UseID         string `json:"use_id"`
	Message       string `json:"message"`
	DownloadLabel string `json:"download_label"`
	UseLabel      string `json:"use_label"`
}

type onboardingSpeechViewState struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Ready       bool   `json:"ready"`
	OnDisk      bool   `json:"on_disk"`
	InstalledBy string `json:"installed_by"`
	SizeLabel   string `json:"size_label"`

	ShowDownload     bool                            `json:"show_download"`
	DownloadLabel    string                          `json:"download_label"`
	Downloading      bool                            `json:"downloading"`
	DownloadFraction float64                         `json:"download_fraction"`
	About            string                          `json:"about"`
	Alternative      *onboardingAlternativeViewState `json:"alternative,omitempty"`
	Error            *onboardingNoticeViewState      `json:"error,omitempty"`
}

type onboardingModelsViewState struct {
	ReuseShown    bool                      `json:"reuse_shown"`
	ReuseIDs      []string                  `json:"reuse_ids"`
	ReuseSubtitle string                    `json:"reuse_subtitle"`
	ReuseNames    string                    `json:"reuse_names"`
	Speech        onboardingSpeechViewState `json:"speech"`
	LanguageReady bool                      `json:"language_ready"`
	LanguageName  string                    `json:"language_name"`
	NeedsDownload bool                      `json:"needs_download"`
	DownloadLeft  string                    `json:"download_left"`
}

type onboardingKeyboardViewState struct {
	Visible  bool            `json:"visible"`
	LitKeys  map[string]bool `json:"lit_keys"`
	ExtraKey string          `json:"extra_key"`
}

type onboardingShortcutViewState struct {
	Present              bool                        `json:"present"`
	FeatureTitle         string                      `json:"feature_title"`
	Text                 string                      `json:"text"`
	How                  string                      `json:"how"`
	AccessibilityWarning bool                        `json:"accessibility_warning"`
	Keyboard             onboardingKeyboardViewState `json:"keyboard"`
}

type onboardingTryCardViewState struct {
	Mine        bool                       `json:"mine"`
	State       string                     `json:"state"`
	SessionKind string                     `json:"session_kind"`
	Live        bool                       `json:"live"`
	Ended       bool                       `json:"ended"`
	Recording   bool                       `json:"recording"`
	Composing   bool                       `json:"composing"`
	Cancelled   bool                       `json:"cancelled"`
	Failed      bool                       `json:"failed"`
	Levels      []float64                  `json:"levels"`
	Heard       bool                       `json:"heard"`
	ElapsedText string                     `json:"elapsed_text"`
	Partial     string                     `json:"partial"`
	Draft       string                     `json:"draft"`
	WorkingText string                     `json:"working_text"`
	BusyText    string                     `json:"busy_text"`
	Block       *onboardingNoticeViewState `json:"block,omitempty"`
	Failure     *onboardingNoticeViewState `json:"failure,omitempty"`
	FailureText string                     `json:"failure_text"`
	Warning     *onboardingNoticeViewState `json:"warning,omitempty"`
	ShowResult  bool                       `json:"show_result"`
	Result      string                     `json:"result"`
	Raw         string                     `json:"raw"`
	ShowRaw     bool                       `json:"show_raw"`
	RawTitle    string                     `json:"raw_title"`
	Timings     string                     `json:"timings"`
	ProblemCode string                     `json:"problem_code"`
}

type onboardingTryViewState struct {
	Prefix        string                      `json:"prefix"`
	Shortcut      onboardingShortcutViewState `json:"shortcut"`
	Conflicts     []string                    `json:"conflicts"`
	Passed        bool                        `json:"passed"`
	SampleTitle   string                      `json:"sample_title"`
	SampleText    string                      `json:"sample_text"`
	SampleAction  bool                        `json:"sample_action"`
	Hint          string                      `json:"hint"`
	LanguageGuide *onboardingNoticeViewState  `json:"language_guide,omitempty"`
	SpeechGuide   *onboardingNoticeViewState  `json:"speech_guide,omitempty"`
	Card          onboardingTryCardViewState  `json:"card"`
}

type onboardingOptionViewState struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

type onboardingShortcutRowViewState struct {
	Title    string   `json:"title"`
	Keys     string   `json:"keys"`
	Problems []string `json:"problems"`
}

type onboardingMoreViewState struct {
	PresetOptions  []onboardingOptionViewState      `json:"preset_options"`
	PresetSelected string                           `json:"preset_selected"`
	PresetAbout    string                           `json:"preset_about"`
	PresetWarnings []string                         `json:"preset_warnings"`
	Shortcuts      []onboardingShortcutRowViewState `json:"shortcuts"`
}

type onboardingFooterViewState struct {
	Mode             string `json:"mode"`
	Busy             bool   `json:"busy"`
	Back             bool   `json:"back"`
	ContinueLabel    string `json:"continue_label"`
	ContinueTooltip  string `json:"continue_tooltip"`
	ContinueDisabled bool   `json:"continue_disabled"`
}

type onboardingViewState struct {
	Open        bool                            `json:"open"`
	Step        int                             `json:"step"`
	StepTitles  [obSteps]string                 `json:"step_titles"`
	Passed      [obSteps]bool                   `json:"passed"`
	Title       string                          `json:"title"`
	Subtitle    string                          `json:"subtitle"`
	Recording   bool                            `json:"recording"`
	Welcome     *onboardingWelcomeViewState     `json:"welcome,omitempty"`
	Permissions *onboardingPermissionsViewState `json:"permissions,omitempty"`
	Models      *onboardingModelsViewState      `json:"models,omitempty"`
	Try         *onboardingTryViewState         `json:"try,omitempty"`
	More        *onboardingMoreViewState        `json:"more,omitempty"`
	Footer      onboardingFooterViewState       `json:"footer"`
}

type onboardingBackViewState struct {
	Visible bool   `json:"visible"`
	Text    string `json:"text"`
}

type onboardingModelStateInput struct {
	SpeechID      string
	PackIDs       []string
	Status        map[string]models.Status
	Found         []models.Found
	InstalledASR  []local.Model
	Language      local.Model
	LanguageReady bool
}

type onboardingWelcomeCacheKey struct {
	locale   string
	revision uint64
	step     int
	triesNil bool
}

type onboardingViewStateInput struct {
	Open                bool
	Step                int
	Passed              map[int]bool
	Tries               map[int]string
	Session             session.Snapshot
	SessionProblem      *sessionNotice
	Settings            settings.Settings
	PermissionMic       permission.State
	PermissionSpeech    permission.State
	PermissionAccess    bool
	Models              onboardingModelStateInput
	SpeechAbout         string
	AlternativeUseLabel string
	SpeechError         *sessionNotice
	Block               *sessionNotice
	Hint                string
	Blocker             *sessionNotice
	SpeechWait          *sessionNotice
	Legacy              legacyReport
	LegacyAvailable     bool
	LegacyDone          bool
	TryKeys             string
	Conflicts           []shortcuts.Conflict
	MoreKeys            map[string]string
	TargetLanguage      string
	HotkeysInstalled    bool
	Now                 time.Time
}

func onboardingStepTitles(translator *i18n.Translator) [obSteps]string {
	return [obSteps]string{translator.T("Welcome"), translator.T("Permissions"), translator.T("Models"), translator.T("Transcribe"),
		translator.T("Translate"), translator.T("Rewrite"), translator.T("More")}
}

func obStepTitles() [obSteps]string { return onboardingStepTitles(i18n.CurrentTranslator()) }

func onboardingKindTitle(translator *i18n.Translator, kind store.Kind) string {
	switch kind {
	case store.KindTranslate:
		return translator.T("Translate")
	case store.KindRewrite:
		return translator.T("Rewrite")
	}
	return translator.T("Transcribe")
}

func onboardingFeatureTitle(translator *i18n.Translator, feature string) string {
	switch feature {
	case hotkey.Translate:
		return translator.T("Translate")
	case hotkey.Rewrite:
		return translator.T("Rewrite")
	case hotkey.PasteLast:
		return translator.T("Paste the last result")
	case hotkey.CancelID:
		return translator.T("Cancel recording")
	case hotkey.Note:
		return translator.T("Notes")
	case hotkey.Meeting:
		return translator.T("Meetings")
	}
	return translator.T("Transcribe")
}

func onboardingNoticeActionLabel(translator *i18n.Translator, action string) string {
	switch action {
	case actionSignIn:
		return translator.T("Sign in")
	case actionUsage:
		return translator.T("Manage usage")
	case actionModels:
		return translator.T("Open Speech models")
	case actionLanguage:
		return translator.T("Open Language models")
	case actionPermissions:
		return translator.T("Open Permissions")
	case actionHelper:
		return translator.T("Restart helper")
	case actionRetry:
		return translator.T("Retry")
	}
	return ""
}

func onboardingNotice(translator *i18n.Translator, notice *sessionNotice) *onboardingNoticeViewState {
	if notice == nil {
		return nil
	}
	return &onboardingNoticeViewState{
		Text: notice.text, More: notice.more, Action: notice.action,
		ActionLabel: onboardingNoticeActionLabel(translator, notice.action),
	}
}

func buildOnboardingViewState(translator *i18n.Translator, input onboardingViewStateInput) onboardingViewState {
	state := onboardingViewState{Open: input.Open, Step: input.Step, StepTitles: onboardingStepTitles(translator), Recording: input.Session.Try && input.Session.State == session.Recording}
	for step := 0; step < obSteps; step++ {
		state.Passed[step] = input.Passed[step]
	}
	switch input.Step {
	case obWelcome:
		state.Title = translator.T("Welcome to Voxt")
		state.Subtitle = translator.T("Speak in any app and get text where your cursor is: transcribe, translate and rewrite. There are a few short steps, and you can skip any of them.")
	case obPermissions:
		state.Title = translator.T("Permissions")
		state.Subtitle = translator.T("Voxt needs the microphone to hear you, and Accessibility to type the result where your cursor is. Allow them here, or later in Settings → Permissions.")
	case obModels:
		state.Title = translator.T("Models")
		state.Subtitle = translator.T("Voxt recognises speech with a model on this Mac and cleans up, translates and rewrites with a language model you set up. The model below is the best start, with others in Settings → Speech models.")
	case obTranscribe:
		state.Title = translator.T("Try Transcribe")
		state.Subtitle = translator.T("Transcribe writes down what you say. Press the shortcut (or Start), read the sentence below, then press it again to stop.")
	case obTranslate:
		state.Title = translator.T("Try Translate")
		state.Subtitle = translator.T("Translate turns the selected text, or what you say, into the language you choose. Translate the sample below, or press the shortcut (or Start) and say a sentence.")
	case obRewrite:
		state.Title = translator.T("Try Rewrite")
		state.Subtitle = translator.T("Rewrite writes what you ask for, or changes the selected text as you say. Press the shortcut (or Start) and ask for something short.")
	default:
		state.Title = translator.T("More to explore")
		state.Subtitle = translator.T("Voxt is ready. Two more things make it yours.")
	}
	if input.Step == obWelcome {
		state.Welcome = &onboardingWelcomeViewState{LegacyAvailable: input.LegacyAvailable}
		state.Welcome.Legacy = onboardingLegacyViewState{Done: input.LegacyDone, Failed: input.Legacy.failed, Imported: input.Legacy.imported, Skipped: input.Legacy.skipped}
		for _, todo := range input.Legacy.todo {
			state.Welcome.Legacy.Todos = append(state.Welcome.Legacy.Todos, onboardingLegacyTodoViewState{
				Text: todo.text, Kind: string(todo.kind), ID: todo.id, ServiceName: onboardingServiceName(translator, todo.kind, todo.id),
			})
		}
	}
	switch input.Step {
	case obPermissions:
		permissions := buildOnboardingPermissions(translator, input)
		state.Permissions = &permissions
	case obModels:
		models := buildOnboardingModels(translator, input)
		state.Models = &models
	case obTranscribe, obTranslate, obRewrite:
		tryState := buildOnboardingTry(translator, input)
		state.Try = &tryState
	case obMore:
		more := buildOnboardingMore(translator, input)
		state.More = &more
	}
	ownsSession := false
	for _, id := range input.Tries {
		if id != "" && id == input.Session.ID {
			ownsSession = true
		}
	}
	state.Footer = onboardingFooterViewState{Back: input.Step > 0, Busy: input.Session.State.Live() && ownsSession}
	_, tryStep := obStepKind(input.Step)
	switch {
	case input.Step == obMore:
		state.Footer.Mode = "finish"
		state.Footer.ContinueLabel = translator.T("Start using Voxt")
	case input.Step == obWelcome:
		state.Footer.Mode = "welcome"
		state.Footer.ContinueLabel = translator.T("Get started")
	case tryStep && !input.Passed[input.Step]:
		state.Footer.Mode = "try-wait"
		state.Footer.ContinueLabel = translator.T("Continue")
		state.Footer.ContinueTooltip = translator.T("Continue is available once a test gives a result")
		state.Footer.ContinueDisabled = true
	case input.Step == obModels && state.Models.ReuseShown:
		state.Footer.Mode = "reuse"
		state.Footer.ContinueLabel = translator.T("Continue")
	case input.Step == obModels && state.Models.NeedsDownload:
		state.Footer.Mode = "download"
		state.Footer.ContinueLabel = translator.T("Download and continue (%s)", state.Models.DownloadLeft)
		state.Footer.ContinueDisabled = state.Footer.Busy
	case input.Step == obModels && input.Models.Status[input.Models.SpeechID].Downloading:
		state.Footer.Mode = "downloading"
		state.Footer.ContinueLabel = translator.T("Continue")
		state.Footer.ContinueTooltip = translator.T("The speech model is still downloading. Later steps work once it is ready, or you can skip them.")
		state.Footer.ContinueDisabled = state.Footer.Busy
	default:
		state.Footer.Mode = "next"
		state.Footer.ContinueLabel = translator.T("Continue")
		state.Footer.ContinueDisabled = state.Footer.Busy
	}
	return state
}

func buildOnboardingBackViewState(translator *i18n.Translator, open bool, awayTo, path string, step int) onboardingBackViewState {
	return onboardingBackViewState{
		Visible: awayTo == path && !open,
		Text:    translator.T("The setup guide is paused at the %s step.", onboardingStepTitles(translator)[step]),
	}
}

func (a *app) onboardingBackViewState(path string) onboardingBackViewState {
	return buildOnboardingBackViewState(i18n.CurrentTranslator(), a.ob.open, a.ob.awayTo, path, a.ob.step)
}

func (a *app) onboardingModelsInput() onboardingModelStateInput {
	language, languageReady := a.featureModel(local.LLM, hotkey.Transcribe)
	return onboardingModelStateInput{
		SpeechID: a.obSpeech(), PackIDs: a.obPack(), Status: a.modelStatus, Found: a.ob.found,
		InstalledASR: a.installed(local.ASR), Language: language, LanguageReady: languageReady,
	}
}

func (a *app) onboardingViewState() onboardingViewState {
	return a.onboardingViewStateForTranslator(i18n.CurrentTranslator())
}

func (a *app) onboardingViewStateForTranslator(translator *i18n.Translator) onboardingViewState {
	cacheWelcome := a.ob.step == obWelcome && !a.ob.legacyAvailable
	cacheKey := onboardingWelcomeCacheKey{locale: translator.Locale(), revision: a.sess.Rev, step: a.ob.step, triesNil: a.ob.tries == nil}
	if cacheWelcome && a.ob.viewCached && a.ob.viewKey == cacheKey {
		return a.ob.view
	}
	input := onboardingViewStateInput{
		Open: a.ob.open, Step: a.ob.step, Passed: a.ob.passed, Tries: a.ob.tries, Session: a.sess,
		Settings: a.prefs, PermissionMic: a.perms.mic, PermissionSpeech: a.perms.speech, PermissionAccess: a.perms.access,
		Block: a.ob.block, Hint: a.ob.hint, Legacy: a.ob.legacy,
		LegacyAvailable: a.ob.legacyAvailable, LegacyDone: a.ob.legacyDone, HotkeysInstalled: a.hotkeys.Installed, Now: time.Now(),
	}
	if a.ob.step == obModels {
		input.Models = a.onboardingModelsInput()
	}
	if kind, ok := obStepKind(a.ob.step); ok {
		input.Blocker = a.sessionBlocker(kind)
		input.SpeechWait = a.speechWaits(kind)
		if input.SpeechWait != nil {
			wait := *input.SpeechWait
			wait.text = translator.T("The speech model is still downloading (%d%%).", downloadPercent(a.modelStatus[a.obSpeech()]))
			input.SpeechWait = &wait
		}
		input.Blocker = onboardingStartNotice(translator, input.Blocker, input.SpeechWait)
		input.Block = onboardingStartNotice(translator, input.Block, input.SpeechWait)
		input.Conflicts = a.famConflicts
	}
	if a.ob.step == obMore {
		input.Conflicts = a.famConflicts
	}
	localizeOnboardingViewStateInput(translator, &input)
	state := buildOnboardingViewState(translator, input)
	if cacheWelcome {
		a.ob.view, a.ob.viewKey, a.ob.viewCached = state, cacheKey, true
	}
	return state
}

func localizeOnboardingViewStateInput(translator *i18n.Translator, input *onboardingViewStateInput) {
	if input.Step == obModels {
		input.SpeechAbout = onboardingModelAbout(translator, input.Models.SpeechID)
		if alternative, ok := onboardingModelAlternative(input.Models); ok {
			input.AlternativeUseLabel = translator.X("alternative", "Use %s", onboardingModelName(alternative.ID))
		}
		if input.Models.Status[input.Models.SpeechID].Err != nil {
			input.SpeechError = onboardingDownloadErrorNotice(translator, input.Models.Status[input.Models.SpeechID].Err)
		}
	}
	if input.Session.Problem != nil {
		input.SessionProblem = onboardingSessionProblemNotice(translator, input.Session.Problem)
	}
	if kind, ok := obStepKind(input.Step); ok {
		input.TryKeys = ""
		if hotkeys := hotkey.HotkeysOf(input.Settings.Features, string(kind)); len(hotkeys) > 0 {
			input.TryKeys = onboardingHotkeyText(translator, hotkeys[0].Keys)
		}
		if kind == store.KindTranslate {
			target := session.TranslateTarget(input.Settings, translator.Locale(), "")
			input.TargetLanguage = compose.LanguageName(target, translator.Locale())
		}
	}
	if input.Step == obMore && input.MoreKeys == nil {
		input.MoreKeys = map[string]string{}
		for _, feature := range append(append([]string{}, hotkey.Features...), hotkey.PasteLast) {
			if hotkeys := hotkey.HotkeysOf(input.Settings.Features, feature); len(hotkeys) > 0 {
				input.MoreKeys[feature] = onboardingHotkeyText(translator, hotkeys[0].Keys)
			}
		}
	}
}
