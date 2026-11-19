package main

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (m overlayMode) MarshalJSON() ([]byte, error) { return json.Marshal(m.String()) }

func (m *overlayMode) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value == "hidden" {
		value = ""
	}
	*m = overlayMode(value)
	return nil
}

// sessionOverlayActions are the intents the overlay controls can originate.
type sessionOverlayActions interface {
	skipLLM()
	stopSession()
	stopLLM()
	followUpAnswer(hold string) bool
	retryAnswer(language string)
	insertAnswer()
	copyAnswer()
	closeAnswer()
	noticeAction(action string)
	toggleLangMenu()
	chooseLanguage(code string)
	toggleThinkMenu()
	togglePillThinkMenu()
	chooseThinking(value sessionThinkingValue)
	cycleOverlayCardSize()
	toggleQuote(turn int)
	copyBubble(block sessionCardBlock)
	// overlayTranscriptResult shows formatted text in the overlay's WebView
	// and says whether it did; the view uses its native renderer when not.
	overlayTranscriptResult(context *uiContext, request sessionOverlayPreviewRequest) bool
}

// sessionOverlayViewState is the serializable display state of a session
// pill, notice, answer card, or menu. Controller inputs and native state stay outside this DTO.
type sessionOverlayViewState struct {
	Revision    uint64                  `json:"revision"`
	SessionID   string                  `json:"session_id"`
	State       string                  `json:"state"`
	ProblemCode string                  `json:"problem_code,omitempty"`
	Mode        overlayMode             `json:"mode"`
	Pill        sessionPillViewState    `json:"pill"`
	Notice      *sessionNoticeViewState `json:"notice,omitempty"`
	Card        *sessionCardViewState   `json:"card,omitempty"`
	Menu        *sessionMenuViewState   `json:"menu,omitempty"`
}

type sessionPillViewState struct {
	Kind          string                `json:"kind"`
	Title         string                `json:"title"`
	WorkingLabel  string                `json:"working_label"`
	WaitingLabel  string                `json:"waiting_label,omitempty"`
	SceneText     string                `json:"scene_text,omitempty"`
	SceneLabel    string                `json:"scene_label,omitempty"`
	LiveText      string                `json:"live_text,omitempty"`
	ElapsedText   string                `json:"elapsed_text"`
	ThinkingLabel string                `json:"thinking_label,omitempty"`
	SkipLabel     string                `json:"skip_label,omitempty"`
	SkipTooltip   string                `json:"skip_tooltip,omitempty"`
	Levels        [session.Bars]float64 `json:"levels"`
	LiveHint      bool                  `json:"live_hint,omitempty"`
	Heard         bool                  `json:"heard"`
	Starting      bool                  `json:"starting"`
	Waiting       string                `json:"waiting"`
	CanSkip       bool                  `json:"can_skip"`
	CanThink      bool                  `json:"can_think"`
}

const (
	sessionNoticeError   = "error"
	sessionNoticeWarning = "warning"
	sessionNoticeSuccess = "success"
)

type sessionNoticeViewState struct {
	Text        string `json:"text"`
	Action      string `json:"action,omitempty"`
	ActionLabel string `json:"action_label,omitempty"`
	Tone        string `json:"tone"`
}

// sessionOverlayProjectionInput contains the controller values the projection
// reads. None of these domain values cross into the view state.
type sessionOverlayProjectionInput struct {
	Snapshot      session.Snapshot
	Mode          overlayMode
	HideLiveText  bool
	ShowTime      bool
	PillNoteID    string
	PillNoteText  string
	CanThink      bool
	ThinkingLabel string
	Notice        *sessionNotice
	Card          sessionCardProjectionInput
	Menu          *sessionMenuProjectionInput
}

func buildSessionOverlayViewState(translator *i18n.Translator, input sessionOverlayProjectionInput, now time.Time) sessionOverlayViewState {
	snapshot := input.Snapshot
	state := sessionOverlayViewState{
		Revision:  snapshot.Rev,
		SessionID: snapshot.ID,
		State:     string(snapshot.State),
		Mode:      input.Mode,
	}
	if snapshot.Problem != nil {
		state.ProblemCode = snapshot.Problem.Code
	}
	switch input.Mode {
	case ovRecording, ovWorking:
		state.Pill = buildSessionPillViewState(translator, input, now)
	case ovNotice:
		if input.Notice != nil {
			notice := buildSessionNoticeViewState(translator, *input.Notice)
			state.Notice = &notice
		}
	case ovAnswer:
		card := buildSessionCardViewState(translator, input.Card)
		state.Card = &card
	}
	if input.Menu != nil {
		menu := buildSessionMenuViewState(translator, *input.Menu)
		state.Menu = &menu
	}
	return state
}

func buildSessionPillViewState(translator *i18n.Translator, input sessionOverlayProjectionInput, now time.Time) sessionPillViewState {
	snapshot := pillSessionFor(input.Snapshot, input.HideLiveText, input.PillNoteID, input.PillNoteText)
	state := sessionPillViewState{
		Kind:          string(snapshot.Kind),
		Title:         uitext.KindTitleFor(translator, snapshot.Kind),
		WorkingLabel:  uitext.WorkingTextFor(translator, snapshot),
		WaitingLabel:  uitext.WaitTextFor(translator, snapshot),
		SceneText:     uitext.SceneShort(snapshot),
		SceneLabel:    uitext.SceneLabel(snapshot),
		ElapsedText:   clock(snapshot.Elapsed(now)),
		ThinkingLabel: input.ThinkingLabel,
		Waiting:       string(snapshot.Waiting),
		Heard:         !snapshot.HeardAt.IsZero(),
		Starting:      snapshot.Waiting != session.WaitNone,
		CanSkip:       canSkip(snapshot, input.Mode),
		CanThink:      input.CanThink,
	}
	state.LiveText, state.LiveHint = pillLineFor(translator, snapshot, now)
	if state.CanSkip {
		state.SkipLabel = translator.T("Use transcript")
		state.SkipTooltip = translator.T("Insert the recognised text without waiting for the language model")
	}
	if !state.CanThink {
		state.ThinkingLabel = ""
	}
	copySessionLevels(&state.Levels, snapshot.Levels)
	return state
}

func buildSessionNoticeViewState(translator *i18n.Translator, notice sessionNotice) sessionNoticeViewState {
	tone := sessionNoticeError
	switch {
	case notice.ok:
		tone = sessionNoticeSuccess
	case notice.warn:
		tone = sessionNoticeWarning
	}
	return sessionNoticeViewState{
		Text:        notice.text,
		Action:      notice.action,
		ActionLabel: noticeActionLabelFor(translator, notice.action),
		Tone:        tone,
	}
}

// pillSessionFor applies the overlay's privacy switch and temporary pill note.
func pillSessionFor(snapshot session.Snapshot, hideLiveText bool, noteID, noteText string) session.Snapshot {
	if hideLiveText {
		snapshot.Partial, snapshot.Draft, snapshot.Selection = "", "", ""
	}
	if noteText != "" && noteID == snapshot.ID && snapshot.State == session.Recording {
		snapshot.Partial = noteText
	}
	return snapshot
}

// canSkip: the pill in mode offers Use transcript (session.Manager's
// SkipLLM, #3821).
func canSkip(snapshot session.Snapshot, mode overlayMode) bool {
	return mode == ovWorking && snapshot.State == session.Composing && snapshot.CanSkip
}

// pillLine is the compatibility form used by unmigrated session callers.
func pillLine(snapshot session.Snapshot) (text string, hint bool) {
	return pillLineFor(i18n.CurrentTranslator(), snapshot, time.Now())
}

// pillLineFor is the line under the pill's row: the live text, else (hint)
// "Starting the microphone…" while the microphone gives no audio (what
// is said then is lost); "" for none. A speech model that is still
// loading shows only the starting icon (the user, 10-21): the recording
// works meanwhile, so no line is needed.
func pillLineFor(translator *i18n.Translator, snapshot session.Snapshot, now time.Time) (text string, hint bool) {
	if note := limitNoteFor(translator, snapshot, now); note != "" {
		return note, true
	}
	if text := liveLineText(snapshot); text != "" {
		return text, false
	}
	if snapshot.Waiting == session.WaitMic {
		if waiting := uitext.WaitTextFor(translator, snapshot); waiting != "" {
			return waiting, true
		}
	}
	return "", false
}

// limitNote warns while recording that a limit of Settings → Audio is
// near (#3840): in the last minute before the length limit stops the
// recording, and from 30 s before it passes the upload size limit (it
// then goes to the speech model in pieces); "" for none.
func limitNote(snapshot session.Snapshot, now time.Time) string {
	return limitNoteFor(i18n.CurrentTranslator(), snapshot, now)
}

func limitNoteFor(translator *i18n.Translator, snapshot session.Snapshot, now time.Time) string {
	if snapshot.State != session.Recording {
		return ""
	}
	elapsed := snapshot.Elapsed(now)
	switch {
	case snapshot.MaxLen > 0 && snapshot.MaxLen-elapsed <= time.Minute:
		return translator.T("The length limit stops the recording in %s", clock(max(0, snapshot.MaxLen-elapsed)))
	case snapshot.UploadLen > 0 && elapsed >= snapshot.UploadLen:
		return translator.T("Over the upload size limit: Voxt sends the recording in pieces")
	case snapshot.UploadLen > 0 && snapshot.UploadLen-elapsed <= 30*time.Second:
		return translator.T("Near the upload size limit: Voxt will send the recording in pieces")
	}
	return ""
}

// liveLineText is the live text as its line shows it: its spaces and
// line breaks one space each.
func liveLineText(snapshot session.Snapshot) string {
	return strings.Join(strings.Fields(liveText(snapshot)), " ")
}

// liveText is the text the pill shows under it. The preview carries only
// live ASR partials, not the offline Final nor language-model drafts
// (FB6). While the language model works, the live text stays as it last
// was (Rewrite's spoken instruction too): the pill went blank then
// (#3644); without one, the selection worked on (selected text to
// translate).
func liveText(snapshot session.Snapshot) string {
	partial := strings.TrimSpace(snapshot.Partial)
	switch snapshot.State {
	case session.Recording, session.Recognizing:
		return partial
	case session.Composing, session.Delivering:
		// A stream's words become the recognised text as it was sent to the
		// model (its last partial can differ); without live words, the selection.
		if partial != "" {
			return cmpOr(strings.TrimSpace(snapshot.Spoken), partial)
		}
		return strings.TrimSpace(snapshot.Selection)
	}
	return ""
}

// clock shows a duration as m:ss.
func clock(duration time.Duration) string {
	seconds := int(duration.Seconds())
	return strconv.Itoa(seconds/60) + ":" + string(rune('0'+seconds%60/10)) + string(rune('0'+seconds%10))
}

func copySessionLevels(destination *[session.Bars]float64, levels []float64) {
	*destination = [session.Bars]float64{}
	if len(levels) > session.Bars {
		levels = levels[len(levels)-session.Bars:]
	}
	copy(destination[session.Bars-len(levels):], levels)
}

type sessionPillProjectionKey struct {
	spoken                      string // liveText shows it while the model works
	sessionID, kind, state      string
	partial, selection          string
	waiting                     string
	sceneText, sceneLabel       string
	pillNoteID, pillNoteText    string
	thinkingLabel, locale       string
	mode                        overlayMode
	hideLiveText, showTime      bool
	canSkip, canThink, heard    bool
	maxLen, uploadLen           time.Duration
	elapsedSecond, lengthSecond int
	lengthNear, uploadNear      bool
	uploadOver                  bool
}

type sessionNoticeProjectionKey struct {
	sequence      int
	text, action  string
	locale        string
	warn, success bool
}

type sessionOverlayProjection struct {
	view        sessionOverlayViewState
	pillKey     sessionPillProjectionKey
	noticeKey   sessionNoticeProjectionKey
	cardKey     sessionCardProjectionKey
	cardLiveKey sessionCardLiveProjectionKey
	menuKey     sessionMenuProjectionKey
	notice      sessionNoticeViewState
	card        sessionCardViewState
	menu        sessionMenuViewState
	translator  *i18n.Translator
	locale      string
	pillReady   bool
	noticeReady bool
	cardReady   bool
	menuReady   bool
	stillReady  bool
	stillKey    sessionOverlayStillKey
}

// sessionOverlayStillKey is everything a still notice frame reads: when
// it matches, refreshSessionOverlay leaves the projection as it is
// instead of rebuilding the header and comparing the sub-keys every
// frame.
type sessionOverlayStillKey struct {
	sessionID, state, problemCode string
	revision                      uint64
	sequence                      int
	text, action                  string
	locale                        string
	warn, success, over           bool
}

func sessionPillKeyFor(input sessionOverlayProjectionInput, locale string, now time.Time) sessionPillProjectionKey {
	snapshot := input.Snapshot
	elapsed := snapshot.Elapsed(now)
	key := sessionPillProjectionKey{
		sessionID: snapshot.ID, kind: string(snapshot.Kind), state: string(snapshot.State),
		partial: snapshot.Partial, spoken: snapshot.Spoken, selection: snapshot.Selection, waiting: string(snapshot.Waiting),
		sceneText: uitext.SceneShort(snapshot), sceneLabel: uitext.SceneLabel(snapshot),
		pillNoteID: input.PillNoteID, pillNoteText: input.PillNoteText,
		thinkingLabel: input.ThinkingLabel, locale: locale, mode: input.Mode,
		hideLiveText: input.HideLiveText, showTime: input.ShowTime,
		canSkip: snapshot.CanSkip, canThink: input.CanThink, heard: !snapshot.HeardAt.IsZero(),
		maxLen: snapshot.MaxLen, uploadLen: snapshot.UploadLen,
	}
	if input.ShowTime {
		key.elapsedSecond = int(elapsed.Seconds())
	}
	if snapshot.MaxLen > 0 && snapshot.MaxLen-elapsed <= time.Minute {
		key.lengthNear = true
		key.lengthSecond = int(max(0, snapshot.MaxLen-elapsed).Seconds())
	}
	if snapshot.UploadLen > 0 {
		key.uploadOver = elapsed >= snapshot.UploadLen
		key.uploadNear = snapshot.UploadLen-elapsed <= 30*time.Second
	}
	return key
}

// refreshSessionOverlay updates the cached pill or notice before measurement
// and drawing. Level updates reuse the cached text projection.
func (a *app) refreshSessionOverlay(now time.Time) {
	projection := &a.ov.projection
	locale := i18n.Locale()
	mode := a.overlayModeNow()
	if mode == ovNotice && a.notice != nil && !a.answer.ui.langOpen {
		key := sessionOverlayStillKey{
			sessionID: a.sess.ID, state: string(a.sess.State), revision: a.sess.Rev,
			sequence: a.notice.seq, text: a.notice.text, action: a.notice.action,
			locale: locale, warn: a.notice.warn, success: a.notice.ok, over: a.notice.over,
		}
		if a.sess.Problem != nil {
			key.problemCode = a.sess.Problem.Code
		}
		if projection.stillReady && key == projection.stillKey && projection.view.Mode == ovNotice {
			return
		}
		projection.stillKey, projection.stillReady = key, true
	} else {
		projection.stillReady = false
	}
	if projection.translator == nil || projection.locale != locale {
		projection.locale = locale
		projection.translator = i18n.NewTranslator(locale)
		projection.cardReady, projection.menuReady = false, false
	}
	input := sessionOverlayProjectionInput{
		Snapshot: a.sess, Mode: mode,
		HideLiveText: a.prefs.Overlay.HideLiveText, ShowTime: a.prefs.Overlay.ShowTime,
		PillNoteID: a.pillNote.id, PillNoteText: a.pillNote.text, Notice: a.notice,
	}
	if input.Mode == ovRecording || input.Mode == ovWorking {
		input.CanThink = a.pillThinks(input.Mode)
		if input.CanThink {
			input.ThinkingLabel = a.pillThinkText().text
		}
	}
	next := &projection.view
	next.Revision, next.SessionID, next.State, next.Mode = a.sess.Rev, a.sess.ID, string(a.sess.State), input.Mode
	next.ProblemCode = ""
	if a.sess.Problem != nil {
		next.ProblemCode = a.sess.Problem.Code
	}
	if input.Mode == ovRecording || input.Mode == ovWorking || input.Mode == ovHidden {
		key := sessionPillKeyFor(input, locale, now)
		if !projection.pillReady || key != projection.pillKey {
			next.Pill = buildSessionPillViewState(projection.translator, input, now)
			projection.pillKey, projection.pillReady = key, true
		}
		copySessionLevels(&next.Pill.Levels, a.sess.Levels)
	} else {
		next.Pill = sessionPillViewState{}
		projection.pillReady = false
	}
	next.Card = nil
	if input.Mode == ovAnswer {
		a.refreshSessionCardProjection(projection, next)
	}
	next.Menu = nil
	a.refreshSessionMenuProjection(projection, next, locale)
	next.Notice = nil
	if input.Mode == ovNotice && a.notice != nil {
		key := sessionNoticeProjectionKey{
			sequence: a.notice.seq, text: a.notice.text, action: a.notice.action,
			locale: locale, warn: a.notice.warn, success: a.notice.ok,
		}
		if !projection.noticeReady || key != projection.noticeKey {
			projection.notice = buildSessionNoticeViewState(projection.translator, *a.notice)
			projection.noticeKey, projection.noticeReady = key, true
		}
		next.Notice = &projection.notice
	}
}
