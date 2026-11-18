package main

import "github.com/egoist/mygo/ui"

type featureDemoState struct {
	local     featureUI
	block     *tryNoticeViewState
	keysHeard bool
}

func (host *meetingDemoHost) selectFeatureFixture(state *featureViewState) {
	host.feature = featureDemoState{}
	if state == nil {
		return
	}
	host.feature.local = featureUI{
		prompt: state.Prompt.Rules.Prompt, rules: state.Prompt.Rules.Open, preview: state.Prompt.Rules.PreviewOpen,
		find: state.Prompt.Rules.FindQuery, asr: state.Models.ASR.Value, llm: state.Models.LLM.Value,
	}
	if state.Prompt.Target != nil {
		host.feature.local.lang = state.Prompt.Target.Value
	}
	if state.Models.Fallback != nil {
		host.feature.local.fallback = state.Models.Fallback.Value
	}
	host.feature.local.rows = make([]keyRow, len(state.Shortcuts.Rows))
	for index, row := range state.Shortcuts.Rows {
		host.feature.local.rows[index] = featureDemoKeyRow(row)
	}
	if state.Shortcuts.Paste != nil {
		host.feature.local.paste = featureDemoKeyRow(*state.Shortcuts.Paste)
	}
	if state.Shortcuts.Cancel != nil {
		host.feature.local.cancel = featureDemoKeyRow(*state.Shortcuts.Cancel)
	}
	if state.Shortcuts.EndWord != nil {
		host.feature.local.endWord = keyRow{sel: state.Shortcuts.EndWord.Value, custom: state.Shortcuts.EndWord.Custom, text: state.Shortcuts.EndWord.CustomText}
	}
	if state.Try.Block != nil {
		block := *state.Try.Block
		host.feature.block = &block
	}
}

func featureDemoKeyRow(state featureKeyRowViewState) keyRow {
	text := state.Keys
	if state.CustomEditing {
		text = state.Custom.InputText
	}
	return keyRow{
		sel: state.Value, text: text, trig: state.Trigger, custom: state.CustomEditing,
		typing: state.Typing, edited: state.Edited, recordDenied: state.RecordDenied, unknown: state.Unknown, recFocused: state.RecorderFocused,
	}
}

func (host *meetingDemoHost) openFeaturePermissions() { host.record("openFeaturePermissions") }
func (host *meetingDemoHost) addFeatureShortcut(feature string) {
	host.record("addFeatureShortcut %s", feature)
}
func (host *meetingDemoHost) chooseFeatureShortcut(feature string, index int, keys string) {
	if index < len(host.feature.local.rows) {
		host.feature.local.rows[index].custom, host.feature.local.rows[index].text = false, keys
	}
	host.record("chooseFeatureShortcut %s %d %q", feature, index, keys)
}
func (host *meetingDemoHost) setFeatureShortcutTrigger(feature string, index, trigger int) {
	host.record("setFeatureShortcutTrigger %s %d %d", feature, index, trigger)
}
func (host *meetingDemoHost) removeFeatureShortcut(feature string, index int) {
	host.record("removeFeatureShortcut %s %d", feature, index)
}
func (host *meetingDemoHost) saveFeatureShortcutKeys(feature string, index int, keys string) {
	host.record("saveFeatureShortcutKeys %s %d %q", feature, index, keys)
}
func (host *meetingDemoHost) chooseFeaturePasteLast(keys string) {
	host.feature.local.paste.custom, host.feature.local.paste.text = false, keys
	host.record("chooseFeaturePasteLast %q", keys)
}
func (host *meetingDemoHost) saveFeaturePasteLast(keys string) {
	host.record("saveFeaturePasteLast %q", keys)
}
func (host *meetingDemoHost) chooseFeatureCancelKeys(keys string) {
	host.feature.local.cancel.custom, host.feature.local.cancel.text = false, keys
	host.record("chooseFeatureCancelKeys %q", keys)
}
func (host *meetingDemoHost) saveFeatureCancelKeys(keys string) {
	host.record("saveFeatureCancelKeys %q", keys)
}
func (host *meetingDemoHost) recordFeatureKeys(row *keyRow, event ui.InputEvent, save func(string)) {
	host.record("recordFeatureKeys")
}
func (host *meetingDemoHost) markFeatureKeysHeard() {
	if !host.feature.keysHeard {
		host.feature.keysHeard = true
		host.record("markFeatureKeysHeard")
	}
}
func (host *meetingDemoHost) openFeatureSpeechModels() { host.record("openFeatureSpeechModels") }
func (host *meetingDemoHost) openFeatureLanguageModels() {
	host.record("openFeatureLanguageModels")
}
func (host *meetingDemoHost) selectFeatureModel(feature, kind, id string) {
	host.record("selectFeatureModel %s %s %q", feature, kind, id)
}
func (host *meetingDemoHost) selectFeatureFallback(id string) {
	host.record("selectFeatureFallback %q", id)
}
func (host *meetingDemoHost) setFeatureEnhance(on bool) { host.record("setFeatureEnhance %t", on) }
func (host *meetingDemoHost) setFeatureCleanup(on bool) { host.record("setFeatureCleanup %t", on) }
func (host *meetingDemoHost) setFeatureStructure(on bool) {
	host.record("setFeatureStructure %t", on)
}
func (host *meetingDemoHost) selectFeatureTarget(code string) {
	host.record("selectFeatureTarget %q", code)
}
func (host *meetingDemoHost) setFeatureTranslateCleanup(on bool) {
	host.record("setFeatureTranslateCleanup %t", on)
}
func (host *meetingDemoHost) setFeatureRewriteCleanup(on bool) {
	host.record("setFeatureRewriteCleanup %t", on)
}
func (host *meetingDemoHost) setFeatureShowCard(feature string, on bool) {
	host.record("setFeatureShowCard %s %t", feature, on)
}
func (host *meetingDemoHost) setCardSize(feature string, size int) {
	host.record("setCardSize %s %d", feature, size)
}
func (host *meetingDemoHost) addFeatureTargetVariable() { host.record("addFeatureTargetVariable") }
func (host *meetingDemoHost) savePrompt(feature string) {
	host.feature.local.promptDirty = false
	host.record("savePrompt %s", feature)
}
func (host *meetingDemoHost) resetFeaturePrompt(feature string) {
	host.record("resetFeaturePrompt %s", feature)
}
func (host *meetingDemoHost) scheduleFeaturePromptSave(feature string, sequence int) {
	host.feature.local.promptDirty = true
	host.record("scheduleFeaturePromptSave %s %d", feature, sequence)
}
func (host *meetingDemoHost) copyFeatureText(text string) { host.record("copyFeatureText %q", text) }
func (host *meetingDemoHost) setFeatureEndWord(word string) {
	host.record("setFeatureEndWord %q", word)
}

func (host *meetingDemoHost) startTry()  { host.record("startTry") }
func (host *meetingDemoHost) stopTry()   { host.record("stopTry") }
func (host *meetingDemoHost) cancelTry() { host.record("cancelTry") }
func (host *meetingDemoHost) copyTryResult(text string) {
	host.record("copyTryResult %q", text)
}
func (host *meetingDemoHost) clearTryNotice() {
	host.feature.block = nil
	host.record("clearTryNotice")
}
func (host *meetingDemoHost) runTryNoticeAction(action string) {
	host.feature.block = nil
	host.record("runTryNoticeAction %s", action)
}

// transcriptResult (tryCardActions, historyActions): the demo has no WebView, so results draw as plain text.
func (host *meetingDemoHost) transcriptResult(*uiContext, string, string, bool, bool) bool {
	return false
}
