package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func onboardingWorkingText(translator *i18n.Translator, snapshot session.Snapshot) string {
	switch snapshot.State {
	case session.Reading:
		return translator.T("Reading the selection…")
	case session.Composing:
		switch {
		case snapshot.Kind == store.KindTranslate && snapshot.Selection != "":
			return translator.T("Translating the selection…")
		case snapshot.Kind == store.KindTranslate:
			return translator.T("Translating…")
		case snapshot.Kind == store.KindRewrite && snapshot.Selection != "":
			return translator.T("Rewriting the selection…")
		case snapshot.Kind == store.KindRewrite:
			return translator.T("Writing…")
		}
		return translator.T("Cleaning up…")
	case session.Delivering:
		return translator.T("Inserting…")
	}
	return translator.T("Recognising…")
}

func onboardingTimingsText(translator *i18n.Translator, snapshot session.Snapshot) string {
	seconds := func(duration time.Duration) string { return translator.T("%.1f s", duration.Seconds()) }
	var parts []string
	timings := snapshot.Timings
	if timings.Audio > 0 {
		parts = append(parts, translator.T("Recording %s", seconds(timings.Audio)))
	}
	if timings.ASR > 0 {
		parts = append(parts, translator.T("Recognition %s", seconds(timings.ASR)))
	}
	if timings.LLM > 0 {
		switch snapshot.Kind {
		case store.KindTranslate:
			parts = append(parts, translator.T("Translation %s", seconds(timings.LLM)))
		case store.KindRewrite:
			parts = append(parts, translator.T("Writing %s", seconds(timings.LLM)))
		default:
			parts = append(parts, translator.T("Clean-up %s", seconds(timings.LLM)))
		}
	}
	if timings.Total > 0 {
		parts = append(parts, translator.T("Total %s", seconds(timings.Total)))
	}
	var serviceNames []string
	for _, model := range []session.Model{snapshot.ASR, snapshot.LLM} {
		if model.Name != "" {
			serviceNames = append(serviceNames, model.Name)
		}
	}
	if len(serviceNames) > 0 {
		parts = append(parts, strings.Join(serviceNames, ", "))
	}
	return strings.Join(parts, " · ")
}

func onboardingHotkeyText(translator *i18n.Translator, keys string) string {
	chord, err := hotkey.Parse(keys)
	if err != nil {
		return strings.TrimSpace(keys)
	}
	glyphs := map[string]string{proto.ModControl: "⌃", proto.ModOption: "⌥", proto.ModShift: "⇧", proto.ModCommand: "⌘"}
	var parts []string
	if _, ok := chord.Mods[proto.ModFn]; ok {
		parts = append(parts, "fn")
	}
	for _, family := range []string{proto.ModControl, proto.ModOption, proto.ModShift, proto.ModCommand} {
		side, ok := chord.Mods[family]
		if !ok {
			continue
		}
		switch side {
		case proto.SideLeft:
			parts = append(parts, translator.T("Left %s", glyphs[family]))
		case proto.SideRight:
			parts = append(parts, translator.T("Right %s", glyphs[family]))
		default:
			parts = append(parts, glyphs[family])
		}
	}
	if chord.Key != "" {
		parts = append(parts, onboardingKeyGlyph(translator, chord.Key))
	}
	var text strings.Builder
	for index, part := range parts {
		key := index == len(parts)-1 && chord.Key != "" && !chord.Mouse()
		if index > 0 && (len([]rune(part)) > 1 && !key || len([]rune(parts[index-1])) > 1) {
			text.WriteString(" ")
		}
		text.WriteString(part)
	}
	return text.String()
}

func onboardingKeyGlyph(translator *i18n.Translator, key string) string {
	switch key {
	case "Space":
		return translator.T("Space")
	case "Return":
		return translator.T("Return")
	case "Tab":
		return translator.T("Tab")
	case "Escape":
		return translator.T("Esc")
	case "Delete":
		return translator.T("Delete")
	case "ForwardDelete":
		return "⌦"
	case "Left":
		return "←"
	case "Right":
		return "→"
	case "Up":
		return "↑"
	case "Down":
		return "↓"
	case "PageUp":
		return "⇞"
	case "PageDown":
		return "⇟"
	case "Home":
		return "↖"
	case "End":
		return "↘"
	case "Mouse3":
		return translator.T("Middle mouse button")
	}
	if number, ok := strings.CutPrefix(key, "Mouse"); ok {
		return translator.T("Mouse button %s", number)
	}
	return key
}

func keyboardLit(chord hotkey.Chord) map[string]bool {
	lit := map[string]bool{}
	for family, side := range chord.Mods {
		if family == proto.ModFn {
			lit["fn"] = true
			continue
		}
		if side != proto.SideRight {
			lit[family+"-left"] = true
		}
		if side != proto.SideLeft {
			lit[family+"-right"] = true
		}
	}
	if chord.Key == "Space" {
		lit["space"] = true
	}
	return lit
}

func onboardingKeyboard(keys string) onboardingKeyboardViewState {
	chord, err := hotkey.Parse(keys)
	if err != nil || chord.Mouse() {
		return onboardingKeyboardViewState{}
	}
	state := onboardingKeyboardViewState{Visible: true, LitKeys: keyboardLit(chord)}
	if chord.Key != "" && chord.Key != "Space" {
		state.ExtraKey = chord.Key
	}
	return state
}

func buildOnboardingTry(translator *i18n.Translator, input onboardingViewStateInput) onboardingTryViewState {
	kind, _ := obStepKind(input.Step)
	feature := string(kind)
	prefix := "ob-" + uitext.FeaturePrefix(feature)
	state := onboardingTryViewState{Prefix: prefix, Passed: input.Passed[input.Step], Hint: input.Hint, Conflicts: onboardingConflictLines(translator, input.Conflicts, feature, 0)}
	hotkeys := hotkey.HotkeysOf(input.Settings.Features, feature)
	state.Shortcut.FeatureTitle = onboardingKindTitle(translator, kind)
	if len(hotkeys) > 0 {
		state.Shortcut.Present = true
		state.Shortcut.Text = input.TryKeys
		state.Shortcut.AccessibilityWarning = hotkey.NeedsTap(hotkeys[0]) && !input.HotkeysInstalled
		state.Shortcut.Keyboard = onboardingKeyboard(hotkeys[0].Keys)
		switch hotkeys[0].Trigger {
		case settings.TriggerDouble:
			state.Shortcut.How = translator.T("Press twice quickly to start; once more to stop.")
		case settings.TriggerLong:
			state.Shortcut.How = translator.T("Record while you hold the keys; release to stop.")
		default:
			state.Shortcut.How = translator.T("Press to start, press again to stop.")
		}
	}
	switch kind {
	case store.KindTranscribe:
		state.SampleTitle = translator.T("Read this, or say anything")
		state.SampleText = translator.T("Um, so the design review moves to Thursday afternoon, no, Friday morning, and I'll send the agenda tonight.")
	case store.KindTranslate:
		state.SampleTitle = translator.T("Sample text · into %s", input.TargetLanguage)
		state.SampleText = translator.T("Let's go camping in the mountains this weekend; remember to bring a warm jacket and a torch.")
		state.SampleAction = true
	case store.KindRewrite:
		state.SampleTitle = translator.T("Say, for example")
		state.SampleText = translator.T("Write a short note to the team saying I'll be ten minutes late to the stand-up.")
	}
	if input.Blocker != nil && input.Blocker.action == actionLanguage {
		detail := translator.T("No language model is set up.")
		if kind == store.KindTranslate {
			detail = translator.T("No language model is set up. Set one up to translate.")
		} else if kind == store.KindRewrite {
			detail = translator.T("No language model is set up. Set one up to rewrite.")
		}
		state.LanguageGuide = &onboardingNoticeViewState{
			Text: detail, More: translator.T("Set one up in Settings → Language models, then come back and try again. You can also skip this step."),
			Action: actionLanguage, ActionLabel: translator.T("Set up a language model"),
		}
	} else if kind != store.KindTranslate && input.SpeechWait != nil {
		state.SpeechGuide = &onboardingNoticeViewState{
			Text: input.SpeechWait.text,
			More: translator.T("The model downloads in the background. You can continue the guide and try again when the bar fills, or skip this step."),
		}
	}
	snapshot := input.Session
	mine := input.Tries[input.Step] != "" && snapshot.ID == input.Tries[input.Step]
	card := onboardingTryCardViewState{
		Mine: mine, State: string(snapshot.State), SessionKind: string(snapshot.Kind), Live: snapshot.State.Live(), Ended: snapshot.State.Ended(),
		Recording: snapshot.State == session.Recording, Composing: snapshot.State == session.Composing,
		Cancelled: snapshot.State == session.Cancelled, Failed: snapshot.State == session.Failed,
		Levels: snapshot.Levels, Heard: !snapshot.HeardAt.IsZero(), Partial: strings.TrimSpace(snapshot.Partial),
		Block: onboardingNotice(translator, input.Block),
	}
	if mine && card.Recording {
		elapsed := snapshot.Elapsed(input.Now)
		seconds := int(elapsed.Seconds())
		card.ElapsedText = fmt.Sprintf("%d:%02d", seconds/60, seconds%60)
	}
	if mine && snapshot.State.Live() && !card.Recording {
		card.WorkingText = onboardingWorkingText(translator, snapshot)
		if card.Composing {
			card.Draft = strings.TrimSpace(snapshot.Draft)
		}
	}
	if snapshot.State.Live() {
		card.BusyText = translator.T("Another task is running.")
		if snapshot.Kind == store.KindTranscribe {
			card.BusyText = translator.T("A transcription is running.")
		}
	}
	if mine && snapshot.State.Ended() {
		card.Result, card.Raw = snapshot.Result, strings.TrimSpace(snapshot.Raw)
		card.ShowRaw = card.Raw != "" && card.Raw != strings.TrimSpace(snapshot.Result)
		card.RawTitle = translator.T("As recognised")
		if snapshot.Kind == store.KindRewrite {
			card.RawTitle = translator.T("Your instruction")
		}
		card.Timings = onboardingTimingsText(translator, snapshot)
		if snapshot.Problem != nil {
			card.ProblemCode = snapshot.Problem.Code
		}
		problem := onboardingNotice(translator, input.SessionProblem)
		switch {
		case card.Cancelled:
		case card.Failed && problem != nil && problem.Text != "":
			card.Failure = problem
		case card.Failed:
			card.FailureText = translator.T("Nothing was heard.")
		default:
			if problem != nil && problem.Text != "" {
				card.Warning = problem
			}
			card.ShowResult = true
		}
	}
	state.Card = card
	return state
}
