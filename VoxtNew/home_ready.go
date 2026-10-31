package main

// Home's readiness list (home_page.go): what is missing before the first
// session (permissions, models, shortcuts), shown until all of it is
// ready.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// ---- The page ----

// readiness is one thing VoxtNew needs, as Home lists it.
type readiness struct {
	key   string
	title string // what it is
	ok    bool
	warn  bool   // ok, but it waits for something another line asks for
	text  string // ok: what is used; else what is missing
	kind  noticeKind
	// action names the way to fix it ("" for none), which fix takes: the
	// whole line is its button.
	action string
	fix    func()
}

// readinessList says what is ready and what is missing (memory only). A
// helper that is starting is neither: it is ready in a moment.
func (a *app) readinessList() []readiness {
	var out []readiness
	openModels := func() { a.navigate(modelsPath) }
	openPerms := func() { a.navigate("/settings/permissions") }
	ma, okA := a.featureModel(local.ASR, hotkey.Transcribe)
	ml, okL := a.featureModel(local.LLM, hotkey.Transcribe)
	switch {
	case !okA && !okL:
		// Nothing installed: one line, one way.
		out = append(out, readiness{key: "home-models", title: i18n.T("Models"), kind: noticeWarn,
			text: i18n.T("No speech model or language model is installed: install one of each to start."), action: i18n.T("Open Local models"), fix: openModels})
	case !okA:
		out = append(out, readiness{key: "home-asr", title: i18n.T("Speech recognition"), kind: noticeWarn,
			text: i18n.T("No speech model is installed: install one to transcribe, translate and rewrite by voice."), action: i18n.T("Open Local models"), fix: openModels})
	default:
		out = append(out, readiness{key: "home-asr", title: i18n.T("Speech recognition"), ok: true, text: ma.Name})
	}
	switch {
	case !okA && !okL:
	case !okL:
		out = append(out, readiness{key: "home-llm", title: i18n.T("Language model"), kind: noticeWarn,
			text: i18n.T("No language model is installed: Translate and Rewrite need one, and Transcribe types the text as recognised."), action: i18n.T("Open Local models"), fix: openModels})
	default:
		out = append(out, readiness{key: "home-llm", title: i18n.T("Language model"), ok: true, text: ml.Name})
	}
	// The Microphone.
	switch {
	case !a.perms.checked:
	case a.perms.mic == permission.Granted:
		out = append(out, readiness{key: "home-mic", title: i18n.T("Microphone"), ok: true, text: i18n.T("Allowed")})
	case a.perms.mic == permission.NotDetermined:
		out = append(out, readiness{key: "home-mic", title: i18n.T("Microphone"), kind: noticeInfo,
			text: i18n.T("macOS asks for the microphone the first time you record."), action: i18n.T("Ask now"),
			fix: func() { a.askPermission(proto.PermissionMicrophone) }})
	default:
		out = append(out, readiness{key: "home-mic", title: i18n.T("Microphone"), kind: noticeError,
			text: i18n.T("VoxtNew may not use the microphone: allow it in System Settings."), action: i18n.T("Open Permissions"), fix: openPerms})
	}
	// Accessibility.
	switch {
	case !a.perms.checked:
	case a.perms.access:
		out = append(out, readiness{key: "home-access", title: i18n.T("Accessibility"), ok: true, text: i18n.T("Allowed")})
	default:
		out = append(out, readiness{key: "home-access", title: i18n.T("Accessibility"), kind: noticeWarn,
			text: i18n.T("Without Accessibility results are copied for you to paste, and shortcuts such as Right ⌥ do not work."), action: i18n.T("Open Permissions"), fix: openPerms})
	}
	// The shortcut.
	out = append(out, a.shortcutReadiness())
	// The helper.
	switch st := a.helperStatus(); {
	case st.State == helper.Ready, st.State == helper.Starting:
	case st.Reason == helper.FailUpdated:
		out = append(out, readiness{key: "home-helper", title: i18n.T("Helper"), kind: noticeWarn,
			text: uitext.HelperFailText(st), action: i18n.T("Restart VoxtNew"), fix: a.relaunch})
	default:
		out = append(out, readiness{key: "home-helper", title: i18n.T("Helper"), kind: noticeError,
			text: uitext.HelperFailText(st), action: i18n.T("Restart the helper"), fix: a.restartHelper})
	}
	return out
}

// shortcutReadiness says whether Transcribe's shortcut works: there is
// one, nothing else takes its keys, and the event tap it may need runs.
func (a *app) shortcutReadiness() readiness {
	r := readiness{key: "home-keys", title: i18n.T("Shortcut")}
	openTr := func() { a.navigate("/transcribe") }
	hs := hotkey.HotkeysOf(a.prefs.Features, hotkey.Transcribe)
	if len(hs) == 0 {
		r.kind, r.text, r.action, r.fix = noticeWarn, i18n.T("Transcribe has no shortcut: add one, or start from the menu bar."), i18n.T("Open Transcribe"), openTr
		return r
	}
	// Its own problems only: a clash is reported on the later binding,
	// and the first one (Transcribe's) wins.
	for _, p := range hotkey.Check(a.prefs.Features) {
		if p.Feature == hotkey.Transcribe && p.Index == 0 {
			r.kind, r.text, r.action, r.fix = noticeWarn, i18n.T("Transcribe's shortcut %s: %s", uitext.HotkeyText(hs[0].Keys), uitext.HotkeyProblemText(p)), i18n.T("Open Transcribe"), openTr
			return r
		}
	}
	r.ok, r.text = true, uitext.HotkeyText(hs[0].Keys)
	if hotkey.NeedsTap(hs[0]) && !a.hotkeys.Installed {
		// The Accessibility line says what to do; this one says it waits.
		r.warn = true
		r.text = i18n.T("%s (once Accessibility is allowed)", uitext.HotkeyText(hs[0].Keys))
	}
	return r
}

// homeMissing is what is still missing, from the readiness list.
func homeMissing(list []readiness) (missing, ready []readiness) {
	for _, r := range list {
		if r.ok {
			ready = append(ready, r)
		} else {
			missing = append(missing, r)
		}
	}
	return missing, ready
}

// homeReadiness is what is still missing, a line each (the whole line
// fixes it), and below it, quieter, what is ready. Nothing when all is
// ready: Home is then the day's numbers and the recent results.
func (a *app) homeReadiness(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	missing, ready := homeMissing(a.readinessList())
	if len(missing) == 0 {
		return
	}
	sectionTitle(c, i18n.N(len(missing), "%d step to go", "%d steps to go", len(missing)))
	card(c.Key("home-missing")).Padding(px(t, sp1)).Gap(0).Children(func() {
		for _, r := range missing {
			ic, col := icons.InfoCircle, k.Text2
			switch r.kind {
			case noticeWarn:
				ic, col = icons.DangerTriangle, k.Warning
			case noticeError:
				ic, col = icons.DangerCircle, k.Danger
			}
			row := ui.ButtonBase(c.Key(r.key)).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).
				Gap(px(t, sp3)).AlignItems(ui.Center).Label(i18n.T("%s: %s", r.title, r.action)).Tooltip(r.action).Disabled(r.fix == nil)
			if row.Hovered() && r.fix != nil {
				row.Background(k.Hover)
			}
			row.Children(func() {
				ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(col).Shrink(0)
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
					ui.Text(c, r.title).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0)
					ui.Text(c, r.text).FontSize(px(t, fsCaption)).TextColor(k.Text2).MinWidth(0)
				})
				if r.fix != nil {
					ui.Row(c).Gap(px(t, 2)).AlignItems(ui.Center).Shrink(0).Children(func() {
						ui.Text(c, r.action).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine()
						ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3)
					})
				}
			})
			if row.Clicked() && r.fix != nil {
				r.fix()
			}
		}
	})
	if len(ready) == 0 {
		return
	}
	// What is ready, quieter: a mark and a name each.
	ui.Row(c.Key("home-ready")).Wrap().GapX(px(t, sp4)).GapY(px(t, sp1)).Padding(0, px(t, sp1)).Children(func() {
		for _, r := range ready {
			ic, col := icons.CheckCircle, k.Success
			if r.warn {
				ic, col = icons.DangerTriangle, k.Warning
			}
			ui.Row(c.Key(r.key)).Gap(px(t, 6)).AlignItems(ui.Center).MinWidth(0).Tooltip(r.text).Children(func() {
				ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0)
				ui.Text(c, r.title).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine()
				ui.Text(c, r.text).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0).MaxWidth(px(t, 220))
			})
		}
	})
}
