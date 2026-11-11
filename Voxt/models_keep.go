package main

// How long the speech models' servers keep a model loaded unused
// (settings.Models, the Speech models page's Advanced section): Automatic
// leaves their own idle time (10 min); the other choices set another, or
// keep them until VoxtNew quits. A change applies to the running servers
// at once (their idle timers start again). The ActNow family's memory rules stay above it: another app's
// memory request and the budget before a load unload idle models
// whatever the choice (family_runtime.go, internal/actnow).

import (
	"slices"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// keepIdle is the idle time of the speech models' servers that m sets.
func keepIdle(m settings.Models) time.Duration {
	if d, auto := m.KeepFor(); !auto {
		return d
	}
	return omni.DefaultIdleTimeout
}

// keepSetting is the saved choice (goroutine-safe: the runtimes are made
// on sessions' goroutines).
func (a *app) keepSetting() settings.Models {
	if a.settings == nil {
		return settings.Models{}
	}
	return a.settings.Get().Models
}

// applyKeepLoaded gives the running servers m's idle time
// (settingsChanged).
func (a *app) applyKeepLoaded(m settings.Models) {
	idle := keepIdle(m)
	a.async(func() {
		a.omniMu.Lock()
		rt := a.omniRT
		a.omniMu.Unlock()
		if rt != nil {
			rt.SetIdleTimeout(idle)
		}
	})
}

// keepLabels are the choices' labels, in settings.KeepChoices' order.
func keepLabels() []string {
	return []string{
		i18n.T("Automatic"), i18n.T("5 minutes"), i18n.T("15 minutes"),
		i18n.T("30 minutes"), i18n.T("1 hour"), i18n.T("Until Voxt quits"),
	}
}

// keepLabel is the label of choice v (an unknown one is Automatic's).
func keepLabel(v string) string {
	i := max(slices.Index(settings.KeepChoices, v), 0)
	return keepLabels()[i]
}

// modelsAdvanced is the Speech models page's Advanced section, folded
// until opened: how long models stay loaded.
func (a *app) modelsAdvanced(c *ui.Context) {
	t := c.Theme()
	collapsible(c.Key("models-advanced"), i18n.T("Advanced"), &a.modelAdvanced, nil, func() {
		help := i18n.T("Loaded models start at once but keep using memory. Automatic unloads them after 10 minutes unused.")
		settingRow(c, "models-keep", i18n.T("Keep models loaded"), help, func() {
			// The label follows the setting and the language, unless a
			// choice of this select is on its way to the setting.
			v := a.prefs.Models.KeepLoaded
			if want := v + "\x00" + i18n.Locale(); a.modelKeepFor != want {
				a.modelKeepSel, a.modelKeepFor = keepLabel(v), want
			}
			opts := keepLabels()
			if ui.Select(c.Key("models-keep-select"), &a.modelKeepSel, opts).Label(i18n.T("Keep models loaded")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 140)).Changed() {
				v = settings.KeepChoices[max(slices.Index(opts, a.modelKeepSel), 0)]
				a.modelKeepFor = v + "\x00" + i18n.Locale()
				a.updateSettings(func(s *settings.Settings) { s.Models.KeepLoaded = v })
			}
		})
	})
}
