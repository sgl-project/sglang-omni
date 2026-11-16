package main

import (
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type appEnhancementCatalog struct {
	apps    []proto.AppInfo
	loading bool
	err     string
	seq     int
}

func (a *app) aeShown() { a.refreshPreview(true) }

func (a *app) aeGroup() (appctx.Group, int) {
	for index, group := range a.aeRules.Groups {
		if group.ID == a.aeUI.sel {
			return group, index
		}
	}
	if len(a.aeRules.Groups) > 0 {
		a.aeUI.sel = a.aeRules.Groups[0].ID
		return a.aeRules.Groups[0], 0
	}
	return appctx.Group{}, -1
}

func (a *app) updateGroup(id string, change func(*appctx.Group)) {
	rules := cloneRules(a.aeRules)
	for index := range rules.Groups {
		if rules.Groups[index].ID == id {
			change(&rules.Groups[index])
		}
	}
	a.setRules(rules)
}

func (a *app) appEnhancementModel() appEnhancementViewState {
	group, index := a.aeGroup()
	if index < 0 {
		a.aeUI.appsOpen, a.aeUI.browsersOpen, a.aeUI.delAsk = false, false, false
	}
	selected := ""
	if index >= 0 {
		selected = group.ID
	}
	return buildAppEnhancementViewState(i18n.CurrentTranslator(), appEnhancementViewStateInput{
		Enabled: a.prefs.AppEnhancement.Enabled, Error: a.aeErr, Rules: a.aeRules, SelectedID: selected,
		CustomBrowsers: a.prefs.AppEnhancement.Browsers, PreviewApp: a.previewApp(), Preview: a.aePrev,
		HelperReady: a.helperReady(), Apps: a.aeCatalog.apps, URLReads: a.urlReads,
		AppsLoading: a.aeCatalog.loading, AppsError: a.aeCatalog.err,
		AppsOpen: a.aeUI.appsOpen, AppsQuery: a.aeUI.appsQuery, BundleInput: a.aeUI.bundleIn,
		BrowsersOpen: a.aeUI.browsersOpen, BrowserIDInput: a.aeUI.brID, BrowserNameInput: a.aeUI.brName,
		DeleteAsk: a.aeUI.delAsk, URLInput: a.aeUI.urlIn, URLError: a.aeUI.urlErr,
		AutoKeyCustom: a.aeUI.keyCustom, AutoKeyText: a.aeUI.keyText, AutoKeyError: a.aeUI.keyErr,
	})
}

func (a *app) setAppEnhancementEnabled(on bool) {
	a.updateSettings(func(settings *settings.Settings) { settings.AppEnhancement.Enabled = on })
}

func (a *app) addAppEnhancementGroup() {
	rules := cloneRules(a.aeRules)
	name := i18n.T("Group %d", len(rules.Groups)+1)
	group := appctx.NewGroup(newGroupID(), name)
	rules.Groups = append(rules.Groups, group)
	a.setRules(rules)
	a.aeUI.sel = group.ID
	a.aeUI.editFor = ""
	a.aeUI.list.open = false
}

func (a *app) deleteAppEnhancementGroup(id string) {
	rules := cloneRules(a.aeRules)
	rules.Groups = slices.DeleteFunc(rules.Groups, func(group appctx.Group) bool { return group.ID == id })
	a.setRules(rules)
	a.aeUI.sel, a.aeUI.editFor = "", ""
	a.toast(i18n.T("Group deleted"))
}

func (a *app) openAppEnhancementApps() {
	a.aeUI.appsOpen, a.aeUI.appsQuery, a.aeUI.bundleIn = true, "", ""
	a.loadApps()
}

func (a *app) openAppEnhancementBrowsers() {
	a.aeUI.browsersOpen = true
	if len(a.aeCatalog.apps) == 0 {
		a.loadApps()
	}
}

func (a *app) loadApps() {
	a.aeCatalog.seq++
	sequence := a.aeCatalog.seq
	if !a.helperReady() {
		a.aeCatalog.loading = false
		a.aeCatalog.err = i18n.T("The helper is not running. Add an app by its bundle ID instead.")
		return
	}
	a.aeCatalog.loading, a.aeCatalog.err = true, ""
	helperCall(a, proto.MethodAppList, proto.AppListParams{Installed: true}, 10*time.Second, func(result proto.AppListResult, err error) {
		if a.aeCatalog.seq != sequence {
			return
		}
		a.aeCatalog.loading = false
		if err != nil {
			a.aeCatalog.err = err.Error()
			return
		}
		a.aeCatalog.apps = slices.DeleteFunc(result.Apps, func(info proto.AppInfo) bool {
			return isSelf(proto.App{PID: info.PID, BundleID: info.BundleID})
		})
	})
}

func (a *app) setAppEnhancementGroupName(id, name string) {
	a.updateGroup(id, func(group *appctx.Group) { group.Name = name })
}

func (a *app) setAppEnhancementGroupEnabled(id string, on bool) {
	a.updateGroup(id, func(group *appctx.Group) { group.Enabled = on })
}

func (a *app) setAppEnhancementGroupFeature(id, feature string, on bool) {
	a.updateGroup(id, func(group *appctx.Group) {
		switch feature {
		case appctx.Translate:
			group.Features.Translate = on
		case appctx.Rewrite:
			group.Features.Rewrite = on
		default:
			group.Features.Transcribe = on
		}
	})
}

func (a *app) setAppEnhancementGroupPrompt(id, prompt string) {
	a.updateGroup(id, func(group *appctx.Group) { group.Prompt = prompt })
}

func (a *app) copyAppEnhancementVariable(token string) { a.copyHistoryText(token) }

func (a *app) toggleAppEnhancementApp(bundleID string) {
	group, index := a.aeGroup()
	if index < 0 {
		return
	}
	var info proto.AppInfo
	for _, candidate := range a.aeCatalog.apps {
		if strings.EqualFold(candidate.BundleID, bundleID) {
			info = candidate
			break
		}
	}
	a.updateGroup(group.ID, func(group *appctx.Group) {
		if group.HasApp(bundleID) {
			group.Apps = slices.DeleteFunc(group.Apps, func(app appctx.App) bool { return strings.EqualFold(app.BundleID, bundleID) })
			return
		}
		group.Apps = append(group.Apps, appctx.App{BundleID: bundleID, Name: info.Name})
	})
}

func (a *app) removeAppEnhancementApp(id, bundleID string) {
	a.updateGroup(id, func(group *appctx.Group) {
		group.Apps = slices.DeleteFunc(group.Apps, func(app appctx.App) bool { return strings.EqualFold(app.BundleID, bundleID) })
	})
}

func (a *app) addAppEnhancementBundle(id, bundleID string) {
	bundleID = strings.TrimSpace(bundleID)
	if bundleID == "" || strings.ContainsAny(bundleID, " /") {
		return
	}
	a.updateGroup(id, func(group *appctx.Group) {
		if !group.HasApp(bundleID) {
			group.Apps = append(group.Apps, appctx.App{BundleID: bundleID})
		}
	})
	a.aeUI.bundleIn = ""
}

func (a *app) addAppEnhancementURL(id, pattern string) {
	pattern = strings.TrimSpace(pattern)
	if pattern == "" {
		return
	}
	if err := appctx.CheckPattern(pattern); err != nil {
		a.aeUI.urlErr = uitext.PatternErrorText(err)
		return
	}
	group, ok := a.aeRules.Group(id)
	if !ok {
		return
	}
	if slices.ContainsFunc(group.URLs, func(existing string) bool {
		return appctx.CanonicalPattern(existing) == appctx.CanonicalPattern(pattern)
	}) {
		a.aeUI.urlErr = i18n.T("This rule is in the group already.")
		return
	}
	a.updateGroup(id, func(group *appctx.Group) { group.URLs = append(group.URLs, pattern) })
	a.aeUI.urlIn, a.aeUI.urlErr = "", ""
}

func (a *app) removeAppEnhancementURL(id string, index int) {
	a.updateGroup(id, func(group *appctx.Group) {
		if index < len(group.URLs) {
			group.URLs = slices.Delete(group.URLs, index, index+1)
		}
	})
}

func (a *app) setAppEnhancementAutoKey(id, keys string) {
	a.aeUI.keyCustom, a.aeUI.keyErr = false, ""
	a.updateGroup(id, func(group *appctx.Group) { group.AutoKey = keys })
}

func (a *app) commitAppEnhancementCustomAutoKey(id, keys string) {
	if _, err := appctx.ParseAutoKey(keys); err != nil || strings.TrimSpace(keys) == "" {
		a.aeUI.keyErr = i18n.T("Enter one key, with or without modifiers: Return, Command+Return, Tab.")
		return
	}
	canonical := appctx.CanonicalAutoKey(keys)
	a.aeUI.keyErr, a.aeUI.keyCustom, a.aeUI.keySel = "", false, uitext.AutoKeyLabel(canonical)
	a.updateGroup(id, func(group *appctx.Group) { group.AutoKey = canonical })
}

func (a *app) addAppEnhancementBrowser(bundleID, name string) {
	bundleID, name = strings.TrimSpace(bundleID), strings.TrimSpace(name)
	if bundleID == "" || strings.ContainsAny(bundleID, " /") || appctx.IsBrowser(bundleID, a.prefs.AppEnhancement.Browsers) {
		return
	}
	a.updateSettings(func(current *settings.Settings) {
		current.AppEnhancement.Browsers = append(slices.Clone(current.AppEnhancement.Browsers), settings.Browser{BundleID: bundleID, Name: name})
	})
	a.aeUI.brID, a.aeUI.brName = "", ""
}

func (a *app) removeAppEnhancementBrowser(bundleID string) {
	a.updateSettings(func(current *settings.Settings) {
		current.AppEnhancement.Browsers = slices.DeleteFunc(slices.Clone(current.AppEnhancement.Browsers), func(browser settings.Browser) bool {
			return strings.EqualFold(browser.BundleID, bundleID)
		})
	})
}

func (a *app) refreshAppEnhancementPreview() { a.refreshPreview(true) }

func (a *app) addAppEnhancementPreviewApp() {
	preview := a.previewApp()
	group, index := a.aeGroup()
	if index < 0 || preview.BundleID == "" || group.HasApp(preview.BundleID) {
		return
	}
	a.updateGroup(group.ID, func(group *appctx.Group) {
		group.Apps = append(group.Apps, appctx.App{BundleID: preview.BundleID, Name: preview.Name})
	})
}

func (a *app) addAppEnhancementPreviewSite() {
	address, ok := appctx.ParseAddress(a.aePrev.url)
	if !ok {
		return
	}
	group, index := a.aeGroup()
	if index < 0 {
		return
	}
	a.updateGroup(group.ID, func(group *appctx.Group) { group.URLs = append(group.URLs, address.Host) })
}
