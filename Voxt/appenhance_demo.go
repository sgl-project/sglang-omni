package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func (host *meetingDemoHost) setAppEnhancementEnabled(on bool) {
	host.record("setAppEnhancementEnabled %t", on)
}
func (host *meetingDemoHost) addAppEnhancementGroup() { host.record("addAppEnhancementGroup") }
func (host *meetingDemoHost) deleteAppEnhancementGroup(id string) {
	host.appEnhancementUI.delAsk = false
	host.record("deleteAppEnhancementGroup %s", id)
}
func (host *meetingDemoHost) openAppEnhancementApps() {
	host.appEnhancementUI.appsOpen = true
	host.record("openAppEnhancementApps")
}
func (host *meetingDemoHost) openAppEnhancementBrowsers() {
	host.appEnhancementUI.browsersOpen = true
	host.record("openAppEnhancementBrowsers")
}
func (host *meetingDemoHost) setAppEnhancementGroupName(id, name string) {
	host.record("setAppEnhancementGroupName %s %q", id, name)
}
func (host *meetingDemoHost) setAppEnhancementGroupEnabled(id string, on bool) {
	host.record("setAppEnhancementGroupEnabled %s %t", id, on)
}
func (host *meetingDemoHost) setAppEnhancementGroupFeature(id, feature string, on bool) {
	host.record("setAppEnhancementGroupFeature %s %s %t", id, feature, on)
}
func (host *meetingDemoHost) setAppEnhancementGroupTranscribeCleanup(id, value string) {
	host.record("setAppEnhancementGroupTranscribeCleanup %s %s", id, value)
}
func (host *meetingDemoHost) setAppEnhancementGroupPrompt(id, prompt string) {
	host.record("setAppEnhancementGroupPrompt %s %q", id, prompt)
}
func (host *meetingDemoHost) copyAppEnhancementVariable(token string) {
	host.record("copyAppEnhancementVariable %q", token)
}
func (host *meetingDemoHost) toggleAppEnhancementApp(bundleID string) {
	host.record("toggleAppEnhancementApp %s", bundleID)
}
func (host *meetingDemoHost) removeAppEnhancementApp(id, bundleID string) {
	host.record("removeAppEnhancementApp %s %s", id, bundleID)
}
func (host *meetingDemoHost) addAppEnhancementBundle(id, bundleID string) {
	host.appEnhancementUI.bundleIn = ""
	host.record("addAppEnhancementBundle %s %s", id, bundleID)
}
func (host *meetingDemoHost) addAppEnhancementURL(id, pattern string) {
	host.record("addAppEnhancementURL %s %q", id, pattern)
}
func (host *meetingDemoHost) removeAppEnhancementURL(id string, index int) {
	host.record("removeAppEnhancementURL %s %d", id, index)
}
func (host *meetingDemoHost) setAppEnhancementAutoKey(id, keys string) {
	host.appEnhancementUI.keyCustom, host.appEnhancementUI.keyErr = false, ""
	host.record("setAppEnhancementAutoKey %s %q", id, keys)
}
func (host *meetingDemoHost) commitAppEnhancementCustomAutoKey(id, keys string) {
	host.record("commitAppEnhancementCustomAutoKey %s %q", id, keys)
}
func (host *meetingDemoHost) addAppEnhancementBrowser(bundleID, name string) {
	host.appEnhancementUI.brID, host.appEnhancementUI.brName = "", ""
	host.record("addAppEnhancementBrowser %s %q", strings.TrimSpace(bundleID), strings.TrimSpace(name))
}
func (host *meetingDemoHost) removeAppEnhancementBrowser(bundleID string) {
	host.record("removeAppEnhancementBrowser %s", bundleID)
}
func (host *meetingDemoHost) refreshAppEnhancementPreview() {
	host.record("refreshAppEnhancementPreview")
}
func (host *meetingDemoHost) addAppEnhancementPreviewApp() {
	host.record("addAppEnhancementPreviewApp")
}
func (host *meetingDemoHost) addAppEnhancementPreviewSite() {
	host.record("addAppEnhancementPreviewSite")
}
func (host *meetingDemoHost) appEnhancementIcon(ref appEnhancementIconRef, want bool) (*appIconBitmap, bool) {
	return nil, true
}

func (host *meetingDemoHost) appEnhancementState() *appEnhancementViewState {
	fixture := host.current()
	state := *fixture.state.AppEnhancement
	local := &host.appEnhancementUI
	if local.sel != state.Selected {
		state.Selected = local.sel
		state.Groups = append([]appEnhancementGroupRow(nil), state.Groups...)
		for index := range state.Groups {
			state.Groups[index].Selected = state.Groups[index].ID == local.sel
		}
		for _, detail := range host.appEnhancementDetails[fixture.locale] {
			if detail.ID == local.sel {
				detail := detail
				state.Detail = &detail
				break
			}
		}
	}
	synthesizedApps := false
	if !local.appsOpen {
		state.Apps = nil
	} else if state.Apps == nil && state.Detail != nil {
		synthesizedApps = true
		state.Apps = &appEnhancementAppsDialog{
			GroupID: state.Detail.ID, Title: i18n.T("Add apps to %s", state.Detail.Name), BundleInput: local.bundleIn,
		}
	}
	if state.Apps != nil && (synthesizedApps || local.appsQuery != state.Apps.Query) {
		catalog := state.Apps.Rows
		if library := host.appEnhancementApps[fixture.locale]; len(library) > 0 && (synthesizedApps || state.Apps.HasCatalog) {
			catalog = library
		}
		catalog = append([]appEnhancementAppChoice(nil), catalog...)
		for index := range catalog {
			catalog[index].Member = false
			if state.Detail != nil {
				for _, app := range state.Detail.Apps {
					if strings.EqualFold(app.Ref.BundleID, catalog[index].Ref.BundleID) {
						catalog[index].Member = true
						break
					}
				}
			}
		}
		apps := *state.Apps
		apps.Rows = filterAppEnhancementApps(catalog, local.appsQuery)
		apps.HasCatalog = len(catalog) > 0
		apps.NoMatches = len(apps.Rows) == 0 && !apps.Loading && apps.HasCatalog
		apps.Query = local.appsQuery
		state.Apps = &apps
	}
	if !local.browsersOpen {
		state.Browsers = nil
	} else if state.Browsers == nil {
		state.Browsers = &appEnhancementBrowsersDialog{BundleInput: local.brID, NameInput: local.brName, AddDisabled: true}
	}
	if !local.delAsk {
		state.Delete = nil
	} else if state.Delete == nil && state.Detail != nil {
		state.Delete = &appEnhancementDeleteDialog{GroupID: state.Detail.ID, Title: i18n.T("Delete “%s”?", state.Detail.Name)}
	}
	appEnhancementPreviewActionsFor(i18n.CurrentTranslator(), &state.Preview, state.Detail)
	return &state

}
