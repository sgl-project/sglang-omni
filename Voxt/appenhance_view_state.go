package main

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type appEnhancementIconRef struct {
	BundleID string `json:"bundle_id"`
	Name     string `json:"name,omitempty"`
	Path     string `json:"path,omitempty"`
}

type appEnhancementGroupRow struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Summary  string `json:"summary"`
	Enabled  bool   `json:"enabled"`
	Selected bool   `json:"selected"`
	Matches  bool   `json:"matches"`
}

type appEnhancementFeatureOption struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Checked bool   `json:"checked"`
}

type appEnhancementVariable struct {
	Name    string `json:"name"`
	Token   string `json:"token"`
	Tooltip string `json:"tooltip"`
}

type appEnhancementAppRow struct {
	Ref     appEnhancementIconRef `json:"ref"`
	Browser bool                  `json:"browser"`
}

type appEnhancementURLRow struct {
	Text        string `json:"text"`
	Canonical   string `json:"canonical,omitempty"`
	Warning     string `json:"warning,omitempty"`
	WarningCode string `json:"warning_code,omitempty"`
}

type appEnhancementAutoKeyOption struct {
	Label string `json:"label"`
	Value string `json:"value"`
}

type appEnhancementAutoKey struct {
	Options      []appEnhancementAutoKeyOption `json:"options"`
	Value        string                        `json:"value"`
	CurrentLabel string                        `json:"current_label"`
	CustomLabel  string                        `json:"custom_label"`
	Custom       bool                          `json:"custom"`
	Text         string                        `json:"text"`
	Error        string                        `json:"error,omitempty"`
	ErrorCode    string                        `json:"error_code,omitempty"`
}

type appEnhancementGroupDetail struct {
	ID        string                        `json:"id"`
	Name      string                        `json:"name"`
	Enabled   bool                          `json:"enabled"`
	Prompt    string                        `json:"prompt"`
	Features  []appEnhancementFeatureOption `json:"features"`
	Variables []appEnhancementVariable      `json:"variables"`
	Apps      []appEnhancementAppRow        `json:"apps"`
	URLs      []appEnhancementURLRow        `json:"urls"`
	URLInput  string                        `json:"url_input"`
	URLError  string                        `json:"url_error,omitempty"`
	AutoKey   appEnhancementAutoKey         `json:"auto_key"`
}

type appEnhancementPreview struct {
	App           appEnhancementIconRef `json:"app"`
	Empty         bool                  `json:"empty"`
	HasBundleID   bool                  `json:"has_bundle_id"`
	FrontText     string                `json:"front_text"`
	Browser       bool                  `json:"browser"`
	Address       string                `json:"address,omitempty"`
	AddressStatus string                `json:"address_status"`
	Matched       bool                  `json:"matched"`
	MatchSource   string                `json:"match_source"`
	MatchPattern  string                `json:"match_pattern,omitempty"`
	MatchText     string                `json:"match_text"`
	CanRefresh    bool                  `json:"can_refresh"`
	AddApp        bool                  `json:"add_app"`
	AddAppLabel   string                `json:"add_app_label,omitempty"`
	AddSite       bool                  `json:"add_site"`
	AddSiteLabel  string                `json:"add_site_label,omitempty"`
}

type appEnhancementAppChoice struct {
	Ref     appEnhancementIconRef `json:"ref"`
	Running bool                  `json:"running"`
	Member  bool                  `json:"member"`
}

type appEnhancementAppsDialog struct {
	GroupID     string                    `json:"group_id"`
	Title       string                    `json:"title"`
	Loading     bool                      `json:"loading"`
	Error       string                    `json:"error,omitempty"`
	Rows        []appEnhancementAppChoice `json:"rows"`
	HasCatalog  bool                      `json:"has_catalog"`
	Query       string                    `json:"query"`
	BundleInput string                    `json:"bundle_input"`
}

type appEnhancementBrowserRow struct {
	Ref        appEnhancementIconRef `json:"ref"`
	Custom     bool                  `json:"custom"`
	Engine     string                `json:"engine"`
	StatusText string                `json:"status_text"`
	StatusTone string                `json:"status_tone"`
}

type appEnhancementBrowsersDialog struct {
	Rows        []appEnhancementBrowserRow `json:"rows"`
	OthersText  string                     `json:"others_text,omitempty"`
	BundleInput string                     `json:"bundle_input"`
	NameInput   string                     `json:"name_input"`
	AddDisabled bool                       `json:"add_disabled"`
}

type appEnhancementDeleteDialog struct {
	GroupID string `json:"group_id"`
	Title   string `json:"title"`
}

type appEnhancementViewState struct {
	Enabled     bool                          `json:"enabled"`
	EnabledText string                        `json:"enabled_text"`
	Error       string                        `json:"error,omitempty"`
	Selected    string                        `json:"selected"`
	Groups      []appEnhancementGroupRow      `json:"groups"`
	Detail      *appEnhancementGroupDetail    `json:"detail,omitempty"`
	Preview     appEnhancementPreview         `json:"preview"`
	Apps        *appEnhancementAppsDialog     `json:"apps,omitempty"`
	Browsers    *appEnhancementBrowsersDialog `json:"browsers,omitempty"`
	Delete      *appEnhancementDeleteDialog   `json:"delete,omitempty"`
}

type appEnhancementViewStateInput struct {
	Enabled           bool
	Error             string
	Rules             appctx.Rules
	SelectedID        string
	EditorGroupID     string
	CustomBrowsers    []settings.Browser
	PreviewApp        proto.App
	Preview           aePreview
	HelperReady       bool
	Apps              []proto.AppInfo
	URLReads          map[string]urlRead
	AppsLoading       bool
	AppsError         string
	AppsHelperStopped bool
	AppsOpen          bool
	AppsQuery         string
	BundleInput       string
	BrowsersOpen      bool
	BrowserIDInput    string
	BrowserNameInput  string
	DeleteAsk         bool
	URLInput          string
	URLError          string
	AutoKeyCustom     bool
	AutoKeyText       string
	AutoKeyError      string
}

var autoKeys = []string{"Return", "Command+Return", "Shift+Return", "Control+Return", "Option+Return"}

func buildAppEnhancementViewState(translator *i18n.Translator, input appEnhancementViewStateInput) appEnhancementViewState {
	state := appEnhancementViewState{Enabled: input.Enabled, EnabledText: translator.T("Off"), Error: input.Error, Selected: input.SelectedID}
	if input.Enabled {
		state.EnabledText = translator.T("On")
	}
	var selected *appctx.Group
	for index := range input.Rules.Groups {
		group := &input.Rules.Groups[index]
		if group.ID == input.SelectedID {
			selected = group
			break
		}
	}
	if selected == nil && len(input.Rules.Groups) > 0 {
		selected = &input.Rules.Groups[0]
		state.Selected = selected.ID
	}
	match := appEnhancementPreviewMatch(input)
	state.Groups = make([]appEnhancementGroupRow, len(input.Rules.Groups))
	for index, group := range input.Rules.Groups {
		name := group.Name
		if name == "" {
			name = translator.T("Untitled")
		}
		state.Groups[index] = appEnhancementGroupRow{
			ID: group.ID, Name: name, Summary: appEnhancementGroupSummary(translator, group),
			Enabled: group.Enabled, Selected: selected != nil && group.ID == selected.ID, Matches: match.GroupID() == group.ID,
		}
	}
	if selected != nil {
		detailInput := input
		if input.EditorGroupID != selected.ID {
			detailInput.URLInput, detailInput.URLError = "", ""
			detailInput.AutoKeyCustom, detailInput.AutoKeyError = false, ""
			detailInput.AutoKeyText = selected.AutoKey
		}
		detail := buildAppEnhancementGroupDetail(translator, *selected, detailInput)
		state.Detail = &detail
	}
	state.Preview = buildAppEnhancementPreview(translator, input, selected, match)
	if input.AppsOpen && selected != nil {
		apps := buildAppEnhancementAppsDialog(translator, input, *selected)
		state.Apps = &apps
	}
	if input.BrowsersOpen {
		browsers := buildAppEnhancementBrowsersDialog(translator, input)
		state.Browsers = &browsers
	}
	if input.DeleteAsk && selected != nil {
		state.Delete = &appEnhancementDeleteDialog{GroupID: selected.ID, Title: translator.T("Delete “%s”?", selected.Name)}
	}
	return state
}

func appEnhancementPreviewMatch(input appEnhancementViewStateInput) appctx.Match {
	browser := appctx.IsBrowser(input.PreviewApp.BundleID, input.CustomBrowsers)
	context := appctx.Context{App: input.PreviewApp, Browser: browser}
	if browser && input.Preview.app == input.PreviewApp {
		context.URL = input.Preview.url
	}
	return appctx.Resolve(input.Rules, context, appctx.Transcribe)
}

func appEnhancementGroupSummary(translator *i18n.Translator, group appctx.Group) string {
	var parts []string
	if count := len(group.Apps); count > 0 {
		parts = append(parts, translator.N(count, "%d app", "%d apps", count))
	}
	if count := len(group.URLs); count > 0 {
		parts = append(parts, translator.N(count, "%d website", "%d websites", count))
	}
	if len(parts) == 0 {
		return translator.T("Empty")
	}
	return strings.Join(parts, " · ")
}

func buildAppEnhancementGroupDetail(translator *i18n.Translator, group appctx.Group, input appEnhancementViewStateInput) appEnhancementGroupDetail {
	detail := appEnhancementGroupDetail{
		ID: group.ID, Name: group.Name, Enabled: group.Enabled, Prompt: group.Prompt,
		Features: []appEnhancementFeatureOption{
			{ID: appctx.Transcribe, Label: translator.T("Transcribe"), Checked: group.Features.Transcribe},
			{ID: appctx.Translate, Label: translator.T("Translate"), Checked: group.Features.Translate},
			{ID: appctx.Rewrite, Label: translator.T("Rewrite"), Checked: group.Features.Rewrite},
		},
		URLInput: input.URLInput, URLError: input.URLError,
		Apps: make([]appEnhancementAppRow, len(group.Apps)), URLs: make([]appEnhancementURLRow, len(group.URLs)),
	}
	for _, name := range compose.VariableNames {
		detail.Variables = append(detail.Variables, appEnhancementVariable{
			Name: name, Token: "{{" + name + "}}", Tooltip: appEnhancementVariableInfo(translator, name),
		})
	}
	for index, app := range group.Apps {
		name := app.Name
		if name == "" {
			name = app.BundleID
		}
		detail.Apps[index] = appEnhancementAppRow{
			Ref:     appEnhancementIconRef{BundleID: app.BundleID, Name: name, Path: appEnhancementPath(input.Apps, app.BundleID)},
			Browser: appctx.IsBrowser(app.BundleID, input.CustomBrowsers),
		}
	}
	for index, pattern := range group.URLs {
		row := appEnhancementURLRow{Text: pattern}
		if err := appctx.CheckPattern(pattern); err != nil {
			row.Warning = uitext.PatternErrorTextFor(translator, err)
			row.WarningCode = appEnhancementPatternErrorCode(err)
		} else if canonical := appctx.CanonicalPattern(pattern); canonical != strings.ToLower(strings.TrimSpace(pattern)) {
			row.Canonical = translator.T("matches %s", canonical)
		}
		detail.URLs[index] = row
	}
	detail.AutoKey = buildAppEnhancementAutoKey(translator, group.AutoKey, input)
	return detail
}

func appEnhancementVariableInfo(translator *i18n.Translator, name string) string {
	switch name {
	case "USER_MAIN_LANGUAGE":
		return translator.T("Your main language, from General settings. Every feature.")
	case "USER_OTHER_LANGUAGES":
		return translator.T("Your other languages, from General settings. Every feature.")
	case "TARGET_LANGUAGE":
		return translator.T("The Target language menu on the Translate page. Translate only.")
	case "APP_NAME":
		return translator.T("The app in front, when an App enhancement group matches it. The features that group applies to.")
	}
	return ""
}

func buildAppEnhancementAutoKey(translator *i18n.Translator, current string, input appEnhancementViewStateInput) appEnhancementAutoKey {
	text := input.AutoKeyText
	if !input.AutoKeyCustom && text == "" {
		text = current
	}
	state := appEnhancementAutoKey{
		Value: current, CurrentLabel: uitext.AutoKeyLabelFor(translator, current), CustomLabel: translator.T("Custom…"),
		Custom: input.AutoKeyCustom, Text: text, Error: input.AutoKeyError,
	}
	if state.Error != "" {
		state.ErrorCode = "autoKey"
	}
	state.Options = append(state.Options, appEnhancementAutoKeyOption{Label: translator.T("None")})
	for _, keys := range autoKeys {
		state.Options = append(state.Options, appEnhancementAutoKeyOption{Label: uitext.AutoKeyLabelFor(translator, keys), Value: keys})
	}
	canonical := appctx.CanonicalAutoKey(current)
	if canonical != "" && !slices.Contains(autoKeys, canonical) {
		state.Options = append(state.Options, appEnhancementAutoKeyOption{Label: uitext.AutoKeyLabelFor(translator, canonical), Value: canonical})
	}
	state.Options = append(state.Options, appEnhancementAutoKeyOption{Label: state.CustomLabel})
	return state
}

func buildAppEnhancementPreview(translator *i18n.Translator, input appEnhancementViewStateInput, selected *appctx.Group, match appctx.Match) appEnhancementPreview {
	app := input.PreviewApp
	name := app.Name
	if name == "" {
		name = app.BundleID
	}
	state := appEnhancementPreview{
		App: appEnhancementIconRef{BundleID: app.BundleID, Name: name}, Empty: name == "", HasBundleID: app.BundleID != "",
		Browser: appctx.IsBrowser(app.BundleID, input.CustomBrowsers), CanRefresh: app.BundleID != "", AddressStatus: "none",
	}
	if state.Empty {
		state.FrontText = translator.T("Switch to another app to see which group it falls in.")
	} else {
		state.FrontText = translator.T("Frontmost: %s", name)
	}
	if !state.Empty && state.Browser {
		switch {
		case input.Preview.app != app || input.Preview.reading:
			state.Address = translator.T("Reading the address…")
			state.AddressStatus = "loading"
		case input.Preview.url != "":
			if address, ok := appctx.ParseAddress(input.Preview.url); ok {
				state.Address = address.String()
				state.AddressStatus = "address"
			} else {
				state.Address = input.Preview.url
				state.AddressStatus = "raw-address"
			}
		case !input.HelperReady:
			state.Address = translator.T("The helper is not running, so the address cannot be read.")
			state.AddressStatus = "helper-stopped"
		default:
			state.Address = translator.T("%s does not show its address right now, so it matches as an app.", app.Name)
			state.AddressStatus = "no-address"
		}
	}
	state.Matched = match.Matched()
	state.MatchSource, state.MatchPattern = string(match.Source), match.Pattern
	if state.Matched {
		state.MatchText = translator.T("Matches %s, %s", match.GroupName(), uitext.SourceTextFor(translator, match.Source, match.Pattern))
	} else {
		state.MatchText = translator.T("No group matches, so only the features' own rules apply.")
	}
	if selected == nil || !state.HasBundleID {
		return state
	}
	state.AddApp = !selected.HasApp(app.BundleID)
	if state.AddApp {
		state.AddAppLabel = translator.T("Add %s to %s", app.Name, selected.Name)
	}
	address, hasAddress := appctx.ParseAddress(input.Preview.url)
	state.AddSite = hasAddress && state.Browser && input.Preview.app == app && !slices.ContainsFunc(selected.URLs, func(pattern string) bool {
		return appctx.MatchPattern(pattern, address)
	})
	if state.AddSite {
		state.AddSiteLabel = translator.T("Add %s to %s", address.Host, selected.Name)
	}
	return state
}

func buildAppEnhancementAppsDialog(translator *i18n.Translator, input appEnhancementViewStateInput, selected appctx.Group) appEnhancementAppsDialog {
	errorText := input.AppsError
	if input.AppsHelperStopped {
		errorText = translator.T("The helper is not running. Add an app by its bundle ID instead.")
	}
	state := appEnhancementAppsDialog{
		GroupID: selected.ID, Title: translator.T("Add apps to %s", selected.Name), Loading: input.AppsLoading, Error: errorText,
		HasCatalog: len(input.Apps) > 0, Query: input.AppsQuery, BundleInput: input.BundleInput,
	}
	apps := slices.Clone(input.Apps)
	slices.SortStableFunc(apps, func(left, right proto.AppInfo) int {
		switch {
		case left.Running == right.Running:
			return 0
		case left.Running:
			return -1
		}
		return 1
	})
	for _, app := range apps {
		state.Rows = append(state.Rows, appEnhancementAppChoice{
			Ref: appEnhancementIconRef{BundleID: app.BundleID, Name: app.Name, Path: app.Path}, Running: app.Running, Member: selected.HasApp(app.BundleID),
		})
	}
	return state
}

func buildAppEnhancementBrowsersDialog(translator *i18n.Translator, input appEnhancementViewStateInput) appEnhancementBrowsersDialog {
	state := appEnhancementBrowsersDialog{BundleInput: input.BrowserIDInput, NameInput: input.BrowserNameInput}
	installed := map[string]bool{}
	for _, app := range input.Apps {
		installed[strings.ToLower(app.BundleID)] = true
	}
	var others []string
	for _, browser := range appctx.Browsers(input.CustomBrowsers) {
		if len(input.Apps) > 0 && !installed[strings.ToLower(browser.BundleID)] && appctx.IsBuiltin(browser.BundleID) {
			others = append(others, browser.Name)
			continue
		}
		statusText, statusTone := appEnhancementBrowserStatus(translator, browser, input.URLReads)
		state.Rows = append(state.Rows, appEnhancementBrowserRow{
			Ref:    appEnhancementIconRef{BundleID: browser.BundleID, Name: browser.Name, Path: appEnhancementPath(input.Apps, browser.BundleID)},
			Custom: !appctx.IsBuiltin(browser.BundleID), Engine: browser.Engine, StatusText: statusText, StatusTone: statusTone,
		})
	}
	if len(others) > 0 {
		state.OthersText = translator.T("Also recognised when installed: %s.", strings.Join(others, translator.T(", ")))
	}
	bundleID := strings.TrimSpace(input.BrowserIDInput)
	state.AddDisabled = bundleID == "" || strings.ContainsAny(bundleID, " /") || appctx.IsBrowser(bundleID, input.CustomBrowsers)
	return state
}

func appEnhancementBrowserStatus(translator *i18n.Translator, browser appctx.Browser, reads map[string]urlRead) (string, string) {
	if read, ok := reads[browser.BundleID]; ok {
		if read.ok {
			return translator.T("Address read"), "success"
		}
		return translator.T("No address: app only"), "warning"
	}
	if browser.Engine == appctx.Gecko || browser.Engine == "" {
		return translator.T("Not tried; may be app only"), "info"
	}
	return translator.T("Not tried yet"), "info"
}

func appEnhancementURLError(translator *i18n.Translator, group appctx.Group, pattern string) string {
	pattern = strings.TrimSpace(pattern)
	if pattern == "" {
		return ""
	}
	if err := appctx.CheckPattern(pattern); err != nil {
		return uitext.PatternErrorTextFor(translator, err)
	}
	if slices.ContainsFunc(group.URLs, func(existing string) bool {
		return appctx.CanonicalPattern(existing) == appctx.CanonicalPattern(pattern)
	}) {
		return translator.T("This rule is in the group already.")
	}
	return ""
}

func appEnhancementCustomAutoKeyError(translator *i18n.Translator, keys string) string {
	if _, err := appctx.ParseAutoKey(keys); err != nil || strings.TrimSpace(keys) == "" {
		return translator.T("Enter one key, with or without modifiers: Return, Command+Return, Tab.")
	}
	return ""
}

func appEnhancementPatternErrorCode(err error) string {
	switch err {
	case appctx.ErrPatternEmpty:
		return "empty"
	case appctx.ErrPatternSpace:
		return "space"
	case appctx.ErrPatternScheme:
		return "scheme"
	default:
		return "host"
	}
}

func appEnhancementPath(apps []proto.AppInfo, bundleID string) string {
	for _, app := range apps {
		if strings.EqualFold(app.BundleID, bundleID) {
			return app.Path
		}
	}
	return ""
}
