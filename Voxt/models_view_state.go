package main

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type modelActionViewState struct {
	Kind               modelActionKind `json:"kind"`
	ModelID            string          `json:"model_id"`
	Label              string          `json:"label"`
	AccessibilityLabel string          `json:"accessibility_label"`
	Tooltip            string          `json:"tooltip,omitempty"`
	Disabled           bool            `json:"disabled,omitempty"`
}

type modelProgressViewState struct {
	Fraction float64 `json:"fraction"`
	Label    string  `json:"label"`
}

type modelErrorViewState struct {
	Message    string `json:"message"`
	Detail     string `json:"detail"`
	RetryLabel string `json:"retry_label"`
}

type modelDraftViewState struct {
	ModelID       string                  `json:"model_id"`
	ParentID      string                  `json:"parent_id"`
	Help          string                  `json:"help"`
	Installed     bool                    `json:"installed"`
	NeedsDownload bool                    `json:"needs_download"`
	SizeText      string                  `json:"size_text"`
	On            bool                    `json:"on"`
	SwitchLabel   string                  `json:"switch_label"`
	Actions       []modelActionViewState  `json:"actions"`
	Progress      *modelProgressViewState `json:"progress,omitempty"`
	Error         *modelErrorViewState    `json:"error,omitempty"`
}

type modelRowViewState struct {
	ID         string                  `json:"id"`
	Kind       string                  `json:"kind"`
	Name       string                  `json:"name"`
	Default    bool                    `json:"default"`
	Installed  bool                    `json:"installed"`
	SizeText   string                  `json:"size_text"`
	UsedBy     string                  `json:"used_by,omitempty"`
	MemoryText string                  `json:"memory_text,omitempty"`
	FamilyText string                  `json:"family_text,omitempty"`
	About      string                  `json:"about"`
	Languages  string                  `json:"languages,omitempty"`
	Tags       []string                `json:"tags"`
	Actions    []modelActionViewState  `json:"actions"`
	Progress   *modelProgressViewState `json:"progress,omitempty"`
	Error      *modelErrorViewState    `json:"error,omitempty"`
	Draft      *modelDraftViewState    `json:"draft,omitempty"`
}

type modelCardViewState struct {
	Key        string              `json:"key"`
	FamilyName string              `json:"family_name,omitempty"`
	CountText  string              `json:"count_text,omitempty"`
	Languages  string              `json:"languages,omitempty"`
	Rows       []modelRowViewState `json:"rows"`
}

type modelsChoiceViewState struct {
	Label string `json:"label"`
	Value string `json:"value"`
}

type modelsAdvancedViewState struct {
	Open      bool                    `json:"open"`
	Locale    string                  `json:"locale"`
	Value     string                  `json:"value"`
	Selection string                  `json:"selection"`
	Options   []modelsChoiceViewState `json:"options"`
}

type modelPlainDeleteViewState struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Prompt string `json:"prompt"`
	Detail string `json:"detail"`
}

type modelSharedDeleteViewState struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	ForAll       bool   `json:"for_all"`
	RemovePrompt string `json:"remove_prompt"`
	RemoveDetail string `json:"remove_detail"`
	AllPrompt    string `json:"all_prompt"`
	AllDetail    string `json:"all_detail"`
}

type modelsDeleteViewState struct {
	Plain  *modelPlainDeleteViewState  `json:"plain,omitempty"`
	Shared *modelSharedDeleteViewState `json:"shared,omitempty"`
}

type modelsNetworkFamilyViewState struct {
	Key         string `json:"key"`
	Kind        string `json:"kind"`
	Text        string `json:"text"`
	ActionLabel string `json:"action_label,omitempty"`
	Follow      bool   `json:"follow,omitempty"`
}

type modelsNetworkViewState struct {
	Open          bool                         `json:"open"`
	ProxyMode     int                          `json:"proxy_mode"`
	ProxyError    string                       `json:"proxy_error,omitempty"`
	ProxyCaption  string                       `json:"proxy_caption"`
	Source        int                          `json:"source"`
	SourceURL     string                       `json:"source_url,omitempty"`
	SourceLabels  []string                     `json:"source_labels"`
	SourceCaption string                       `json:"source_caption"`
	ProxyFamily   modelsNetworkFamilyViewState `json:"proxy_family"`
	SourceFamily  modelsNetworkFamilyViewState `json:"source_family"`
}

type modelsViewState struct {
	ASR           []modelCardViewState    `json:"asr"`
	Dictation     *modelCardViewState     `json:"dictation,omitempty"`
	LLM           []modelCardViewState    `json:"llm"`
	ShowLLM       bool                    `json:"show_llm"`
	Hidden        []modelCardViewState    `json:"hidden"`
	HiddenOpen    bool                    `json:"hidden_open"`
	Advanced      modelsAdvancedViewState `json:"advanced"`
	Delete        modelsDeleteViewState   `json:"delete"`
	Network       modelsNetworkViewState  `json:"network"`
	SpeechContext string                  `json:"speech_context"`
}

type modelsViewStateInput struct {
	Catalog        []models.Model
	Status         map[string]models.Status
	Memory         map[string]int64
	Settings       settings.Settings
	InstalledASR   []local.Model
	InstalledLLM   []local.Model
	ModelAsk       string
	ModelAskFree   int64
	ModelShared    *sharedDelete
	ModelDeciding  string
	NetworkOpen    bool
	NetworkForm    networkForm
	FamilyOthers   bool
	FamilyReadOnly bool
	AdvancedOpen   bool
	HiddenOpen     bool
}

func buildModelsViewState(translator *i18n.Translator, input modelsViewStateInput) modelsViewState {
	state := modelsViewState{
		ASR: buildModelCards(translator, input, models.ASR), LLM: buildModelCards(translator, input, models.LLM),
		Hidden: buildHiddenModelCards(translator, input), HiddenOpen: input.HiddenOpen,
		Advanced: buildModelsAdvancedViewState(translator, input), Delete: buildModelsDeleteViewState(translator, input),
		Network: buildModelsNetworkViewState(translator, input), SpeechContext: input.Settings.Dictionary.SpeechContext,
	}
	state.Dictation = buildDictationCard(translator, input, false)
	for _, model := range input.Catalog {
		if model.Kind == models.LLM && modelShown(model, input.Status[model.ID]) {
			state.ShowLLM = true
			break
		}
	}
	return state
}

func buildModelCards(translator *i18n.Translator, input modelsViewStateInput, kind models.Kind) []modelCardViewState {
	if kind != models.ASR {
		var cards []modelCardViewState
		for _, model := range input.Catalog {
			if model.Kind == kind && modelShown(model, input.Status[model.ID]) && !modelHiddenFor(translator, input, model.ID) {
				cards = append(cards, buildSingleModelCard(translator, input, model))
			}
		}
		return cards
	}
	var families []string
	byFamily := map[string][]models.Model{}
	for _, model := range input.Catalog {
		if model.Kind != kind || !modelShown(model, input.Status[model.ID]) || modelHiddenFor(translator, input, model.ID) {
			continue
		}
		if byFamily[model.Family] == nil {
			families = append(families, model.Family)
		}
		byFamily[model.Family] = append(byFamily[model.Family], model)
	}
	var cards []modelCardViewState
	for _, family := range families {
		list := byFamily[family]
		if len(list) == 1 {
			cards = append(cards, buildSingleModelCard(translator, input, list[0]))
			continue
		}
		installed := 0
		rows := make([]modelRowViewState, len(list))
		for index, model := range list {
			if input.Status[model.ID].Installed {
				installed++
			}
			rows[index] = buildModelRowViewState(translator, input, model, false)
		}
		count := translator.N(len(list), "%d model", "%d models", len(list))
		if installed > 0 {
			count += " · " + translator.T("%d installed", installed)
		}
		cards = append(cards, modelCardViewState{
			Key: "family-" + family, FamilyName: uitext.FamilyName(family), CountText: count,
			Languages: uitext.ModelLanguagesFor(translator, list[0].ID), Rows: rows,
		})
	}
	return cards
}

func buildSingleModelCard(translator *i18n.Translator, input modelsViewStateInput, model models.Model) modelCardViewState {
	return modelCardViewState{Key: "model-" + model.ID, Rows: []modelRowViewState{buildModelRowViewState(translator, input, model, true)}}
}

func buildHiddenModelCards(translator *i18n.Translator, input modelsViewStateInput) []modelCardViewState {
	var cards []modelCardViewState
	for _, model := range input.Catalog {
		if model.Kind == models.ASR && modelShown(model, input.Status[model.ID]) && modelHiddenFor(translator, input, model.ID) {
			cards = append(cards, buildSingleModelCard(translator, input, model))
		}
	}
	if card := buildDictationCard(translator, input, true); card != nil {
		cards = append(cards, *card)
	}
	return cards
}

func modelHiddenFor(translator *i18n.Translator, input modelsViewStateInput, id string) bool {
	return slices.Contains(input.Settings.Models.Hidden, id) && modelsUsedBy(translator, input.Settings, input.InstalledASR, input.InstalledLLM, local.ASR, id) == ""
}

func dictationListed(input modelsViewStateInput) bool {
	return slices.ContainsFunc(input.InstalledASR, func(model local.Model) bool { return model.ID == dictation.ModelID })
}

func buildDictationCard(translator *i18n.Translator, input modelsViewStateInput, hidden bool) *modelCardViewState {
	if !dictationListed(input) || modelHiddenFor(translator, input, dictation.ModelID) != hidden {
		return nil
	}
	usedBy := modelsUsedBy(translator, input.Settings, input.InstalledASR, input.InstalledLLM, local.ASR, dictation.ModelID)
	row := modelRowViewState{
		ID: dictation.ModelID, Kind: string(local.ASR), Name: uitext.ModelNameFor(translator, dictation.ModelID),
		SizeText: translator.T("Built-in"), UsedBy: usedBy, About: uitext.ModelAboutFor(translator, dictation.ModelID),
		Languages: uitext.ModelLanguagesFor(translator, dictation.ModelID), Tags: uitext.ModelTagsFor(translator, dictation.ModelID),
	}
	row.Actions = buildHideAction(translator, input, dictation.ModelID, usedBy)
	return &modelCardViewState{Key: "model-" + dictation.ModelID, Rows: []modelRowViewState{row}}
}

func buildHideAction(translator *i18n.Translator, input modelsViewStateInput, id, usedBy string) []modelActionViewState {
	if usedBy != "" {
		return nil
	}
	name := uitext.ModelNameFor(translator, id)
	if slices.Contains(input.Settings.Models.Hidden, id) {
		return []modelActionViewState{{
			Kind: modelActionShow, ModelID: id, Label: translator.T("Show"), AccessibilityLabel: translator.T("Show %s", name),
			Tooltip: translator.T("Show it in the features' menus again"),
		}}
	}
	return []modelActionViewState{{
		Kind: modelActionHide, ModelID: id, Label: translator.T("Hide"), AccessibilityLabel: translator.T("Hide %s", name),
		Tooltip: translator.T("Hide it from the features' menus"),
	}}
}

func buildModelRowViewState(translator *i18n.Translator, input modelsViewStateInput, model models.Model, alone bool) modelRowViewState {
	status := input.Status[model.ID]
	usedBy := modelsUsedBy(translator, input.Settings, input.InstalledASR, input.InstalledLLM, local.Kind(model.Kind), model.ID)
	sizeText, familyText := modelSizeTextFor(translator, model, status), modelFamilyTextFor(translator, model, status)
	if status.Downloading && status.WaitingFor != "" {
		sizeText, familyText = familyText, ""
	}
	row := modelRowViewState{
		ID: model.ID, Kind: string(model.Kind), Name: uitext.ModelNameFor(translator, model.ID),
		Default: model.Default && !model.Retired, Installed: status.Installed, SizeText: sizeText, UsedBy: usedBy,
		FamilyText: familyText, About: uitext.ModelAboutFor(translator, model.ID),
	}
	if model.Kind == models.ASR {
		row.Actions = buildHideAction(translator, input, model.ID, usedBy)
	}
	if alone {
		row.Languages = uitext.ModelLanguagesFor(translator, model.ID)
	}
	if memory, ok := input.Memory[model.ID]; ok && status.Installed {
		row.MemoryText = modelMemTextFor(translator, memory)
	}
	if !model.Retired {
		row.Tags = uitext.ModelTagsFor(translator, model.ID)
		if spec, ok := omni.Lookup(model.ID); ok && len(spec.Languages) > 0 {
			if slices.Contains(spec.Languages, lang.Canonical(session.MainLanguage(input.Settings, translator.Locale()))) {
				row.Tags = append(row.Tags, translator.T("Supports your main language"))
			} else {
				row.Tags = append(row.Tags, translator.T("Does not support your main language"))
			}
		}
	}
	row.Actions = append(row.Actions, buildModelActions(translator, input, model, status)...)
	row.Progress = buildModelProgress(translator, model, status)
	if status.Err != nil && !status.Downloading {
		message, detail := uitext.DownloadErrorTextFor(translator, status.Err)
		row.Error = &modelErrorViewState{Message: message, Detail: detail, RetryLabel: translator.T("Try again")}
	}
	draftStatus := input.Status[model.Draft]
	if model.Draft != "" && (status.Installed || draftStatus.Installed || draftStatus.OnDisk || draftStatus.Downloading || draftStatus.Done > 0) {
		row.Draft = buildModelDraftViewState(translator, input, model, draftStatus)
	}
	return row
}

func buildModelProgress(translator *i18n.Translator, model models.Model, status models.Status) *modelProgressViewState {
	if !status.Downloading {
		return nil
	}
	fraction := 0.0
	if status.Total > 0 {
		fraction = float64(status.Done) / float64(status.Total)
	}
	return &modelProgressViewState{Fraction: fraction, Label: translator.T("Downloading %s", uitext.ModelNameFor(translator, model.ID))}
}

func buildModelActions(translator *i18n.Translator, input modelsViewStateInput, model models.Model, status models.Status) []modelActionViewState {
	name := uitext.ModelNameFor(translator, model.ID)
	switch {
	case status.Downloading:
		return []modelActionViewState{{Kind: modelActionCancel, ModelID: model.ID, Label: translator.T("Cancel"), AccessibilityLabel: translator.T("Cancel downloading %s", name)}}
	case status.Installed:
		label, tooltip := translator.T("Delete"), translator.T("Delete %s…", name)
		if model.Borrowed {
			label, tooltip = translator.T("Remove"), translator.T("Remove %s…", name)
		}
		return []modelActionViewState{{Kind: modelActionDelete, ModelID: model.ID, Label: label, AccessibilityLabel: tooltip, Disabled: input.ModelDeciding == model.ID}}
	case model.Retired:
		if status.Done > 0 {
			return []modelActionViewState{{Kind: modelActionDiscard, ModelID: model.ID, Label: translator.T("Delete"), AccessibilityLabel: translator.T("Delete the partial download of %s", name)}}
		}
	case model.Borrowed || status.OnDisk:
		if status.OnDisk {
			return []modelActionViewState{{Kind: modelActionUse, ModelID: model.ID, Label: translator.T("Use it"), AccessibilityLabel: translator.T("Use %s", name)}}
		}
	default:
		var actions []modelActionViewState
		elsewhere := status.Elsewhere != ""
		if status.Done > 0 && !elsewhere {
			actions = append(actions, modelActionViewState{Kind: modelActionDiscard, ModelID: model.ID, Label: translator.T("Delete"), AccessibilityLabel: translator.T("Delete the partial download of %s", name)})
		}
		label := translator.T("Download")
		switch {
		case status.DeletedBy != "":
			label = translator.T("Download again")
		case status.Done > 0 && !elsewhere:
			label = translator.T("Resume")
		}
		if status.Err == nil {
			actions = append(actions, modelActionViewState{Kind: modelActionDownload, ModelID: model.ID, Label: label, AccessibilityLabel: translator.T("Download %s", name), Disabled: elsewhere})
		}
		return actions
	}
	return nil
}

func buildModelDraftViewState(translator *i18n.Translator, input modelsViewStateInput, parent models.Model, status models.Status) *modelDraftViewState {
	draft, _ := models.Lookup(parent.Draft)
	on, set := input.Settings.Models.Faster[parent.ID]
	if !set {
		on = status.Installed
	}
	state := &modelDraftViewState{
		ModelID: draft.ID, ParentID: parent.ID, Installed: status.Installed, NeedsDownload: on && !status.Installed && !status.Downloading,
		SizeText: modelSizeTextFor(translator, draft, status), On: on, SwitchLabel: translator.T("Faster transcription for %s", uitext.ModelNameFor(translator, parent.ID)),
		Help: translator.T("Uses speculative decoding: a small draft model (%s) proposes several words at a time, and the speech model checks them. It needs the draft downloaded.", uitext.HumanBytes(draft.Size())),
	}
	if on || status.Installed || status.OnDisk || status.Downloading || status.Done > 0 {
		state.Actions = buildModelActions(translator, input, draft, status)
	}
	state.Progress = buildModelProgress(translator, draft, status)
	if status.Err != nil && !status.Downloading {
		message, detail := uitext.DownloadErrorTextFor(translator, status.Err)
		state.Error = &modelErrorViewState{Message: message, Detail: detail, RetryLabel: translator.T("Try again")}
	}
	return state
}

func buildModelsAdvancedViewState(translator *i18n.Translator, input modelsViewStateInput) modelsAdvancedViewState {
	labels := keepLabelsFor(translator)
	options := make([]modelsChoiceViewState, len(settings.KeepChoices))
	for index, value := range settings.KeepChoices {
		options[index] = modelsChoiceViewState{Label: labels[index], Value: value}
	}
	value := input.Settings.Models.KeepLoaded
	return modelsAdvancedViewState{
		Open: input.AdvancedOpen, Locale: translator.Locale(), Value: value,
		Selection: keepLabelFor(translator, value), Options: options,
	}
}

func buildModelsDeleteViewState(translator *i18n.Translator, input modelsViewStateInput) modelsDeleteViewState {
	state := modelsDeleteViewState{}
	if input.ModelAsk != "" {
		free := input.ModelAskFree
		if free <= 0 {
			model, _ := models.Lookup(input.ModelAsk)
			free = model.Size()
		}
		state.Plain = &modelPlainDeleteViewState{
			ID: input.ModelAsk, Name: uitext.ModelNameFor(translator, input.ModelAsk), Prompt: translator.T("Delete %s?", uitext.ModelNameFor(translator, input.ModelAsk)),
			Detail: translator.T("This frees %s. You can download it again later.", uitext.HumanBytes(free)),
		}
	}
	if shared := input.ModelShared; shared != nil {
		name, apps, size := uitext.ModelNameFor(translator, shared.id), uitext.NamesFor(translator, shared.apps), uitext.HumanBytes(shared.bytes)
		state.Shared = &modelSharedDeleteViewState{
			ID: shared.id, Name: name, ForAll: shared.forAll,
			RemovePrompt: translator.T("Remove \"%s\" from Voxt?", name),
			RemoveDetail: translator.N(len(shared.apps), "%s also uses this model, so its files (%s) stay on this Mac.", "%s also use this model, so its files (%s) stay on this Mac.", apps, size),
			AllPrompt:    translator.T("Delete \"%s\" for all apps?", name), AllDetail: translator.T("%s will need to download it again (%s).", apps, size),
		}
	}
	return state
}

func buildModelsNetworkViewState(translator *i18n.Translator, input modelsViewStateInput) modelsNetworkViewState {
	state := modelsNetworkViewState{Open: input.NetworkOpen}
	updateModelsNetworkFormViewState(translator, &state, input.NetworkForm)
	state.ProxyFamily = buildModelsNetworkFamilyViewState(translator, input, actnow.KeyProxy, translator.T("Other apps that share this model library use a different proxy."))
	state.SourceFamily = buildModelsNetworkFamilyViewState(translator, input, actnow.KeyHFEndpoints, translator.T("Other apps that share this model library use a different download source."))
	return state
}

func updateModelsNetworkFormViewState(translator *i18n.Translator, state *modelsNetworkViewState, form networkForm) {
	state.ProxyMode, state.ProxyError = form.proxyMode, form.proxyErr
	state.Source, state.SourceURL = form.source, form.sourceURL
	state.SourceLabels = []string{translator.T("Automatic"), translator.T("Hugging Face only"), translator.T("hf-mirror.com only")}
	state.ProxyCaption = ""
	switch form.proxyMode {
	case proxyAuto:
		state.ProxyCaption = translator.T("The proxy in this Mac's network settings, or one set by an environment variable.")
	case proxyNone:
		state.ProxyCaption = translator.T("Downloads connect to the server directly.")
	}
	if form.sourceURL != "" {
		state.SourceLabels = append(state.SourceLabels, translator.T("Other"))
	}
	switch {
	case form.source == 0:
		state.SourceCaption = translator.T("Whichever answers first. Every file is checked.")
	case form.source < len(networkSources):
		state.SourceCaption = translator.T("Only this server. If it cannot be reached, downloads fail.")
	default:
		state.SourceCaption = translator.T("Custom servers: %s", form.sourceURL)
	}
}

func buildModelsNetworkFamilyViewState(translator *i18n.Translator, input modelsViewStateInput, key, conflictText string) modelsNetworkFamilyViewState {
	state := modelsNetworkFamilyViewState{Key: key}
	follows := input.Settings.Family.Follows(key)
	if slices.Contains(input.Settings.Family.Conflicts, key) && !follows {
		state.Kind, state.Text, state.ActionLabel, state.Follow = "conflict", conflictText, translator.T("Use theirs"), true
		return state
	}
	if !input.FamilyOthers && !(follows && input.FamilyReadOnly) {
		return state
	}
	switch {
	case follows && input.FamilyReadOnly:
		state.Kind, state.Text = "readonly", translator.T("Another app updated the shared format. Update Voxt to change these settings.")
	case follows:
		state.Kind, state.Text, state.ActionLabel = "follows", translator.T("Applies to all apps that use this model library"), translator.T("Use a different setting for this app")
	default:
		state.Kind, state.Text, state.ActionLabel, state.Follow = "own", translator.T("Only Voxt uses this setting."), translator.T("Use the same setting as the other apps that share this model library"), true
	}
	return state
}

func modelsViewCacheKey(input modelsViewStateInput, locale string) string {
	var key strings.Builder
	fmt.Fprintf(&key, "%s|%#v|%#v|%#v|%#v|%#v|", locale, input.Settings, input.InstalledASR, input.InstalledLLM, input.NetworkForm.proxyMode, input.NetworkForm.source)
	fmt.Fprintf(&key, "%s|%s|%t|%t|%t|%t|%s|%d|%#v|%s|", input.NetworkForm.sourceURL, input.NetworkForm.proxyErr, input.NetworkOpen, input.FamilyOthers, input.FamilyReadOnly, input.AdvancedOpen, input.ModelAsk, input.ModelAskFree, input.ModelShared, input.ModelDeciding)
	for _, model := range input.Catalog {
		status := input.Status[model.ID]
		fmt.Fprintf(&key, "%s:%t:%t:%t:%d:%d:%s:%s:%v:%t:%s:%s:%s;", model.ID, status.Installed, status.OnDisk, status.Downloading, status.Done, status.Total, status.WaitingFor, status.Elsewhere, status.Err, status.Mine, strings.Join(status.Users, ","), status.DeletedBy, status.Revision)
	}
	ids := make([]string, 0, len(input.Memory))
	for id := range input.Memory {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		fmt.Fprintf(&key, "%s:%d;", id, input.Memory[id])
	}
	fmt.Fprintf(&key, "|%t|%t", input.HiddenOpen, input.Settings.Dictionary.SpeechContext != "")
	return key.String()
}
