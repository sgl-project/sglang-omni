package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func (host *meetingDemoHost) modelsViewState() *modelsViewState {
	state := *host.current().state.Models
	updateModelsNetworkFormViewState(i18n.CurrentTranslator(), &state.Network, host.modelsUI.netForm)
	state.Network.Open = host.modelsUI.netOpen
	return &state
}

func (host *meetingDemoHost) openNetwork() {
	host.modelsUI.netOpen = true
	host.modelsUI.netWas = host.modelsUI.netForm
	host.record("openNetwork")
}
func (host *meetingDemoHost) saveNetwork() bool {
	host.record("saveNetwork")
	return true
}
func (host *meetingDemoHost) closeNetwork() {
	host.modelsUI.netOpen = false
	host.record("closeNetwork")
}
func (host *meetingDemoHost) followNetwork(key string, on bool) {
	host.record("followNetwork %s %t", key, on)
}
func (host *meetingDemoHost) downloadModel(id string) { host.record("downloadModel %s", id) }
func (host *meetingDemoHost) cancelModel(id string)   { host.record("cancelModel %s", id) }
func (host *meetingDemoHost) askDeleteModel(id string) {
	host.record("askDeleteModel %s", id)
}
func (host *meetingDemoHost) discardModelDownload(id string) {
	host.record("discardModelDownload %s", id)
}
func (host *meetingDemoHost) useModel(id string) { host.record("useModel %s", id) }
func (host *meetingDemoHost) confirmModelDelete(id string, allApps bool) {
	host.record("confirmModelDelete %s %t", id, allApps)
}
func (host *meetingDemoHost) setModelHidden(id string, hidden bool) {
	host.record("setModelHidden %s %t", id, hidden)
}
func (host *meetingDemoHost) setModelFaster(id string, on bool) {
	host.record("setModelFaster %s %t", id, on)
}
func (host *meetingDemoHost) setModelsKeepLoaded(value string) {
	host.record("setModelsKeepLoaded %q", value)
}
func (host *meetingDemoHost) autosaveSpeechContext(changed, focused bool) {
	state := &host.modelsUI.speechContext
	if changed {
		state.dirty = true
		state.seq++
		host.record("autosaveSpeechContext %t %t", changed, focused)
		return
	}
	if state.dirty && !focused {
		state.text = strings.TrimSpace(state.text)
		state.seen = state.text
		state.dirty = false
		host.record("autosaveSpeechContext %t %t", changed, focused)
	}
}
