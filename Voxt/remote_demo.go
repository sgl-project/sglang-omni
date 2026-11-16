package main

// Recording-only remote actions for the fixture demo. No method performs
// settings, credential, browser, model or network work.

func (host *meetingDemoHost) remoteDebugState() llmDebugViewState {
	return llmDebugStateForLocal(*host.current().state.RemoteDebug, &host.remoteDebugUI)
}

func (host *meetingDemoHost) openRemoteProvider(kind, id string) {
	host.record("openRemoteProvider %s %s", kind, id)
}

func (host *meetingDemoHost) testRemoteProvider(kind, id string) {
	host.record("testRemoteProvider %s %s", kind, id)
}

func (host *meetingDemoHost) chooseLLMWait(value string, custom bool, minutes int) {
	host.record("chooseLLMWait %q %t %d", value, custom, minutes)
}

func (host *meetingDemoHost) changeLLMWaitMinutes(minutes int) {
	host.record("changeLLMWaitMinutes %d", minutes)
}

func (host *meetingDemoHost) openOnboarding()        { host.record("openOnboarding") }
func (host *meetingDemoHost) dismissOnboardingBack() { host.record("dismissOnboardingBack") }
func (host *meetingDemoHost) openLLMDebug()          { host.record("openLLMDebug") }
func (host *meetingDemoHost) testDialog()            { host.record("testDialog") }
func (host *meetingDemoHost) saveProviderDialog()    { host.record("saveProviderDialog") }
func (host *meetingDemoHost) removeProvider()        { host.record("removeProvider") }
func (host *meetingDemoHost) openRemoteKeyURL()      { host.record("openRemoteKeyURL") }
func (host *meetingDemoHost) openChatGPTUsage()      { host.record("openChatGPTUsage") }
func (host *meetingDemoHost) signInChatGPT()         { host.record("signInChatGPT") }
func (host *meetingDemoHost) cancelChatGPTLogin()    { host.record("cancelChatGPTLogin") }
func (host *meetingDemoHost) signOutChatGPT()        { host.record("signOutChatGPT") }
func (host *meetingDemoHost) dismissChatGPTError() {
	host.record("dismissChatGPTError")
}

func (host *meetingDemoHost) openRemoteLanguageModels() { host.record("openRemoteLanguageModels") }
func (host *meetingDemoHost) runRemoteDebug()           { host.record("runRemoteDebug") }
func (host *meetingDemoHost) useRemoteDebugPrompt()     { host.record("useRemoteDebugPrompt") }
func (host *meetingDemoHost) dismissLLMDebugResult()    { host.record("dismissLLMDebugResult") }
