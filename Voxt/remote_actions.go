package main

// User intents exposed by the Language models page and its dialog.
type remoteCardsActions interface {
	openRemoteProvider(kind, id string)
	testRemoteProvider(kind, id string)
	openModelAssignment(kind, id string)
}

type remoteWaitActions interface {
	chooseLLMWait(value string)
	setLLMSwitch(on bool)
}

type remoteDialogActions interface {
	testDialog()
	saveProviderDialog()
	removeProvider()
	openRemoteKeyURL()
	openChatGPTUsage()
	signInChatGPT()
	cancelChatGPTLogin()
	signOutChatGPT()
	dismissChatGPTError()
}

type remotePageActions interface {
	remoteCardsActions
	remoteWaitActions
	openOnboarding()
	dismissOnboardingBack()
	openLLMDebug()
}

type remoteActions interface {
	remotePageActions
	remoteDialogActions
}

type llmDebugActions interface {
	openRemoteLanguageModels()
	runRemoteDebug()
	useRemoteDebugPrompt()
	dismissLLMDebugResult()
}

// remoteLocalUI points at the independent widgets' state; the dialog's
// controller owns the surrounding saved setup and async operations.
type remoteLocalUI struct {
	wait   *llmWaitLocalUI
	dialog *providerDialogLocalUI
}
