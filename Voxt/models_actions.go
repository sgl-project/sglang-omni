package main

type modelActionKind string

const (
	modelActionHide     modelActionKind = "hide"
	modelActionShow     modelActionKind = "show"
	modelActionCancel   modelActionKind = "cancel"
	modelActionDelete   modelActionKind = "delete"
	modelActionDiscard  modelActionKind = "discard"
	modelActionUse      modelActionKind = "use"
	modelActionDownload modelActionKind = "download"
)

type modelsActions interface {
	openNetwork()
	saveNetwork() bool
	closeNetwork()
	followNetwork(key string, on bool)
	downloadModel(id string)
	cancelModel(id string)
	askDeleteModel(id string)
	discardModelDownload(id string)
	useModel(id string)
	confirmModelDelete(id string, allApps bool)
	setModelHidden(id string, hidden bool)
	setModelFaster(id string, on bool)
	setModelsKeepLoaded(value string)
	autosaveSpeechContext(changed, focused bool)
}
