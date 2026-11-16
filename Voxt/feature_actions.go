package main

import "github.com/egoist/mygo/ui"

type featureActions interface {
	openFeaturePermissions()
	addFeatureShortcut(feature string)
	chooseFeatureShortcut(feature string, index int, keys string)
	setFeatureShortcutTrigger(feature string, index, trigger int)
	removeFeatureShortcut(feature string, index int)
	saveFeatureShortcutKeys(feature string, index int, keys string)
	chooseFeaturePasteLast(keys string)
	saveFeaturePasteLast(keys string)
	chooseFeatureCancelKeys(keys string)
	saveFeatureCancelKeys(keys string)
	beginHotkeyRecord(row *keyRow, save func(string))
	cancelHotkeyRecord()
	recordFeatureKeys(row *keyRow, event ui.InputEvent, save func(string))
	markFeatureKeysHeard()
	openFeatureSpeechModels()
	openFeatureLanguageModels()
	selectFeatureModel(feature, kind, id string)
	selectFeatureFallback(id string)
	setFeatureEnhance(on bool)
	setFeatureCleanup(on bool)
	setFeatureStructure(on bool)
	setFeatureEndWord(word string)
}
