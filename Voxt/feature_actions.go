package main

type featureActions interface {
	openFeatureSpeechModels()
	openFeatureLanguageModels()
	selectFeatureModel(feature, kind, id string)
	selectFeatureFallback(id string)
	setFeatureEnhance(on bool)
	setFeatureCleanup(on bool)
	setFeatureStructure(on bool)
	setFeatureEndWord(word string)
}
