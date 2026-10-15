// MeetingTranslationSupport.swift
// Provides Meeting Translation Support for meeting transcript processing.

import Foundation

enum MeetingTranslationSupport {
    static func resolvedProvider(
        selectedProvider: TranslationModelProvider,
        fallbackProvider: TranslationModelProvider,
        transcriptionEngine: TranscriptionEngine,
        targetLanguage: TranslationTargetLanguage
    ) -> TranslationProviderResolution {
        TranslationProviderResolver.resolve(
            selectedProvider: selectedProvider,
            fallbackProvider: fallbackProvider,
            transcriptionEngine: transcriptionEngine,
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: false
        )
    }
}
