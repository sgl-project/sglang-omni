import Foundation

// Shared hint/dictionary context for file requests and streaming providers.
extension RemoteASRTranscriber {
    func resolvedHintPayload(
        for provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> ResolvedASRHintPayload {
        let settingsRaw = UserDefaults.standard.string(forKey: AppPreferenceKey.asrHintSettings)
        let settings = ASRHintSettingsStore.resolvedSettings(
            for: ASRHintTarget.from(engine: .remote, remoteProvider: provider),
            rawValue: settingsRaw
        )
        let userLanguageCodes = UserMainLanguageOption.storedSelection(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.userMainLanguageCodes)
        )
        return ASRHintResolver.resolve(
            target: ASRHintTarget.from(engine: .remote, remoteProvider: provider),
            settings: settings,
            userLanguageCodes: userLanguageCodes,
            mlxModelRepo: configuration.model,
            dictionaryTerms: resolvedDictionaryTermsTemplateValue()
        )
    }

    private func resolvedDictionaryTermsTemplateValue() -> String {
        DictionaryEntryCollection.asrPromptTermsText(from: dictionaryEntryProvider?() ?? [])
    }

    func doubaoRequestPayload(
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload,
        requestID: String,
        userID: String,
        audioFormat: String,
        enableNonstream: Bool = false
    ) -> [String: Any] {
        let dictionaryPayload = DoubaoDictionaryRequestPayloadBuilder.build(
            configuration: configuration,
            entries: doubaoDictionaryEntryProvider?() ?? [],
            dictionaryEnabled: true
        )
        return DoubaoASRConfiguration.fullRequestPayload(
            requestID: requestID,
            userID: userID,
            language: hintPayload.language,
            chineseOutputVariant: hintPayload.chineseOutputVariant,
            audioFormat: audioFormat,
            enableNonstream: enableNonstream,
            dictionaryPayload: dictionaryPayload
        )
    }
}
