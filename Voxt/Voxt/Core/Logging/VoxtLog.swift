// VoxtLog.swift
// Provides the compatibility logging facade for the app.

import Darwin
import AppKit
import Foundation
import Logging

enum VoxtLog {
    struct ExportPayload {
        let filename: String
        let content: String
    }

    nonisolated(unsafe) static var verboseEnabled = false

    nonisolated static func info(_ message: @autoclosure () -> String, verbose: Bool = false) {
        appLogger.info(message(), verbose: verbose)
    }

    nonisolated static func audio(_ message: @autoclosure () -> String, verbose: Bool = false) {
        audioLogger.info(message(), verbose: verbose)
    }

    nonisolated static func audioWarning(_ message: @autoclosure () -> String) {
        audioLogger.warning(message())
    }

    nonisolated static func dictionary(_ message: @autoclosure () -> String, verbose: Bool = false) {
        dictionaryLogger.info(message(), verbose: verbose)
    }

    nonisolated static func dictionaryWarning(_ message: @autoclosure () -> String) {
        dictionaryLogger.warning(message())
    }

    nonisolated static func history(_ message: @autoclosure () -> String, verbose: Bool = false) {
        historyLogger.info(message(), verbose: verbose)
    }

    nonisolated static func historyWarning(_ message: @autoclosure () -> String) {
        historyLogger.warning(message())
    }

    nonisolated static func hotkey(_ message: @autoclosure () -> String, verbose: Bool = false) {
        guard isHotkeyDebugLoggingEnabled,
              !verbose || verboseEnabled
        else {
            return
        }
        hotkeyLogger.info(message())
    }

    nonisolated static var isHotkeyDebugLoggingEnabled: Bool {
        UserDefaults.standard.bool(forKey: AppPreferenceKey.hotkeyDebugLoggingEnabled)
    }

    nonisolated static func input(_ message: @autoclosure () -> String, verbose: Bool = false) {
        inputLogger.info(message(), verbose: verbose)
    }

    nonisolated static func inputWarning(_ message: @autoclosure () -> String) {
        inputLogger.warning(message())
    }

    nonisolated static func asr(_ message: @autoclosure () -> String, verbose: Bool = false) {
        asrLogger.info(message(), verbose: verbose)
    }

    nonisolated static func asrWarning(_ message: @autoclosure () -> String) {
        asrLogger.warning(message())
    }

    nonisolated static func asrError(_ message: @autoclosure () -> String) {
        asrLogger.error(message())
    }

    nonisolated static func llm(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        // LLM prompts, user input, and generated output can contain arbitrary private
        // content. The regular model-debug switch must never make that content part of
        // persistent diagnostics; callers should log dimensions and timings separately.
        // Do not evaluate the autoclosure either, which avoids constructing and scanning
        // large prompt/response strings merely to redact them.
        _ = message
    }

    nonisolated static func llmDebug(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        llmLogger.info(message())
    }

    nonisolated static func llmInfo(_ message: @autoclosure () -> String, verbose: Bool = false) {
        llmLogger.info(message(), verbose: verbose)
    }

    nonisolated static func llmWarning(_ message: @autoclosure () -> String) {
        llmLogger.warning(message())
    }

    nonisolated static func model(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        modelLogger.info(message(), privacy: .preview(limit: 4_000))
    }

    nonisolated static func vad(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        modelLogger.info(message(), privacy: .preview(limit: 4_000))
    }

    nonisolated static func modelInfo(_ message: @autoclosure () -> String, verbose: Bool = false) {
        modelLogger.info(message(), verbose: verbose)
    }

    nonisolated static func modelWarning(_ message: @autoclosure () -> String) {
        modelLogger.warning(message())
    }

    nonisolated static func modelError(_ message: @autoclosure () -> String) {
        modelLogger.error(message())
    }

    nonisolated static func meeting(_ message: @autoclosure () -> String, verbose: Bool = false) {
        meetingLogger.info(message(), verbose: verbose)
    }

    nonisolated static func meetingWarning(_ message: @autoclosure () -> String) {
        meetingLogger.warning(message())
    }

    nonisolated static func meetingError(_ message: @autoclosure () -> String) {
        meetingLogger.error(message())
    }

    nonisolated static func network(_ message: @autoclosure () -> String, verbose: Bool = false) {
        networkLogger.info(message(), verbose: verbose)
    }

    nonisolated static func networkWarning(_ message: @autoclosure () -> String) {
        networkLogger.warning(message())
    }

    nonisolated static func settings(_ message: @autoclosure () -> String, verbose: Bool = false) {
        settingsLogger.info(message(), verbose: verbose)
    }

    nonisolated static func settingsWarning(_ message: @autoclosure () -> String) {
        settingsLogger.warning(message())
    }

    nonisolated static func persistence(_ message: @autoclosure () -> String, verbose: Bool = false) {
        persistenceLogger.info(message(), verbose: verbose)
    }

    nonisolated static func persistenceWarning(_ message: @autoclosure () -> String) {
        persistenceLogger.warning(message())
    }

    nonisolated static func persistenceError(_ message: @autoclosure () -> String) {
        persistenceLogger.error(message())
    }

    nonisolated static func translation(_ message: @autoclosure () -> String, verbose: Bool = false) {
        translationLogger.info(message(), verbose: verbose)
    }

    nonisolated static func translationWarning(_ message: @autoclosure () -> String) {
        translationLogger.warning(message())
    }

    nonisolated static func update(_ message: @autoclosure () -> String, verbose: Bool = false) {
        updateLogger.info(message(), verbose: verbose)
    }

    nonisolated static func updateWarning(_ message: @autoclosure () -> String) {
        updateLogger.warning(message())
    }

    nonisolated static func updateError(_ message: @autoclosure () -> String) {
        updateLogger.error(message())
    }

    nonisolated static func securityWarning(_ message: @autoclosure () -> String) {
        securityLogger.warning(message())
    }

    nonisolated static func llmPreview(_ text: String, limit: Int = 1200) -> String {
        VoxtLogRedactor.preview(text, limit: limit)
    }

    nonisolated static func warning(_ message: @autoclosure () -> String) {
        appLogger.warning(message())
    }

    nonisolated static func error(_ message: @autoclosure () -> String) {
        appLogger.error(message())
    }

    nonisolated static func latestLogUpdateDate() -> Date? {
        VoxtLogExportStore.latestLogUpdateDate()
    }

    nonisolated static func latestLogExportPayload(limit: Int = 1000) -> ExportPayload {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let filename = "voxt-log-\(formatter.string(from: Date())).txt"
        let content = VoxtLogExportStore.latestLogDisplayText(limit: limit) {
            MainActorSync.run {
                diagnosticsMetadataText()
            }
        }
        return ExportPayload(filename: filename, content: content)
    }

    nonisolated static func latestLogDisplayText(limit: Int = 1000) -> String {
        VoxtLogExportStore.latestLogDisplayText(limit: limit) {
            MainActorSync.run {
                diagnosticsMetadataText()
            }
        }
    }

    nonisolated static func exportLatestLogs(limit: Int = 1000) throws -> URL {
        let payload = latestLogExportPayload(limit: limit)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(payload.filename)
        try payload.content.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private nonisolated static let appLogger = VoxtLogger(category: .app)
    private nonisolated static let audioLogger = VoxtLogger(category: .audio)
    private nonisolated static let dictionaryLogger = VoxtLogger(category: .dictionary)
    private nonisolated static let historyLogger = VoxtLogger(category: .history)
    private nonisolated static let hotkeyLogger = VoxtLogger(category: .hotkey)
    private nonisolated static let inputLogger = VoxtLogger(category: .input)
    private nonisolated static let asrLogger = VoxtLogger(category: .asr)
    private nonisolated static let llmLogger = VoxtLogger(category: .llm)
    private nonisolated static let modelLogger = VoxtLogger(category: .model)
    private nonisolated static let meetingLogger = VoxtLogger(category: .meeting)
    private nonisolated static let networkLogger = VoxtLogger(category: .network)
    private nonisolated static let settingsLogger = VoxtLogger(category: .settings)
    private nonisolated static let persistenceLogger = VoxtLogger(category: .persistence)
    private nonisolated static let translationLogger = VoxtLogger(category: .translation)
    private nonisolated static let updateLogger = VoxtLogger(category: .update)
    private nonisolated static let securityLogger = VoxtLogger(category: .security)

    @MainActor
    private static func diagnosticsMetadataText(defaults: UserDefaults = .standard) -> String {
        let featureSettings = FeatureSettingsStore.load(defaults: defaults)
        let proxySettings = VoxtNetworkSession.currentProxySettings
        let systemProxyStatus = VoxtNetworkSession.currentSystemProxyStatus
        let remoteASRProvider = RemoteASRProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteASRSelectedProvider) ?? "")
            ?? .openAIWhisper
        let remoteLLMProvider = RemoteLLMProvider(rawValue: defaults.string(forKey: AppPreferenceKey.remoteLLMSelectedProvider) ?? "")
            ?? .openAI
        let remoteASRConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: defaults.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? "",
            sensitiveValueLoading: .metadataOnly
        )
        let remoteLLMConfigurations = RemoteModelConfigurationStore.loadConfigurations(
            from: defaults.string(forKey: AppPreferenceKey.remoteLLMProviderConfigurations) ?? "",
            sensitiveValueLoading: .metadataOnly
        )
        let resolvedRemoteASRConfiguration = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: remoteASRProvider,
            stored: remoteASRConfigurations
        )
        let resolvedRemoteLLMConfiguration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: remoteLLMProvider,
            stored: remoteLLMConfigurations
        )
        let selectedEngine = TranscriptionEngine.resolved(
            rawValue: defaults.string(forKey: AppPreferenceKey.transcriptionEngine)
        )
        let enhancementMode = EnhancementMode(rawValue: defaults.string(forKey: AppPreferenceKey.enhancementMode) ?? "")
            ?? .off
        let mlxModelRepo = MLXModelManager.canonicalModelRepo(
            defaults.string(forKey: AppPreferenceKey.mlxModelRepo) ?? MLXModelCatalog.defaultModelRepo
        )
        let translationProvider = TranslationModelProvider(
            rawValue: defaults.string(forKey: AppPreferenceKey.translationModelProvider) ?? ""
        ) ?? .customLLM
        let rewriteProvider = RewriteModelProvider(
            rawValue: defaults.string(forKey: AppPreferenceKey.rewriteModelProvider) ?? ""
        ) ?? .customLLM
        let translationRemoteProvider = RemoteLLMProvider(
            rawValue: defaults.string(forKey: AppPreferenceKey.translationRemoteLLMProvider) ?? ""
        )
        let rewriteRemoteProvider = RemoteLLMProvider(
            rawValue: defaults.string(forKey: AppPreferenceKey.rewriteRemoteLLMProvider) ?? ""
        )
        let hotkeyPreset = HotkeyPreference.Preset(
            rawValue: defaults.string(forKey: AppPreferenceKey.hotkeyPreset) ?? ""
        ) ?? .fnCombo
        let hotkeyTriggerMode = HotkeyPreference.TriggerMode(
            rawValue: defaults.string(forKey: AppPreferenceKey.hotkeyTriggerMode) ?? ""
        ) ?? .longPress
        let interactionSoundPreset = InteractionSoundPreset(
            rawValue: defaults.string(forKey: AppPreferenceKey.interactionSoundPreset) ?? ""
        ) ?? .soft
        let voiceEndPreset = VoiceEndCommandPreset(
            rawValue: defaults.string(forKey: AppPreferenceKey.voiceEndCommandPreset) ?? ""
        ) ?? .over

        var lines: [String] = []
        lines.append("generatedAt: \(VoxtLogFormatter.timestamp(for: Date()))")
        lines.append("appVersion: \(bundleVersionText())")
        lines.append("bundleID: \(Bundle.main.bundleIdentifier ?? "unknown")")
        lines.append("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
        lines.append("machine: \(machineSummary())")
        lines.append("locale: \(Locale.current.identifier)")
        lines.append("preferredLanguages: \(Locale.preferredLanguages.joined(separator: ", "))")
        lines.append("timeZone: \(TimeZone.current.identifier)")
        lines.append("selectedTranscriptionEngine: \(selectedEngine.rawValue) [\(selectedEngine.title)]")
        lines.append("enhancementMode: \(enhancementMode.rawValue) [\(enhancementMode.title)]")
        lines.append("realtimeTextDisplayEnabled: \(defaults.bool(forKey: AppPreferenceKey.realtimeTextDisplayEnabled))")
        lines.append("voiceEndCommandEnabled: \(defaults.bool(forKey: AppPreferenceKey.voiceEndCommandEnabled))")
        lines.append("voiceEndCommandPreset: \(voiceEndPreset.rawValue) [\(voiceEndPreset.title)]")
        lines.append("voiceEndCommandText: \(nonEmptyOrPlaceholder(defaults.string(forKey: AppPreferenceKey.voiceEndCommandText), placeholder: "<preset>"))")
        lines.append("escapeKeyCancelsOverlaySession: \(defaults.object(forKey: AppPreferenceKey.escapeKeyCancelsOverlaySession) as? Bool ?? true)")
        lines.append("hotkeyPreset: \(hotkeyPreset.rawValue) [\(hotkeyPreset.title)]")
        lines.append("hotkeyTriggerMode: \(hotkeyTriggerMode.rawValue) [\(hotkeyTriggerMode.title)]")
        lines.append("hotkeyDebugLoggingEnabled: \(defaults.bool(forKey: AppPreferenceKey.hotkeyDebugLoggingEnabled))")
        lines.append("modelDebugLoggingEnabled: \(defaults.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled))")
        lines.append("activeInputDeviceUID: \(nonEmptyOrPlaceholder(defaults.string(forKey: AppPreferenceKey.activeInputDeviceUID)))")
        lines.append("microphoneAutoSwitchEnabled: \(defaults.object(forKey: AppPreferenceKey.microphoneAutoSwitchEnabled) as? Bool ?? true)")
        lines.append("muteSystemAudioWhileRecording: \(defaults.bool(forKey: AppPreferenceKey.muteSystemAudioWhileRecording))")
        lines.append("interactionSoundsEnabled: \(defaults.object(forKey: AppPreferenceKey.interactionSoundsEnabled) as? Bool ?? true)")
        lines.append("interactionSoundPreset: \(interactionSoundPreset.rawValue) [\(interactionSoundPreset.title)]")
        lines.append("historyEnabled: \(defaults.object(forKey: AppPreferenceKey.historyEnabled) as? Bool ?? true)")
        lines.append("historyAudioStorageEnabled: \(defaults.object(forKey: AppPreferenceKey.historyAudioStorageEnabled) as? Bool ?? false)")
        lines.append("useHfMirror: \(defaults.bool(forKey: AppPreferenceKey.useHfMirror))")
        let selectedDownloadSources = defaults.data(forKey: AppPreferenceKey.modelDownloadSourceSelections)
            .flatMap { String(data: $0, encoding: .utf8) }
            ?? "{}"
        lines.append("modelDownloadSourceSelections: \(selectedDownloadSources)")
        lines.append("localModelIdleUnloadDelaySeconds: \(AppPreferenceKey.resolvedLocalModelIdleUnloadDelaySeconds(defaults: defaults))")
        lines.append("mlxModel: \(mlxModelRepo) [\(MLXModelCatalog.displayTitle(for: mlxModelRepo))]")
        lines.append("customLLMModelRepo: \(nonEmptyOrPlaceholder(defaults.string(forKey: AppPreferenceKey.customLLMModelRepo)))")
        lines.append("translationCustomLLMModelRepo: \(nonEmptyOrPlaceholder(defaults.string(forKey: AppPreferenceKey.translationCustomLLMModelRepo)))")
        lines.append("rewriteCustomLLMModelRepo: \(nonEmptyOrPlaceholder(defaults.string(forKey: AppPreferenceKey.rewriteCustomLLMModelRepo)))")
        lines.append("proxyMode: \(proxySettings.mode.rawValue)")
        lines.append("proxyRoute: \(proxyRouteSummary(settings: proxySettings, systemStatus: systemProxyStatus))")
        lines.append("translationProvider: \(translationProvider.rawValue) [\(translationProvider.title)]")
        lines.append("translationRemoteProvider: \(resolvedProviderSummary(translationRemoteProvider))")
        lines.append("rewriteProvider: \(rewriteProvider.rawValue) [\(rewriteProvider.title)]")
        lines.append("rewriteRemoteProvider: \(resolvedProviderSummary(rewriteRemoteProvider))")
        lines.append("selectedRemoteASR: \(remoteASRProvider.rawValue) [\(remoteASRProvider.title)]")
        lines.append("selectedRemoteASRConfiguration: \(remoteASRConfigurationSummary(provider: remoteASRProvider, configuration: resolvedRemoteASRConfiguration))")
        lines.append("selectedRemoteLLM: \(remoteLLMProvider.rawValue) [\(remoteLLMProvider.title)]")
        lines.append("selectedRemoteLLMConfiguration: \(remoteLLMConfigurationSummary(provider: remoteLLMProvider, configuration: resolvedRemoteLLMConfiguration))")
        lines.append("feature.transcription.asrSelectionID: \(featureSettings.transcription.asrSelectionID.rawValue)")
        lines.append("feature.transcription.llmEnabled: \(featureSettings.transcription.llmEnabled)")
        lines.append("feature.transcription.llmSelectionID: \(featureSettings.transcription.llmSelectionID.rawValue)")
        lines.append("feature.transcription.notes.enabled: \(featureSettings.transcription.notes.enabled)")
        lines.append("feature.transcription.notes.titleModelSelectionID: \(featureSettings.transcription.notes.titleModelSelectionID.rawValue)")
        lines.append("feature.transcription.notes.panel.corner: \(featureSettings.transcription.notes.panel.corner.rawValue)")
        lines.append("feature.transcription.notes.panel.revealDelay: \(featureSettings.transcription.notes.panel.revealDelay)")
        lines.append("feature.transcription.notes.panel.hideDelay: \(featureSettings.transcription.notes.panel.hideDelay)")
        lines.append("feature.transcription.notes.panel.isTranslucent: \(featureSettings.transcription.notes.panel.isTranslucent)")
        lines.append("feature.translation.asrSelectionID: \(featureSettings.translation.asrSelectionID.rawValue)")
        lines.append("feature.translation.modelSelectionID: \(featureSettings.translation.modelSelectionID.rawValue)")
        lines.append("feature.translation.targetLanguage: \(featureSettings.translation.targetLanguage.rawValue)")
        lines.append("feature.translation.showResultWindow: \(featureSettings.translation.showResultWindow)")
        lines.append("feature.rewrite.asrSelectionID: \(featureSettings.rewrite.asrSelectionID.rawValue)")
        lines.append("feature.rewrite.llmSelectionID: \(featureSettings.rewrite.llmSelectionID.rawValue)")
        lines.append("feature.rewrite.appEnhancementEnabled: \(featureSettings.rewrite.appEnhancementEnabled)")
        lines.append("feature.rewrite.continueShortcut: \(shortcutSummary(featureSettings.rewrite.continueShortcut))")

        let configuredASRSummaries = RemoteASRProvider.allCases.compactMap { provider -> String? in
            guard let configuration = remoteASRConfigurations[provider.rawValue],
                  configuration.isConfigured || configuration.hasUsableModel || !configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return nil
            }
            return remoteASRConfigurationSummary(provider: provider, configuration: configuration)
        }
        let configuredLLMSummaries = RemoteLLMProvider.allCases.compactMap { provider -> String? in
            guard let configuration = remoteLLMConfigurations[provider.rawValue],
                  configuration.isConfigured || configuration.hasUsableModel || !configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return nil
            }
            return remoteLLMConfigurationSummary(provider: provider, configuration: configuration)
        }
        lines.append("configuredRemoteASRProviders: \(configuredASRSummaries.isEmpty ? "<none>" : configuredASRSummaries.joined(separator: " | "))")
        lines.append("configuredRemoteLLMProviders: \(configuredLLMSummaries.isEmpty ? "<none>" : configuredLLMSummaries.joined(separator: " | "))")

        return lines.joined(separator: "\n")
    }

    private nonisolated static func bundleVersionText() -> String {
        let bundle = Bundle.main
        let shortVersion = (bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let buildVersion = (bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        switch (shortVersion?.isEmpty == false ? shortVersion : nil, buildVersion?.isEmpty == false ? buildVersion : nil) {
        case let (shortVersion?, buildVersion?):
            return "\(shortVersion) (\(buildVersion))"
        case let (shortVersion?, nil):
            return shortVersion
        case let (nil, buildVersion?):
            return buildVersion
        default:
            return "unknown"
        }
    }

    private nonisolated static func machineSummary() -> String {
        let model = sysctlString(named: "hw.model")
        let machine = sysctlString(named: "hw.machine")
        if model == machine {
            return model
        }
        return "\(model) / \(machine)"
    }

    private nonisolated static func sysctlString(named name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
            return "unknown"
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            return "unknown"
        }
        let value = String(cString: buffer).trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? "unknown" : value
    }

    @MainActor
    private static func proxyRouteSummary(
        settings: VoxtNetworkSession.ProxySettings,
        systemStatus: VoxtNetworkSession.SystemProxyStatus
    ) -> String {
        switch settings.mode {
        case .system:
            return systemStatus.preferredSummary ?? "<none>"
        case .disabled:
            return "direct"
        case .custom:
            let host = settings.host.isEmpty ? "<missing-host>" : settings.host
            let port = settings.port.map(String.init) ?? "<missing-port>"
            let auth = settings.hasCredentials ? "auth" : "noauth"
            return "\(settings.scheme.rawValue)://\(host):\(port) [\(auth)]"
        }
    }

    @MainActor
    private static func remoteASRConfigurationSummary(
        provider: RemoteASRProvider,
        configuration: RemoteProviderConfiguration
    ) -> String {
        let model = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let endpoint = configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        var extras: [String] = [
            "provider=\(provider.rawValue)",
            "configured=\(configuration.isConfigured)",
            "model=\(model.isEmpty ? "<empty>" : model)",
            "endpoint=\(endpoint.isEmpty ? "<default>" : endpoint)"
        ]
        if provider == .openAIWhisper {
            extras.append("pseudoRealtime=\(configuration.openAIChunkPseudoRealtimeEnabled)")
        }
        if provider == .aliyunBailianASR {
            extras.append("route=\(aliyunASRRouteSummary(for: configuration.model))")
        }
        return extras.joined(separator: ", ")
    }

    @MainActor
    private static func remoteLLMConfigurationSummary(
        provider: RemoteLLMProvider,
        configuration: RemoteProviderConfiguration
    ) -> String {
        let model = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let endpoint = configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        var extras: [String] = [
            "provider=\(provider.rawValue)",
            "configured=\(configuration.isConfigured)",
            "model=\(model.isEmpty ? "<empty>" : model)",
            "endpoint=\(endpoint.isEmpty ? "<default>" : endpoint)",
            "searchEnabled=\(configuration.searchEnabled)"
        ]
        if provider == .codex {
            extras.append("fastMode=\(configuration.codexFastModeEnabled)")
        }
        return extras.joined(separator: ", ")
    }

    @MainActor
    private static func aliyunASRRouteSummary(for model: String) -> String {
        if let kind = RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: model) {
            switch kind {
            case .qwenASR:
                return "qwen-realtime"
            case .omniASR:
                return "omni-realtime"
            }
        }
        if RemoteASREndpointSupport.isAliyunFunRealtimeModel(model) {
            return "fun-realtime"
        }
        if RemoteASREndpointSupport.isAliyunFileTranscriptionModel(model) {
            return "file-transcription"
        }
        return "unknown"
    }

    @MainActor
    private static func resolvedProviderSummary(_ provider: RemoteLLMProvider?) -> String {
        guard let provider else { return "<unset>" }
        return "\(provider.rawValue) [\(provider.title)]"
    }

    @MainActor
    private static func shortcutSummary(_ shortcut: FeatureShortcutSettings) -> String {
        "keyCode=\(shortcut.keyCode), modifiers=\(shortcut.modifiers.rawValue), sidedModifiers=\(shortcut.sidedModifiers.rawValue)"
    }

    private nonisolated static func nonEmptyOrPlaceholder(_ value: String?, placeholder: String = "<empty>") -> String {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? placeholder : trimmed
    }
}
