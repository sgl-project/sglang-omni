// LLMWarmup.swift
// Provides LLMWarmup for app lifecycle and routing.

import Foundation

extension AppDelegate {
    func scheduleLLMIdleWarmupIfNeeded() {
        let remoteWarmupKeys = remoteLLMWarmupContextsForIdle().map(\.key)
        cancelLLMWarmupTasks(except: [])
        cancelRemoteLLMWarmupTasks(except: remoteWarmupKeys)
        for context in remoteLLMWarmupContextsForIdle() {
            startRemoteLLMWarmupIfNeeded(context: context, reason: "idle")
        }
    }

    func prewarmLLMForUpcomingSession(outputMode: SessionOutputMode) {
        for repo in customLLMWarmupReposForSession(outputMode: outputMode) {
            startCustomLLMWarmupIfNeeded(repo: repo, reason: "session-\(RecordingSessionSupport.outputLabel(for: outputMode))")
        }
        for context in remoteLLMWarmupContextsForSession(outputMode: outputMode) {
            startRemoteLLMWarmupIfNeeded(
                context: context,
                reason: "session-\(RecordingSessionSupport.outputLabel(for: outputMode))"
            )
        }
    }

    func prewarmLLMForPendingPostASRProcessing(outputMode: SessionOutputMode) {
        for repo in customLLMWarmupReposForSession(outputMode: outputMode) {
            startCustomLLMWarmupIfNeeded(repo: repo, reason: "post-asr-\(RecordingSessionSupport.outputLabel(for: outputMode))")
        }
        for context in remoteLLMWarmupContextsForSession(outputMode: outputMode) {
            startRemoteLLMWarmupIfNeeded(
                context: context,
                reason: "post-asr-\(RecordingSessionSupport.outputLabel(for: outputMode))"
            )
        }
    }

    func prewarmLLMForCurrentActiveSessionIfNeeded() {
        guard isSessionActive else { return }
        prewarmLLMForPendingPostASRProcessing(outputMode: sessionOutputMode)
    }

    func prewarmSelectedTextTranslationLLMIfNeeded(targetLanguage: TranslationTargetLanguage) {
        let resolution = resolvedTranslationProviderResolution(
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: true
        )
        switch resolution.provider {
        case .customLLM:
            startCustomLLMWarmupIfNeeded(repo: translationCustomLLMRepo, reason: "selected-text-translation")
        case .localGGUF:
            break
        case .remoteLLM:
            let context = resolvedRemoteLLMContext(forTranslation: true)
            startRemoteLLMWarmupIfNeeded(
                context: RemoteWarmupContext(
                    provider: context.provider,
                    configuration: context.configuration
                ),
                reason: "selected-text-translation"
            )
        }
    }

    func cancelAllLLMWarmupTasks() {
        cancelLLMWarmupTasks(except: [])
    }

    private func customLLMWarmupReposForSession(outputMode: SessionOutputMode) -> Set<String> {
        let availability = FeatureSettingsStore.availability()
        var repos = Set<String>()

        switch outputMode {
        case .transcription:
            if let enhancementRepo = resolvedTranscriptionEnhancementLocalRepo() {
                repos.insert(enhancementRepo)
            }
        case .translation:
            guard availability.translationEnabled else { break }
            let resolution = resolvedTranslationProviderResolution(
                targetLanguage: effectiveSessionTranslationTargetLanguage,
                isSelectedTextTranslation: false
            )
            if resolution.provider == .customLLM {
                repos.insert(translationCustomLLMRepo)
            }
        case .rewrite:
            guard availability.rewriteEnabled else { break }
            if rewriteModelProvider == .customLLM {
                repos.insert(rewriteCustomLLMRepo)
            }
        }

        return repos.filter { customLLMManager.canAttemptInference(repo: $0) }
    }

    private struct RemoteWarmupContext {
        let provider: RemoteLLMProvider
        let configuration: RemoteProviderConfiguration

        var key: String {
            let endpoint = configuration.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
            let model = configuration.model.trimmingCharacters(in: .whitespacesAndNewlines)
            return "\(provider.rawValue)|\(endpoint)|\(model)"
        }
    }

    private func remoteLLMWarmupContextsForIdle() -> [RemoteWarmupContext] {
        let availability = FeatureSettingsStore.availability()
        var contexts: [RemoteWarmupContext] = []

        if enhancementMode == .remoteLLM {
            let context = resolvedRemoteLLMContext(forTranslation: false)
            if isStoredRemoteLLMConfigured(context.provider) {
                contexts.append(RemoteWarmupContext(provider: context.provider, configuration: context.configuration))
            }
        }
        if availability.translationEnabled, translationModelProvider == .remoteLLM {
            let context = resolvedRemoteLLMContext(forTranslation: true)
            if isStoredRemoteLLMConfigured(context.provider) {
                contexts.append(RemoteWarmupContext(provider: context.provider, configuration: context.configuration))
            }
        }
        if availability.rewriteEnabled, rewriteModelProvider == .remoteLLM {
            let context = resolvedRemoteLLMContext(forRewrite: true)
            if isStoredRemoteLLMConfigured(context.provider) {
                contexts.append(RemoteWarmupContext(provider: context.provider, configuration: context.configuration))
            }
        }

        return deduplicatedRemoteWarmupContexts(contexts)
    }

    private func remoteLLMWarmupContextsForSession(outputMode: SessionOutputMode) -> [RemoteWarmupContext] {
        let availability = FeatureSettingsStore.availability()
        switch outputMode {
        case .transcription:
            guard enhancementMode == .remoteLLM else { return [] }
            let context = resolvedRemoteLLMContext(forTranslation: false)
            guard isStoredRemoteLLMConfigured(context.provider) else { return [] }
            return [RemoteWarmupContext(provider: context.provider, configuration: context.configuration)]
        case .translation:
            guard availability.translationEnabled else { return [] }
            let resolution = resolvedTranslationProviderResolution(
                targetLanguage: effectiveSessionTranslationTargetLanguage,
                isSelectedTextTranslation: false
            )
            guard resolution.provider == .remoteLLM else { return [] }
            let context = resolvedRemoteLLMContext(forTranslation: true)
            guard isStoredRemoteLLMConfigured(context.provider) else { return [] }
            return [RemoteWarmupContext(provider: context.provider, configuration: context.configuration)]
        case .rewrite:
            guard availability.rewriteEnabled else { return [] }
            guard rewriteModelProvider == .remoteLLM else { return [] }
            let context = resolvedRemoteLLMContext(forRewrite: true)
            guard isStoredRemoteLLMConfigured(context.provider) else { return [] }
            return [RemoteWarmupContext(provider: context.provider, configuration: context.configuration)]
        }
    }

    private func startCustomLLMWarmupIfNeeded(repo: String, reason: String) {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        guard customLLMManager.canAttemptInference(repo: canonicalRepo) else { return }
        guard llmWarmupTasksByRepo[canonicalRepo] == nil else { return }
        guard !customLLMManager.isModelLoaded(repo: canonicalRepo) else { return }

        llmWarmupTasksByRepo[canonicalRepo] = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.llmWarmupTasksByRepo[canonicalRepo] = nil }
            do {
                try await self.customLLMManager.prewarmModel(repo: canonicalRepo)
                VoxtLog.llmInfo("Custom LLM warmup completed. repo=\(canonicalRepo), reason=\(reason)", verbose: true)
            } catch {
                VoxtLog.llmWarning("Custom LLM warmup failed. repo=\(canonicalRepo), reason=\(reason), error=\(error.localizedDescription)")
            }
        }
    }

    private func cancelLLMWarmupTasks(except reposToKeep: Set<String>) {
        let reposToCancel = llmWarmupTasksByRepo.keys.filter { !reposToKeep.contains($0) }
        for repo in reposToCancel {
            llmWarmupTasksByRepo[repo]?.cancel()
            llmWarmupTasksByRepo.removeValue(forKey: repo)
        }
    }

    private func startRemoteLLMWarmupIfNeeded(context: RemoteWarmupContext, reason: String) {
        guard remoteLLMWarmupTasksByKey[context.key] == nil else { return }

        remoteLLMWarmupTasksByKey[context.key] = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.remoteLLMWarmupTasksByKey[context.key] = nil }
            do {
                try await RemoteLLMRuntimeClient().warmupConnection(
                    provider: context.provider,
                    configuration: context.configuration
                )
                VoxtLog.llmInfo(
                    "Remote LLM warmup completed. provider=\(context.provider.rawValue), model=\(context.configuration.model), reason=\(reason)",
                    verbose: true
                )
            } catch {
                VoxtLog.llmWarning(
                    "Remote LLM warmup failed. provider=\(context.provider.rawValue), model=\(context.configuration.model), reason=\(reason), error=\(error.localizedDescription)"
                )
            }
        }
    }

    private func cancelRemoteLLMWarmupTasks(except keysToKeep: [String]) {
        let keep = Set(keysToKeep)
        let keysToCancel = remoteLLMWarmupTasksByKey.keys.filter { !keep.contains($0) }
        for key in keysToCancel {
            remoteLLMWarmupTasksByKey[key]?.cancel()
            remoteLLMWarmupTasksByKey.removeValue(forKey: key)
        }
    }

    private func deduplicatedRemoteWarmupContexts(_ contexts: [RemoteWarmupContext]) -> [RemoteWarmupContext] {
        var seen = Set<String>()
        return contexts.filter { context in
            seen.insert(context.key).inserted
        }
    }
}
