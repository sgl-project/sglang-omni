// NoteTitleSupport.swift
// Provides Note Title Support for app lifecycle and routing.

import Foundation
import AppKit

private enum VoxtNoteTitleModel {
    case appleIntelligence
    case customLLM(repo: String)
    case remoteLLM(provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration)
}

extension AppDelegate {
    @discardableResult
    func appendVoxtNote(
        text: String,
        sessionID: UUID,
        source: VoxtNoteSource = .transcription
    ) -> Bool {
        let fallbackTitle = VoxtNoteTitleSupport.fallbackTitle(from: text)
        let resolvedTitleModel = resolvedVoxtNoteTitleModel()
        let shouldEnhanceText = source == .transcription && enhancementMode != .off
        let initialState: NoteTitleGenerationState = resolvedTitleModel == nil && !shouldEnhanceText
            ? .fallback
            : .pending

        guard let item = noteStore.append(
            sessionID: sessionID,
            text: text,
            title: fallbackTitle,
            titleGenerationState: initialState,
            source: source
        ) else {
            return false
        }

        noteWindowManager.show()

        if shouldEnhanceText {
            Task { @MainActor [weak self] in
                await self?.enhanceVoxtNoteAndGenerateTitle(
                    for: item.id,
                    rawText: item.text,
                    titleModel: resolvedTitleModel
                )
            }
            return true
        }

        guard let resolvedTitleModel else { return true }
        Task { @MainActor [weak self] in
            await self?.generateVoxtNoteTitle(
                for: item.id,
                text: item.text,
                fallbackTitle: fallbackTitle,
                model: resolvedTitleModel
            )
        }
        return true
    }

    private func enhanceVoxtNoteAndGenerateTitle(
        for noteID: UUID,
        rawText: String,
        titleModel: VoxtNoteTitleModel?
    ) async {
        let resolvedText: String
        do {
            let enhancedText = try await enhanceTextForCurrentMode(rawText)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            resolvedText = enhancedText.isEmpty ? rawText : enhancedText
            VoxtLog.history(
                "Voxt note enhancement completed. noteID=\(noteID.uuidString), inputChars=\(rawText.count), outputChars=\(resolvedText.count)"
            )
        } catch {
            resolvedText = rawText
            VoxtLog.historyWarning(
                "Voxt note enhancement failed; preserving raw text. noteID=\(noteID.uuidString), error=\(error.localizedDescription)"
            )
        }

        guard let currentItem = noteStore.updateText(
            resolvedText,
            ifUnchangedFrom: rawText,
            for: noteID
        ) else { return }

        let finalText = currentItem.text
        let fallbackTitle = VoxtNoteTitleSupport.fallbackTitle(from: finalText)
        guard let titleModel else {
            _ = noteStore.updateTitle(fallbackTitle, state: .fallback, for: noteID)
            return
        }
        await generateVoxtNoteTitle(
            for: noteID,
            text: finalText,
            fallbackTitle: fallbackTitle,
            model: titleModel
        )
    }

    private func generateVoxtNoteTitle(
        for noteID: UUID,
        text: String,
        fallbackTitle: String,
        model: VoxtNoteTitleModel
    ) async {
        do {
            let generatedTitle = try await runVoxtNoteTitlePrompt(
                voxtNoteTitlePrompt(for: text),
                model: model
            )
            let normalizedTitle = VoxtNoteTitleSupport.normalizedGeneratedTitle(generatedTitle)
            let resolvedTitle = normalizedTitle.isEmpty ? fallbackTitle : normalizedTitle
            let resolvedState: NoteTitleGenerationState = normalizedTitle.isEmpty ? .fallback : .generated
            _ = noteStore.updateTitle(resolvedTitle, state: resolvedState, for: noteID)
            VoxtLog.history(
                "Voxt note title generated. noteID=\(noteID.uuidString), state=\(resolvedState.rawValue), titleChars=\(resolvedTitle.count)"
            )
        } catch {
            _ = noteStore.updateTitle(fallbackTitle, state: .fallback, for: noteID)
            VoxtLog.historyWarning("Voxt note title generation failed. noteID=\(noteID.uuidString), error=\(error.localizedDescription)")
        }
    }

    private func resolvedVoxtNoteTitleModel() -> VoxtNoteTitleModel? {
        switch noteFeatureSettings.titleModelSelectionID.textSelection {
        case .appleIntelligence:
            guard let enhancer else { return nil }
            if #available(macOS 26.0, *) {
                guard TextEnhancer.isAvailable else { return nil }
                _ = enhancer
                return .appleIntelligence
            }
            return nil
        case .localLLM(let repo):
            guard customLLMManager.canAttemptInference(repo: repo) else { return nil }
            return .customLLM(repo: repo)
        case .remoteLLM(let provider):
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            ) else {
                return nil
            }
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            return .remoteLLM(provider: provider, configuration: configuration)
        case .none:
            return nil
        }
    }

    private func runVoxtNoteTitlePrompt(
        _ prompt: String,
        model: VoxtNoteTitleModel
    ) async throws -> String {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return "" }

        switch model {
        case .appleIntelligence:
            guard let enhancer else {
                throw NSError(
                    domain: "Voxt.NoteTitle",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence is unavailable.")]
                )
            }
            if #available(macOS 26.0, *) {
                return try await enhancer.enhance(userPrompt: trimmedPrompt)
            }
            throw NSError(
                domain: "Voxt.NoteTitle",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence requires macOS 26 or later.")]
            )
        case .customLLM(let repo):
            return try await customLLMManager.enhance(userPrompt: trimmedPrompt, repo: repo)
        case .remoteLLM(let provider, let configuration):
            return try await RemoteLLMRuntimeClient().enhance(
                userPrompt: trimmedPrompt,
                provider: provider,
                configuration: configuration
            )
        }
    }

    private func voxtNoteTitlePrompt(for text: String) -> String {
        """
        You are Voxt's note title generator.

        Generate a very short plain-text title for the note below.

        Rules:
        1. Reply in the user's main language.
        2. Return one line only.
        3. Keep it concise and specific.
        4. Avoid quotes, numbering, markdown, or extra explanation.
        5. Prefer 4-8 words, or under 20 Chinese characters.

        User main language: \(userMainLanguagePromptValue)

        Note text:
        \(text)
        """
    }
}
