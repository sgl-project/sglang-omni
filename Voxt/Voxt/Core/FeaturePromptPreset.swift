// FeaturePromptPreset.swift
// Provides built-in editable prompt presets for feature settings.

import Foundation

struct FeaturePromptPreset: Identifiable, Equatable {
    let id: String
    let title: String
    let prompt: String
}

enum FeaturePromptPresetCatalog {
    static let customPresetID = "custom"
    private static let supportedLanguages: [AppInterfaceLanguage] = [
        .english,
        .chineseSimplified,
        .japanese
    ]

    static var localizedResourceNames: Set<String> {
        Set(
            AppPromptKind.allCases.flatMap { kind in
                definitions(for: kind).compactMap { definition in
                    guard definition.id != defaultPresetID(for: kind) else { return nil }
                    return "\(kind.promptResource.rawValue)-\(definition.id)"
                }
            }
        )
    }

    static func defaultPresetID(for kind: AppPromptKind) -> String? {
        switch kind {
        case .enhancement:
            return "precise"
        case .translation:
            return "precise"
        case .rewrite:
            return "strict"
        default:
            return nil
        }
    }

    static func presets(
        for kind: AppPromptKind,
        language: AppInterfaceLanguage = AppLocalization.language
    ) -> [FeaturePromptPreset] {
        definitions(for: kind).compactMap { definition in
            let prompt: String?
            if definition.id == defaultPresetID(for: kind) {
                prompt = AppPromptDefaults.text(for: kind, language: language)
            } else {
                prompt = AppPromptResourceStore.presetText(
                    for: kind,
                    presetID: definition.id,
                    language: language
                )
            }

            guard let prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return nil
            }
            return FeaturePromptPreset(
                id: definition.id,
                title: AppLocalization.localizedString(definition.titleKey),
                prompt: prompt
            )
        }
    }

    static func preset(
        id: String?,
        for kind: AppPromptKind,
        language: AppInterfaceLanguage = AppLocalization.language
    ) -> FeaturePromptPreset? {
        guard let id else { return nil }
        return presets(for: kind, language: language).first { $0.id == id }
    }

    static func inferredPresetID(
        storedID: String?,
        prompt: String,
        kind: AppPromptKind,
        language: AppInterfaceLanguage
    ) -> String? {
        if let storedID,
           preset(id: storedID, for: kind, language: language) != nil {
            return storedID
        }

        let normalizedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedPrompt.isEmpty else {
            return defaultPresetID(for: kind)
        }
        return presets(for: kind, language: language)
            .first { $0.prompt.trimmingCharacters(in: .whitespacesAndNewlines) == normalizedPrompt }?
            .id
    }

    static func resolvedPrompt(
        storedID: String?,
        prompt: String,
        kind: AppPromptKind,
        language: AppInterfaceLanguage
    ) -> String {
        guard let storedID,
              let localizedPreset = preset(id: storedID, for: kind, language: language)
        else {
            return prompt
        }

        let normalizedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let matchesBuiltInTemplate = supportedLanguages.contains { candidateLanguage in
            preset(id: storedID, for: kind, language: candidateLanguage)?
                .prompt
                .trimmingCharacters(in: .whitespacesAndNewlines) == normalizedPrompt
        }
        return matchesBuiltInTemplate ? localizedPreset.prompt : prompt
    }

    private struct Definition {
        let id: String
        let titleKey: String
    }

    private static func definitions(for kind: AppPromptKind) -> [Definition] {
        switch kind {
        case .enhancement:
            return [
                Definition(id: "precise", titleKey: "Precise Cleanup"),
                Definition(id: "structured", titleKey: "Clear Structure")
            ]
        case .translation:
            return [
                Definition(id: "precise", titleKey: "Precise Cleanup"),
                Definition(id: "natural", titleKey: "Natural Translation")
            ]
        case .rewrite:
            return [
                Definition(id: "strict", titleKey: "Strict Instructions")
            ]
        default:
            return []
        }
    }
}
