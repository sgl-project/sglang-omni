// ASRHintResolver.swift
// Provides ASRHint Resolver for transcription processing.

import Foundation

struct ResolvedASRHintPayload {
    var language: String?
    var languageHints: [String] = []
    var chineseOutputVariant: String?
    var prompt: String?
    var otherLanguages: [String] = []
    var multilingualContext: String?
    var contextualPhrases: [String] = []
}

struct ResolvedDictationSettings: Equatable {
    var localeIdentifier: String?
    var contextualPhrases: [String]
    var prefersOnDeviceRecognition: Bool
    var addsPunctuation: Bool
    var reportsPartialResults: Bool
}

@MainActor
enum ASRHintResolver {
    static func resolve(
        target: ASRHintTarget,
        settings: ASRHintSettings,
        userLanguageCodes: [String],
        mlxModelRepo: String? = nil,
        dictionaryTerms: String = ""
    ) -> ResolvedASRHintPayload {
        let selectedOptions = selectedLanguageOptions(userLanguageCodes)
        let mainLanguage = selectedOptions.first ?? UserMainLanguageOption.fallbackOption()
        let otherLanguageOptions = Array(selectedOptions.dropFirst())
        let prompt = resolvePrompt(
            for: target,
            template: settings.promptTemplate,
            mainLanguage: mainLanguage,
            otherLanguages: otherLanguageOptions,
            dictionaryTerms: dictionaryTerms
        )
        let otherLanguages = otherLanguageOptions.map(\.promptName)
        let contextualPhrases = ASRHintSettingsStore.contextualPhrases(from: settings)
        let usesExplicitSingleLanguageHint = settings.followsUserMainLanguage && otherLanguageOptions.isEmpty
        let mlxResolvedLanguage = settings.followsUserMainLanguage
            ? resolvedMLXLanguageHint(
                mainLanguage: mainLanguage,
                otherLanguages: otherLanguageOptions,
                modelRepo: mlxModelRepo
            )
            : nil
        let multilingualContext = settings.followsUserMainLanguage
            ? resolvedMultilingualContext(mainLanguage: mainLanguage, otherLanguages: otherLanguageOptions)
            : nil

        switch target {
        case .dictation:
            return ResolvedASRHintPayload()
        case .mlxAudio:
            return ResolvedASRHintPayload(
                language: mlxResolvedLanguage,
                prompt: nil,
                otherLanguages: otherLanguages,
                multilingualContext: multilingualContext
            )
        case .openAIWhisper:
            return ResolvedASRHintPayload(
                language: usesExplicitSingleLanguageHint ? resolvedOpenAILanguage(mainLanguage) : nil,
                prompt: prompt,
                otherLanguages: otherLanguages
            )
        case .glmASR:
            return ResolvedASRHintPayload(
                language: nil,
                prompt: prompt,
                otherLanguages: otherLanguages
            )
        case .doubaoASR:
            return ResolvedASRHintPayload(
                language: usesExplicitSingleLanguageHint ? resolvedDoubaoLanguage(mainLanguage) : nil,
                chineseOutputVariant: resolvedDoubaoChineseVariant(mainLanguage),
                prompt: nil,
                otherLanguages: otherLanguages
            )
        case .aliyunBailianASR:
            let hints = settings.followsUserMainLanguage
                ? resolvedAliyunLanguageHints(options: selectedOptions, model: mlxModelRepo)
                : []
            let terms = resolvedAliyunFunTerms(
                contextualPhrases: contextualPhrases,
                dictionaryTerms: dictionaryTerms
            )
            return ResolvedASRHintPayload(
                language: hints.first,
                languageHints: hints,
                prompt: nil,
                otherLanguages: otherLanguages,
                contextualPhrases: terms
            )
        case .stepFunASR:
            let terms = resolvedStepFunTerms(
                contextualPhrases: contextualPhrases,
                dictionaryTerms: dictionaryTerms
            )
            return ResolvedASRHintPayload(
                language: usesExplicitSingleLanguageHint ? resolvedStepFunLanguage(mainLanguage) : nil,
                prompt: resolvedStepFunPrompt(terms: terms),
                otherLanguages: otherLanguages,
                contextualPhrases: terms
            )
        case .xiaomiMiMoASR:
            return ResolvedASRHintPayload(
                language: usesExplicitSingleLanguageHint ? resolvedXiaomiMiMoLanguage(mainLanguage) : "auto",
                prompt: nil,
                otherLanguages: otherLanguages
            )
        case .googleGeminiASR:
            let hints = settings.followsUserMainLanguage
                ? resolvedGeminiLanguageHints(options: selectedOptions)
                : []
            let terms = resolvedGeminiTerms(
                contextualPhrases: contextualPhrases,
                dictionaryTerms: dictionaryTerms
            )
            return ResolvedASRHintPayload(
                language: hints.first,
                languageHints: hints,
                prompt: nil,
                otherLanguages: otherLanguages,
                contextualPhrases: terms
            )
        }
    }

    static func selectedLanguageOptions(_ userLanguageCodes: [String]) -> [UserMainLanguageOption] {
        UserMainLanguageOption
            .sanitizedSelection(userLanguageCodes)
            .compactMap(UserMainLanguageOption.option(for:))
    }

    static func selectedLanguageSummary(_ userLanguageCodes: [String]) -> String {
        selectedLanguageOptions(userLanguageCodes)
            .map(\.promptName)
            .joined(separator: ", ")
    }

    static func secondaryLanguageSummary(_ userLanguageCodes: [String]) -> String {
        let secondary = selectedLanguageOptions(userLanguageCodes)
            .dropFirst()
            .map(\.promptName)
        return secondary.isEmpty ? AppLocalization.localizedString("Not applied") : secondary.joined(separator: ", ")
    }

    static func outputVariantDescription(for mainLanguage: UserMainLanguageOption) -> String {
        guard mainLanguage.isChinese else {
            return AppLocalization.localizedString("Not applied")
        }
        return mainLanguage.isTraditionalChinese
            ? AppLocalization.localizedString("Traditional Chinese")
            : AppLocalization.localizedString("Simplified Chinese")
    }

    static func resolveDictationSettings(
        settings: ASRHintSettings,
        userLanguageCodes: [String]
    ) -> ResolvedDictationSettings {
        let mainLanguage = UserMainLanguageOption
            .sanitizedSelection(userLanguageCodes)
            .compactMap(UserMainLanguageOption.option(for:))
            .first ?? UserMainLanguageOption.fallbackOption()

        return ResolvedDictationSettings(
            localeIdentifier: settings.followsUserMainLanguage ? resolvedDictationLocaleIdentifier(mainLanguage) : nil,
            contextualPhrases: ASRHintSettingsStore.contextualPhrases(from: settings),
            prefersOnDeviceRecognition: settings.prefersOnDeviceRecognition,
            addsPunctuation: settings.addsPunctuation,
            reportsPartialResults: settings.reportsPartialResults
        )
    }

    static func resolveTemplateVariables(
        in template: String,
        userLanguageCodes: [String],
        dictionaryTerms: String = "",
        appendOtherLanguagesWhenMissing: Bool = false
    ) -> String {
        let selectedOptions = selectedLanguageOptions(userLanguageCodes)
        let mainLanguage = selectedOptions.first ?? UserMainLanguageOption.fallbackOption()
        let otherLanguages = Array(selectedOptions.dropFirst())
        return resolveTemplateVariables(
            in: template,
            mainLanguage: mainLanguage,
            otherLanguages: otherLanguages,
            dictionaryTerms: dictionaryTerms,
            appendOtherLanguagesWhenMissing: appendOtherLanguagesWhenMissing
        )
    }

    private static func resolvePrompt(
        for target: ASRHintTarget,
        template: String,
        mainLanguage: UserMainLanguageOption,
        otherLanguages: [UserMainLanguageOption],
        dictionaryTerms: String
    ) -> String? {
        guard target.supportsPromptEditor else { return nil }
        let trimmed = template.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmed.isEmpty else { return nil }

        let resolved = resolveTemplateVariables(
            in: trimmed,
            mainLanguage: mainLanguage,
            otherLanguages: otherLanguages,
            dictionaryTerms: dictionaryTerms,
            appendOtherLanguagesWhenMissing: false
        )
        let compact = resolved.trimmingCharacters(in: .whitespacesAndNewlines)
        return compact.isEmpty ? nil : compact
    }

    private static func resolveTemplateVariables(
        in template: String,
        mainLanguage: UserMainLanguageOption,
        otherLanguages: [UserMainLanguageOption],
        dictionaryTerms: String,
        appendOtherLanguagesWhenMissing: Bool
    ) -> String {
        let trimmed = template.trimmingCharacters(in: .whitespacesAndNewlines)
        let otherLanguagesSummary = otherLanguages.isEmpty
            ? "None specified"
            : otherLanguages.map(\.promptName).joined(separator: ", ")

        var resolved = trimmed
            .replacingOccurrences(
                of: AppPreferenceKey.asrUserMainLanguageTemplateVariable,
                with: mainLanguage.promptName
            )
            .replacingOccurrences(
                of: AppPreferenceKey.asrUserOtherLanguagesTemplateVariable,
                with: otherLanguagesSummary
            )
            .replacingOccurrences(
                of: AppPreferenceKey.asrDictionaryTermsTemplateVariable,
                with: dictionaryTerms.trimmingCharacters(in: .whitespacesAndNewlines)
            )

        if appendOtherLanguagesWhenMissing,
           !otherLanguages.isEmpty,
           !trimmed.contains(AppPreferenceKey.asrUserOtherLanguagesTemplateVariable) {
            resolved += "\nOther frequently used languages: \(otherLanguagesSummary)."
        }

        return resolved
    }

    private static func resolvedMultilingualContext(
        mainLanguage: UserMainLanguageOption,
        otherLanguages: [UserMainLanguageOption]
    ) -> String? {
        guard !otherLanguages.isEmpty else { return nil }
        let otherLanguagesSummary = otherLanguages.map(\.promptName).joined(separator: ", ")
        return """
            Primary language: \(mainLanguage.promptName)
            Other frequently used languages: \(otherLanguagesSummary)
            Mixed-language speech may appear. Preserve names, brands, URLs, and code-like text exactly as spoken.
            """
    }

    private static func resolvedStepFunTerms(
        contextualPhrases: [String],
        dictionaryTerms: String
    ) -> [String] {
        mergedTermLines(
            contextualPhrases + dictionaryTerms.components(separatedBy: .newlines)
        )
    }

    private static func resolvedAliyunFunTerms(
        contextualPhrases: [String],
        dictionaryTerms: String
    ) -> [String] {
        mergedTermLines(
            contextualPhrases + dictionaryTerms.components(separatedBy: .newlines)
        )
    }

    private static func resolvedStepFunPrompt(
        terms: [String]
    ) -> String? {
        guard !terms.isEmpty else { return nil }

        return """
            Prefer these terms when they match the audio. Preserve names, product terms, technical terms, URLs, and code-like text exactly as spoken. Do not translate them.
            \(terms.joined(separator: "\n"))
            """
    }

    private static func mergedTermLines(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let key = trimmed.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            guard seen.insert(key).inserted else { continue }
            result.append(trimmed)
        }
        return result
    }

    private static func resolvedOpenAILanguage(_ language: UserMainLanguageOption) -> String {
        language.baseLanguageCode
    }

    private static func resolvedDoubaoLanguage(_ language: UserMainLanguageOption) -> String? {
        switch language.baseLanguageCode {
        case "zh":
            return "zh-CN"
        case "en":
            return "en-US"
        case "ja":
            return "ja-JP"
        case "ko":
            return "ko-KR"
        case "id":
            return "id-ID"
        case "es":
            return "es-MX"
        default:
            return nil
        }
    }

    private static func resolvedDoubaoChineseVariant(_ language: UserMainLanguageOption) -> String? {
        guard language.isChinese else { return nil }
        return language.isTraditionalChinese ? "zh-Hant" : "zh-Hans"
    }

    private static func resolvedAliyunLanguageHints(
        options: [UserMainLanguageOption],
        model: String?
    ) -> [String] {
        let modelID = model?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? model!
            : RemoteASRProvider.aliyunBailianASR.suggestedModel
        let capabilities = AliyunASRModelCapabilities.forModel(modelID)
        let supportedCodes = Set(capabilities.supportedLanguageCodes)
        var seen = Set<String>()
        let mapped = options.compactMap { option -> String? in
            let code = option.baseLanguageCode
            let aliyunCode = code == "tl" && supportedCodes.contains("fil") ? "fil" : code
            return supportedCodes.contains(aliyunCode) ? aliyunCode : nil
        }

        let deduped = mapped.filter { seen.insert($0).inserted }
        return Array(deduped.prefix(capabilities.maximumLanguageHints))
    }

    private static func resolvedStepFunLanguage(_ language: UserMainLanguageOption) -> String {
        language.baseLanguageCode
    }

    /// Gemini takes BCP-47 tags; an empty list is what enables auto-detection
    /// and mid-sentence code-mixing.
    private static func resolvedGeminiLanguageHints(options: [UserMainLanguageOption]) -> [String] {
        var seen = Set<String>()
        return options.compactMap { option in
            let code = option.baseLanguageCode
            guard !code.isEmpty, seen.insert(code).inserted else { return nil }
            return code
        }
    }

    private static func resolvedGeminiTerms(
        contextualPhrases: [String],
        dictionaryTerms: String
    ) -> [String] {
        mergedTermLines(
            contextualPhrases + dictionaryTerms.components(separatedBy: .newlines)
        )
    }

    private static func resolvedXiaomiMiMoLanguage(_ language: UserMainLanguageOption) -> String {
        switch language.baseLanguageCode {
        case "zh", "en":
            return language.baseLanguageCode
        default:
            return "auto"
        }
    }

    private static func resolvedMLXLanguageHint(
        mainLanguage: UserMainLanguageOption,
        otherLanguages: [UserMainLanguageOption],
        modelRepo: String?
    ) -> String? {
        guard let modelRepo else { return nil }
        let capability = MLXModelCatalog.capability(for: modelRepo)
        if capability.requiresExplicitPrimaryLanguage {
            return capability.resolvedLanguage(for: mainLanguage)
        }
        guard otherLanguages.isEmpty else { return nil }
        return capability.resolvedLanguage(for: mainLanguage)
    }

    private static func resolvedDictationLocaleIdentifier(_ mainLanguage: UserMainLanguageOption) -> String {
        switch mainLanguage.code {
        case "zh-hans":
            return "zh-CN"
        case "zh-hant":
            return "zh-TW"
        default:
            return mainLanguage.baseLanguageCode
        }
    }
}
