// AppPromptDefaults.swift
// Provides App Prompt Defaults for core app behavior.

import Foundation
import CryptoKit

enum AppPromptKind: CaseIterable {
    case enhancement
    case translation
    case rewrite
    case transcriptSummary
    case dictionaryIngest
    case dictionaryAutoLearning
    case qwenASRContextBias
    case openAIASRHint
    case glmASRHint
}

enum AppPromptDefaults {
    private static let transcriptPromptCurrentToken = TranscriptSummarySupport.transcriptRecordTemplateVariable
    private static let transcriptPromptLegacyToken = "{{MEETING_RECORD}}"

    static func interfaceLanguage(from defaults: UserDefaults = .standard) -> AppInterfaceLanguage {
        let rawValue = defaults.string(forKey: AppPreferenceKey.interfaceLanguage)
        return AppInterfaceLanguage(rawValue: rawValue ?? "") ?? .system
    }

    static func text(for kind: AppPromptKind, language: AppInterfaceLanguage = AppLocalization.language) -> String {
        switch resolvedLanguage(language) {
        case .english:
            return englishText(for: kind)
        case .chineseSimplified:
            return chineseSimplifiedText(for: kind)
        case .japanese:
            return japaneseText(for: kind)
        case .system:
            return englishText(for: kind)
        }
    }

    static func text(for kind: AppPromptKind, resolvedFrom defaults: UserDefaults) -> String {
        text(for: kind, language: interfaceLanguage(from: defaults))
    }

    static func resolvedStoredText(
        _ storedText: String?,
        kind: AppPromptKind,
        defaults: UserDefaults = .standard
    ) -> String {
        let normalizedStoredText = normalizeStoredText(storedText, kind: kind)
        let trimmedText = normalizedStoredText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmedText.isEmpty || matchesKnownDefault(trimmedText, kind: kind) {
            return text(for: kind, resolvedFrom: defaults)
        }
        return normalizedStoredText ?? ""
    }

    static func canonicalStoredText(_ text: String, kind: AppPromptKind) -> String {
        let normalizedText = normalizeStoredText(text, kind: kind) ?? text
        let trimmedText = normalizedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return "" }
        return matchesKnownDefault(trimmedText, kind: kind) ? "" : normalizedText
    }

    static func matchesKnownDefault(_ text: String, kind: AppPromptKind) -> Bool {
        let normalizedText = normalizeStoredText(text, kind: kind) ?? text
        let trimmedText = normalizedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return false }

        let localizedDefaults = [AppInterfaceLanguage.english, .chineseSimplified, .japanese]
            .map { self.text(for: kind, language: $0).trimmingCharacters(in: .whitespacesAndNewlines) }

        if localizedDefaults.contains(trimmedText) {
            return true
        }

        if legacyDefaultDigests(for: kind).contains(promptDigest(trimmedText)) {
            return true
        }

        if kind == .qwenASRContextBias,
           matchesLegacyQwenASRContextBiasText(trimmedText) {
            return true
        }

        if matchesLegacyASRLanguagePromptText(trimmedText, kind: kind) {
            return true
        }

        return false
    }

    private static func normalizeStoredText(_ text: String?, kind: AppPromptKind) -> String? {
        guard let text else { return nil }
        guard kind == .transcriptSummary else { return text }
        return text.replacingOccurrences(of: transcriptPromptLegacyToken, with: transcriptPromptCurrentToken)
    }

    private static func resolvedLanguage(_ language: AppInterfaceLanguage) -> AppInterfaceLanguage {
        switch language {
        case .system:
            return .resolvedSystemLanguage
        case .english, .chineseSimplified, .japanese:
            return language
        }
    }

    private static func legacyDefaultDigests(for kind: AppPromptKind) -> Set<String> {
        switch kind {
        case .enhancement:
            return [
                "1653d67b2661988289508482c4d9ecbdef404251308fe38140e1a737248893c5",
                "d0426c874bf54ab9d4cc6d4947d348b78fc158eed8564c7fbe60496c11f25f3b",
                "b9f9680a04c0bcb363170cde0eccbd7e84f0f5a3a5f7e19066ce5eac0b1349ad",
                "57b7358c481764cc653b82b38034d5bc51425464437e126704dfc94a2a37f459",
                "360bd4e764fb6bd1ea4080e170655e31d240b2beff600189446c6316a4ddc02c",
                "30fc6631aeeedb02325d1a2dbb0f0d2d93cc24c7b5654970c688d119c452cf6e",
                "21c438a9a05a34699dbb5199f715e62086f2746858b846fdbe2e6f51e735cfe0",
                "55fa7253c76d272e011c40d7775cfa1bd62c99ff6edd56a2cc04e2930fc32008",
                "6ae4ab890cc3fc5b9d0c147953c2067bada9934ef80df8de2aa498e0f41095a7",
                "63ac6dc353ce8e7e5902f3fa5383c7d23df2b4460461dbc9b749218d42795abf"
            ]
        case .translation:
            return [
                "18f5d0be6a61366d16508f5a31f3b3576141e4feba9b18d0a96d007d1958e416",
                "a0ceeb27b2352b08c98065c24bdd68b8127913aa9a2516de3c290de02ffbc32d",
                "998a0a8a67786a9ab76ca5ff81b46b43b984f32cbfce5b5c22fa938da9f89cdc",
                "e0319e193a842e00609a9e9ba4212013bbe7a909a2974f2fbe5426e21448f566",
                "cd193cee3027a3ba4a6f24dd548c268fb177b76c86ae6abca27c5e70f70cb9fd",
                "7949566e9e8848e1989800a593c719eedea2a7934d015dc0244aadc63cdca7f4",
                "f59232f69bf85ac67b7deb0ff30d5e529363a4589c662fec4184efd921e9d0ca",
                "0a00a63561f0670abbdd5dc57efd96829124fd68cb1f691539c93cfa68f3373f",
                "585d0f4bf8a21d55aa669c5b16aa7c7ef77adb2417acbc88c85c36a85e8b3aec"
            ]
        case .rewrite:
            return [
                // Defaults retired with automatic window-context capture.
                "896482a35261df59cd0e37f50265e7ca9b28b942796f250381bf26d5bbbf0b74",
                "d662ecb5d34cada98d06f112cd51b7bd299b5114f5531a5b8d21a0a18d08c4d1",
                "5ed3a58145fa218cac67277b95042d84a4d153eb30de7e411668555d4378972f",
                "09f35f472123a67b863f54e0d8f9429a097047060cbe82b97145efa85b4bb03d",
                "5b3bb1cd6779e3ca43b727fb8e91066c749ac4c9b87dce66f1da9606b8ad3bb5",
                "821bb73ad9372d231078bee8ac4d8ab8223c5c4010cce0d8fc25510ca12696cb"
            ]
        case .qwenASRContextBias:
            return [
                "faa073ae8ad5b56d7ae66b78eee186ebc198f0490235cea6ad7e7f6a1dcdb8de",
                "fe4f1c8dd5529ae266e3f4df3d5773ec1d799855eb8736bb3c6e0f5422377497",
                "857db2aa4d9011bf501af532934cc572084de0c2473f8be15de6d7ef094ae963"
            ]
        default:
            return []
        }
    }

    private static func promptDigest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func matchesLegacyQwenASRContextBiasText(_ text: String) -> Bool {
        let markerSets = [
            [
                "The speaker's primary language is",
                "Other commonly used languages",
                "Bias recognition toward correct spelling",
                "Prefer these dictionary terms"
            ],
            [
                "说话者的主要语言",
                "其他常用语言",
                "请将识别偏向于",
                "词典词汇"
            ],
            [
                "話者の主要言語",
                "その他のよく使う言語",
                "認識を寄せてください",
                "辞書語"
            ]
        ]

        return markerSets.contains { markers in
            markers.allSatisfy { text.contains($0) }
        }
    }

    private static func matchesLegacyASRLanguagePromptText(_ text: String, kind: AppPromptKind) -> Bool {
        guard [.openAIASRHint, .glmASRHint].contains(kind) else { return false }
        let markerSets = [
            [
                "The speaker's primary language is",
                "Preserve",
                "exactly as spoken"
            ],
            [
                "说话者的主要语言",
                "按原样保留"
            ],
            [
                "話者の主要言語",
                "発話どおり"
            ]
        ]

        return markerSets.contains { markers in
            markers.allSatisfy { text.contains($0) }
        }
    }

    private static func englishText(for kind: AppPromptKind) -> String {
        resourceText(for: kind, language: .english)
    }

    private static func chineseSimplifiedText(for kind: AppPromptKind) -> String {
        resourceText(for: kind, language: .chineseSimplified)
    }

    private static func japaneseText(for kind: AppPromptKind) -> String {
        resourceText(for: kind, language: .japanese)
    }

    private static func resourceText(for kind: AppPromptKind, language: AppInterfaceLanguage) -> String {
        guard let text = AppPromptResourceStore.text(for: kind, language: language),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            preconditionFailure("Missing bundled prompt resource for \(kind) in \(language.rawValue)")
        }
        return text
    }
}
