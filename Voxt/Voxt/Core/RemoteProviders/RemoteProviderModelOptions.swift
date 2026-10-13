import Foundation

enum OllamaResponseFormat: String, CaseIterable, Identifiable {
    case plain
    case json
    case jsonSchema

    var id: String { rawValue }

    var title: String {
        switch self {
        case .plain:
            return AppLocalization.localizedString("Plain Text")
        case .json:
            return "JSON"
        case .jsonSchema:
            return AppLocalization.localizedString("JSON Schema")
        }
    }
}

enum OllamaThinkMode: String, CaseIterable, Identifiable {
    case off
    case on
    case low
    case medium
    case high

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off:
            return AppLocalization.localizedString("Off")
        case .on:
            return AppLocalization.localizedString("On")
        case .low:
            return AppLocalization.localizedString("Low")
        case .medium:
            return AppLocalization.localizedString("Medium")
        case .high:
            return AppLocalization.localizedString("High")
        }
    }
}

enum OMLXResponseFormat: String, CaseIterable, Identifiable {
    case plain
    case jsonSchema

    var id: String { rawValue }

    var title: String {
        switch self {
        case .plain:
            return AppLocalization.localizedString("Plain Text")
        case .jsonSchema:
            return AppLocalization.localizedString("JSON Schema")
        }
    }
}

enum OpenAIReasoningEffort: String, CaseIterable, Identifiable {
    case automatic
    case none
    case minimal
    case low
    case medium
    case high
    case xhigh

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            return AppLocalization.localizedString("Default")
        case .none:
            return AppLocalization.localizedString("None")
        case .minimal:
            return AppLocalization.localizedString("Minimal")
        case .low:
            return AppLocalization.localizedString("Low")
        case .medium:
            return AppLocalization.localizedString("Medium")
        case .high:
            return AppLocalization.localizedString("High")
        case .xhigh:
            return AppLocalization.localizedString("Extra High")
        }
    }

    static func supportedCases(forModel model: String) -> [OpenAIReasoningEffort] {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized == "gpt-5.2-pro" || normalized.hasPrefix("gpt-5.2-pro-") {
            return [.automatic, .medium, .high, .xhigh]
        }
        if normalized == "gpt-5-pro" || normalized.hasPrefix("gpt-5-pro-") {
            return [.automatic, .high]
        }
        if normalized == "gpt-5.2-codex" || normalized.hasPrefix("gpt-5.2-codex-") {
            return [.automatic, .low, .medium, .high, .xhigh]
        }
        if normalized == "gpt-5.3-codex-spark" || normalized.hasPrefix("gpt-5.3-codex-spark-") {
            return [.automatic, .none, .low, .medium, .high]
        }
        if normalized == "gpt-5.1-codex-max" || normalized.hasPrefix("gpt-5.1-codex-max-") {
            return [.automatic, .none, .medium, .high, .xhigh]
        }
        if normalized == "gpt-5.2" || normalized.hasPrefix("gpt-5.2-") {
            return [.automatic, .none, .low, .medium, .high, .xhigh]
        }
        if normalized == "gpt-5.1" || normalized.hasPrefix("gpt-5.1-") {
            return [.automatic, .none, .low, .medium, .high]
        }
        if normalized.hasPrefix("gpt-5") {
            return [.automatic, .minimal, .low, .medium, .high]
        }
        if normalized.hasPrefix("o1") ||
            normalized.hasPrefix("o3") ||
            normalized.hasPrefix("o4") {
            return [.automatic, .low, .medium, .high]
        }
        return [.automatic]
    }

    static func apiValue(selection: String, model: String) -> String? {
        guard let value = OpenAIReasoningEffort(rawValue: selection),
              value != .automatic,
              supportedCases(forModel: model).contains(value)
        else {
            return nil
        }
        return value.rawValue
    }
}

enum OpenAITextVerbosity: String, CaseIterable, Identifiable {
    case automatic
    case low
    case medium
    case high

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            return AppLocalization.localizedString("Default")
        case .low:
            return AppLocalization.localizedString("Low")
        case .medium:
            return AppLocalization.localizedString("Medium")
        case .high:
            return AppLocalization.localizedString("High")
        }
    }

    static func supportsModel(_ model: String) -> Bool {
        model.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .hasPrefix("gpt-5")
    }

    static func apiValue(selection: String, model: String) -> String? {
        guard supportsModel(model),
              let value = OpenAITextVerbosity(rawValue: selection),
              value != .automatic
        else {
            return nil
        }
        return value.rawValue
    }
}


struct AliyunASRModelSettings: Codable, Hashable {
    var maxSentenceSilenceMilliseconds: Int = 1300
    var serverVADThreshold: Double = 0.35
    var serverVADSilenceDurationMilliseconds: Int = 800
    var useManualCommit: Bool = false
    var semanticPunctuationEnabled: Bool = false
    var punctuationPredictionEnabled: Bool = true
    var inverseTextNormalizationEnabled: Bool = true
    var disfluencyRemovalEnabled: Bool = false

    enum CodingKeys: String, CodingKey {
        case maxSentenceSilenceMilliseconds
        case serverVADThreshold
        case serverVADSilenceDurationMilliseconds
        case useManualCommit
        case semanticPunctuationEnabled
        case punctuationPredictionEnabled
        case inverseTextNormalizationEnabled
        case disfluencyRemovalEnabled
    }

    init(
        maxSentenceSilenceMilliseconds: Int = 1300,
        serverVADThreshold: Double = 0.35,
        serverVADSilenceDurationMilliseconds: Int = 800,
        useManualCommit: Bool = false,
        semanticPunctuationEnabled: Bool = false,
        punctuationPredictionEnabled: Bool = true,
        inverseTextNormalizationEnabled: Bool = true,
        disfluencyRemovalEnabled: Bool = false
    ) {
        self.maxSentenceSilenceMilliseconds = maxSentenceSilenceMilliseconds
        self.serverVADThreshold = serverVADThreshold
        self.serverVADSilenceDurationMilliseconds = serverVADSilenceDurationMilliseconds
        self.useManualCommit = useManualCommit
        self.semanticPunctuationEnabled = semanticPunctuationEnabled
        self.punctuationPredictionEnabled = punctuationPredictionEnabled
        self.inverseTextNormalizationEnabled = inverseTextNormalizationEnabled
        self.disfluencyRemovalEnabled = disfluencyRemovalEnabled
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        maxSentenceSilenceMilliseconds = try container.decodeIfPresent(Int.self, forKey: .maxSentenceSilenceMilliseconds) ?? 1300
        serverVADThreshold = try container.decodeIfPresent(Double.self, forKey: .serverVADThreshold) ?? 0.35
        serverVADSilenceDurationMilliseconds = try container.decodeIfPresent(Int.self, forKey: .serverVADSilenceDurationMilliseconds) ?? 800
        useManualCommit = try container.decodeIfPresent(Bool.self, forKey: .useManualCommit) ?? false
        semanticPunctuationEnabled = try container.decodeIfPresent(Bool.self, forKey: .semanticPunctuationEnabled) ?? false
        punctuationPredictionEnabled = try container.decodeIfPresent(Bool.self, forKey: .punctuationPredictionEnabled) ?? true
        inverseTextNormalizationEnabled = try container.decodeIfPresent(Bool.self, forKey: .inverseTextNormalizationEnabled) ?? true
        disfluencyRemovalEnabled = try container.decodeIfPresent(Bool.self, forKey: .disfluencyRemovalEnabled) ?? false
    }
}
