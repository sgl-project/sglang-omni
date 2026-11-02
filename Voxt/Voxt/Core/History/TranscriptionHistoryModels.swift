import Foundation

enum TranscriptionHistoryKind: Codable, Hashable, Sendable {
    case normal
    case translation
    case rewrite
    case transcript

    var rawValue: String {
        switch self {
        case .normal:
            return "normal"
        case .translation:
            return "translation"
        case .rewrite:
            return "rewrite"
        case .transcript:
            return "transcript"
        }
    }

    init?(rawValue: String) {
        switch rawValue {
        case "normal":
            self = .normal
        case "translation":
            self = .translation
        case "rewrite":
            self = .rewrite
        case "transcript":
            self = .transcript
        default:
            return nil
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        switch rawValue {
        case "normal":
            self = .normal
        case "translation":
            self = .translation
        case "rewrite":
            self = .rewrite
        case "transcript":
            self = .transcript
        default:
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unknown TranscriptionHistoryKind value: \(rawValue)"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

struct WhisperHistoryWordTiming: Codable, Hashable {
    let word: String
    let startSeconds: Double
    let endSeconds: Double
    let probability: Double
}

struct TranscriptionHistoryEntry: Identifiable, Codable, Hashable {
    let id: UUID
    let text: String
    let createdAt: Date
    let transcriptionEngine: String
    let transcriptionModel: String
    let enhancementMode: String
    let enhancementModel: String
    let kind: TranscriptionHistoryKind
    let isTranslation: Bool
    let audioDurationSeconds: TimeInterval?
    let transcriptionProcessingDurationSeconds: TimeInterval?
    let llmDurationSeconds: TimeInterval?
    let focusedAppName: String?
    let focusedAppBundleID: String?
    let browserURLHost: String?
    let browserURLOrigin: String?
    let matchedGroupID: UUID?
    let matchedGroupName: String?
    let matchedAppGroupName: String?
    let matchedURLGroupName: String?
    let remoteASRProvider: String?
    let remoteASRModel: String?
    let remoteASREndpoint: String?
    let remoteLLMProvider: String?
    let remoteLLMModel: String?
    let remoteLLMEndpoint: String?
    let audioRelativePath: String?
    let whisperWordTimings: [WhisperHistoryWordTiming]?
    let senseVoiceMetadata: SenseVoiceTranscriptMetadata?
    let transcriptSegments: [TranscriptSegment]?
    let transcriptAudioRelativePath: String?
    let meetingCaptureMode: MeetingCaptureMode?
    let transcriptSummary: TranscriptSummarySnapshot?
    let transcriptSummaryStale: Bool
    let transcriptSummaryChatMessages: [TranscriptSummaryChatMessage]?
    let displayTitle: String?
    let transcriptionChatMessages: [TranscriptSummaryChatMessage]?
    let dictionaryHitTerms: [String]
    let dictionaryCorrectedTerms: [String]
    let dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot]
    let dictionarySuggestedTerms: [DictionarySuggestionSnapshot]

    enum CodingKeys: String, CodingKey {
        case id
        case text
        case createdAt
        case transcriptionEngine
        case transcriptionModel
        case enhancementMode
        case enhancementModel
        case kind
        case isTranslation
        case audioDurationSeconds
        case transcriptionProcessingDurationSeconds
        case llmDurationSeconds
        case focusedAppName
        case focusedAppBundleID
        case browserURLHost
        case browserURLOrigin
        case matchedGroupID
        case matchedGroupName
        case matchedAppGroupName
        case matchedURLGroupName
        case remoteASRProvider
        case remoteASRModel
        case remoteASREndpoint
        case remoteLLMProvider
        case remoteLLMModel
        case remoteLLMEndpoint
        case audioRelativePath
        case whisperWordTimings
        case senseVoiceMetadata
        case transcriptSegments
        case transcriptAudioRelativePath
        case meetingCaptureMode
        case transcriptSummary
        case transcriptSummaryStale
        case transcriptSummaryChatMessages
        case displayTitle
        case transcriptionChatMessages
        case dictionaryHitTerms
        case dictionaryCorrectedTerms
        case dictionaryCorrectionSnapshots
        case dictionarySuggestedTerms
    }

    init(
        id: UUID,
        text: String,
        createdAt: Date,
        transcriptionEngine: String,
        transcriptionModel: String,
        enhancementMode: String,
        enhancementModel: String,
        kind: TranscriptionHistoryKind,
        isTranslation: Bool,
        audioDurationSeconds: TimeInterval?,
        transcriptionProcessingDurationSeconds: TimeInterval?,
        llmDurationSeconds: TimeInterval?,
        focusedAppName: String?,
        focusedAppBundleID: String?,
        browserURLHost: String? = nil,
        browserURLOrigin: String? = nil,
        matchedGroupID: UUID?,
        matchedGroupName: String?,
        matchedAppGroupName: String?,
        matchedURLGroupName: String?,
        remoteASRProvider: String?,
        remoteASRModel: String?,
        remoteASREndpoint: String?,
        remoteLLMProvider: String?,
        remoteLLMModel: String?,
        remoteLLMEndpoint: String?,
        audioRelativePath: String? = nil,
        whisperWordTimings: [WhisperHistoryWordTiming]?,
        senseVoiceMetadata: SenseVoiceTranscriptMetadata? = nil,
        transcriptSegments: [TranscriptSegment]? = nil,
        transcriptAudioRelativePath: String? = nil,
        meetingCaptureMode: MeetingCaptureMode? = nil,
        transcriptSummary: TranscriptSummarySnapshot? = nil,
        transcriptSummaryStale: Bool = false,
        transcriptSummaryChatMessages: [TranscriptSummaryChatMessage]? = nil,
        displayTitle: String? = nil,
        transcriptionChatMessages: [TranscriptSummaryChatMessage]? = nil,
        dictionaryHitTerms: [String],
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot] = [],
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot]
    ) {
        self.id = id
        self.text = text
        self.createdAt = createdAt
        self.transcriptionEngine = transcriptionEngine
        self.transcriptionModel = transcriptionModel
        self.enhancementMode = enhancementMode
        self.enhancementModel = enhancementModel
        self.kind = kind
        self.isTranslation = isTranslation
        self.audioDurationSeconds = audioDurationSeconds
        self.transcriptionProcessingDurationSeconds = transcriptionProcessingDurationSeconds
        self.llmDurationSeconds = llmDurationSeconds
        self.focusedAppName = focusedAppName
        self.focusedAppBundleID = focusedAppBundleID
        self.browserURLHost = browserURLHost
        self.browserURLOrigin = browserURLOrigin
        self.matchedGroupID = matchedGroupID
        self.matchedGroupName = matchedGroupName
        self.matchedAppGroupName = matchedAppGroupName
        self.matchedURLGroupName = matchedURLGroupName
        self.remoteASRProvider = remoteASRProvider
        self.remoteASRModel = remoteASRModel
        self.remoteASREndpoint = remoteASREndpoint
        self.remoteLLMProvider = remoteLLMProvider
        self.remoteLLMModel = remoteLLMModel
        self.remoteLLMEndpoint = remoteLLMEndpoint
        self.audioRelativePath = audioRelativePath ?? transcriptAudioRelativePath
        self.whisperWordTimings = whisperWordTimings
        self.senseVoiceMetadata = senseVoiceMetadata
        self.transcriptSegments = transcriptSegments
        self.transcriptAudioRelativePath = transcriptAudioRelativePath
        self.meetingCaptureMode = meetingCaptureMode
        self.transcriptSummary = transcriptSummary
        self.transcriptSummaryStale = transcriptSummaryStale
        self.transcriptSummaryChatMessages = transcriptSummaryChatMessages
        self.displayTitle = displayTitle
        self.transcriptionChatMessages = transcriptionChatMessages
        self.dictionaryHitTerms = dictionaryHitTerms
        self.dictionaryCorrectedTerms = dictionaryCorrectedTerms
        self.dictionaryCorrectionSnapshots = dictionaryCorrectionSnapshots
        self.dictionarySuggestedTerms = dictionarySuggestedTerms
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        text = try container.decode(String.self, forKey: .text)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        transcriptionEngine = try container.decode(String.self, forKey: .transcriptionEngine)
        transcriptionModel = try container.decode(String.self, forKey: .transcriptionModel)
        enhancementMode = try container.decode(String.self, forKey: .enhancementMode)
        enhancementModel = try container.decode(String.self, forKey: .enhancementModel)
        let decodedIsTranslation = try container.decodeIfPresent(Bool.self, forKey: .isTranslation) ?? false
        isTranslation = decodedIsTranslation
        kind = try container.decodeIfPresent(TranscriptionHistoryKind.self, forKey: .kind)
            ?? (decodedIsTranslation ? .translation : .normal)
        audioDurationSeconds = try container.decodeIfPresent(TimeInterval.self, forKey: .audioDurationSeconds)
        transcriptionProcessingDurationSeconds = try container.decodeIfPresent(TimeInterval.self, forKey: .transcriptionProcessingDurationSeconds)
        llmDurationSeconds = try container.decodeIfPresent(TimeInterval.self, forKey: .llmDurationSeconds)
        focusedAppName = try container.decodeIfPresent(String.self, forKey: .focusedAppName)
        focusedAppBundleID = try container.decodeIfPresent(String.self, forKey: .focusedAppBundleID)
        browserURLHost = try container.decodeIfPresent(String.self, forKey: .browserURLHost)
        browserURLOrigin = try container.decodeIfPresent(String.self, forKey: .browserURLOrigin)
        let decodedMatchedAppGroupName = try container.decodeIfPresent(String.self, forKey: .matchedAppGroupName)
        let decodedMatchedURLGroupName = try container.decodeIfPresent(String.self, forKey: .matchedURLGroupName)
        matchedGroupID = try container.decodeIfPresent(UUID.self, forKey: .matchedGroupID)
        matchedGroupName = try container.decodeIfPresent(String.self, forKey: .matchedGroupName)
            ?? decodedMatchedURLGroupName
            ?? decodedMatchedAppGroupName
        matchedAppGroupName = decodedMatchedAppGroupName
        matchedURLGroupName = decodedMatchedURLGroupName
        remoteASRProvider = try container.decodeIfPresent(String.self, forKey: .remoteASRProvider)
        remoteASRModel = try container.decodeIfPresent(String.self, forKey: .remoteASRModel)
        remoteASREndpoint = try container.decodeIfPresent(String.self, forKey: .remoteASREndpoint)
        remoteLLMProvider = try container.decodeIfPresent(String.self, forKey: .remoteLLMProvider)
        remoteLLMModel = try container.decodeIfPresent(String.self, forKey: .remoteLLMModel)
        remoteLLMEndpoint = try container.decodeIfPresent(String.self, forKey: .remoteLLMEndpoint)
        let decodedAudioRelativePath = try container.decodeIfPresent(String.self, forKey: .audioRelativePath)
        whisperWordTimings = try container.decodeIfPresent([WhisperHistoryWordTiming].self, forKey: .whisperWordTimings)
        senseVoiceMetadata = try container.decodeIfPresent(SenseVoiceTranscriptMetadata.self, forKey: .senseVoiceMetadata)
        transcriptSegments = try container.decodeIfPresent([TranscriptSegment].self, forKey: .transcriptSegments)
        transcriptAudioRelativePath = try container.decodeIfPresent(String.self, forKey: .transcriptAudioRelativePath)
        meetingCaptureMode = try container.decodeIfPresent(MeetingCaptureMode.self, forKey: .meetingCaptureMode)
        audioRelativePath = decodedAudioRelativePath ?? transcriptAudioRelativePath
        transcriptSummary = try container.decodeIfPresent(TranscriptSummarySnapshot.self, forKey: .transcriptSummary)
        transcriptSummaryStale = try container.decodeIfPresent(Bool.self, forKey: .transcriptSummaryStale) ?? false
        transcriptSummaryChatMessages = try container.decodeIfPresent([TranscriptSummaryChatMessage].self, forKey: .transcriptSummaryChatMessages)
        displayTitle = try container.decodeIfPresent(String.self, forKey: .displayTitle)
        transcriptionChatMessages = try container.decodeIfPresent([TranscriptSummaryChatMessage].self, forKey: .transcriptionChatMessages)
        dictionaryHitTerms = try container.decodeIfPresent([String].self, forKey: .dictionaryHitTerms) ?? []
        dictionaryCorrectedTerms = try container.decodeIfPresent([String].self, forKey: .dictionaryCorrectedTerms) ?? []
        dictionaryCorrectionSnapshots = try container.decodeIfPresent([DictionaryCorrectionSnapshot].self, forKey: .dictionaryCorrectionSnapshots) ?? []
        dictionarySuggestedTerms = try container.decodeIfPresent([DictionarySuggestionSnapshot].self, forKey: .dictionarySuggestedTerms) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(text, forKey: .text)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(transcriptionEngine, forKey: .transcriptionEngine)
        try container.encode(transcriptionModel, forKey: .transcriptionModel)
        try container.encode(enhancementMode, forKey: .enhancementMode)
        try container.encode(enhancementModel, forKey: .enhancementModel)
        try container.encode(kind, forKey: .kind)
        try container.encode(isTranslation, forKey: .isTranslation)
        try container.encodeIfPresent(audioDurationSeconds, forKey: .audioDurationSeconds)
        try container.encodeIfPresent(transcriptionProcessingDurationSeconds, forKey: .transcriptionProcessingDurationSeconds)
        try container.encodeIfPresent(llmDurationSeconds, forKey: .llmDurationSeconds)
        try container.encodeIfPresent(focusedAppName, forKey: .focusedAppName)
        try container.encodeIfPresent(focusedAppBundleID, forKey: .focusedAppBundleID)
        try container.encodeIfPresent(browserURLHost, forKey: .browserURLHost)
        try container.encodeIfPresent(browserURLOrigin, forKey: .browserURLOrigin)
        try container.encodeIfPresent(matchedGroupID, forKey: .matchedGroupID)
        try container.encodeIfPresent(matchedGroupName, forKey: .matchedGroupName)
        try container.encodeIfPresent(matchedAppGroupName, forKey: .matchedAppGroupName)
        try container.encodeIfPresent(matchedURLGroupName, forKey: .matchedURLGroupName)
        try container.encodeIfPresent(remoteASRProvider, forKey: .remoteASRProvider)
        try container.encodeIfPresent(remoteASRModel, forKey: .remoteASRModel)
        try container.encodeIfPresent(remoteASREndpoint, forKey: .remoteASREndpoint)
        try container.encodeIfPresent(remoteLLMProvider, forKey: .remoteLLMProvider)
        try container.encodeIfPresent(remoteLLMModel, forKey: .remoteLLMModel)
        try container.encodeIfPresent(remoteLLMEndpoint, forKey: .remoteLLMEndpoint)
        try container.encodeIfPresent(audioRelativePath, forKey: .audioRelativePath)
        try container.encodeIfPresent(whisperWordTimings, forKey: .whisperWordTimings)
        try container.encodeIfPresent(senseVoiceMetadata, forKey: .senseVoiceMetadata)
        try container.encodeIfPresent(transcriptSegments, forKey: .transcriptSegments)
        try container.encodeIfPresent(transcriptAudioRelativePath, forKey: .transcriptAudioRelativePath)
        try container.encodeIfPresent(meetingCaptureMode, forKey: .meetingCaptureMode)
        try container.encodeIfPresent(transcriptSummary, forKey: .transcriptSummary)
        try container.encode(transcriptSummaryStale, forKey: .transcriptSummaryStale)
        try container.encodeIfPresent(transcriptSummaryChatMessages, forKey: .transcriptSummaryChatMessages)
        try container.encodeIfPresent(displayTitle, forKey: .displayTitle)
        try container.encodeIfPresent(transcriptionChatMessages, forKey: .transcriptionChatMessages)
        try container.encode(dictionaryHitTerms, forKey: .dictionaryHitTerms)
        try container.encode(dictionaryCorrectedTerms, forKey: .dictionaryCorrectedTerms)
        try container.encode(dictionaryCorrectionSnapshots, forKey: .dictionaryCorrectionSnapshots)
        try container.encode(dictionarySuggestedTerms, forKey: .dictionarySuggestedTerms)
    }
}

struct HistoryReportMetrics: Hashable {
    let totalDictationSeconds: TimeInterval
    let totalCharacters: Int
    let totalTranslationCharacters: Int
    let dailyCharacters: [Date: Int]
    let branchItems: [HistoryBranchMetricItem]
}

struct HistoryBranchMetricItem: Identifiable, Hashable {
    enum Kind: String, Hashable {
        case app
        case url
    }

    let kind: Kind
    let title: String
    let subtitle: String?
    let bundleID: String?
    let urlHost: String?
    let urlOrigin: String?
    let characterCount: Int

    var id: String {
        switch kind {
        case .app:
            return "app:\(bundleID ?? title)"
        case .url:
            return "url:\(urlHost ?? title)"
        }
    }
}
