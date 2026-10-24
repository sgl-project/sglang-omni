/// Transport-independent transcription values shared by Voxt ASR clients.
import Foundation

nonisolated enum ASRLanguageProvenance: String, Codable, Hashable, Sendable {
    case detected
    case requested
    case outputTarget
    case modelDefault
    case unknown
}

nonisolated struct ASRTranscriptSegment: Codable, Hashable, Sendable {
    let text: String
    let startSeconds: Double?
    let endSeconds: Double?
    let speakerID: String?
    let language: String?
    let confidence: Double?
    let emotion: String?
    let event: String?

    init(
        text: String,
        startSeconds: Double? = nil,
        endSeconds: Double? = nil,
        speakerID: String? = nil,
        language: String? = nil,
        confidence: Double? = nil,
        emotion: String? = nil,
        event: String? = nil
    ) {
        self.text = text
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.speakerID = speakerID
        self.language = language
        self.confidence = confidence
        self.emotion = emotion
        self.event = event
    }
}

nonisolated struct ASRGenerationMetadata: Codable, Hashable, Sendable {
    let generatedTokenCount: Int
    let hitLengthLimit: Bool?
}

nonisolated struct ASRTranscriptionResult: Codable, Hashable, Sendable {
    let text: String
    let segments: [ASRTranscriptSegment]
    let language: String?
    let languageProvenance: ASRLanguageProvenance
    let generationMetadata: ASRGenerationMetadata?

    init(
        text: String,
        segments: [ASRTranscriptSegment] = [],
        language: String? = nil,
        languageProvenance: ASRLanguageProvenance = .unknown,
        generationMetadata: ASRGenerationMetadata? = nil
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.languageProvenance = languageProvenance
        self.generationMetadata = generationMetadata
    }
}

nonisolated enum ASRStreamingEvent: Equatable, Sendable {
    case displayUpdate(confirmedText: String, provisionalText: String)
    case ended(ASRTranscriptionResult)
    case failed(ASRStreamingFailure)
}

nonisolated struct ASRStreamingFailure: LocalizedError, Equatable, Sendable {
    let message: String

    var errorDescription: String? { message }
}

nonisolated struct ASRGenerationOptions: Equatable, Sendable {
    let maximumTokenCount: Int
    let temperature: Float
    let topProbabilityThreshold: Float
    let topTokenCount: Int
    let verbose: Bool
    let language: String?
    let usePunctuation: Bool?
    let chunkDurationSeconds: Float
    let minimumChunkDurationSeconds: Float
    let kvCacheBitWidth: Int?
    let kvCacheGroupSize: Int
    let kvCacheQuantizedStartTokenCount: Int
}
