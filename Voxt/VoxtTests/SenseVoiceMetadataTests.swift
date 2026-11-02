// SenseVoiceMetadataTests.swift
// Provides Sense Voice Metadata Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class SenseVoiceMetadataTests: XCTestCase {
    func testFromOutputPreservesAllSegmentsAndAggregatesDominantMetadata() {
        let output = ASRTranscriptionResult(
            text: "ignored top-level text",
            segments: [
                ASRTranscriptSegment(
                    text: " first segment ",
                    language: "en",
                    emotion: "happy",
                    event: "speech"
                ),
                ASRTranscriptSegment(
                    text: "second segment",
                    language: "en",
                    emotion: " calm ",
                    event: "speech"
                ),
                ASRTranscriptSegment(
                    text: " third segment ",
                    language: "ja",
                    emotion: "happy",
                    event: "music"
                ),
            ],
            language: "en"
        )

        let metadata = SenseVoiceTranscriptMetadata.fromOutput(
            output,
            startSeconds: 0,
            endSeconds: 9,
            usedVADSegmentation: false
        )

        XCTAssertEqual(metadata?.language, "en")
        XCTAssertEqual(metadata?.emotion, "happy")
        XCTAssertEqual(metadata?.event, "speech")
        XCTAssertEqual(metadata?.segments.count, 3)
        XCTAssertEqual(metadata?.segments.map(\.text), ["first segment", "second segment", "third segment"])
        XCTAssertEqual(metadata?.segments.map(\.language), ["en", "en", "ja"])
        XCTAssertEqual(metadata?.segments.map(\.emotion), ["happy", "calm", "happy"])
        XCTAssertEqual(metadata?.segments.map(\.event), ["speech", "speech", "music"])
        XCTAssertEqual(metadata?.segments.map(\.startSeconds), [0, 3, 6])
        XCTAssertEqual(metadata?.segments.map(\.endSeconds), [3, 6, 9])
    }

    func testFromOutputUsesFallbackLanguageAndExplicitSegmentTimingWhenPresent() throws {
        let output = ASRTranscriptionResult(
            text: "overall transcript",
            segments: [
                ASRTranscriptSegment(
                    text: "alpha",
                    startSeconds: 1.5,
                    endSeconds: 2.75
                ),
                ASRTranscriptSegment(
                    text: "beta",
                    startSeconds: 3.0,
                    endSeconds: 4.25,
                    emotion: "focused"
                ),
            ],
            language: "zh"
        )

        let metadata = try XCTUnwrap(
            SenseVoiceTranscriptMetadata.fromOutput(
                output,
                startSeconds: 0,
                endSeconds: 5,
                usedVADSegmentation: true
            )
        )

        XCTAssertEqual(metadata.language, "zh")
        XCTAssertEqual(metadata.emotion, "focused")
        XCTAssertTrue(metadata.usedVADSegmentation)
        XCTAssertEqual(metadata.segments[0].startSeconds, 1.5, accuracy: 0.0001)
        XCTAssertEqual(metadata.segments[0].endSeconds, 2.75, accuracy: 0.0001)
        XCTAssertEqual(metadata.segments[0].language, "zh")
        XCTAssertEqual(metadata.segments[1].startSeconds, 3.0, accuracy: 0.0001)
        XCTAssertEqual(metadata.segments[1].endSeconds, 4.25, accuracy: 0.0001)
        XCTAssertEqual(metadata.segments[1].language, "zh")
        XCTAssertEqual(metadata.segments[1].emotion, "focused")
    }

    func testFromOutputWithoutSegmentsUsesTranscriptAndLanguage() throws {
        let output = ASRTranscriptionResult(
            text: " hello world ",
            language: " en ",
            languageProvenance: .requested
        )
        let metadata = try XCTUnwrap(
            SenseVoiceTranscriptMetadata.fromOutput(
                output,
                startSeconds: 2,
                endSeconds: 5,
                usedVADSegmentation: false
            )
        )

        XCTAssertEqual(metadata.language, "en")
        XCTAssertNil(metadata.emotion)
        XCTAssertNil(metadata.event)
        XCTAssertFalse(metadata.usedVADSegmentation)
        XCTAssertEqual(metadata.segments.count, 1)
        XCTAssertEqual(metadata.segments[0].text, "hello world")
        XCTAssertEqual(metadata.segments[0].startSeconds, 2)
        XCTAssertEqual(metadata.segments[0].endSeconds, 5)
    }

    func testTranscriptSegmentsCodableRoundTripPreservesAllMetadata() throws {
        let segments = [
            ASRTranscriptSegment(
                text: "Hello world",
                startSeconds: 1.25,
                endSeconds: 2.75,
                speakerID: "S02",
                language: "en",
                confidence: 0.95,
                emotion: "happy",
                event: "speech"
            ),
            ASRTranscriptSegment(text: "Text only"),
        ]

        let encoded = try JSONEncoder().encode(segments)
        let decoded = try JSONDecoder().decode([ASRTranscriptSegment].self, from: encoded)

        XCTAssertEqual(decoded, segments)
        XCTAssertEqual(Set(decoded), Set(segments))
    }

    func testTranscriptSegmentDecodesWithoutOptionalMetadata() throws {
        let encoded = Data(#"{"text":"Text only"}"#.utf8)
        let segment = try JSONDecoder().decode(ASRTranscriptSegment.self, from: encoded)

        XCTAssertEqual(segment.text, "Text only")
        XCTAssertNil(segment.startSeconds)
        XCTAssertNil(segment.endSeconds)
        XCTAssertNil(segment.speakerID)
        XCTAssertNil(segment.language)
        XCTAssertNil(segment.confidence)
        XCTAssertNil(segment.emotion)
        XCTAssertNil(segment.event)
    }

    func testTranscriptionResultDefaultsToEmptySegmentsAndUnknownLanguageProvenance() throws {
        let result = ASRTranscriptionResult(text: "Text only")

        XCTAssertEqual(result.text, "Text only")
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertNil(result.language)
        XCTAssertEqual(result.languageProvenance, .unknown)
        XCTAssertNil(result.generationMetadata)
        XCTAssertEqual(ASRTranscriptionResult(text: "Hello", language: "en").languageProvenance, .unknown)

        let encoded = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(ASRTranscriptionResult.self, from: encoded)
        XCTAssertEqual(decoded, result)
    }

    func testTranscriptionResultCodableRoundTripPreservesSegmentsAndEachLanguageProvenance() throws {
        let segments = [
            ASRTranscriptSegment(text: "Hello", startSeconds: 0.5, endSeconds: 1.5, language: "en"),
            ASRTranscriptSegment(text: "世界", language: "zh", emotion: "calm", event: "speech"),
        ]
        let generationMetadata = ASRGenerationMetadata(generatedTokenCount: 24, hitLengthLimit: false)
        let provenances: [ASRLanguageProvenance] = [.detected, .requested, .outputTarget, .modelDefault, .unknown]
        for provenance in provenances {
            let result = ASRTranscriptionResult(
                text: "Hello 世界",
                segments: segments,
                language: "en",
                languageProvenance: provenance,
                generationMetadata: generationMetadata
            )
            let encoded = try JSONEncoder().encode(result)
            let decoded = try JSONDecoder().decode(ASRTranscriptionResult.self, from: encoded)

            XCTAssertEqual(decoded.text, "Hello 世界")
            XCTAssertEqual(decoded.segments, segments)
            XCTAssertEqual(decoded.language, "en")
            XCTAssertEqual(decoded.languageProvenance, provenance)
            XCTAssertEqual(decoded.generationMetadata, generationMetadata)
        }
    }

    func testMLXConfigurationSummaryOmitsPresetForSenseVoice() {
        let tuning = MLXLocalTuningSettings(
            preset: .accuracyFirst,
            senseVoiceUseITN: true
        )

        XCTAssertEqual(
            MLXConfigurationSummarySupport.summary(
                for: "mlx-community/SenseVoiceSmall",
                tuning: tuning
            ),
            AppLocalization.localizedString("ITN On")
        )
    }

    func testSequentialSegmentMergeDeduplicatesBoundaryOverlap() {
        let originalSegmentID = UUID()
        let merged = SenseVoiceTranscriptMetadata.mergeSequentialSegments(
            base: [
                SenseVoiceSegmentMetadata(
                    id: originalSegmentID,
                    startSeconds: 0,
                    endSeconds: 2.4,
                    text: "hello world",
                    language: "en",
                    emotion: "calm",
                    event: "speech"
                )
            ],
            next: [
                SenseVoiceSegmentMetadata(
                    startSeconds: 2.2,
                    endSeconds: 4.0,
                    text: "world again",
                    language: "en",
                    emotion: "calm",
                    event: "speech"
                ),
                SenseVoiceSegmentMetadata(
                    startSeconds: 4.0,
                    endSeconds: 5.0,
                    text: "and again",
                    language: "en",
                    emotion: "calm",
                    event: "speech"
                )
            ]
        )

        XCTAssertEqual(merged.map(\.text), ["hello world again", "and again"])
        XCTAssertEqual(merged[0].id, originalSegmentID)
        XCTAssertEqual(merged[0].startSeconds, 0, accuracy: 0.0001)
        XCTAssertEqual(merged[0].endSeconds, 4.0, accuracy: 0.0001)
    }

    func testSequentialSegmentMergeIgnoresFallbackTimingDriftWhenTextOverlaps() {
        let merged = SenseVoiceTranscriptMetadata.mergeSequentialSegments(
            base: [
                SenseVoiceSegmentMetadata(
                    startSeconds: 0,
                    endSeconds: 2.4,
                    text: "alpha beta gamma",
                    language: "en",
                    emotion: nil,
                    event: "speech"
                )
            ],
            next: [
                SenseVoiceSegmentMetadata(
                    startSeconds: 3.6,
                    endSeconds: 5.0,
                    text: "beta gamma delta",
                    language: "en",
                    emotion: nil,
                    event: "speech"
                )
            ]
        )

        XCTAssertEqual(merged.map(\.text), ["alpha beta gamma delta"])
        XCTAssertEqual(merged[0].startSeconds, 0, accuracy: 0.0001)
        XCTAssertEqual(merged[0].endSeconds, 5.0, accuracy: 0.0001)
    }

    func testSequentialSegmentMergeKeepsDisjointBoundarySegments() {
        let merged = SenseVoiceTranscriptMetadata.mergeSequentialSegments(
            base: [
                SenseVoiceSegmentMetadata(
                    startSeconds: 0,
                    endSeconds: 1.0,
                    text: "first sentence",
                    language: "en",
                    emotion: nil,
                    event: "speech"
                )
            ],
            next: [
                SenseVoiceSegmentMetadata(
                    startSeconds: 1.5,
                    endSeconds: 2.5,
                    text: "second sentence",
                    language: "en",
                    emotion: nil,
                    event: "speech"
                )
            ]
        )

        XCTAssertEqual(merged.map(\.text), ["first sentence", "second sentence"])
    }
}
