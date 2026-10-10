// TranscriptionHistoryEntryAudioTests.swift
// Provides Transcription History Entry Audio Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class TranscriptionHistoryEntryAudioTests: XCTestCase {
    private static var retainedObjects: [AnyObject] = []

    func testDecodingTranscriptAudioPathPopulatesGenericAudioPath() throws {
        let createdAt = Date(timeIntervalSinceReferenceDate: 321)
        let payload: [String: Any] = [
            "id": UUID().uuidString,
            "text": "Transcript",
            "createdAt": createdAt.timeIntervalSince1970,
            "transcriptionEngine": "MLX Whisper",
            "transcriptionModel": "whisper-base-mlx",
            "enhancementMode": "Off",
            "enhancementModel": "None",
            "kind": "transcript",
            "isTranslation": false,
            "transcriptAudioRelativePath": "transcript/clip.wav",
            "dictionaryHitTerms": [],
            "dictionaryCorrectedTerms": [],
            "dictionarySuggestedTerms": []
        ]

        let data = try JSONSerialization.data(withJSONObject: payload)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970

        let entry = try decoder.decode(TranscriptionHistoryEntry.self, from: data)

        XCTAssertEqual(entry.kind, .transcript)
        XCTAssertEqual(entry.transcriptAudioRelativePath, "transcript/clip.wav")
        XCTAssertEqual(entry.audioRelativePath, "transcript/clip.wav")
        XCTAssertTrue(entry.dictionaryCorrectionSnapshots.isEmpty)
    }

    func testEncodingGenericAudioPathOmitsTranscriptSpecificFieldWhenUnset() throws {
        let entry = TranscriptionHistoryEntry(
            id: UUID(),
            text: "Transcript",
            createdAt: Date(timeIntervalSinceReferenceDate: 456),
            transcriptionEngine: "MLX Whisper",
            transcriptionModel: "whisper-large-v3-turbo",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: .normal,
            isTranslation: false,
            audioDurationSeconds: 2,
            transcriptionProcessingDurationSeconds: 1,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            audioRelativePath: "transcription/sample.wav",
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )

        let data = try JSONEncoder().encode(entry)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(object["audioRelativePath"] as? String, "transcription/sample.wav")
        XCTAssertNil(object["transcriptAudioRelativePath"])
        XCTAssertTrue((object["dictionaryCorrectionSnapshots"] as? [Any])?.isEmpty == true)
    }

    func testHistoryStoreAllowsAudioOnlyMeetingEntryWhenExplicitlyEnabled() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-audio-history-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let database = retain(VoxtDatabase(databaseURL: directoryURL.appendingPathComponent("history.sqlite")))
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let store = retain(TranscriptionHistoryStore(repository: repository))

        let entryID = store.append(
            text: "",
            transcriptionEngine: "MLX Audio",
            transcriptionModel: "MOSS",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: .transcript,
            isTranslation: false,
            audioDurationSeconds: 12,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            audioRelativePath: "transcript/audio-only.wav",
            whisperWordTimings: nil,
            transcriptSegments: [],
            transcriptAudioRelativePath: "transcript/audio-only.wav",
            meetingCaptureMode: .recording,
            displayTitle: "Recording",
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: [],
            allowEmptyTextWithAudio: true
        )

        let entry = try XCTUnwrap(entryID.flatMap { store.entry(id: $0) })
        XCTAssertEqual(entry.text, "")
        XCTAssertEqual(entry.meetingCaptureMode, .recording)
        XCTAssertEqual(entry.audioRelativePath, "transcript/audio-only.wav")
    }

    func testDeletingAllMeetingSegmentsKeepsAudioAndNormalEntriesRequireText() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-history-segment-delete-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let database = retain(VoxtDatabase(databaseURL: directoryURL.appendingPathComponent("history.sqlite")))
        let repository = retain(HistoryRepository(database: database, legacyJSONURL: nil, migrateLegacyJSON: false))
        let store = retain(TranscriptionHistoryStore(repository: repository))
        let segment = TranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1,
            text: "Meeting text"
        )

        let meetingID = try XCTUnwrap(store.append(
            text: segment.text,
            transcriptionEngine: "MLX Audio",
            transcriptionModel: "MOSS",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: .transcript,
            isTranslation: false,
            audioDurationSeconds: 1,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            audioRelativePath: "transcript/meeting.wav",
            whisperWordTimings: nil,
            transcriptSegments: [segment],
            transcriptAudioRelativePath: "transcript/meeting.wav",
            meetingCaptureMode: .recording,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        ))

        let updatedMeeting = try XCTUnwrap(
            store.updateTranscriptSegments([], for: meetingID, allowsEmptyText: true)
        )
        XCTAssertEqual(updatedMeeting.text, "")
        XCTAssertEqual(updatedMeeting.audioRelativePath, "transcript/meeting.wav")

        let summary = TranscriptSummarySnapshot(
            title: "Saved summary",
            body: "Body",
            todoItems: [],
            generatedAt: Date(timeIntervalSince1970: 2),
            settingsSnapshot: TranscriptSummarySettingsSnapshot(autoGenerate: true)
        )
        _ = store.updateTranscriptSummary(summary, for: meetingID)
        XCTAssertFalse(store.entry(id: meetingID)?.transcriptSummaryStale ?? true)
        let staleEntry = try XCTUnwrap(store.updateTranscriptSummaryStale(true, for: meetingID))
        XCTAssertTrue(staleEntry.transcriptSummaryStale)
        let freshEntry = try XCTUnwrap(store.updateTranscriptSummary(summary, for: meetingID))
        XCTAssertFalse(freshEntry.transcriptSummaryStale)

        let normalID = try XCTUnwrap(store.append(
            text: "Normal transcription",
            transcriptionEngine: "MLX Whisper",
            transcriptionModel: "base",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: .normal,
            isTranslation: false,
            audioDurationSeconds: nil,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        ))
        XCTAssertNil(store.updateTranscriptSegments([], for: normalID, allowsEmptyText: true))
        XCTAssertEqual(store.entry(id: normalID)?.text, "Normal transcription")
    }

    private func retain<Value: AnyObject>(_ value: Value) -> Value {
        Self.retainedObjects.append(value)
        return value
    }
}
