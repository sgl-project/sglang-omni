// MeetingAudioArchiveTests.swift
// Provides Meeting Audio Archive Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MeetingAudioArchiveTests: XCTestCase {
    func testExportWAVPreservesStartOffsetsAcrossSpeakers() async throws {
        let archive = MeetingAudioArchive()
        let oneSecond = [Float](repeating: 1.0, count: 16_000)
        let halfSecond = [Float](repeating: 0.5, count: 16_000)
        let destinationURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("MeetingAudioArchiveTests-\(UUID().uuidString)")
            .appendingPathExtension("wav")

        defer {
            try? FileManager.default.removeItem(at: destinationURL)
        }

        await archive.append(samples: oneSecond, sampleRate: 16_000, speaker: .me, startSeconds: 0)
        await archive.append(samples: halfSecond, sampleRate: 16_000, speaker: .them, startSeconds: 1.0)

        let didExport = try await archive.exportWAV(to: destinationURL)
        XCTAssertTrue(didExport)

        let samples = try decodeMono16BitWAVSamples(from: destinationURL)
        XCTAssertEqual(samples.count, 32_000)
        XCTAssertEqual(samples[1_000], 0.5, accuracy: 0.02)
        XCTAssertEqual(samples[20_000], 0.25, accuracy: 0.02)
    }

    func testAnalysisAssetsExposeMixedFullMeetingAudio() async {
        let archive = MeetingAudioArchive()
        await archive.append(samples: [Float](repeating: 0.4, count: 1_600), sampleRate: 16_000, speaker: .me, startSeconds: 0)
        await archive.append(samples: [Float](repeating: 0.8, count: 1_600), sampleRate: 16_000, speaker: .them, startSeconds: 0.1)

        let assets = await archive.analysisAssets()

        XCTAssertEqual(assets.map(\.source), [.mixed])
        XCTAssertEqual(assets[0].sampleRate, 16_000)
        XCTAssertEqual(assets[0].durationSeconds, 0.2, accuracy: 0.001)
        XCTAssertEqual(assets[0].samples[800], 0.2, accuracy: 0.001)
        XCTAssertEqual(assets[0].samples[2_400], 0.4, accuracy: 0.001)
    }

    func testFinalTranscriptionAssetsPreserveAudioSources() async {
        let archive = MeetingAudioArchive()
        await archive.append(samples: [Float](repeating: 0.4, count: 1_600), sampleRate: 16_000, speaker: .me, startSeconds: 0)
        await archive.append(samples: [Float](repeating: 0.8, count: 1_600), sampleRate: 16_000, speaker: .them, startSeconds: 0.1)

        let assets = await archive.finalTranscriptionAssets()

        XCTAssertEqual(assets.map(\.source), [.microphone, .systemAudio])
        XCTAssertEqual(assets[0].durationSeconds, 0.1, accuracy: 0.001)
        XCTAssertEqual(assets[1].durationSeconds, 0.1, accuracy: 0.001)
        XCTAssertEqual(assets[0].sessionStartOffset, 0, accuracy: 0.001)
        XCTAssertEqual(assets[1].sessionStartOffset, 0.1, accuracy: 0.001)
        XCTAssertEqual(assets[0].samples[800], 0.4, accuracy: 0.001)
        XCTAssertEqual(assets[1].samples[800], 0.8, accuracy: 0.001)
    }

    func testAssetDescriptorsLoadAudioOnDemand() async throws {
        let archive = MeetingAudioArchive()
        await archive.append(samples: [Float](repeating: 0.4, count: 1_600), sampleRate: 16_000, speaker: .me, startSeconds: 0)
        await archive.append(samples: [Float](repeating: 0.8, count: 1_600), sampleRate: 16_000, speaker: .them, startSeconds: 0.1)

        let analysisDescriptors = await archive.analysisAssetDescriptors()
        let finalDescriptors = await archive.finalTranscriptionAssetDescriptors()

        XCTAssertEqual(analysisDescriptors.count, 1)
        XCTAssertEqual(finalDescriptors.map(\.source), [.microphone, .systemAudio])
        let loadedMixedAsset = await archive.loadAsset(analysisDescriptors[0])
        let mixedAsset = try XCTUnwrap(loadedMixedAsset)
        XCTAssertEqual(mixedAsset.source, .mixed)
        XCTAssertEqual(mixedAsset.durationSeconds, 0.2, accuracy: 0.001)
        XCTAssertEqual(mixedAsset.samples[800], 0.2, accuracy: 0.001)
        XCTAssertEqual(mixedAsset.samples[2_400], 0.4, accuracy: 0.001)

        let loadedSystemAsset = await archive.loadAssetWindow(
            source: TranscriptAudioSource.systemAudio,
            startSeconds: 0.12,
            endSeconds: 0.18,
            paddingSeconds: 0.05
        )
        let asset = try XCTUnwrap(loadedSystemAsset)

        XCTAssertEqual(asset.source, TranscriptAudioSource.systemAudio)
        XCTAssertEqual(asset.sessionStartOffset, 0.07, accuracy: 0.001)
        XCTAssertEqual(asset.durationSeconds, 0.16, accuracy: 0.001)
        XCTAssertEqual(asset.samples[800], 0.8, accuracy: 0.001)
    }

    func testModeSpecificAnalysisDescriptorsRouteToExpectedSources() async {
        let archive = MeetingAudioArchive()
        await archive.append(samples: [Float](repeating: 0.4, count: 1_600), sampleRate: 16_000, speaker: .me, startSeconds: 0)
        await archive.append(samples: [Float](repeating: 0.8, count: 1_600), sampleRate: 16_000, speaker: .them, startSeconds: 0.1)

        let meetingDescriptors = await archive.analysisAssetDescriptors(for: .meeting)
        let subtitlesDescriptors = await archive.analysisAssetDescriptors(for: .subtitles)
        let recordingDescriptors = await archive.analysisAssetDescriptors(for: .recording)

        XCTAssertEqual(meetingDescriptors.map(\.source), [.systemAudio])
        XCTAssertTrue(subtitlesDescriptors.isEmpty)
        XCTAssertTrue(recordingDescriptors.isEmpty)
    }

    func testSmallContinuousAppendsUseBatchedPersistentWrites() async {
        let archive = MeetingAudioArchive()
        let frame = [Float](repeating: 0.25, count: 160)

        for index in 0..<100 {
            await archive.append(
                samples: frame,
                sampleRate: 16_000,
                speaker: .me,
                startSeconds: Double(index * frame.count) / 16_000
            )
        }

        let statistics = await archive.currentIOStatistics()
        XCTAssertEqual(statistics.appendCount, 100)
        XCTAssertEqual(statistics.resampleCount, 0)
        XCTAssertEqual(statistics.writeOperationCount, 1)
        XCTAssertEqual(statistics.writeHandleOpenCount, 1)
        XCTAssertEqual(statistics.writtenByteCount, Int64(16_000 * MemoryLayout<Float>.size))
    }

    func testWindowReadOnlyLoadsRequestedFloatSamples() async throws {
        let archive = MeetingAudioArchive()
        await archive.append(
            samples: [Float](repeating: 0.4, count: 32_000),
            sampleRate: 16_000,
            speaker: .me,
            startSeconds: 0
        )

        let loadedAsset = await archive.loadAssetWindow(
            source: .microphone,
            startSeconds: 0.5,
            endSeconds: 0.6
        )
        let asset = try XCTUnwrap(loadedAsset)
        let statistics = await archive.currentIOStatistics()

        XCTAssertEqual(asset.samples.count, 1_600)
        XCTAssertEqual(statistics.readOperationCount, 1)
        XCTAssertEqual(statistics.readByteCount, Int64(1_600 * MemoryLayout<Float>.size))
    }

    private func decodeMono16BitWAVSamples(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let dataRange = try findDataChunk(in: data)
        let pcm = data[dataRange]
        let sampleCount = pcm.count / 2
        var samples: [Float] = []
        samples.reserveCapacity(sampleCount)

        for index in stride(from: 0, to: pcm.count, by: 2) {
            let lower = UInt16(pcm[pcm.startIndex + index])
            let upper = UInt16(pcm[pcm.startIndex + index + 1]) << 8
            let value = Int16(bitPattern: lower | upper)
            samples.append(Float(value) / Float(Int16.max))
        }

        return samples
    }

    private func findDataChunk(in data: Data) throws -> Range<Data.Index> {
        var cursor = 12
        while cursor + 8 <= data.count {
            let chunkID = String(data: data[cursor..<(cursor + 4)], encoding: .ascii)
            let chunkSize = Int(readUInt32LE(from: data, at: cursor + 4))
            let chunkStart = cursor + 8
            let chunkEnd = chunkStart + chunkSize
            guard chunkEnd <= data.count else {
                throw NSError(domain: "MeetingAudioArchiveTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid WAV chunk bounds."])
            }
            if chunkID == "data" {
                return chunkStart..<chunkEnd
            }
            cursor = chunkEnd + (chunkSize % 2)
        }

        throw NSError(domain: "MeetingAudioArchiveTests", code: 2, userInfo: [NSLocalizedDescriptionKey: "WAV data chunk not found."])
    }

    private func readUInt32LE(from data: Data, at offset: Int) -> UInt32 {
        let b0 = UInt32(data[data.startIndex + offset])
        let b1 = UInt32(data[data.startIndex + offset + 1]) << 8
        let b2 = UInt32(data[data.startIndex + offset + 2]) << 16
        let b3 = UInt32(data[data.startIndex + offset + 3]) << 24
        return b0 | b1 | b2 | b3
    }
}
