// MeetingImportedAudioFileTests.swift
// Covers imported meeting media normalization and bounded asset loading.

import XCTest
@testable import Voxt

@MainActor
final class MeetingImportedAudioFileTests: XCTestCase {
    func testImportSupportAcceptsCommonAudioAndVideoExtensions() {
        XCTAssertTrue(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/demo.mp3")))
        XCTAssertTrue(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/demo.m4a")))
        XCTAssertTrue(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/demo.wav")))
        XCTAssertTrue(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/demo.mp4")))
        XCTAssertTrue(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/demo.mov")))
        XCTAssertFalse(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/notes.txt")))
        XCTAssertFalse(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/slides.pdf")))
        XCTAssertFalse(MeetingFileImportSupport.isSupportedImportFile(at: URL(fileURLWithPath: "/tmp/archive.zip")))
    }

    func testImportSupportParsesDroppedFileURLItems() {
        let fileURL = URL(fileURLWithPath: "/tmp/meeting-demo.m4a")
        // Keep the provider URL object intact (no standardizedFileURL) so sandbox
        // security scope from Finder drops is not stripped.
        let fromURL = MeetingFileImportSupport.fileURL(fromDropItem: fileURL as NSURL)
        XCTAssertEqual(fromURL, fileURL)
        XCTAssertEqual(fromURL?.path, fileURL.path)

        let fromData = MeetingFileImportSupport.fileURL(fromDropItem: fileURL.dataRepresentation)
        XCTAssertEqual(fromData?.path, fileURL.path)
        XCTAssertNil(MeetingFileImportSupport.fileURL(fromDropItem: nil))
    }

    func testAnalysisProgressMapsStageProgressToMonotonicOverallProgress() {
        let samples = [
            MeetingFileAnalysisProgress(stage: .preparing, stageFraction: 0),
            MeetingFileAnalysisProgress(stage: .preparing, stageFraction: 1),
            MeetingFileAnalysisProgress(stage: .transcribing, stageFraction: 0.5),
            MeetingFileAnalysisProgress(stage: .identifyingSpeakers, stageFraction: 1),
            MeetingFileAnalysisProgress(stage: .saving, stageFraction: 1),
        ]

        for (actual, expected) in zip(samples.map(\.fractionCompleted), [0, 0.15, 0.465, 0.96, 1]) {
            XCTAssertEqual(actual, expected, accuracy: 0.000_001)
        }
        XCTAssertEqual(samples.map(\.stage), [.preparing, .preparing, .transcribing, .identifyingSpeakers, .saving])
    }

    func testPrepareNormalizesAudioAndCreatesBoundedDescriptors() async throws {
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("Voxt-Meeting-Import-Test-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        let sourceSampleRate = 8_000
        let durationSeconds = 61
        let samples = (0..<(sourceSampleRate * durationSeconds)).map { index in
            Float(sin(Double(index) * 2 * .pi * 220 / Double(sourceSampleRate))) * 0.2
        }
        try MeetingAudioChunkWAVExporter.write(
            samples: samples,
            sampleRate: sourceSampleRate,
            to: sourceURL
        )
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let imported = try await MeetingImportedAudioFile.prepare(from: sourceURL)
        defer { try? FileManager.default.removeItem(at: imported.standardizedAudioURL) }

        XCTAssertEqual(imported.durationSeconds, 61, accuracy: 0.05)
        XCTAssertEqual(imported.assetDescriptors.count, 2)
        XCTAssertEqual(imported.assetDescriptors[0].durationSeconds, 60, accuracy: 0.001)
        XCTAssertEqual(imported.assetDescriptors[1].sessionStartOffset, 60, accuracy: 0.001)
        XCTAssertEqual(imported.assetDescriptors[1].durationSeconds, 1, accuracy: 0.05)

        let finalAsset = try XCTUnwrap(imported.loadAsset(imported.assetDescriptors[1]))
        XCTAssertEqual(finalAsset.sampleRate, 16_000)
        XCTAssertEqual(finalAsset.sessionStartOffset, 60, accuracy: 0.001)
        XCTAssertEqual(finalAsset.durationSeconds, 1, accuracy: 0.05)
        XCTAssertTrue(finalAsset.samples.contains { abs($0) > 0.01 })
        XCTAssertNil(imported.loadAsset(MeetingAudioAssetDescriptor(
            source: .mixed, sampleRate: 16_000, startSample: 0, sampleCount: 61 * 16_000
        )))
        XCTAssertNil(imported.loadAsset(MeetingAudioAssetDescriptor(
            source: .mixed, sampleRate: 16_000, startSample: Int.max, sampleCount: 1
        )))
    }

    func testPreparationPublishesOnlyValidatedOutputAndPreservesOriginal() async throws {
        let directory = try TemporaryDirectory()
        let source = try makeSmallSource(in: directory.url)
        let original = try Data(contentsOf: source)
        let destination = directory.url.appendingPathComponent("prepared.wav")
        let audio = try await MeetingImportedAudioFile.prepare(from: source, to: destination)
        XCTAssertEqual(audio.standardizedAudioURL, destination)
        XCTAssertEqual(try MeetingImportedAudioFile.validatedPreparedFile(at: destination).sampleCount, audio.sampleCount)
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathExtension("partial").path))
        XCTAssertEqual(try Data(contentsOf: source), original)

        let archive = try await MeetingImportedAudioFile.copyPreparedForAnalysis(from: destination)
        defer { try? FileManager.default.removeItem(at: archive.standardizedAudioURL) }
        XCTAssertNotEqual(archive.standardizedAudioURL, destination)
        XCTAssertEqual(archive.sampleCount, audio.sampleCount)
        XCTAssertEqual(try Data(contentsOf: archive.standardizedAudioURL), try Data(contentsOf: destination))
        try FileManager.default.removeItem(at: archive.standardizedAudioURL)
        XCTAssertNoThrow(try MeetingImportedAudioFile.validatedPreparedFile(at: destination))
    }

    func testCancelledPreparationRemovesPartialFile() async throws {
        let directory = try TemporaryDirectory()
        let source = try makeSmallSource(in: directory.url)
        let destination = directory.url.appendingPathComponent("cancelled.wav")
        let partial = destination.appendingPathExtension("partial")
        do {
            _ = try await MeetingImportedAudioFile.prepare(from: source, to: destination, checkpoint: {
                if FileManager.default.fileExists(atPath: partial.path) { throw CancellationError() }
            })
            XCTFail("Expected cancellation after the partial file was created")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
    }

    func testPreparationRejectsSourceDurationOutputAndDiskLimitsWithoutPublishing() async throws {
        let directory = try TemporaryDirectory()
        let source = try makeSmallSource(in: directory.url)
        var sourceLimit = MeetingFilePreparationLimits()
        sourceLimit.maximumSourceBytes = 1
        var durationLimit = MeetingFilePreparationLimits()
        durationLimit.maximumDurationSeconds = 0.01
        var outputLimit = MeetingFilePreparationLimits()
        outputLimit.maximumOutputBytes = 46
        var diskLimit = MeetingFilePreparationLimits()
        diskLimit.minimumFreeDiskBytes = .max
        for (index, limits) in [sourceLimit, durationLimit, outputLimit, diskLimit].enumerated() {
            let destination = directory.url.appendingPathComponent("rejected-\(index).wav")
            do {
                _ = try await MeetingImportedAudioFile.prepare(from: source, to: destination, limits: limits)
                XCTFail("Unsafe input was accepted")
            } catch {}
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathExtension("partial").path))
        }
    }

    func testPreparedCacheRejectsTruncationAndWrongSampleRate() throws {
        let directory = try TemporaryDirectory()
        let source = try makeSmallSource(in: directory.url)
        XCTAssertThrowsError(try MeetingImportedAudioFile.validatedPreparedFile(at: source)) // 8 kHz, not canonical
        let truncated = directory.url.appendingPathComponent("truncated.wav")
        try Data(repeating: 0, count: 46).write(to: truncated)
        XCTAssertThrowsError(try MeetingImportedAudioFile.validatedPreparedFile(at: truncated))
    }

    func testDecoderBuffersAndActualDecodedSampleCountAreBounded() throws {
        XCTAssertEqual(try MeetingImportedWAVFormat.validatedIncomingSampleCount(
            floatByteCount: 16, currentSampleCount: 6, maximumSampleCount: 10
        ), 4)
        XCTAssertThrowsError(try MeetingImportedWAVFormat.validatedIncomingSampleCount(
            floatByteCount: 16, currentSampleCount: 7, maximumSampleCount: 10
        ))
        XCTAssertThrowsError(try MeetingImportedWAVFormat.validatedIncomingSampleCount(
            floatByteCount: MeetingFilePreparationLimits.maximumDecoderBufferBytes + 4,
            currentSampleCount: 0, maximumSampleCount: .max
        ))
        XCTAssertThrowsError(try MeetingImportedWAVFormat.validatedIncomingSampleCount(
            floatByteCount: 3, currentSampleCount: 0, maximumSampleCount: 10
        ))
        XCTAssertLessThanOrEqual(MeetingFilePreparationLimits.conversionBufferBytes, 64 * 1024)
        XCTAssertLessThanOrEqual(MeetingFilePreparationLimits.copyBufferBytes, 1024 * 1024)
    }

    private func makeSmallSource(in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent("source.wav")
        try MeetingAudioChunkWAVExporter.write(
            samples: Array(repeating: Float(0.1), count: 8_000), sampleRate: 8_000, to: url
        )
        return url
    }

    func testWAVDataByteCountRejectsValuesThatOverflowRIFFChunkSize() throws {
        let maximumSampleCount = Int(MeetingImportedWAVFormat.maximumDataByteCount) /
            MemoryLayout<Int16>.size

        let acceptedByteCount = try MeetingImportedWAVFormat.dataByteCount(
            sampleCount: maximumSampleCount
        )
        XCTAssertLessThanOrEqual(
            UInt64(acceptedByteCount) + UInt64(MeetingImportedWAVFormat.riffSizeOverhead),
            UInt64(UInt32.max)
        )
        XCTAssertThrowsError(
            try MeetingImportedWAVFormat.dataByteCount(sampleCount: maximumSampleCount + 1)
        ) { error in
            XCTAssertEqual(error as? MeetingImportedAudioFileError, .fileTooLarge)
        }
    }
}
