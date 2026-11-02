// MeetingImportedAudioFile.swift
// Normalizes imported meeting media and exposes bounded analysis windows.

import AVFoundation
import CoreMedia
import Foundation
import UniformTypeIdentifiers

enum MeetingFileImportSupport {
    static let allowedContentTypes: [UTType] = [.audio, .movie]
    nonisolated static let maximumAnalysisDurationSeconds: TimeInterval = 12 * 60 * 60

    static func isSupportedImportFile(at url: URL) -> Bool {
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .contentTypeKey])
        if values?.isDirectory == true {
            return false
        }
        if let isRegularFile = values?.isRegularFile, !isRegularFile {
            return false
        }

        if let contentType = values?.contentType,
           conformsToAllowedTypes(contentType) {
            return true
        }

        let ext = url.pathExtension.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !ext.isEmpty, let inferred = UTType(filenameExtension: ext) else {
            return false
        }
        return conformsToAllowedTypes(inferred)
    }

    static func mediaDurationSeconds(at url: URL) async -> TimeInterval? {
        let asset = AVURLAsset(url: url)
        guard let duration = try? await asset.load(.duration),
              duration.seconds.isFinite,
              duration.seconds > 0
        else {
            return nil
        }
        return duration.seconds
    }

    /// Preserves the system-provided URL object. Do not standardize or rebuild from a
    /// path string before `startAccessingSecurityScopedResource()` — that strips the
    /// sandbox security scope carried by Finder / NSItemProvider drop URLs.
    static func fileURL(fromDropItem item: Any?) -> URL? {
        if let url = item as? URL {
            return url
        }
        if let data = item as? Data {
            return URL(dataRepresentation: data, relativeTo: nil)
        }
        if let path = item as? String {
            return fileURL(fromPath: path)
        }
        if let path = item as? NSString {
            return fileURL(fromPath: path as String)
        }
        return nil
    }

    private static func fileURL(fromPath path: String) -> URL? {
        if path.hasPrefix("file:") {
            return URL(string: path)
        }
        return URL(fileURLWithPath: path)
    }

    private static func conformsToAllowedTypes(_ type: UTType) -> Bool {
        allowedContentTypes.contains { type.conforms(to: $0) }
    }
}

nonisolated enum MeetingFileTaskStagingError: LocalizedError {
    case sourceUnavailable
    case sourceTooLarge
    case stagingLimitExceeded
    case mediaDurationUnavailable
    case mediaTooLong
    case insufficientDiskSpace

    var errorDescription: String? {
        switch self {
        case .sourceUnavailable:
            return AppLocalization.localizedString("The selected meeting file is no longer available.")
        case .sourceTooLarge:
            return AppLocalization.localizedString("The selected meeting file is too large to stage safely.")
        case .stagingLimitExceeded:
            return AppLocalization.localizedString("The meeting file queue has reached its storage limit.")
        case .mediaDurationUnavailable:
            return AppLocalization.localizedString("The selected meeting file duration could not be read.")
        case .mediaTooLong:
            return AppLocalization.localizedString("The selected meeting file is longer than the supported 12-hour limit.")
        case .insufficientDiskSpace:
            return AppLocalization.localizedString("There is not enough free disk space to safely stage this meeting file.")
        }
    }
}

nonisolated struct MeetingImportedAudioFile: Sendable {
    static let targetSampleRate = 16_000
    private static let analysisWindowSeconds: TimeInterval = 60

    let standardizedAudioURL: URL
    let sampleCount: Int

    var durationSeconds: TimeInterval {
        TimeInterval(sampleCount) / TimeInterval(Self.targetSampleRate)
    }

    var assetDescriptors: [MeetingAudioAssetDescriptor] {
        let samplesPerWindow = max(
            Int(Self.analysisWindowSeconds * TimeInterval(Self.targetSampleRate)),
            1
        )
        var descriptors: [MeetingAudioAssetDescriptor] = []
        var startSample = 0
        while startSample < sampleCount {
            let windowSampleCount = min(samplesPerWindow, sampleCount - startSample)
            descriptors.append(
                MeetingAudioAssetDescriptor(
                    source: .mixed,
                    sampleRate: Double(Self.targetSampleRate),
                    startSample: startSample,
                    sampleCount: windowSampleCount
                )
            )
            startSample += windowSampleCount
        }
        return descriptors
    }

    static func prepare(
        from sourceURL: URL,
        to destination: URL? = nil,
        limits: MeetingFilePreparationLimits = .init(),
        checkpoint: (@Sendable () async throws -> Void)? = nil,
        progress: (@Sendable (Double) async -> Void)? = nil
    ) async throws -> MeetingImportedAudioFile {
        let destinationURL = destination ?? temporaryAudioURL()
        let partialURL = destinationURL.appendingPathExtension("partial")
        let resources = MeetingFilePreparationResources()
        try Task.checkCancellation()
        guard !FileManager.default.fileExists(atPath: destinationURL.path) else {
            throw CocoaError(.fileWriteFileExists)
        }

        do {
            try await resources.waitUntilAvailable()
            try await checkpoint?()
            try limits.checkDiskSpace(at: destinationURL.deletingLastPathComponent())
            let sourceAttributes = try FileManager.default.attributesOfItem(atPath: sourceURL.path)
            guard sourceAttributes[.type] as? FileAttributeType == .typeRegular,
                  let sourceBytes = (sourceAttributes[.size] as? NSNumber)?.int64Value,
                  sourceBytes > 0 else { throw MeetingFileTaskStagingError.sourceUnavailable }
            guard sourceBytes <= limits.maximumSourceBytes else { throw MeetingFileTaskStagingError.sourceTooLarge }
            let asset = AVURLAsset(url: sourceURL)
            guard let audioTrack = try await asset.loadTracks(withMediaType: .audio).first else {
                throw MeetingImportedAudioFileError.noAudioTrack
            }
            let durationSeconds = try await asset.load(.duration).seconds
            guard durationSeconds.isFinite, durationSeconds > 0 else {
                throw MeetingFileTaskStagingError.mediaDurationUnavailable
            }
            guard durationSeconds <= limits.maximumDurationSeconds else {
                throw MeetingFileTaskStagingError.mediaTooLong
            }
            let estimatedSampleCount = durationSeconds * Double(targetSampleRate)
            guard estimatedSampleCount <= Double(limits.maximumSampleCount) else {
                throw MeetingFileTaskStagingError.stagingLimitExceeded
            }
            try limits.checkDiskSpace(
                at: destinationURL.deletingLastPathComponent(),
                additionalBytes: Int64(estimatedSampleCount.rounded(.up)) * 2 + 44
            )
            await progress?(0)

            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: audioTrack,
                outputSettings: [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVSampleRateKey: targetSampleRate,
                    AVNumberOfChannelsKey: 1,
                    AVLinearPCMBitDepthKey: 32,
                    AVLinearPCMIsFloatKey: true,
                    AVLinearPCMIsBigEndianKey: false
                ]
            )
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else {
                throw MeetingImportedAudioFileError.unsupportedMedia
            }
            reader.add(output)

            guard FileManager.default.createFile(
                atPath: partialURL.path,
                contents: Data(count: MeetingImportedWAVWriter.headerByteCount),
                attributes: [.posixPermissions: 0o600]
            ) else { throw CocoaError(.fileWriteUnknown) }
            let writer = try MeetingImportedWAVWriter(
                destinationURL: partialURL,
                sampleRate: targetSampleRate,
                maximumSampleCount: limits.maximumSampleCount
            )
            defer { writer.close() }
            defer { reader.cancelReading() }
            try Task.checkCancellation()

            guard reader.startReading() else {
                throw reader.error ?? MeetingImportedAudioFileError.unableToDecode
            }

            var lastReportedProgress = 0.0
            var lastSafetyCheck = ContinuousClock.now
            var samplesAtSafetyCheck = 0
            while true {
                try Task.checkCancellation()
                if writer.sampleCount == 0
                    || writer.sampleCount - samplesAtSafetyCheck >= 512_000
                    || lastSafetyCheck.duration(to: .now) >= .milliseconds(250) {
                    try await resources.waitUntilAvailable()
                    try await checkpoint?()
                    try limits.checkDiskSpace(
                        at: destinationURL.deletingLastPathComponent(),
                        additionalBytes: Int64(MeetingFilePreparationLimits.maximumDecoderBufferBytes)
                    )
                    samplesAtSafetyCheck = writer.sampleCount
                    lastSafetyCheck = .now
                }
                // Drain ObjC decoder temporaries after each buffer, not at the end
                // of a multi-hour import. Reading and writing are strictly serial.
                let hasBuffer = try autoreleasepool {
                    guard let buffer = output.copyNextSampleBuffer() else { return false }
                    try writer.append(sampleBuffer: buffer)
                    return true
                }
                guard hasBuffer else { break }
                let currentProgress = min(Double(writer.sampleCount) / estimatedSampleCount, 1)
                if currentProgress - lastReportedProgress >= 0.01 {
                    lastReportedProgress = currentProgress
                    await progress?(currentProgress)
                }
            }

            try Task.checkCancellation()
            guard reader.status == .completed else {
                throw reader.error ?? MeetingImportedAudioFileError.unableToDecode
            }
            guard writer.sampleCount > 0 else { throw MeetingImportedAudioFileError.emptyAudio }
            try writer.finish()
            let validated = try validatedPreparedFile(at: partialURL, limits: limits)
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: partialURL, to: destinationURL)
            await progress?(1)
            return MeetingImportedAudioFile(standardizedAudioURL: destinationURL, sampleCount: validated.sampleCount)
        } catch {
            try? FileManager.default.removeItem(at: partialURL)
            throw error
        }
    }

    private static func temporaryAudioURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("Voxt-Imported-Meeting-\(UUID().uuidString).wav")
    }

    /// Accept only our canonical PCM format. Read the 44-byte header, never the
    /// complete cache file; reject truncated, oversized or incompatible caches.
    static func validatedPreparedFile(
        at url: URL,
        limits: MeetingFilePreparationLimits = .init()
    ) throws -> MeetingImportedAudioFile {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              let size = (attributes[.size] as? NSNumber)?.int64Value,
              size > 44, size <= limits.maximumOutputBytes,
              (size - 44).isMultiple(of: 2),
              (size - 44) / 2 <= Int64(limits.maximumSampleCount)
        else { throw MeetingImportedAudioFileError.unableToDecode }
        let sampleCount = Int((size - 44) / 2)
        let dataByteCount = try MeetingImportedWAVFormat.dataByteCount(sampleCount: sampleCount)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let header = try handle.read(upToCount: 44)
        guard header == MeetingImportedWAVWriter.wavHeader(sampleRate: targetSampleRate, dataByteCount: dataByteCount) else {
            throw MeetingImportedAudioFileError.unableToDecode
        }
        return MeetingImportedAudioFile(standardizedAudioURL: url, sampleCount: sampleCount)
    }

    /// History takes ownership of its audio by moving it. Give it an independent,
    /// bounded-copy archive so failures/retries cannot consume the queue's cache.
    static func copyPreparedForAnalysis(from sourceURL: URL) async throws -> MeetingImportedAudioFile {
        let limits = MeetingFilePreparationLimits()
        let source = try validatedPreparedFile(at: sourceURL)
        let destination = temporaryAudioURL()
        let partial = destination.appendingPathExtension("partial")
        let resources = MeetingFilePreparationResources()
        let expectedBytes = Int64(source.sampleCount) * 2 + 44
        try await resources.waitUntilAvailable()
        try limits.checkDiskSpace(at: destination.deletingLastPathComponent(), additionalBytes: expectedBytes)
        do {
            let input = try FileHandle(forReadingFrom: sourceURL)
            defer { try? input.close() }
            guard FileManager.default.createFile(atPath: partial.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            let output = try FileHandle(forWritingTo: partial)
            defer { try? output.close() }
            var copied: Int64 = 0
            while copied < expectedBytes {
                try await resources.waitUntilAvailable()
                let copiedBytes = try autoreleasepool {
                    try limits.checkDiskSpace(at: destination.deletingLastPathComponent(), additionalBytes: Int64(MeetingFilePreparationLimits.copyBufferBytes))
                    guard let data = try input.read(upToCount: min(MeetingFilePreparationLimits.copyBufferBytes, Int(expectedBytes - copied))),
                          !data.isEmpty else { throw MeetingImportedAudioFileError.unableToDecode }
                    try output.write(contentsOf: data)
                    return Int64(data.count)
                }
                copied += copiedBytes
            }
            try output.synchronize()
            try output.close()
            _ = try validatedPreparedFile(at: partial)
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: partial, to: destination)
            return MeetingImportedAudioFile(standardizedAudioURL: destination, sampleCount: source.sampleCount)
        } catch {
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
    }

    func loadAsset(_ descriptor: MeetingAudioAssetDescriptor) -> MeetingAudioAsset? {
        guard descriptor.sampleRate == Double(Self.targetSampleRate),
              descriptor.startSample >= 0, descriptor.startSample < sampleCount,
              descriptor.sampleCount > 0,
              descriptor.sampleCount <= Int(Self.analysisWindowSeconds * Double(Self.targetSampleRate)),
              descriptor.sampleCount <= sampleCount - descriptor.startSample
        else {
            return nil
        }

        do {
            let file = try AVAudioFile(forReading: standardizedAudioURL)
            guard AVAudioFramePosition(descriptor.startSample) < file.length else { return nil }
            file.framePosition = AVAudioFramePosition(descriptor.startSample)
            let availableFrames = max(Int(file.length - file.framePosition), 0)
            guard availableFrames >= descriptor.sampleCount else { return nil }
            let frameCount = descriptor.sampleCount
            guard frameCount > 0,
                  let buffer = AVAudioPCMBuffer(
                    pcmFormat: file.processingFormat,
                    frameCapacity: AVAudioFrameCount(frameCount)
                  )
            else {
                return nil
            }
            try file.read(into: buffer, frameCount: AVAudioFrameCount(frameCount))
            guard let samples = AudioLevelMeter.monoSamples(from: buffer), !samples.isEmpty else {
                return nil
            }
            return MeetingAudioAsset(
                source: descriptor.source,
                samples: samples,
                sampleRate: descriptor.sampleRate,
                sessionStartOffset: descriptor.sessionStartOffset
            )
        } catch {
            VoxtLog.meetingWarning("Imported meeting audio window could not be loaded.")
            return nil
        }
    }
}

nonisolated enum MeetingImportedAudioFileError: LocalizedError, Equatable {
    case noAudioTrack
    case unsupportedMedia
    case unableToDecode
    case emptyAudio
    case fileTooLarge

    var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            return AppLocalization.localizedString("The selected file does not contain an audio track.")
        case .unsupportedMedia:
            return AppLocalization.localizedString("This audio or video file cannot be analyzed.")
        case .unableToDecode:
            return AppLocalization.localizedString("Voxt could not decode the selected media file.")
        case .emptyAudio:
            return AppLocalization.localizedString("The selected file does not contain usable audio.")
        case .fileTooLarge:
            return AppLocalization.localizedString("The meeting audio is too long to store as a WAV file.")
        }
    }
}
