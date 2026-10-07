// MeetingAudioArchive.swift
// Provides Meeting Audio Archive for meeting capture.

import Foundation

nonisolated struct MeetingAudioArchiveIOStatistics: Equatable, Sendable {
    var appendCount = 0
    var resampleCount = 0
    var writeOperationCount = 0
    var writeHandleOpenCount = 0
    var writtenByteCount: Int64 = 0
    var readOperationCount = 0
    var readHandleOpenCount = 0
    var readByteCount: Int64 = 0
}

actor MeetingAudioArchive {
    private static let segmentDurationSeconds: TimeInterval = 300
    private static let maxAssetDurationSeconds: TimeInterval = 60
    private static let exportWindowDurationSeconds: TimeInterval = 60
    private static let writeBatchDurationSeconds: TimeInterval = 1
    private static let silenceThreshold: Float = 0.0001
    private static let diskCapacityCheckInterval: TimeInterval = 60
    private static let minimumDiskReserveBytes: Int64 = 2 * 1_024 * 1_024 * 1_024
    private static let diskCapacityFailureMessage = "Recording stopped to protect this Mac because meeting audio could not be written safely. Free some disk space, then try again."

    private struct PendingWrite {
        let segmentIndex: Int
        let sampleOffset: Int
        var samples: [Float]

        var endSampleOffset: Int {
            sampleOffset + samples.count
        }
    }

    private struct ActiveWriteHandle {
        let segmentIndex: Int
        let handle: FileHandle
    }

    private let targetSampleRate: Double = HistoryAudioArchiveSupport.targetSampleRate
    private let fileManager = FileManager.default
    private var tempDirectoryURL: URL?
    private var meWrittenRange: Range<Int>?
    private var themWrittenRange: Range<Int>?
    private var pendingWrites: [MeetingSpeaker: PendingWrite] = [:]
    private var activeWriteHandles: [MeetingSpeaker: ActiveWriteHandle] = [:]
    private var ioStatistics = MeetingAudioArchiveIOStatistics()
    private var lastDiskCapacityCheckAt: Date?
    private var safetyFailureMessage: String?

    deinit {
        for writer in activeWriteHandles.values {
            try? writer.handle.close()
        }
        if let tempDirectoryURL {
            try? fileManager.removeItem(at: tempDirectoryURL)
        }
    }

    func append(
        samples: [Float],
        sampleRate: Double,
        speaker: MeetingSpeaker,
        startSeconds: TimeInterval
    ) {
        guard !samples.isEmpty else { return }
        ioStatistics.appendCount += 1
        if abs(sampleRate - targetSampleRate) > 1 {
            ioStatistics.resampleCount += 1
        }
        let preparedSamples = Self.resample(samples: samples, from: sampleRate, to: targetSampleRate)
        guard !preparedSamples.isEmpty else { return }

        let startIndex = max(Int((startSeconds * targetSampleRate).rounded()), 0)
        do {
            try validateDiskCapacityIfNeeded(additionalSampleCount: preparedSamples.count)
            try enqueueWrite(preparedSamples, at: startIndex, speaker: speaker)
            switch speaker {
            case .me:
                meWrittenRange = Self.union(meWrittenRange, with: startIndex..<(startIndex + preparedSamples.count))
            case .them:
                themWrittenRange = Self.union(themWrittenRange, with: startIndex..<(startIndex + preparedSamples.count))
            }
        } catch {
            safetyFailureMessage = Self.diskCapacityFailureMessage
            VoxtLog.meetingWarning("Meeting audio archive append failed: \(error.localizedDescription)")
        }
    }

    func exportWAV(to destinationURL: URL) throws -> Bool {
        try prepareForReading()
        guard let range = combinedWrittenRange(), range.lowerBound < range.upperBound else {
            return false
        }

        let sampleCount = range.count
        let dataByteCount = sampleCount * MemoryLayout<Int16>.size
        guard dataByteCount <= Int(UInt32.max) else {
            throw NSError(
                domain: "Voxt.MeetingAudioArchive",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Meeting audio is too large to export as a single WAV file."]
            )
        }

        fileManager.createFile(
            atPath: destinationURL.path,
            contents: Self.wavHeader(sampleCount: sampleCount, sampleRate: targetSampleRate),
            attributes: nil
        )
        let handle = try FileHandle(forWritingTo: destinationURL)
        defer { try? handle.close() }
        try handle.seekToEnd()

        let windowSamples = max(Int(Self.exportWindowDurationSeconds * targetSampleRate), 1)
        var start = range.lowerBound
        while start < range.upperBound {
            let end = min(start + windowSamples, range.upperBound)
            let samples = readMixedSamples(in: start..<end)
            try handle.write(contentsOf: Self.pcm16Data(from: samples))
            start = end
        }

        return true
    }

    func analysisAssetDescriptors() -> [MeetingAudioAssetDescriptor] {
        guard let range = combinedWrittenRange() else { return [] }
        return descriptors(source: .mixed, range: range)
    }

    func analysisAssetDescriptors(for mode: MeetingCaptureMode) -> [MeetingAudioAssetDescriptor] {
        let capabilities = mode.capabilities
        guard !capabilities.finalDiarizationSources.isEmpty else { return [] }

        var output: [MeetingAudioAssetDescriptor] = []
        if capabilities.shouldRunFinalDiarization(for: .microphone), let meWrittenRange {
            output.append(contentsOf: descriptors(source: .microphone, range: meWrittenRange))
        }
        if capabilities.shouldRunFinalDiarization(for: .systemAudio), let themWrittenRange {
            output.append(contentsOf: descriptors(source: .systemAudio, range: themWrittenRange))
        }

        if output.isEmpty, let range = combinedWrittenRange() {
            return descriptors(source: .mixed, range: range)
        }
        return output
    }

    func finalTranscriptionAssetDescriptors() -> [MeetingAudioAssetDescriptor] {
        [
            meWrittenRange.map { descriptors(source: .microphone, range: $0) } ?? [],
            themWrittenRange.map { descriptors(source: .systemAudio, range: $0) } ?? []
        ]
        .flatMap { $0 }
    }

    func analysisAssets() -> [MeetingAudioAsset] {
        analysisAssetDescriptors().compactMap { loadAsset($0) }
    }

    func finalTranscriptionAssets() -> [MeetingAudioAsset] {
        finalTranscriptionAssetDescriptors().compactMap { loadAsset($0) }
    }

    func loadAsset(_ descriptor: MeetingAudioAssetDescriptor) -> MeetingAudioAsset? {
        guard let asset = loadVoiceActivityAsset(descriptor) else { return nil }
        guard asset.samples.contains(where: { abs($0) > Self.silenceThreshold }) else { return nil }
        return asset
    }

    func loadVoiceActivityAsset(_ descriptor: MeetingAudioAssetDescriptor) -> MeetingAudioAsset? {
        do {
            try prepareForReading()
        } catch {
            VoxtLog.meetingWarning("Meeting audio archive flush before read failed: \(error.localizedDescription)")
            return nil
        }
        let range = descriptor.startSample..<(descriptor.startSample + descriptor.sampleCount)
        guard range.lowerBound >= 0, range.lowerBound < range.upperBound else { return nil }
        let samples: [Float]
        switch descriptor.source {
        case .mixed:
            samples = readMixedSamples(in: range)
        case .microphone:
            samples = readSamples(speaker: .me, in: range)
        case .systemAudio:
            samples = readSamples(speaker: .them, in: range)
        }
        guard !samples.isEmpty else { return nil }
        return MeetingAudioAsset(
            source: descriptor.source,
            samples: samples,
            sampleRate: descriptor.sampleRate,
            sessionStartOffset: descriptor.sessionStartOffset
        )
    }

    func loadAssetWindow(
        source: TranscriptAudioSource,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        paddingSeconds: TimeInterval = 0
    ) -> MeetingAudioAsset? {
        let lowerSeconds = max(startSeconds - paddingSeconds, 0)
        let upperSeconds = max(endSeconds + paddingSeconds, lowerSeconds)
        let startSample = max(Int((lowerSeconds * targetSampleRate).rounded()), 0)
        let endSample = max(Int((upperSeconds * targetSampleRate).rounded()), startSample)
        guard endSample > startSample else { return nil }
        return loadAsset(
            MeetingAudioAssetDescriptor(
                source: source,
                sampleRate: targetSampleRate,
                startSample: startSample,
                sampleCount: endSample - startSample
            )
        )
    }

    func reset() {
        closeActiveWriteHandles()
        if let tempDirectoryURL {
            try? fileManager.removeItem(at: tempDirectoryURL)
        }
        tempDirectoryURL = nil
        meWrittenRange = nil
        themWrittenRange = nil
        pendingWrites.removeAll(keepingCapacity: false)
        ioStatistics = MeetingAudioArchiveIOStatistics()
        lastDiskCapacityCheckAt = nil
        safetyFailureMessage = nil
    }

    func preflightFailureMessage() -> String? {
        do {
            try validateDiskCapacityIfNeeded(additionalSampleCount: 0, force: true)
            return nil
        } catch {
            return Self.diskCapacityFailureMessage
        }
    }

    func currentIOStatistics() -> MeetingAudioArchiveIOStatistics {
        ioStatistics
    }

    func consumeSafetyFailureMessage() -> String? {
        defer { safetyFailureMessage = nil }
        return safetyFailureMessage
    }

    private func validateDiskCapacityIfNeeded(additionalSampleCount: Int, force: Bool = false) throws {
        let now = Date()
        if !force,
           let lastDiskCapacityCheckAt,
           now.timeIntervalSince(lastDiskCapacityCheckAt) < Self.diskCapacityCheckInterval {
            return
        }
        lastDiskCapacityCheckAt = now

        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let values = try directory.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey
        ])
        guard let available = values.volumeAvailableCapacityForImportantUsage else { return }
        let pendingBytes = Int64(max(additionalSampleCount, 0) * MemoryLayout<Float>.size)
        guard available - pendingBytes >= Self.minimumDiskReserveBytes else {
            throw NSError(
                domain: "Voxt.MeetingAudioArchive",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Insufficient disk reserve for safe meeting capture."]
            )
        }
    }

    private func descriptors(
        source: TranscriptAudioSource,
        range: Range<Int>
    ) -> [MeetingAudioAssetDescriptor] {
        let windowSamples = max(Int(Self.maxAssetDurationSeconds * targetSampleRate), 1)
        var output: [MeetingAudioAssetDescriptor] = []
        var start = max(range.lowerBound, 0)
        while start < range.upperBound {
            let count = min(windowSamples, range.upperBound - start)
            output.append(
                MeetingAudioAssetDescriptor(
                    source: source,
                    sampleRate: targetSampleRate,
                    startSample: start,
                    sampleCount: count
                )
            )
            start += count
        }
        return output
    }

    private func enqueueWrite(_ samples: [Float], at startIndex: Int, speaker: MeetingSpeaker) throws {
        guard !samples.isEmpty else { return }
        let samplesPerSegment = max(Int(Self.segmentDurationSeconds * targetSampleRate), 1)
        let writeBatchSamples = max(Int(Self.writeBatchDurationSeconds * targetSampleRate), 1)
        var remainingStart = startIndex
        var sampleOffset = 0

        while sampleOffset < samples.count {
            let segmentIndex = remainingStart / samplesPerSegment
            let offsetInSegment = remainingStart % samplesPerSegment
            let writableCount = min(samplesPerSegment - offsetInSegment, samples.count - sampleOffset)
            let newSamples = samples[sampleOffset..<(sampleOffset + writableCount)]

            if var pending = pendingWrites[speaker],
               pending.segmentIndex == segmentIndex,
               pending.endSampleOffset == offsetInSegment {
                pending.samples.append(contentsOf: newSamples)
                pendingWrites[speaker] = pending
            } else {
                try flushPendingWrite(for: speaker)
                pendingWrites[speaker] = PendingWrite(
                    segmentIndex: segmentIndex,
                    sampleOffset: offsetInSegment,
                    samples: Array(newSamples)
                )
            }

            if let pending = pendingWrites[speaker],
               pending.samples.count >= writeBatchSamples || pending.endSampleOffset >= samplesPerSegment {
                try flushPendingWrite(for: speaker)
            }
            sampleOffset += writableCount
            remainingStart += writableCount
        }
    }

    private func flushPendingWrites() throws {
        for speaker in [MeetingSpeaker.me, .them] {
            try flushPendingWrite(for: speaker)
        }
    }

    private func flushPendingWrite(for speaker: MeetingSpeaker) throws {
        guard let pending = pendingWrites[speaker], !pending.samples.isEmpty else {
            pendingWrites[speaker] = nil
            return
        }
        let url = try segmentURL(for: speaker, index: pending.segmentIndex)
        if !fileManager.fileExists(atPath: url.path) {
            fileManager.createFile(atPath: url.path, contents: nil, attributes: nil)
        }
        let handle = try writeHandle(for: speaker, segmentIndex: pending.segmentIndex, url: url)
        let byteOffset = UInt64(pending.sampleOffset * MemoryLayout<Float>.size)
        try handle.seek(toOffset: byteOffset)
        let data = Self.floatData(from: pending.samples)
        try handle.write(contentsOf: data)
        ioStatistics.writeOperationCount += 1
        ioStatistics.writtenByteCount += Int64(data.count)
        pendingWrites[speaker] = nil
    }

    private func writeHandle(
        for speaker: MeetingSpeaker,
        segmentIndex: Int,
        url: URL
    ) throws -> FileHandle {
        if let active = activeWriteHandles[speaker], active.segmentIndex == segmentIndex {
            return active.handle
        }
        if let active = activeWriteHandles.removeValue(forKey: speaker) {
            try? active.handle.close()
        }
        let handle = try FileHandle(forWritingTo: url)
        activeWriteHandles[speaker] = ActiveWriteHandle(segmentIndex: segmentIndex, handle: handle)
        ioStatistics.writeHandleOpenCount += 1
        return handle
    }

    private func prepareForReading() throws {
        try flushPendingWrites()
        closeActiveWriteHandles()
    }

    private func closeActiveWriteHandles() {
        for writer in activeWriteHandles.values {
            try? writer.handle.close()
        }
        activeWriteHandles.removeAll(keepingCapacity: false)
    }

    private func readMixedSamples(in range: Range<Int>) -> [Float] {
        let me = readSamples(speaker: .me, in: range)
        let them = readSamples(speaker: .them, in: range)
        guard !me.isEmpty || !them.isEmpty else { return [] }

        let count = max(me.count, them.count)
        var output = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let meSample = index < me.count ? me[index] : 0
            let themSample = index < them.count ? them[index] : 0
            output[index] = max(-1, min(1, (meSample + themSample) * 0.5))
        }
        return output
    }

    private func readSamples(speaker: MeetingSpeaker, in range: Range<Int>) -> [Float] {
        guard range.lowerBound < range.upperBound else { return [] }
        let samplesPerSegment = max(Int(Self.segmentDurationSeconds * targetSampleRate), 1)
        var output = [Float](repeating: 0, count: range.count)
        var cursor = range.lowerBound

        while cursor < range.upperBound {
            let segmentIndex = cursor / samplesPerSegment
            let offsetInSegment = cursor % samplesPerSegment
            let readableCount = min(samplesPerSegment - offsetInSegment, range.upperBound - cursor)
            if let url = segmentURLIfPresent(for: speaker, index: segmentIndex),
               let handle = try? FileHandle(forReadingFrom: url) {
                ioStatistics.readHandleOpenCount += 1
                defer { try? handle.close() }
                let byteStart = UInt64(offsetInSegment * MemoryLayout<Float>.size)
                let requestedBytes = readableCount * MemoryLayout<Float>.size
                do {
                    try handle.seek(toOffset: byteStart)
                    if let data = try handle.read(upToCount: requestedBytes), !data.isEmpty {
                        ioStatistics.readOperationCount += 1
                        ioStatistics.readByteCount += Int64(data.count)
                        let count = min(readableCount, data.count / MemoryLayout<Float>.size)
                        if count > 0 {
                            let samples = Self.floatSamples(from: data, sampleOffset: 0, count: count)
                            output.replaceSubrange((cursor - range.lowerBound)..<(cursor - range.lowerBound + count), with: samples)
                        }
                    }
                } catch {
                    VoxtLog.meetingWarning("Meeting audio archive window read failed: \(error.localizedDescription)")
                }
            }
            cursor += readableCount
        }

        return output
    }

    private func combinedWrittenRange() -> Range<Int>? {
        switch (meWrittenRange, themWrittenRange) {
        case let (lhs?, rhs?):
            return min(lhs.lowerBound, rhs.lowerBound)..<max(lhs.upperBound, rhs.upperBound)
        case let (lhs?, nil):
            return lhs
        case let (nil, rhs?):
            return rhs
        case (nil, nil):
            return nil
        }
    }

    private func ensureTempDirectoryURL() throws -> URL {
        if let tempDirectoryURL {
            return tempDirectoryURL
        }
        let url = fileManager.temporaryDirectory
            .appendingPathComponent("Voxt-MeetingAudioArchive-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        tempDirectoryURL = url
        return url
    }

    private func segmentURL(for speaker: MeetingSpeaker, index: Int) throws -> URL {
        try ensureTempDirectoryURL()
            .appendingPathComponent("\(speakerStorageName(speaker))-\(index).f32")
    }

    private func segmentURLIfPresent(for speaker: MeetingSpeaker, index: Int) -> URL? {
        guard let tempDirectoryURL else { return nil }
        return tempDirectoryURL.appendingPathComponent("\(speakerStorageName(speaker))-\(index).f32")
    }

    private func speakerStorageName(_ speaker: MeetingSpeaker) -> String {
        switch speaker {
        case .me:
            return "me"
        case .them:
            return "them"
        }
    }

    private static func union(_ existing: Range<Int>?, with next: Range<Int>) -> Range<Int> {
        guard let existing else { return next }
        return min(existing.lowerBound, next.lowerBound)..<max(existing.upperBound, next.upperBound)
    }

    private static func floatData(from samples: [Float]) -> Data {
        samples.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }

    private static func floatSamples(from data: Data, sampleOffset: Int, count: Int) -> [Float] {
        let byteStart = sampleOffset * MemoryLayout<Float>.size
        let byteEnd = byteStart + count * MemoryLayout<Float>.size
        guard byteStart >= 0, byteEnd <= data.count else { return [] }
        var output = [Float](repeating: 0, count: count)
        _ = output.withUnsafeMutableBytes { outputBuffer in
            data.copyBytes(to: outputBuffer, from: byteStart..<byteEnd)
        }
        return output
    }

    private static func pcm16Data(from samples: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(samples.count * MemoryLayout<Int16>.size)
        for sample in samples {
            let clamped = max(-1, min(1, sample))
            let value = Int16((clamped * Float(Int16.max)).rounded())
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    private static func wavHeader(sampleCount: Int, sampleRate: Double) -> Data {
        let byteRate = UInt32(sampleRate.rounded()) * UInt32(MemoryLayout<Int16>.size)
        let dataSize = UInt32(sampleCount * MemoryLayout<Int16>.size)
        let riffSize = UInt32(36) + dataSize
        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        data.appendUInt32LE(riffSize)
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.appendUInt32LE(16)
        data.appendUInt16LE(1)
        data.appendUInt16LE(1)
        data.appendUInt32LE(UInt32(sampleRate.rounded()))
        data.appendUInt32LE(byteRate)
        data.appendUInt16LE(UInt16(MemoryLayout<Int16>.size))
        data.appendUInt16LE(16)
        data.append(contentsOf: "data".utf8)
        data.appendUInt32LE(dataSize)
        return data
    }

    private static func resample(samples: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        guard !samples.isEmpty, inputRate > 0, outputRate > 0 else { return samples }
        if abs(inputRate - outputRate) <= 1 {
            return samples
        }

        let ratio = outputRate / inputRate
        let outputCount = max(Int(Double(samples.count) * ratio), 1)
        var output = [Float](repeating: 0, count: outputCount)

        for index in 0..<outputCount {
            let position = Double(index) / ratio
            let lowerIndex = Int(position)
            let upperIndex = min(lowerIndex + 1, samples.count - 1)
            let fraction = Float(position - Double(lowerIndex))
            let lower = samples[min(lowerIndex, samples.count - 1)]
            let upper = samples[upperIndex]
            output[index] = lower + (upper - lower) * fraction
        }

        return output
    }
}

private extension Data {
    nonisolated mutating func appendUInt16LE(_ value: UInt16) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }

    nonisolated mutating func appendUInt32LE(_ value: UInt32) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}
