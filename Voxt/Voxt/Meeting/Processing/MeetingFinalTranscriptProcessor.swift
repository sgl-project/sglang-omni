// MeetingFinalTranscriptProcessor.swift
// Provides Meeting Final Transcript Processor for meeting transcript processing.

import Foundation

enum MeetingFinalTranscriptionPass {
    enum Failure: LocalizedError, Equatable {
        case assetUnavailable(TranscriptAudioSource)

        var errorDescription: String? {
            switch self {
            case let .assetUnavailable(source):
                return "Meeting audio asset is unavailable for \(source.rawValue)."
            }
        }
    }

    struct Options: Equatable, Sendable {
        var maxChunkSeconds: TimeInterval = 22
        var overlapSeconds: TimeInterval = 1.0
        var minimumChunkSeconds: TimeInterval = 0.6
        var minimumRMS: Float = 0.002
        var silenceSplitSeconds: TimeInterval = 0.85
        var silenceWindowSeconds: TimeInterval = 0.10
        var speechPaddingSeconds: TimeInterval = 0.12

        nonisolated init(
            maxChunkSeconds: TimeInterval = 22,
            overlapSeconds: TimeInterval = 1.0,
            minimumChunkSeconds: TimeInterval = 0.6,
            minimumRMS: Float = 0.002,
            silenceSplitSeconds: TimeInterval = 0.85,
            silenceWindowSeconds: TimeInterval = 0.10,
            speechPaddingSeconds: TimeInterval = 0.12
        ) {
            self.maxChunkSeconds = maxChunkSeconds
            self.overlapSeconds = overlapSeconds
            self.minimumChunkSeconds = minimumChunkSeconds
            self.minimumRMS = minimumRMS
            self.silenceSplitSeconds = silenceSplitSeconds
            self.silenceWindowSeconds = silenceWindowSeconds
            self.speechPaddingSeconds = speechPaddingSeconds
        }
    }

    static func transcribe(
        assets: [MeetingAudioAsset],
        transcriber: any MeetingSegmentTranscribing,
        options: Options = Options(),
        requiresCompleteTranscription: Bool = false
    ) async throws -> [MeetingTranscriptSegment] {
        var segments: [MeetingTranscriptSegment] = []
        for asset in assets {
            if let wholeAssetSegments = try await transcriber.transcribeWholeAsset(asset) {
                appendCleaned(wholeAssetSegments, to: &segments)
                continue
            }
            let chunks = chunks(for: asset, options: options)
            for chunk in chunks {
                let chunkSegments = if requiresCompleteTranscription {
                    try await transcriber.transcribeSegmentsStrict(chunk: chunk)
                } else {
                    await transcriber.transcribeSegments(chunk: chunk)
                }
                appendCleaned(chunkSegments, to: &segments)
            }
        }
        return MeetingTranscriptPostProcessor.process(segments)
    }

    static func transcribe(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        transcriber: any MeetingSegmentTranscribing,
        options: Options = Options(),
        requiresCompleteTranscription: Bool = false,
        progress: (@Sendable (Double) async -> Void)? = nil,
        processedDurationProgress: (@Sendable (Double, TimeInterval) async -> Void)? = nil,
        initialSegments: [MeetingTranscriptSegment] = [],
        startingDescriptorIndex: Int = 0,
        checkpoint: (@Sendable ([MeetingTranscriptSegment], Int) async throws -> Void)? = nil
    ) async throws -> [MeetingTranscriptSegment] {
        var segments = initialSegments
        let descriptorCount = max(descriptors.count, 1)
        let startIndex = min(max(startingDescriptorIndex, 0), descriptors.count)
        var completedDurationBeforeDescriptor = descriptors
            .prefix(startIndex)
            .reduce(0) { $0 + max($1.durationSeconds, 0) }
        await progress?(startIndex == 0 ? 0 : Double(startIndex) / Double(descriptorCount))
        await processedDurationProgress?(
            startIndex == 0 ? 0 : Double(startIndex) / Double(descriptorCount),
            completedDurationBeforeDescriptor
        )
        for (relativeDescriptorIndex, descriptor) in descriptors.dropFirst(startIndex).enumerated() {
            let descriptorIndex = relativeDescriptorIndex + startIndex
            try Task.checkCancellation()
            let descriptorDuration = max(descriptor.durationSeconds, 0)
            defer { completedDurationBeforeDescriptor += descriptorDuration }
            guard let asset = await loadAsset(descriptor) else {
                throw Failure.assetUnavailable(descriptor.source)
            }
            try Task.checkCancellation()
            if let wholeAssetSegments = try await transcriber.transcribeWholeAsset(asset) {
                appendCleaned(wholeAssetSegments, to: &segments)
                let descriptorProgress = Double(descriptorIndex + 1) / Double(descriptorCount)
                await progress?(descriptorProgress)
                await processedDurationProgress?(
                    descriptorProgress,
                    completedDurationBeforeDescriptor + descriptorDuration
                )
                try await checkpoint?(segments, descriptorIndex + 1)
                continue
            }
            let chunks = chunks(for: asset, options: options)
            let chunkCount = max(chunks.count, 1)
            for (chunkIndex, chunk) in chunks.enumerated() {
                try Task.checkCancellation()
                let chunkSegments = if requiresCompleteTranscription {
                    try await transcriber.transcribeSegmentsStrict(chunk: chunk)
                } else {
                    await transcriber.transcribeSegments(chunk: chunk)
                }
                try Task.checkCancellation()
                appendCleaned(chunkSegments, to: &segments)
                let descriptorProgress = Double(chunkIndex + 1) / Double(chunkCount)
                let overallProgress = (Double(descriptorIndex) + descriptorProgress) / Double(descriptorCount)
                await progress?(overallProgress)
                await processedDurationProgress?(
                    overallProgress,
                    completedDurationBeforeDescriptor + descriptorDuration * descriptorProgress
                )
            }
            if chunks.isEmpty {
                let descriptorProgress = Double(descriptorIndex + 1) / Double(descriptorCount)
                await progress?(descriptorProgress)
                await processedDurationProgress?(
                    descriptorProgress,
                    completedDurationBeforeDescriptor + descriptorDuration
                )
            }
            try await checkpoint?(segments, descriptorIndex + 1)
        }
        try Task.checkCancellation()
        return MeetingTranscriptPostProcessor.process(segments)
    }

    private static func appendCleaned(
        _ newSegments: [MeetingTranscriptSegment],
        to segments: inout [MeetingTranscriptSegment]
    ) {
        for segment in newSegments {
            let cleaned = segment.updatingText(
                MeetingTranscriptTextPostProcessor.normalizedFinalText(segment.text)
            )
            if !cleaned.text.isEmpty {
                segments.append(cleaned)
            }
        }
    }

    struct ChunkPlan {
        let samples: Range<Int>
        let preventsAdjacentMerge: Bool
    }

    /// Retain only sample ranges for upcoming chunks. The lazy collection copies
    /// PCM for the one chunk being consumed, not every overlapping chunk at once.
    static func chunks(
        for asset: MeetingAudioAsset,
        options: Options = Options()
    ) -> LazyMapCollection<[ChunkPlan], BufferedMeetingChunk> {
        chunkPlans(for: asset, options: options).lazy.map { plan in
            BufferedMeetingChunk(
                segmentID: UUID(), speaker: asset.source.defaultSpeaker,
                startSeconds: asset.sessionStartOffset + Double(plan.samples.lowerBound) / asset.sampleRate,
                endSeconds: asset.sessionStartOffset + Double(plan.samples.upperBound) / asset.sampleRate,
                sampleRate: asset.sampleRate, samples: Array(asset.samples[plan.samples]),
                isFinal: true, preventsAdjacentMerge: plan.preventsAdjacentMerge
            )
        }
    }

    static func chunkPlans(for asset: MeetingAudioAsset, options: Options = Options()) -> [ChunkPlan] {
        guard asset.sampleRate > 0, !asset.samples.isEmpty else { return [] }
        let regions = speechRegions(in: asset, options: options)
        guard regions.count > 1 else {
            return fixedWindowChunks(
                for: asset,
                range: 0..<asset.samples.count,
                preventsAdjacentMerge: false,
                options: options
            )
        }

        return regions.flatMap { region in
            fixedWindowChunks(
                for: asset,
                range: region,
                preventsAdjacentMerge: true,
                options: options
            )
        }
    }

    private static func fixedWindowChunks(
        for asset: MeetingAudioAsset,
        range: Range<Int>,
        preventsAdjacentMerge: Bool,
        options: Options
    ) -> [ChunkPlan] {
        let sampleRate = asset.sampleRate
        let maxChunkSamples = max(Int(options.maxChunkSeconds * sampleRate), 1)
        let overlapSamples = max(Int(options.overlapSeconds * sampleRate), 0)
        let stepSamples = max(maxChunkSamples - overlapSamples, 1)
        let minimumSamples = max(Int(options.minimumChunkSeconds * sampleRate), 1)
        var chunks: [ChunkPlan] = []
        var start = range.lowerBound

        while start < range.upperBound {
            let end = min(start + maxChunkSamples, range.upperBound)
            let samples = asset.samples[start..<end]
            if samples.count >= minimumSamples,
               rootMeanSquare(samples) >= options.minimumRMS {
                chunks.append(
                    ChunkPlan(samples: start..<end, preventsAdjacentMerge: preventsAdjacentMerge)
                )
            }
            guard end < range.upperBound else { break }
            start += stepSamples
        }

        return chunks
    }

    private static func speechRegions(
        in asset: MeetingAudioAsset,
        options: Options
    ) -> [Range<Int>] {
        let sampleRate = asset.sampleRate
        let samples = asset.samples
        let windowSamples = max(Int(options.silenceWindowSeconds * sampleRate), 1)
        let silenceWindowsNeeded = max(Int(ceil(options.silenceSplitSeconds / options.silenceWindowSeconds)), 1)
        let paddingSamples = max(Int(options.speechPaddingSeconds * sampleRate), 0)
        let minimumSamples = max(Int(options.minimumChunkSeconds * sampleRate), 1)

        var regions: [Range<Int>] = []
        var currentSpeechStart: Int?
        var lastSpeechEnd = 0
        var consecutiveSilenceWindows = 0

        var windowStart = 0
        while windowStart < samples.count {
            let windowEnd = min(windowStart + windowSamples, samples.count)
            let window = samples[windowStart..<windowEnd]
            let isSpeech = rootMeanSquare(window) >= options.minimumRMS

            if isSpeech {
                if currentSpeechStart == nil {
                    currentSpeechStart = windowStart
                }
                lastSpeechEnd = windowEnd
                consecutiveSilenceWindows = 0
            } else if currentSpeechStart != nil {
                consecutiveSilenceWindows += 1
                if consecutiveSilenceWindows >= silenceWindowsNeeded {
                    let lowerBound = max((currentSpeechStart ?? 0) - paddingSamples, 0)
                    let upperBound = min(lastSpeechEnd + paddingSamples, samples.count)
                    if upperBound - lowerBound >= minimumSamples {
                        regions.append(lowerBound..<upperBound)
                    }
                    currentSpeechStart = nil
                    consecutiveSilenceWindows = 0
                }
            }

            windowStart = windowEnd
        }

        if let currentSpeechStart {
            let lowerBound = max(currentSpeechStart - paddingSamples, 0)
            let upperBound = min(lastSpeechEnd + paddingSamples, samples.count)
            if upperBound - lowerBound >= minimumSamples {
                regions.append(lowerBound..<upperBound)
            }
        }

        return regions
    }

    private static func rootMeanSquare<C: Collection>(_ samples: C) -> Float where C.Element == Float {
        guard !samples.isEmpty else { return 0 }
        var energy: Float = 0
        for sample in samples {
            energy += sample * sample
        }
        return sqrt(energy / Float(samples.count))
    }
}

enum MeetingTranscriptPostProcessor {
    struct Options: Equatable, Sendable {
        var maxSameSpeakerMergeGapSeconds: TimeInterval = 0.55
        var maxMergedDurationSeconds: TimeInterval = 28
        var maxMergedTextCharacters = 420
        var maxSegmentDurationSeconds: TimeInterval = 28
        var maxSegmentTextCharacters = 260
        var minSegmentTextCharacters = 0

        nonisolated init(
            maxSameSpeakerMergeGapSeconds: TimeInterval = 0.55,
            maxMergedDurationSeconds: TimeInterval = 28,
            maxMergedTextCharacters: Int = 420,
            maxSegmentDurationSeconds: TimeInterval = 28,
            maxSegmentTextCharacters: Int = 260,
            minSegmentTextCharacters: Int = 0
        ) {
            self.maxSameSpeakerMergeGapSeconds = maxSameSpeakerMergeGapSeconds
            self.maxMergedDurationSeconds = maxMergedDurationSeconds
            self.maxMergedTextCharacters = maxMergedTextCharacters
            self.maxSegmentDurationSeconds = maxSegmentDurationSeconds
            self.maxSegmentTextCharacters = maxSegmentTextCharacters
            self.minSegmentTextCharacters = minSegmentTextCharacters
        }

        static let liveOverlay = Options(
            maxSameSpeakerMergeGapSeconds: 0.35,
            maxMergedDurationSeconds: 16,
            maxMergedTextCharacters: 240,
            maxSegmentDurationSeconds: 16,
            maxSegmentTextCharacters: 130,
            minSegmentTextCharacters: 42
        )
    }

    static func process(
        _ segments: [MeetingTranscriptSegment],
        options: Options = Options()
    ) -> [MeetingTranscriptSegment] {
        let cleaned = MeetingTranscriptFormatter.meaningfulSegments(for: segments)
            .map { segment in
                segment.updatingText(MeetingTranscriptTextPostProcessor.normalizedFinalText(segment.text))
            }
            .filter { !$0.text.isEmpty }
            .sorted { lhs, rhs in
                if lhs.startSeconds == rhs.startSeconds {
                    return lhs.id.uuidString < rhs.id.uuidString
                }
                return lhs.startSeconds < rhs.startSeconds
            }

        var output: [MeetingTranscriptSegment] = []
        for segment in cleaned {
            if let previous = output.last,
               let merged = mergedAdjacentSegment(previous: previous, next: segment, options: options) {
                output[output.count - 1] = merged
            } else {
                output.append(segment)
            }
        }
        return output.flatMap { splitLongSegment($0, options: options) }
    }

    private static func mergedAdjacentSegment(
        previous: MeetingTranscriptSegment,
        next: MeetingTranscriptSegment,
        options: Options
    ) -> MeetingTranscriptSegment? {
        guard !previous.preventsAdjacentMerge, !next.preventsAdjacentMerge else { return nil }
        guard previous.speakerIdentityKey == next.speakerIdentityKey else { return nil }
        let previousEnd = previous.endSeconds ?? previous.startSeconds
        let nextEnd = next.endSeconds ?? next.startSeconds
        let gap = next.startSeconds - previousEnd
        let textOverlapLength = MeetingTranscriptTextPostProcessor.prefixSuffixOverlapLength(previous.text, next.text)
        guard next.startSeconds >= previous.startSeconds,
              gap <= options.maxSameSpeakerMergeGapSeconds
        else {
            return nil
        }
        if gap < 0, textOverlapLength < 4 {
            return nil
        }

        let mergedDuration = max(previousEnd, nextEnd) - previous.startSeconds
        guard mergedDuration <= options.maxMergedDurationSeconds else { return nil }
        let mergedText = MeetingTranscriptTextPostProcessor.mergedTextRemovingOverlap(previous.text, next.text)
        guard mergedText.count <= options.maxMergedTextCharacters else { return nil }

        return MeetingTranscriptSegment(
            id: previous.id,
            speaker: previous.speaker,
            speakerID: previous.speakerID,
            speakerDisplayName: previous.speakerDisplayName,
            audioSource: previous.audioSource,
            speakerConfidence: [previous.speakerConfidence, next.speakerConfidence]
                .compactMap { $0 }
                .max(),
            startSeconds: previous.startSeconds,
            endSeconds: max(previousEnd, nextEnd),
            text: mergedText,
            translatedText: nil,
            isTranslationPending: false,
            preventsAdjacentMerge: false
        )
    }

    private static func splitLongSegment(
        _ segment: MeetingTranscriptSegment,
        options: Options
    ) -> [MeetingTranscriptSegment] {
        let segmentEnd = segment.endSeconds ?? segment.startSeconds
        let duration = max(0, segmentEnd - segment.startSeconds)
        guard segment.text.count > options.maxSegmentTextCharacters else {
            return [segment]
        }

        let textDrivenChunkCount = Int(ceil(Double(segment.text.count) / Double(max(options.maxSegmentTextCharacters, 1))))
        let targetChunkCount = max(textDrivenChunkCount, 1)
        let targetMaxCharacters = max(
            24,
            min(
                options.maxSegmentTextCharacters,
                Int(ceil(Double(max(segment.text.count, 1)) / Double(targetChunkCount)))
            )
        )

        let chunks = MeetingTranscriptTextPostProcessor.readableChunks(
            segment.text,
            maxCharacters: targetMaxCharacters,
            minCharacters: min(options.minSegmentTextCharacters, targetMaxCharacters)
        )
        guard chunks.count > 1 else { return [segment] }

        let totalWeight = Double(chunks.map { max($0.count, 1) }.reduce(0, +))
        guard totalWeight > 0 else { return [segment] }

        var startSeconds = segment.startSeconds
        return chunks.enumerated().map { index, chunk in
            let isLast = index == chunks.count - 1
            let endSeconds: TimeInterval?
            if duration > 0 {
                if isLast {
                    endSeconds = segmentEnd
                } else {
                    let weight = Double(max(chunk.count, 1))
                    endSeconds = min(segmentEnd, startSeconds + duration * weight / totalWeight)
                }
            } else {
                endSeconds = segment.endSeconds
            }

            let splitSegment = MeetingTranscriptSegment(
                id: index == 0 ? segment.id : UUID(),
                speaker: segment.speaker,
                speakerID: segment.speakerID,
                speakerDisplayName: segment.speakerDisplayName,
                audioSource: segment.audioSource,
                speakerConfidence: segment.speakerConfidence,
                startSeconds: startSeconds,
                endSeconds: endSeconds,
                text: chunk,
                translatedText: nil,
                isTranslationPending: false,
                preventsAdjacentMerge: true
            )
            if let endSeconds {
                startSeconds = endSeconds
            }
            return splitSegment
        }
    }
}

enum MeetingTranscriptTextPostProcessor {
    nonisolated static func normalizedFinalText(_ text: String) -> String {
        let collapsedAcronyms = collapseSpacedAcronyms(in: text)
        return collapsedAcronyms
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: " ,", with: ",")
            .replacingOccurrences(of: " .", with: ".")
            .replacingOccurrences(of: " ，", with: "，")
            .replacingOccurrences(of: " 。", with: "。")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated static func mergedTextRemovingOverlap(_ lhs: String, _ rhs: String) -> String {
        let left = normalizedFinalText(lhs)
        let right = normalizedFinalText(rhs)
        guard !left.isEmpty else { return right }
        guard !right.isEmpty else { return left }
        if left == right || left.hasSuffix(right) {
            return left
        }
        if right.hasPrefix(left) {
            return right
        }

        let maxOverlap = min(left.count, right.count, 80)
        if maxOverlap >= 4 {
            for length in stride(from: maxOverlap, through: 4, by: -1) {
                let leftSuffix = String(left.suffix(length))
                if right.hasPrefix(leftSuffix) {
                    let suffixStart = right.index(right.startIndex, offsetBy: length)
                    return normalizedFinalText(left + String(right[suffixStart...]))
                }
            }
        }

        let separator = needsInlineSeparator(leftLast: left.unicodeScalars.last, rightFirst: right.unicodeScalars.first) ? " " : ""
        return left + separator + right
    }

    nonisolated static func prefixSuffixOverlapLength(_ lhs: String, _ rhs: String) -> Int {
        let left = normalizedFinalText(lhs)
        let right = normalizedFinalText(rhs)
        guard !left.isEmpty, !right.isEmpty else { return 0 }
        if left == right || left.hasSuffix(right) {
            return right.count
        }
        if right.hasPrefix(left) {
            return left.count
        }
        let maxOverlap = min(left.count, right.count, 80)
        guard maxOverlap >= 4 else { return 0 }
        for length in stride(from: maxOverlap, through: 4, by: -1) {
            if right.hasPrefix(String(left.suffix(length))) {
                return length
            }
        }
        return 0
    }

    nonisolated static func readableChunks(
        _ text: String,
        maxCharacters: Int,
        minCharacters: Int = 0
    ) -> [String] {
        let normalized = normalizedFinalText(text)
        guard normalized.count > maxCharacters else {
            return normalized.isEmpty ? [] : [normalized]
        }

        let clauses = readableClauses(in: normalized, maxCharacters: maxCharacters)
        var chunks: [String] = []
        var current = ""
        for clause in clauses {
            guard !clause.isEmpty else { continue }
            if current.isEmpty {
                current = clause
            } else if current.count + clause.count <= maxCharacters {
                let separator = needsInlineSeparator(
                    leftLast: current.unicodeScalars.last,
                    rightFirst: clause.unicodeScalars.first
                ) ? " " : ""
                current = normalizedFinalText(current + separator + clause)
            } else {
                chunks.append(current)
                current = clause
            }
        }
        if !current.isEmpty {
            chunks.append(current)
        }
        return balancedReadableChunks(
            chunks,
            minCharacters: minCharacters,
            maxCharacters: maxCharacters
        )
    }

    nonisolated private static func balancedReadableChunks(
        _ chunks: [String],
        minCharacters: Int,
        maxCharacters: Int
    ) -> [String] {
        let minCharacters = max(minCharacters, 0)
        guard minCharacters > 0, chunks.count > 1 else { return chunks }

        var output: [String] = []
        var index = 0
        while index < chunks.count {
            var current = chunks[index]

            while current.count < minCharacters, index + 1 < chunks.count {
                let next = chunks[index + 1]
                let candidate = joinedReadableChunk(current, next)
                guard candidate.count <= maxCharacters else { break }
                current = candidate
                index += 1
            }

            if current.count < minCharacters,
               let previous = output.last {
                let candidate = joinedReadableChunk(previous, current)
                if candidate.count <= maxCharacters {
                    output[output.count - 1] = candidate
                } else {
                    output.append(current)
                }
            } else {
                output.append(current)
            }
            index += 1
        }

        return output
    }

    nonisolated private static func joinedReadableChunk(_ lhs: String, _ rhs: String) -> String {
        let separator = needsInlineSeparator(
            leftLast: lhs.unicodeScalars.last,
            rightFirst: rhs.unicodeScalars.first
        ) ? " " : ""
        return normalizedFinalText(lhs + separator + rhs)
    }

    nonisolated private static func readableClauses(in text: String, maxCharacters: Int) -> [String] {
        let hardBreaks = Set("。！？!?；;")
        let softBreaks = Set("，,、")
        var clauses: [String] = []
        var current = ""

        for character in text {
            current.append(character)
            let shouldBreak = hardBreaks.contains(character)
                || (softBreaks.contains(character) && current.count >= maxCharacters / 2)
                || (current.count >= maxCharacters && !current.contains(where: \.isWhitespace))
            if shouldBreak {
                clauses.append(contentsOf: splitOversizedClause(current, maxCharacters: maxCharacters))
                current = ""
            }
        }

        if !current.isEmpty {
            clauses.append(contentsOf: splitOversizedClause(current, maxCharacters: maxCharacters))
        }
        return clauses
            .map(normalizedFinalText)
            .filter { !$0.isEmpty }
    }

    nonisolated private static func splitOversizedClause(_ text: String, maxCharacters: Int) -> [String] {
        let normalized = normalizedFinalText(text)
        guard normalized.count > maxCharacters else { return [normalized] }
        if normalized.contains(where: \.isWhitespace) {
            return splitOversizedWhitespaceClause(normalized, maxCharacters: maxCharacters)
        }

        var chunks: [String] = []
        var current = ""
        for character in normalized {
            current.append(character)
            if current.count >= maxCharacters {
                chunks.append(normalizedFinalText(current))
                current = ""
            }
        }
        if !current.isEmpty {
            chunks.append(normalizedFinalText(current))
        }
        return chunks
    }

    nonisolated private static func splitOversizedWhitespaceClause(_ text: String, maxCharacters: Int) -> [String] {
        var chunks: [String] = []
        var current = ""
        for word in text.split(whereSeparator: \.isWhitespace).map(String.init) {
            if word.count > maxCharacters {
                if !current.isEmpty {
                    chunks.append(normalizedFinalText(current))
                    current = ""
                }
                chunks.append(contentsOf: splitOversizedClause(word, maxCharacters: maxCharacters))
            } else if current.isEmpty {
                current = word
            } else if current.count + 1 + word.count <= maxCharacters {
                current += " " + word
            } else {
                chunks.append(normalizedFinalText(current))
                current = word
            }
        }
        if !current.isEmpty {
            chunks.append(normalizedFinalText(current))
        }
        return chunks
    }

    nonisolated private static func collapseSpacedAcronyms(in text: String) -> String {
        let pattern = #"\b(?:[A-Za-z]\s+){1,}[A-Za-z]\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, range: nsRange).reversed()
        var result = text
        for match in matches {
            guard let range = Range(match.range, in: result) else { continue }
            let replacement = result[range].filter { !$0.isWhitespace }
            result.replaceSubrange(range, with: replacement)
        }
        return result
    }

    nonisolated private static func needsInlineSeparator(
        leftLast: UnicodeScalar?,
        rightFirst: UnicodeScalar?
    ) -> Bool {
        guard let leftLast, let rightFirst else { return true }
        let punctuationScalars = CharacterSet(charactersIn: " \t\n\r,.!?;:，。！？；：、)]}\"'》】）")
        if punctuationScalars.contains(leftLast) || punctuationScalars.contains(rightFirst) {
            return false
        }
        return CharacterSet.alphanumerics.contains(leftLast) && CharacterSet.alphanumerics.contains(rightFirst)
    }
}

private extension TranscriptSegment {
    func updatingText(_ text: String) -> TranscriptSegment {
        TranscriptSegment(
            id: id,
            speaker: speaker,
            speakerID: speakerID,
            speakerDisplayName: speakerDisplayName,
            audioSource: audioSource,
            speakerConfidence: speakerConfidence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            text: text,
            translatedText: translatedText,
            isTranslationPending: isTranslationPending,
            preventsAdjacentMerge: preventsAdjacentMerge
        )
    }
}
