import Foundation

/// App scheduling fields only; model compatibility is validated by the inference runtime.
nonisolated struct MeetingSortformerConfiguration: Decodable, Equatable, Sendable {
    let sampleRate: Int
    let featureHopSampleCount: Int
    let subsamplingFactor: Int
    let chunkFrameCount: Int
    let speakerCacheFrameCount: Int
    let speakerCacheUpdateFrameCount: Int
    let usesAOSC: Bool
    let chunkLeftContextFrameCount: Int
    let encoderMaximumFrameCount: Int

    private static let maximumConfigurationByteCount = 16 * 1024 * 1024

    init(
        sampleRate: Int,
        featureHopSampleCount: Int,
        subsamplingFactor: Int,
        chunkFrameCount: Int,
        speakerCacheFrameCount: Int,
        speakerCacheUpdateFrameCount: Int,
        usesAOSC: Bool,
        chunkLeftContextFrameCount: Int,
        encoderMaximumFrameCount: Int
    ) throws {
        guard speakerCacheUpdateFrameCount >= 0, chunkLeftContextFrameCount >= 0,
              encoderMaximumFrameCount > 0 else {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
        self.sampleRate = sampleRate
        self.featureHopSampleCount = featureHopSampleCount
        self.subsamplingFactor = subsamplingFactor
        self.chunkFrameCount = chunkFrameCount
        self.speakerCacheFrameCount = speakerCacheFrameCount
        self.speakerCacheUpdateFrameCount = speakerCacheUpdateFrameCount
        self.usesAOSC = usesAOSC
        self.chunkLeftContextFrameCount = chunkLeftContextFrameCount
        self.encoderMaximumFrameCount = encoderMaximumFrameCount
        _ = try feedPolicy()
    }

    init(from decoder: Decoder) throws {
        do {
            let root = try decoder.container(keyedBy: CodingKeys.self)
            let processor = try Self.section(.processor, in: root)
            let encoder = try Self.section(.encoder, in: root)
            let modules = try Self.section(.modules, in: root)
            let transformer = try Self.section(.transformer, in: root)
            try self.init(
                sampleRate: processor.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000,
                featureHopSampleCount: processor.decodeIfPresent(Int.self, forKey: .featureHopSampleCount) ?? 160,
                subsamplingFactor: encoder.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8,
                chunkFrameCount: modules.decodeIfPresent(Int.self, forKey: .chunkFrameCount) ?? 188,
                speakerCacheFrameCount: modules.decodeIfPresent(Int.self, forKey: .speakerCacheFrameCount) ?? 188,
                speakerCacheUpdateFrameCount: modules.decodeIfPresent(Int.self, forKey: .speakerCacheUpdateFrameCount) ?? 188,
                usesAOSC: modules.decodeIfPresent(Bool.self, forKey: .usesAOSC) ?? false,
                chunkLeftContextFrameCount: modules.decodeIfPresent(Int.self, forKey: .chunkLeftContextFrameCount) ?? 1,
                encoderMaximumFrameCount: transformer.decodeIfPresent(Int.self, forKey: .encoderMaximumFrameCount) ?? 1_500
            )
        } catch {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
    }

    static func load(from directory: URL) throws -> Self {
        do {
            guard directory.isFileURL else { throw MeetingSpeakerFeedError.invalidConfiguration }
            let configurationURL = directory.appendingPathComponent("config.json", isDirectory: false)
                .resolvingSymlinksInPath()
            let attributes = try FileManager.default.attributesOfItem(atPath: configurationURL.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular,
                  FileManager.default.isReadableFile(atPath: configurationURL.path) else {
                throw MeetingSpeakerFeedError.invalidConfiguration
            }
            let handle = try FileHandle(forReadingFrom: configurationURL)
            let configurationBytes: Data
            do {
                configurationBytes = try handle.read(upToCount: maximumConfigurationByteCount + 1) ?? Data()
            } catch {
                try? handle.close()
                throw error
            }
            try handle.close()
            guard !configurationBytes.isEmpty,
                  configurationBytes.count <= maximumConfigurationByteCount else {
                throw MeetingSpeakerFeedError.invalidConfiguration
            }
            return try JSONDecoder().decode(Self.self, from: configurationBytes)
        } catch {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
    }

    func feedPolicy() throws -> MeetingSpeakerFeedPolicy {
        try MeetingSpeakerFeedPolicy(
            sampleRate: sampleRate,
            hopLength: featureHopSampleCount,
            subsamplingFactor: subsamplingFactor,
            chunkFrames: chunkFrameCount,
            cacheFrames: speakerCacheFrameCount,
            updateFrames: speakerCacheUpdateFrameCount,
            usesAOSC: usesAOSC
        )
    }

    func validateEncoderCapacity(cacheMaximumFrameCount: Int, fifoMaximumFrameCount: Int) throws {
        guard cacheMaximumFrameCount >= 0, fifoMaximumFrameCount >= 0 else {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
        var requiredFrameCount = max(cacheMaximumFrameCount, speakerCacheFrameCount)
        for frameCount in [fifoMaximumFrameCount, chunkLeftContextFrameCount, speakerCacheUpdateFrameCount] {
            let (totalFrameCount, overflow) = requiredFrameCount.addingReportingOverflow(frameCount)
            guard !overflow else { throw MeetingSpeakerFeedError.invalidConfiguration }
            requiredFrameCount = totalFrameCount
        }
        guard requiredFrameCount <= encoderMaximumFrameCount else {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
    }

    private enum CodingKeys: String, CodingKey {
        case processor = "processor_config"
        case encoder = "fc_encoder_config"
        case modules = "modules_config"
        case transformer = "tf_encoder_config"
        case sampleRate = "sampling_rate"
        case featureHopSampleCount = "hop_length"
        case subsamplingFactor = "subsampling_factor"
        case chunkFrameCount = "chunk_len"
        case speakerCacheFrameCount = "spkcache_len"
        case speakerCacheUpdateFrameCount = "spkcache_update_period"
        case usesAOSC = "use_aosc"
        case chunkLeftContextFrameCount = "chunk_left_context"
        case encoderMaximumFrameCount = "max_source_positions"
    }

    private static func section(
        _ key: CodingKeys,
        in root: KeyedDecodingContainer<CodingKeys>
    ) throws -> KeyedDecodingContainer<CodingKeys> {
        if try !root.contains(key) || root.decodeNil(forKey: key) {
            return root
        } else {
            return try root.nestedContainer(keyedBy: CodingKeys.self, forKey: key)
        }
    }
}
