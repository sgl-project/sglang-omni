import Foundation
import XCTest
@testable import Voxt

final class MeetingSortformerConfigurationTests: XCTestCase {
    func testDecoderDefaultsAndNullFields() throws {
        let expected = try configuration()
        XCTAssertEqual(try decode("{}"), expected)
        XCTAssertEqual(try decode("""
            {"sampling_rate":null,"hop_length":null,"subsampling_factor":null,
             "chunk_len":null,"spkcache_len":null,"spkcache_update_period":null,
             "use_aosc":null,"chunk_left_context":null,"max_source_positions":null}
            """), expected)
        let policy = try expected.feedPolicy()
        XCTAssertEqual(policy.sampleRate, 16_000)
        XCTAssertEqual(policy.frameSamples, 1_280)
        XCTAssertEqual(policy.samplesPerFeed, 79_360)
        XCTAssertEqual(policy.cacheMaximumFrames, 188)
    }

    func testNestedAndFlatFieldsHaveTheSameProjection() throws {
        let expected = try configuration(featureHopSampleCount: 80, subsamplingFactor: 4,
                                         chunkFrameCount: 12, speakerCacheFrameCount: 64,
                                         speakerCacheUpdateFrameCount: 6, usesAOSC: true,
                                         chunkLeftContextFrameCount: 0, encoderMaximumFrameCount: 600)
        XCTAssertEqual(try decode("""
            {"processor_config":{"sampling_rate":16000,"hop_length":80},
             "fc_encoder_config":{"subsampling_factor":4},
             "modules_config":{"chunk_len":12,"spkcache_len":64,"spkcache_update_period":6,
                               "use_aosc":true,"chunk_left_context":0,"subsampling_factor":32},
             "tf_encoder_config":{"max_source_positions":600}}
            """), expected)
        let flatFields = """
            "sampling_rate":16000,"hop_length":80,"subsampling_factor":4,
            "chunk_len":12,"spkcache_len":64,"spkcache_update_period":6,
            "use_aosc":true,"chunk_left_context":0,"max_source_positions":600
            """
        XCTAssertEqual(try decode("{\(flatFields)}"), expected)
        XCTAssertEqual(try decode("""
            {\(flatFields),"processor_config":null,"fc_encoder_config":null,
             "modules_config":null,"tf_encoder_config":null}
            """), expected)
    }

    func testPresentSectionsDefaultTheirOwnMissingOrNullFields() throws {
        for sectionFields in ["", "\"sampling_rate\":null,\"hop_length\":null"] {
            XCTAssertEqual(try decode("""
                {"sampling_rate":8000,"hop_length":80,"subsampling_factor":4,
                 "chunk_len":12,"spkcache_len":64,"spkcache_update_period":6,
                 "use_aosc":true,"chunk_left_context":0,"max_source_positions":600,
                 "processor_config":{\(sectionFields)},"fc_encoder_config":{"subsampling_factor":null},
                 "modules_config":{"chunk_len":null,"spkcache_len":null,"spkcache_update_period":null,
                                   "use_aosc":null,"chunk_left_context":null},
                 "tf_encoder_config":{"max_source_positions":null}}
                """), try configuration())
        }
    }

    func testEachSectionFallsBackIndependently() throws {
        let decoded = try decode("""
            {"hop_length":80,"subsampling_factor":4,"chunk_len":12,"max_source_positions":600,
             "processor_config":null,"fc_encoder_config":{},
             "modules_config":{"spkcache_len":64},"tf_encoder_config":null}
            """)
        XCTAssertEqual(decoded, try configuration(featureHopSampleCount: 80,
                                                  speakerCacheFrameCount: 64,
                                                  encoderMaximumFrameCount: 600))
    }

    func testRejectsWrongSectionTypes() {
        for section in ["processor_config", "fc_encoder_config", "modules_config", "tf_encoder_config"] {
            for value in ["[]", "true", "1", "\"invalid\""] {
                assertInvalid { _ = try self.decode("{\"\(section)\":\(value)}") }
            }
        }
    }

    func testRejectsWrongTypesFractionsAndIntegersOutsideTheIntRange() {
        let integerFields = [
            ("processor_config", "sampling_rate"), ("processor_config", "hop_length"),
            ("fc_encoder_config", "subsampling_factor"), ("modules_config", "chunk_len"),
            ("modules_config", "spkcache_len"), ("modules_config", "spkcache_update_period"),
            ("modules_config", "chunk_left_context"), ("tf_encoder_config", "max_source_positions")
        ]
        for (section, field) in integerFields {
            for value in ["true", "\"160\"", "[]", "{}", "1.5", "-0.5",
                          "9223372036854775808", "-9223372036854775809", "1e100"] {
                assertInvalid { _ = try self.decode("{\"\(field)\":\(value)}") }
                assertInvalid { _ = try self.decode("{\"\(section)\":{\"\(field)\":\(value)}}") }
            }
        }
        for value in ["0", "1", "\"false\"", "[]", "{}"] {
            assertInvalid { _ = try self.decode("{\"use_aosc\":\(value)}") }
            assertInvalid { _ = try self.decode("{\"modules_config\":{\"use_aosc\":\(value)}}") }
        }
    }

    func testTypedGeneralSettingsReuseActualFeedPolicy() throws {
        for usesAOSC in [false, true] {
            for (hopSampleCount, subsamplingFactor, chunkFrameCount, cacheFrameCount, updateFrameCount) in [
                (80, 4, 12, 64, 6), (160, 8, 188, 188, 188), (1, 64, 1, 4_096, 3),
                (16_000, 1, Int.max, 1, Int.max)
            ] {
                let projected = try configuration(
                    featureHopSampleCount: hopSampleCount, subsamplingFactor: subsamplingFactor,
                    chunkFrameCount: chunkFrameCount, speakerCacheFrameCount: cacheFrameCount,
                    speakerCacheUpdateFrameCount: updateFrameCount, usesAOSC: usesAOSC
                ).feedPolicy()
                let actual = try MeetingSpeakerFeedPolicy(
                    sampleRate: 16_000, hopLength: hopSampleCount, subsamplingFactor: subsamplingFactor,
                    chunkFrames: chunkFrameCount, cacheFrames: cacheFrameCount,
                    updateFrames: updateFrameCount, usesAOSC: usesAOSC
                )
                XCTAssertEqual(projected.samplesPerFeed, actual.samplesPerFeed)
                XCTAssertEqual(projected.frameSamples, actual.frameSamples)
                XCTAssertEqual(projected.sampleRate, actual.sampleRate)
                XCTAssertEqual(projected.cacheMaximumFrames, actual.cacheMaximumFrames)
            }
        }
        XCTAssertNoThrow(try configuration(speakerCacheUpdateFrameCount: 0, usesAOSC: false))
    }

    func testProjectionDoesNotCertifyModelArchitectureCompatibility() throws {
        XCTAssertEqual(try decode("""
            {"model_type":"another-architecture","num_speakers":99,
             "processor_config":{"feature_size":3,"n_fft":7},
             "fc_encoder_config":{"hidden_size":11,"subsampling_factor":4},
             "modules_config":{"num_speakers":99,"subsampling_factor":32,"chunk_len":12},
             "tf_encoder_config":{"d_model":13},"unknown_extension":{"version":999}}
            """), try configuration(subsamplingFactor: 4, chunkFrameCount: 12))
    }

    func testTypedInitializationRejectsUnsafeGeneralSettings() {
        assertInvalid { _ = try self.configuration(sampleRate: 8_000) }
        for hopSampleCount in [Int.min, 0, 16_001, Int.max] {
            assertInvalid { _ = try self.configuration(featureHopSampleCount: hopSampleCount) }
        }
        for subsamplingFactor in [Int.min, 0, 65, Int.max] {
            assertInvalid { _ = try self.configuration(subsamplingFactor: subsamplingFactor) }
        }
        for chunkFrameCount in [Int.min, -1, 0] {
            assertInvalid { _ = try self.configuration(chunkFrameCount: chunkFrameCount) }
        }
        for cacheFrameCount in [Int.min, 0, MeetingSpeakerFeedPolicy.maximumCacheFrames + 1, Int.max] {
            assertInvalid { _ = try self.configuration(speakerCacheFrameCount: cacheFrameCount) }
        }
        for updateFrameCount in [Int.min, -1, 0, 1, 2] {
            assertInvalid {
                _ = try self.configuration(speakerCacheUpdateFrameCount: updateFrameCount, usesAOSC: true)
            }
        }
        assertInvalid { _ = try self.configuration(speakerCacheUpdateFrameCount: -1) }
        assertInvalid { _ = try self.configuration(chunkLeftContextFrameCount: -1) }
        assertInvalid { _ = try self.configuration(encoderMaximumFrameCount: 0) }
        assertInvalid { _ = try self.configuration(encoderMaximumFrameCount: -1) }
        assertInvalid { _ = try self.configuration(featureHopSampleCount: 16_000, subsamplingFactor: 64) }
        assertInvalid { _ = try self.decode("{\"hop_length\":0}") }
    }

    func testEncoderCapacityUsesConfiguredCacheAndAllowsExactBoundary() throws {
        let projected = try configuration(encoderMaximumFrameCount: 565)
        XCTAssertNoThrow(try projected.validateEncoderCapacity(cacheMaximumFrameCount: 0, fifoMaximumFrameCount: 188))
        XCTAssertNoThrow(try projected.validateEncoderCapacity(cacheMaximumFrameCount: 188, fifoMaximumFrameCount: 188))
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: 189, fifoMaximumFrameCount: 188) }
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: 188, fifoMaximumFrameCount: 189) }
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: -1, fifoMaximumFrameCount: 188) }
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: 188, fifoMaximumFrameCount: -1) }
        let smallEncoder = try configuration(encoderMaximumFrameCount: 1)
        assertInvalid { try smallEncoder.validateEncoderCapacity(cacheMaximumFrameCount: 0, fifoMaximumFrameCount: 0) }
    }

    func testEncoderCapacityRejectsOverflowAtEveryAddition() throws {
        let projected = try configuration(encoderMaximumFrameCount: Int.max)
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: Int.max, fifoMaximumFrameCount: 1) }
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: 0, fifoMaximumFrameCount: Int.max) }
        assertInvalid { try projected.validateEncoderCapacity(cacheMaximumFrameCount: Int.max, fifoMaximumFrameCount: 0) }
        let largeUpdate = try configuration(speakerCacheUpdateFrameCount: Int.max,
                                            chunkLeftContextFrameCount: 0, encoderMaximumFrameCount: Int.max)
        assertInvalid { try largeUpdate.validateEncoderCapacity(cacheMaximumFrameCount: 0, fifoMaximumFrameCount: 0) }
        let largeContext = try configuration(chunkLeftContextFrameCount: Int.max, encoderMaximumFrameCount: Int.max)
        assertInvalid { try largeContext.validateEncoderCapacity(cacheMaximumFrameCount: 0, fifoMaximumFrameCount: 0) }
        let exactMaximum = try configuration(speakerCacheUpdateFrameCount: Int.max - 188,
                                             chunkLeftContextFrameCount: 0, encoderMaximumFrameCount: Int.max)
        XCTAssertNoThrow(try exactMaximum.validateEncoderCapacity(cacheMaximumFrameCount: 188, fifoMaximumFrameCount: 0))
    }

    func testLoadNeedsOnlyExactConfigurationNotWeightPayloads() throws {
        let directory = try TemporaryDirectory()
        let configurationURL = directory.url.appendingPathComponent("config.json")
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
        try Data("{}".utf8).write(to: directory.url.appendingPathComponent("other-config.json"))
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
        try Data("{}".utf8).write(to: configurationURL)
        XCTAssertEqual(try MeetingSortformerConfiguration.load(from: directory.url), try configuration())
        let weightsURL = directory.url.appendingPathComponent("model.safetensors")
        try Data("not a model".utf8).write(to: weightsURL)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: weightsURL.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: weightsURL.path) }
        try FileManager.default.createDirectory(at: directory.url.appendingPathComponent("weights"),
                                               withIntermediateDirectories: false)
        XCTAssertEqual(try MeetingSortformerConfiguration.load(from: directory.url), try configuration())
    }

    func testLoadRejectsEmptyMalformedUnreadableAndNonregularConfiguration() throws {
        let directory = try TemporaryDirectory()
        let configurationURL = directory.url.appendingPathComponent("config.json")
        for contents in ["", " ", "{", "null", "[]", "{\"hop_length\":1.5}"] {
            try Data(contents.utf8).write(to: configurationURL)
            assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
        }
        try Data("{}".utf8).write(to: configurationURL)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: configurationURL.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configurationURL.path) }
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configurationURL.path)
        try FileManager.default.removeItem(at: configurationURL)
        try FileManager.default.createDirectory(at: configurationURL, withIntermediateDirectories: false)
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
        try FileManager.default.removeItem(at: configurationURL)
        let targetURL = directory.url.appendingPathComponent("target.json")
        try Data("{}".utf8).write(to: targetURL)
        try FileManager.default.createSymbolicLink(at: configurationURL, withDestinationURL: targetURL)
        XCTAssertEqual(try MeetingSortformerConfiguration.load(from: directory.url), try configuration())
        try FileManager.default.removeItem(at: targetURL)
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
    }

    func testHuggingFaceConfigurationLinksPreserveExternalBlobReads() throws {
        let directory = try TemporaryDirectory()
        let cache = try TemporaryDirectory()
        let configurationURL = directory.url.appendingPathComponent("config.json")
        let blobURL = cache.url.appendingPathComponent("config-blob")
        try Data("{\"modules_config\":{\"use_aosc\":true}}".utf8).write(to: blobURL)
        try FileManager.default.createSymbolicLink(at: configurationURL, withDestinationURL: blobURL)
        XCTAssertEqual(try MeetingSortformerConfiguration.load(from: directory.url), try configuration(usesAOSC: true))
    }

    func testLoadAcceptsExactlySixteenMiBAndRejectsValidJSONPrefixesBeyondLimit() throws {
        let directory = try TemporaryDirectory()
        let configurationURL = directory.url.appendingPathComponent("config.json")
        let maximumByteCount = 16 * 1024 * 1024
        var configurationBytes = Data("{}".utf8)
        configurationBytes.append(Data(repeating: 0x20, count: maximumByteCount - configurationBytes.count))
        try configurationBytes.write(to: configurationURL)
        XCTAssertEqual(try MeetingSortformerConfiguration.load(from: directory.url), try configuration())
        configurationBytes.append(0x20)
        try configurationBytes.write(to: configurationURL)
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
        let handle = try FileHandle(forWritingTo: configurationURL)
        defer { try? handle.close() }
        try handle.truncate(atOffset: 1_024 * 1_024 * 1_024)
        assertInvalid { _ = try MeetingSortformerConfiguration.load(from: directory.url) }
    }

    private func decode(_ json: String) throws -> MeetingSortformerConfiguration {
        try JSONDecoder().decode(MeetingSortformerConfiguration.self, from: Data(json.utf8))
    }

    private func assertInvalid(
        file: StaticString = #filePath, line: UInt = #line, _ operation: () throws -> Void
    ) {
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            guard case MeetingSpeakerFeedError.invalidConfiguration = error else {
                return XCTFail("Expected invalidConfiguration, got \(error)", file: file, line: line)
            }
        }
    }

    private func configuration(
        sampleRate: Int = 16_000,
        featureHopSampleCount: Int = 160,
        subsamplingFactor: Int = 8,
        chunkFrameCount: Int = 188,
        speakerCacheFrameCount: Int = 188,
        speakerCacheUpdateFrameCount: Int = 188,
        usesAOSC: Bool = false,
        chunkLeftContextFrameCount: Int = 1,
        encoderMaximumFrameCount: Int = 1_500
    ) throws -> MeetingSortformerConfiguration {
        try MeetingSortformerConfiguration(
            sampleRate: sampleRate, featureHopSampleCount: featureHopSampleCount,
            subsamplingFactor: subsamplingFactor, chunkFrameCount: chunkFrameCount,
            speakerCacheFrameCount: speakerCacheFrameCount,
            speakerCacheUpdateFrameCount: speakerCacheUpdateFrameCount, usesAOSC: usesAOSC,
            chunkLeftContextFrameCount: chunkLeftContextFrameCount,
            encoderMaximumFrameCount: encoderMaximumFrameCount
        )
    }
}
