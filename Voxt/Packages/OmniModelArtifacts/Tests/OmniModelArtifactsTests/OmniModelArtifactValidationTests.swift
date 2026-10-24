import Foundation
import OmniModelArtifacts
import XCTest

final class OmniModelArtifactValidationTests: XCTestCase {
    private var fixtureRoot: URL!
    private var directory: URL!
    private let scalarHeader = #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#

    override func setUpWithError() throws {
        fixtureRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        directory = fixtureRoot.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"model_type":"fixture"}"#.utf8).write(to: directory.appendingPathComponent("config.json"))
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: fixtureRoot)
    }

    func testAuthenticScalarTensorAndStringMetadata() throws {
        try writeSafetensors(header: #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"__metadata__":{"format":"pt","description":"λ"}}"#)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
    }

    func testMultipleShardsAndOutOfOrderTensors() throws {
        try writeSafetensors(header: scalarHeader, named: "first.safetensors")
        try writeSafetensors(
            header: #"{"later":{"dtype":"U8","shape":[2],"data_offsets":[2,4]},"earlier":{"dtype":"I16","shape":[1],"data_offsets":[0,2]}}"#,
            named: "second.safetensors"
        )
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try Data("broken".utf8).write(to: directory.appendingPathComponent("second.safetensors"))
        assertRejected()
    }

    func testTruncatedLengthPrefixes() throws {
        for byteCount in 0..<8 {
            try Data(repeating: 0, count: byteCount).write(to: directory.appendingPathComponent("model.safetensors"))
            assertRejected()
        }
    }

    func testInvalidHeaderLengths() throws {
        for headerByteCount in [UInt64(0), UInt64(16 * 1024 * 1024), UInt64.max] {
            try writeSafetensors(header: scalarHeader, declaredHeaderByteCount: headerByteCount)
            assertRejected()
        }
        try writeSafetensors(header: scalarHeader, declaredHeaderByteCount: 4096)
        assertRejected()
    }

    func testMalformedJSONAndNonObjectHeaders() throws {
        for header in ["{", "[]", "null", "{}junk", "{\"weight\":null}"] {
            try writeSafetensors(header: header)
            assertRejected()
        }
    }

    func testShapeProductAndDtypeByteCountOverflow() throws {
        for shape in ["[18446744073709551615,2]", "[9223372036854775808]", "[18446744073709551616]"] {
            try writeSafetensors(header: "{\"weight\":{\"dtype\":\"F32\",\"shape\":\(shape),\"data_offsets\":[0,4]}}")
            assertRejected()
        }
    }

    func testMissingOrNonFileDirectoryAndMissingWeights() throws {
        assertRejected()
        for invalidDirectory in [
            fixtureRoot.appendingPathComponent("missing"),
            directory.appendingPathComponent("config.json"),
            URL(string: "https://example.invalid/model")!,
        ] {
            XCTAssertThrowsError(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: invalidDirectory)) { error in
                XCTAssertEqual(error as? OmniModelArtifactValidationError, .invalidDirectory(path: invalidDirectory.path))
            }
        }
    }

    func testExactRootConfigurationNameAndJSONObjectRequired() throws {
        try writeSafetensors(header: scalarHeader)
        let configuration = directory.appendingPathComponent("config.json")
        for json in ["", "[]", "null", "true", "7", #""text""#, "{", "{}trailing"] {
            try Data(json.utf8).write(to: configuration)
            assertRejected()
        }
        for json in ["{}", #"{"unrelated":[1,null,true,{"nested":"value"}]}"#] {
            try Data(json.utf8).write(to: configuration)
            XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        }
        try FileManager.default.moveItem(at: configuration, to: directory.appendingPathComponent("Config.json"))
        assertRejected()
        try FileManager.default.removeItem(at: directory.appendingPathComponent("Config.json"))
        let nestedDirectory = directory.appendingPathComponent("nested")
        try FileManager.default.createDirectory(at: nestedDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: nestedDirectory.appendingPathComponent("config.json"))
        assertRejected()
    }

    func testOnlyLowercaseTopLevelExtensionIsSelectedIncludingHiddenFiles() throws {
        try writeSafetensors(header: scalarHeader, named: "model.SAFETENSORS")
        let nestedDirectory = directory.appendingPathComponent("nested")
        try FileManager.default.createDirectory(at: nestedDirectory, withIntermediateDirectories: true)
        try writeSafetensors(header: scalarHeader, named: "nested/model.safetensors")
        assertRejected()
        try writeSafetensors(header: scalarHeader, named: ".hidden.safetensors")
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try Data().write(to: directory.appendingPathComponent("model.SAFETENSORS"))
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try Data().write(to: directory.appendingPathComponent(".hidden.safetensors"))
        assertRejected()
    }

    func testEverySupportedDtypeUsesCorrectByteWidth() throws {
        let byteWidths = [
            "BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
            "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
            "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
        ]
        for (dtype, byteWidth) in byteWidths {
            let payloadByteCount = 6 * byteWidth
            try writeSafetensors(
                header: "{\"weight\":{\"dtype\":\"\(dtype)\",\"shape\":[2,3],\"data_offsets\":[0,\(payloadByteCount)]}}",
                payload: Data(repeating: 0, count: payloadByteCount)
            )
            XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory), dtype)
        }
    }

    func testUnknownOrIncorrectlyCasedDtypesAreRejected() throws {
        for dtype in ["F4", "F6_E2M3", "F6_E3M2", "F128", "COMPLEX64", "float32", "f32", ""] {
            try writeSafetensors(header: "{\"weight\":{\"dtype\":\"\(dtype)\",\"shape\":[1],\"data_offsets\":[0,4]}}")
            assertRejected()
        }
    }

    func testEmptyTensorsAndBoundaryOffsetsAreValid() throws {
        try writeSafetensors(
            header: #"{"empty":{"dtype":"F64","shape":[3,0,4],"data_offsets":[0,0]}}"#,
            payload: Data()
        )
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try writeSafetensors(
            header: #"{"empty":{"dtype":"U8","shape":[18446744073709551615,0],"data_offsets":[0,0]}}"#,
            payload: Data()
        )
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try writeSafetensors(header: #"{"end":{"dtype":"U8","shape":[0],"data_offsets":[4,4]},"value":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"start":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
    }

    func testEmptyAndMetadataOnlyHeadersAreRejected() throws {
        for header in ["{}", #"{"__metadata__":{"format":"pt"}}"#] {
            try writeSafetensors(header: header, payload: Data())
            assertRejected()
        }
    }

    func testTensorSchemasAreRequiredAndClosed() throws {
        for descriptor in [
            "null", "[]", "{}",
            #"{"shape":[1],"data_offsets":[0,4]}"#,
            #"{"dtype":"F32","data_offsets":[0,4]}"#,
            #"{"dtype":"F32","shape":[1]}"#,
            #"{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":0}"#,
            #"{"dtype":null,"shape":[1],"data_offsets":[0,4]}"#,
            #"{"dtype":"F32","shape":1,"data_offsets":[0,4]}"#,
            #"{"dtype":"F32","shape":[1],"data_offsets":null}"#,
        ] {
            try writeSafetensors(header: "{\"weight\":\(descriptor)}")
            assertRejected()
        }
    }

    func testShapesRequireExactUnsignedIntegerTokens() throws {
        for dimension in ["-1", "-0", "1.0", "1e0", "1.00000000000000000000001", "true", "\"1\"", "null", "[]"] {
            try writeSafetensors(header: "{\"weight\":{\"dtype\":\"F32\",\"shape\":[\(dimension)],\"data_offsets\":[0,4]}}")
            assertRejected()
        }
    }

    func testOffsetsRequireTwoExactUnsignedIntegerTokens() throws {
        for offsets in [
            "[]", "[0]", "[0,4,4]", "[-1,4]", "[-0,4]", "[0,4.0]", "[0,4e0]",
            "[0,3.99999999999999999999]", "[false,4]", "[0,null]", "[\"0\",4]",
            "[0,18446744073709551616]", "[0,18446744073709551615]",
        ] {
            try writeSafetensors(header: "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":\(offsets)}}")
            assertRejected()
        }
    }

    func testTruncatedPayloadAndMismatchedLengthsAreRejected() throws {
        for payloadByteCount in [0, 1, 3, 5, 8] {
            try writeSafetensors(header: scalarHeader, payload: Data(repeating: 0, count: payloadByteCount))
            assertRejected()
        }
        for offsets in ["[0,0]", "[0,3]", "[1,4]", "[4,0]"] {
            try writeSafetensors(header: "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":\(offsets)}}")
            assertRejected()
        }
        try writeSafetensors(header: #"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#)
        assertRejected()
    }

    func testGapsOverlapsAndZeroRangesInsideTensorsAreRejected() throws {
        for header in [
            #"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[1],"data_offsets":[3,4]}}"#,
            #"{"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#,
            #"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
            #"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[0],"data_offsets":[2,2]}}"#,
        ] {
            try writeSafetensors(header: header)
            assertRejected()
        }
    }

    func testMetadataMustBeAStringDictionary() throws {
        for metadata in ["null", "[]", "\"pt\"", #"{"version":1}"#, #"{"nested":{}}"#, #"{"null":null}"#] {
            try writeSafetensors(header: "{\"weight\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]},\"__metadata__\":\(metadata)}")
            assertRejected()
        }
    }

    func testDuplicateFieldsIncludingEscapedSpellingsAreRejected() throws {
        for header in [
            #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
            #"{"weight":{"dtype":"F32","dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
            #"{"weight":{"dtype":"F32","shape":[],"sh\u0061pe":[],"data_offsets":[0,4]}}"#,
            #"{"__metadata__":{"format":"pt","format":"pt"},"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
        ] {
            try writeSafetensors(header: header)
            assertRejected()
        }
    }

    func testEscapedStringsAreNotInterpretedAsSchemaTokens() throws {
        try writeSafetensors(header: #"{"a\"b\\c\u0064":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"__metadata__":{"text":"[{}] 1.0 -42 1e100 \\\" : \uD83D\uDE00","other":"value"}}"#)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
    }

    func testRegularFilesRequiredForConfigurationAndEveryShard() throws {
        try writeSafetensors(header: scalarHeader)
        let extraShard = directory.appendingPathComponent("directory.safetensors")
        try FileManager.default.createDirectory(at: extraShard, withIntermediateDirectories: true)
        assertRejected()
        try FileManager.default.removeItem(at: extraShard)
        let configuration = directory.appendingPathComponent("config.json")
        try FileManager.default.removeItem(at: configuration)
        try FileManager.default.createDirectory(at: configuration, withIntermediateDirectories: true)
        assertRejected()
    }

    func testInternalAndExternalCacheSymlinksAreAllowed() throws {
        try writeSafetensors(header: scalarHeader)
        for blobPath in ["blobs", "../external-blobs"] {
            let blobs = directory.appendingPathComponent(blobPath)
            try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
            for filename in ["config.json", "model.safetensors"] {
                let artifact = directory.appendingPathComponent(filename)
                try FileManager.default.moveItem(at: artifact, to: blobs.appendingPathComponent(filename))
                try FileManager.default.createSymbolicLink(atPath: artifact.path, withDestinationPath: "\(blobPath)/\(filename)")
            }
            XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
            for filename in ["config.json", "model.safetensors"] {
                let artifact = directory.appendingPathComponent(filename)
                try FileManager.default.removeItem(at: artifact)
                try FileManager.default.moveItem(at: blobs.appendingPathComponent(filename), to: artifact)
            }
        }
        let linkedDirectory = fixtureRoot.appendingPathComponent("snapshot-link")
        try FileManager.default.createSymbolicLink(at: linkedDirectory, withDestinationURL: directory)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: linkedDirectory))
    }

    func testBrokenAndNonregularSymlinkTargetsAreRejected() throws {
        try writeSafetensors(header: scalarHeader)
        let weightFile = directory.appendingPathComponent("model.safetensors")
        try FileManager.default.removeItem(at: weightFile)
        for destination in ["missing-blob", ".", "model.safetensors"] {
            try FileManager.default.createSymbolicLink(atPath: weightFile.path, withDestinationPath: destination)
            assertRejected()
            try FileManager.default.removeItem(at: weightFile)
        }
        try writeSafetensors(header: scalarHeader)
        let configuration = directory.appendingPathComponent("config.json")
        try FileManager.default.removeItem(at: configuration)
        try FileManager.default.createSymbolicLink(atPath: configuration.path, withDestinationPath: "missing-config")
        assertRejected()
    }

    func testUnreadableArtifactIsRejected() throws {
        try writeSafetensors(header: scalarHeader)
        let weightFile = directory.appendingPathComponent("model.safetensors")
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: weightFile.path)
        guard !FileManager.default.isReadableFile(atPath: weightFile.path) else {
            throw XCTSkip("This user can read files despite mode 000; permission rejection cannot be exercised.")
        }
        assertRejected()
    }

    func testConfigurationReadLimitIsInclusive() throws {
        let maximumReadByteCount = 16 * 1024 * 1024
        try writeSafetensors(header: scalarHeader)
        let configuration = directory.appendingPathComponent("config.json")
        var configurationBytes = Data("{}".utf8)
        configurationBytes.append(Data(repeating: 0x20, count: maximumReadByteCount - configurationBytes.count))
        try configurationBytes.write(to: configuration)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        configurationBytes.append(0x20)
        try configurationBytes.write(to: configuration)
        assertRejected()
    }

    func testHeaderReadLimitIncludesEightBytePrefix() throws {
        let maximumReadByteCount = 16 * 1024 * 1024
        let header = scalarHeader + String(repeating: " ", count: maximumReadByteCount - 8 - scalarHeader.utf8.count)
        try writeSafetensors(header: header)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try writeSafetensors(header: header + String(repeating: " ", count: 8))
        assertRejected()
    }

    func testMultiGigabyteSparsePayloadRequiresOnlyHeaderReads() throws {
        let payloadByteCount: UInt64 = 8 * 1024 * 1024 * 1024
        try writeSafetensors(
            header: "{\"weight\":{\"dtype\":\"U8\",\"shape\":[\(payloadByteCount)],\"data_offsets\":[0,\(payloadByteCount)]}}",
            payload: Data()
        )
        let handle = try FileHandle(forWritingTo: directory.appendingPathComponent("model.safetensors"))
        defer { try? handle.close() }
        let headerEndByteOffset = try handle.seekToEnd()
        try handle.truncate(atOffset: headerEndByteOffset + payloadByteCount)
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
        try handle.truncate(atOffset: headerEndByteOffset + payloadByteCount - 1)
        assertRejected()
    }

    func testInvalidUTF8HeaderAndBigEndianPrefixAreRejected() throws {
        try writeSafetensors(header: scalarHeader)
        let weightFile = directory.appendingPathComponent("model.safetensors")
        var fileBytes = try Data(contentsOf: weightFile)
        fileBytes[10] = 0xff
        try fileBytes.write(to: weightFile)
        assertRejected()
        try writeSafetensors(header: scalarHeader)
        fileBytes = try Data(contentsOf: weightFile)
        fileBytes.replaceSubrange(0..<8, with: Array(fileBytes.prefix(8).reversed()))
        try fileBytes.write(to: weightFile)
        assertRejected()
    }

    func testHeaderMustUseUTF8NotUTF16() throws {
        let headerBytes = try XCTUnwrap(scalarHeader.data(using: .utf16LittleEndian))
        let headerByteCount = UInt64(headerBytes.count)
        var fileBytes = Data((0..<8).map { UInt8(truncatingIfNeeded: headerByteCount >> ($0 * 8)) })
        fileBytes.append(headerBytes)
        fileBytes.append(contentsOf: [0, 0, 128, 63])
        try fileBytes.write(to: directory.appendingPathComponent("model.safetensors"))
        assertRejected()
    }

    func testTrailingHeaderCommasAreRejected() throws {
        for header in [
            #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4,]}}"#,
            #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4],}}"#,
            #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}, }"#,
            #"{"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"__metadata__":{"format":"pt",}}"#,
        ] {
            try writeSafetensors(header: header)
            assertRejected()
        }
    }

    func testConfigurationTrailingCommasAndDuplicateFieldsAreRejected() throws {
        try writeSafetensors(header: scalarHeader)
        for configuration in [#"{"threshold":0.5,}"#, #"{"thresholds":[0.5, ]}"#,
                              #"{"branch":{"threshold":0.5,}}"#, #"{"threshold":0.5,"threshold":0.6}"#] {
            try Data(configuration.utf8).write(to: directory.appendingPathComponent("config.json"))
            assertRejected()
        }
    }

    func testConfigurationMustUseUTF8() throws {
        try writeSafetensors(header: scalarHeader)
        let configuration = try XCTUnwrap(#"{"threshold":0.5}"#.data(using: .utf16LittleEndian))
        try configuration.write(to: directory.appendingPathComponent("config.json"))
        assertRejected()
    }

    func testConfigurationAllowsSignedFractionalValuesAndCommaLikeStrings() throws {
        try writeSafetensors(header: scalarHeader)
        let configuration = #"{"threshold":0.5,"gain":-1.25e-2,"branch":{"values":[-1,0.5,2]},"label":", } ] ","escaped":"\"}"}"#
        try Data(configuration.utf8).write(to: directory.appendingPathComponent("config.json"))
        XCTAssertNoThrow(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory))
    }

    private func writeSafetensors(
        header: String,
        payload: Data = Data([0, 0, 128, 63]),
        named filename: String = "model.safetensors",
        declaredHeaderByteCount: UInt64? = nil
    ) throws {
        var headerBytes = Data(header.utf8)
        headerBytes.append(Data(repeating: 0x20, count: (8 - headerBytes.count % 8) % 8))
        let headerByteCount = declaredHeaderByteCount ?? UInt64(headerBytes.count)
        var fileBytes = Data((0..<8).map { UInt8(truncatingIfNeeded: headerByteCount >> ($0 * 8)) })
        fileBytes.append(headerBytes)
        fileBytes.append(payload)
        try fileBytes.write(to: directory.appendingPathComponent(filename))
    }

    private func assertRejected(file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory), file: file, line: line) { error in
            XCTAssertTrue(error is OmniModelArtifactValidationError, "Unexpected error: \(error)", file: file, line: line)
            XCTAssertFalse(error.localizedDescription.isEmpty, file: file, line: line)
        }
    }
}
