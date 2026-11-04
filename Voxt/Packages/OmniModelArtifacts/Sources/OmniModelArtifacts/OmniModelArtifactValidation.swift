import Foundation

public nonisolated enum OmniModelArtifactValidationError: LocalizedError, Equatable, Sendable {
    case invalidDirectory(path: String)
    case invalidFile(path: String, reason: String)
    case missingSafetensors(path: String)

    public var errorDescription: String? {
        switch self {
        case .invalidDirectory(let path):
            return "Model artifact directory is not readable: \(path)"
        case .invalidFile(let path, let reason):
            return "Invalid model artifact at \(path): \(reason)"
        case .missingSafetensors(let path):
            return "No top-level .safetensors files in \(path)"
        }
    }
}

public nonisolated enum OmniModelArtifactValidation {
    private static let maximumReadByteCount = 16 * 1024 * 1024
    private static let lengthPrefixByteCount = 8

    /// Checks bounded JSON and safetensors structure, not model compatibility or file integrity.
    public static func validateSafetensorsDirectory(at directory: URL) throws {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard directory.isFileURL,
              fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw OmniModelArtifactValidationError.invalidDirectory(path: directory.path)
        }
        let entries: [URL]
        do {
            entries = try fileManager.contentsOfDirectory(at: directory.resolvingSymlinksInPath(), includingPropertiesForKeys: nil)
        } catch {
            throw OmniModelArtifactValidationError.invalidDirectory(path: directory.path)
        }
        guard let configuration = entries.first(where: { $0.lastPathComponent == "config.json" }) else {
            throw OmniModelArtifactValidationError.invalidFile(
                path: directory.appendingPathComponent("config.json").path,
                reason: "Missing root config.json."
            )
        }
        try validateFile(at: configuration, isConfiguration: true)
        let weightFiles = entries.filter { $0.pathExtension == "safetensors" }
        guard !weightFiles.isEmpty else {
            throw OmniModelArtifactValidationError.missingSafetensors(path: directory.path)
        }
        for weightFile in weightFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            try validateFile(at: weightFile, isConfiguration: false)
        }
    }

    private static func validateFile(at file: URL, isConfiguration: Bool) throws {
        do {
            let resolvedFile = file.resolvingSymlinksInPath()
            let attributes = try FileManager.default.attributesOfItem(atPath: resolvedFile.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular else {
                throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Expected a regular file.")
            }
            let handle = try FileHandle(forReadingFrom: resolvedFile)
            defer { try? handle.close() }
            let fileByteCount = try handle.seekToEnd()
            try handle.seek(toOffset: 0)
            guard fileByteCount > 0 else {
                throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "File is empty.")
            }
            if isConfiguration {
                guard fileByteCount <= UInt64(maximumReadByteCount) else {
                    throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Config exceeds 16 MiB.")
                }
                let configurationBytes = try readExactly(Int(fileByteCount), from: handle, file: file)
                guard try JSONSerialization.jsonObject(with: configurationBytes) is [String: Any] else {
                    throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Config must be a JSON object.")
                }
                try ModelArtifactJSONSyntax.validate(configurationBytes, requiresUnsignedIntegers: false)
            } else {
                guard fileByteCount >= UInt64(lengthPrefixByteCount) else {
                    throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Truncated length prefix.")
                }
                let prefixBytes = try readExactly(lengthPrefixByteCount, from: handle, file: file)
                let headerByteCount = prefixBytes.enumerated().reduce(UInt64(0)) {
                    $0 | (UInt64($1.element) << ($1.offset * 8))
                }
                guard headerByteCount > 0,
                      headerByteCount <= UInt64(maximumReadByteCount - lengthPrefixByteCount),
                      headerByteCount <= fileByteCount - UInt64(lengthPrefixByteCount) else {
                    throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Invalid or truncated header length.")
                }
                let headerBytes = try readExactly(Int(headerByteCount), from: handle, file: file)
                guard headerBytes.first == UInt8(ascii: "{"), !headerBytes.contains(0) else {
                    throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Header must be a UTF-8 JSON object beginning with '{'.")
                }
                let header = try SafetensorsHeader(bytes: headerBytes)
                let payloadByteCount = fileByteCount - UInt64(lengthPrefixByteCount) - headerByteCount
                try header.validate(payloadByteCount: payloadByteCount, file: file)
            }
        } catch let error as OmniModelArtifactValidationError {
            throw error
        } catch {
            throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: error.localizedDescription)
        }
    }

    private static func readExactly(_ byteCount: Int, from handle: FileHandle, file: URL) throws -> Data {
        var bytes = Data()
        while bytes.count < byteCount {
            guard let nextBytes = try handle.read(upToCount: byteCount - bytes.count), !nextBytes.isEmpty else {
                throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Unexpected end of file.")
            }
            bytes.append(nextBytes)
        }
        return bytes
    }
}
