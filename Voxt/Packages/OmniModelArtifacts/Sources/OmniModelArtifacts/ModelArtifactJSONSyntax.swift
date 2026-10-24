import Foundation

nonisolated enum ModelArtifactJSONSyntax {
    // note (OpenAI): Foundation accepts duplicate fields, trailing commas, and non-UTF-8 JSON.
    static func validate(_ bytes: Data, requiresUnsignedIntegers: Bool) throws {
        guard String(data: bytes, encoding: .utf8) != nil, !bytes.contains(0) else {
            throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Expected UTF-8 JSON."))
        }
        let whitespaceBytes = Set(" \t\n\r".utf8)
        let numberMarkerBytes = Set("-+.eE".utf8)
        var objectFields: [Set<String>] = []
        var byteOffset = 0
        while byteOffset < bytes.count {
            switch bytes[byteOffset] {
            case UInt8(ascii: "{"):
                objectFields.append([])
                byteOffset += 1
            case UInt8(ascii: "}"):
                objectFields.removeLast()
                byteOffset += 1
            case UInt8(ascii: "\""):
                let startByteOffset = byteOffset
                byteOffset += 1
                while byteOffset < bytes.count {
                    if bytes[byteOffset] == UInt8(ascii: "\\") {
                        byteOffset += 2
                    } else if bytes[byteOffset] == UInt8(ascii: "\"") {
                        break
                    } else {
                        byteOffset += 1
                    }
                }
                let endByteOffset = byteOffset
                byteOffset += 1
                while byteOffset < bytes.count, whitespaceBytes.contains(bytes[byteOffset]) {
                    byteOffset += 1
                }
                guard byteOffset < bytes.count, bytes[byteOffset] == UInt8(ascii: ":") else { continue }
                let field = try JSONDecoder().decode(String.self, from: bytes[startByteOffset...endByteOffset])
                guard objectFields[objectFields.count - 1].insert(field).inserted else {
                    throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Duplicate JSON field: \(field)"))
                }
            case UInt8(ascii: ","):
                byteOffset += 1
                while byteOffset < bytes.count, whitespaceBytes.contains(bytes[byteOffset]) {
                    byteOffset += 1
                }
                guard byteOffset < bytes.count, bytes[byteOffset] != UInt8(ascii: "}"),
                      bytes[byteOffset] != UInt8(ascii: "]") else {
                    throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Trailing JSON comma."))
                }
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
                let startByteOffset = byteOffset
                while byteOffset < bytes.count,
                      (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(bytes[byteOffset])
                        || numberMarkerBytes.contains(bytes[byteOffset]) {
                    byteOffset += 1
                }
                guard requiresUnsignedIntegers else { continue }
                let number = String(decoding: bytes[startByteOffset..<byteOffset], as: UTF8.self)
                guard !number.hasPrefix("-"), UInt64(number) != nil else {
                    throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Expected an unsigned integer JSON token."))
                }
            default:
                byteOffset += 1
            }
        }
    }
}
