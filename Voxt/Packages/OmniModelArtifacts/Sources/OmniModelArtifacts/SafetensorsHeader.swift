import Foundation

nonisolated struct SafetensorsHeader: Decodable {
    let tensors: [SafetensorsTensor]

    init(bytes: Data) throws {
        self = try JSONDecoder().decode(Self.self, from: bytes)
        try ModelArtifactJSONSyntax.validate(bytes, requiresUnsignedIntegers: true)
    }

    init(from decoder: any Decoder) throws {
        let fields = try decoder.container(keyedBy: SafetensorsField.self)
        var tensors: [SafetensorsTensor] = []
        for field in fields.allKeys {
            if field.stringValue == "__metadata__" {
                _ = try fields.decode([String: String].self, forKey: field)
            } else {
                tensors.append(try fields.decode(SafetensorsTensor.self, forKey: field))
            }
        }
        self.tensors = tensors
    }

    func validate(payloadByteCount: UInt64, file: URL) throws {
        guard !tensors.isEmpty else {
            throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Header contains no tensors.")
        }
        var consumedByteCount: UInt64 = 0
        for tensor in tensors.sorted(by: {
            ($0.dataOffsets[0], $0.dataOffsets[1]) < ($1.dataOffsets[0], $1.dataOffsets[1])
        }) {
            let startByteOffset = tensor.dataOffsets[0]
            let endByteOffset = tensor.dataOffsets[1]
            guard startByteOffset == consumedByteCount, endByteOffset >= startByteOffset,
                  endByteOffset <= payloadByteCount else {
                throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Tensor ranges have gaps, overlap, or exceed the file.")
            }
            var elementCount: UInt64 = 1
            for dimension in tensor.shape {
                let product = elementCount.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else {
                    throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Tensor element count overflows UInt64.")
                }
                elementCount = product.partialValue
            }
            let byteCount = elementCount.multipliedReportingOverflow(by: tensor.dtype.byteWidth)
            guard !byteCount.overflow, byteCount.partialValue == endByteOffset - startByteOffset else {
                throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Tensor byte length does not match its shape and dtype.")
            }
            consumedByteCount = endByteOffset
        }
        guard consumedByteCount == payloadByteCount else {
            throw OmniModelArtifactValidationError.invalidFile(path: file.path, reason: "Unreferenced trailing payload bytes.")
        }
    }
}

private nonisolated struct SafetensorsField: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }

    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

nonisolated struct SafetensorsTensor: Decodable {
    let dtype: SafetensorsDType
    let shape: [UInt64]
    let dataOffsets: [UInt64]

    private enum CodingKeys: String, CodingKey {
        case dtype, shape
        case dataOffsets = "data_offsets"
    }

    init(from decoder: any Decoder) throws {
        let allFields = try decoder.container(keyedBy: SafetensorsField.self)
        guard Set(allFields.allKeys.map(\.stringValue)) == ["dtype", "shape", "data_offsets"] else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Expected dtype, shape, and data_offsets fields."))
        }
        let fields = try decoder.container(keyedBy: CodingKeys.self)
        dtype = try fields.decode(SafetensorsDType.self, forKey: .dtype)
        shape = try fields.decode([UInt64].self, forKey: .shape)
        dataOffsets = try fields.decode([UInt64].self, forKey: .dataOffsets)
        guard dataOffsets.count == 2 else {
            throw DecodingError.dataCorruptedError(forKey: .dataOffsets, in: fields, debugDescription: "Expected two byte offsets.")
        }
    }
}

nonisolated enum SafetensorsDType: String, Decodable {
    case bool = "BOOL"
    case int8 = "I8", uint8 = "U8"
    case int16 = "I16", uint16 = "U16"
    case int32 = "I32", uint32 = "U32"
    case int64 = "I64", uint64 = "U64"
    case float16 = "F16", bfloat16 = "BF16"
    case float32 = "F32", float64 = "F64"
    case float8E4M3 = "F8_E4M3", float8E5M2 = "F8_E5M2", float8E8M0 = "F8_E8M0"

    var byteWidth: UInt64 {
        switch self {
        case .bool, .int8, .uint8, .float8E4M3, .float8E5M2, .float8E8M0: return 1
        case .int16, .uint16, .float16, .bfloat16: return 2
        case .int32, .uint32, .float32: return 4
        case .int64, .uint64, .float64: return 8
        }
    }
}
