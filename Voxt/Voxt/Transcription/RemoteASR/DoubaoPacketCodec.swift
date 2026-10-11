import Foundation
import zlib

/// Wire framing and bounded gzip shared by dictation and meetings. Transcript
/// selection stays with each workflow: whole-text snapshots are not utterance lists.
enum DoubaoPacketCodec {
    static let maxCompressedPayloadBytes = 2 * 1024 * 1024
    static let maxDecompressedPayloadBytes = 8 * 1024 * 1024
    static let maxCompressionRatio = 64

    struct ServerPacket {
        let payload: Data
        let sequence: Int32?
        let flags: UInt8

        var isFinal: Bool {
            (flags & DoubaoProtocol.flagLastAudioPacket) != 0 || (sequence ?? 1) < 0
        }
    }

    enum CodecError: LocalizedError {
        case compressionFailed
        case decompressionFailed
        case payloadTooLarge
        case excessiveExpansion
        case unsupportedCompression

        var errorDescription: String? {
            switch self {
            case .compressionFailed: return "Failed to compress Doubao payload."
            case .decompressionFailed: return "Failed to decode Doubao GZIP response payload."
            case .payloadTooLarge: return "Doubao response payload is too large."
            case .excessiveExpansion: return "Doubao response payload expands beyond the allowed size."
            case .unsupportedCompression: return "Doubao response compression is unsupported in current client."
            }
        }
    }

    static func buildPacket(
        messageType: UInt8,
        messageFlags: UInt8,
        serialization: UInt8,
        compression: UInt8,
        sequence: Int32,
        payload: Data
    ) -> Data {
        var data = Data([
            (DoubaoProtocol.version << 4) | DoubaoProtocol.headerSize,
            (messageType << 4) | messageFlags,
            (serialization << 4) | compression,
            0
        ])
        // Client flag 2 is a final packet WITHOUT sequence (meeting). Flag 3
        // includes the negative sequence used by dictation. Do not conflate them.
        if (messageFlags & DoubaoProtocol.flagPositiveSequence) != 0 {
            data.append(remoteASRBigEndianData(sequence))
        }
        data.append(remoteASRBigEndianData(UInt32(payload.count)))
        data.append(payload)
        return data
    }

    static func encodePayload(_ payload: Data) throws -> (compression: UInt8, payload: Data) {
        guard !payload.isEmpty else { return (DoubaoProtocol.compressionNone, payload) }
        return (DoubaoProtocol.compressionGzip, try gzip(payload))
    }

    static func decodeServerPacket(_ input: Data) throws -> ServerPacket? {
        guard input.count >= 8 else { return nil }
        // At most 15 header words plus sequence/event/error/length fields.
        guard input.count <= maxDecompressedPayloadBytes + 76 else { throw CodecError.payloadTooLarge }
        // Data slices may retain non-zero indices; avoid copying ordinary socket buffers.
        let data = input.startIndex == 0 ? input : input.withUnsafeBytes { Data($0) }
        let headerSize = max(4, Int(data[0] & 0x0F) * 4)
        guard data.count >= headerSize else { return nil }
        let messageType = (data[1] >> 4) & 0x0F
        let flags = data[1] & 0x0F
        let compression = data[2] & 0x0F
        guard [DoubaoProtocol.messageTypeFullServerResponse,
               DoubaoProtocol.messageTypeServerAck,
               DoubaoProtocol.messageTypeServerErrorResponse].contains(messageType) else { return nil }

        var cursor = headerSize
        var sequence: Int32?
        if (flags & DoubaoProtocol.flagPositiveSequence) != 0 {
            guard data.count >= cursor + 4 else { return nil }
            sequence = remoteASRInt32(fromBigEndian: data.subdata(in: cursor..<(cursor + 4)))
            cursor += 4
        }
        if (flags & DoubaoProtocol.flagEvent) != 0 {
            guard data.count >= cursor + 4 else { return nil }
            cursor += 4
        }
        if messageType == DoubaoProtocol.messageTypeServerErrorResponse {
            guard data.count >= cursor + 4 else { return nil }
            cursor += 4 // Provider error code; its description is in the payload.
        }
        let payloadSize: Int
        if messageType == DoubaoProtocol.messageTypeServerAck {
            payloadSize = data.count - cursor
        } else {
            guard data.count >= cursor + 4 else { return nil }
            payloadSize = Int(remoteASRUInt32(fromBigEndian: data.subdata(in: cursor..<(cursor + 4))))
            cursor += 4
            guard payloadSize <= data.count - cursor else { return nil }
        }
        let limit = compression == DoubaoProtocol.compressionGzip
            ? maxCompressedPayloadBytes : maxDecompressedPayloadBytes
        guard payloadSize <= limit else { throw CodecError.payloadTooLarge }
        let rawPayload = data.subdata(in: cursor..<(cursor + payloadSize))
        let payload: Data
        switch compression {
        case DoubaoProtocol.compressionNone: payload = rawPayload
        case DoubaoProtocol.compressionGzip: payload = try gunzip(rawPayload)
        default: throw CodecError.unsupportedCompression
        }
        if messageType == DoubaoProtocol.messageTypeServerErrorResponse {
            throw NSError(
                domain: "Voxt.RemoteASR", code: -7,
                userInfo: [NSLocalizedDescriptionKey: String(data: payload, encoding: .utf8) ?? "Unknown Doubao server error."]
            )
        }
        return ServerPacket(payload: payload, sequence: sequence, flags: flags)
    }

    static func sequence(in object: Any) -> Int32? {
        if let dictionary = object as? [String: Any] {
            for key in ["sequence", "seq", "autoAssignedSequence", "auto_assigned_sequence"] {
                if let number = dictionary[key] as? NSNumber,
                   let value = Int32(exactly: number.doubleValue) {
                    return value
                }
            }
            // Other numeric fields (timestamps, durations) are not sequence IDs.
            for child in dictionary.values {
                if let value = sequence(in: child) { return value }
            }
        } else if let array = object as? [Any] {
            for child in array {
                if let value = sequence(in: child) { return value }
            }
        }
        return nil
    }

    static func isLastPackage(in object: Any) -> Bool? {
        if let dictionary = object as? [String: Any] {
            if let value = dictionary["is_last_package"] {
                return value as? Bool ?? (value as? NSNumber)?.boolValue
            }
            for child in dictionary.values {
                if let value = isLastPackage(in: child) { return value }
            }
        } else if let array = object as? [Any] {
            for child in array {
                if let value = isLastPackage(in: child) { return value }
            }
        }
        return nil
    }

    private static func gzip(_ data: Data) throws -> Data {
        try data.withUnsafeBytes { rawBuffer in
            guard let input = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return data }
            var stream = z_stream()
            stream.next_in = UnsafeMutablePointer<Bytef>(OpaquePointer(input))
            stream.avail_in = uInt(data.count)
            let initialized = deflateInit2_(
                &stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, MAX_WBITS + 16,
                MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)
            )
            guard initialized == Z_OK else { throw CodecError.compressionFailed }
            defer { deflateEnd(&stream) }
            var output = Data()
            var status: Int32 = Z_OK
            while status == Z_OK {
                var chunk = [UInt8](repeating: 0, count: 16_384)
                status = chunk.withUnsafeMutableBytes { buffer in
                    stream.next_out = buffer.bindMemory(to: UInt8.self).baseAddress
                    stream.avail_out = uInt(buffer.count)
                    return deflate(&stream, Z_FINISH)
                }
                guard status == Z_OK || status == Z_STREAM_END else { throw CodecError.compressionFailed }
                output.append(contentsOf: chunk.prefix(chunk.count - Int(stream.avail_out)))
            }
            return output
        }
    }

    private static func gunzip(_ data: Data) throws -> Data {
        guard !data.isEmpty else { return Data() }
        let outputLimit = min(maxDecompressedPayloadBytes, max(data.count * maxCompressionRatio, 1_048_576))
        return try data.withUnsafeBytes { rawBuffer in
            guard let input = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return data }
            var stream = z_stream()
            stream.next_in = UnsafeMutablePointer<Bytef>(OpaquePointer(input))
            stream.avail_in = uInt(data.count)
            let initialized = inflateInit2_(&stream, 16 + MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
            guard initialized == Z_OK else { throw CodecError.decompressionFailed }
            defer { inflateEnd(&stream) }
            var output = Data()
            var status: Int32 = Z_OK
            while status == Z_OK {
                var chunk = [UInt8](repeating: 0, count: 16_384)
                status = chunk.withUnsafeMutableBytes { buffer in
                    stream.next_out = buffer.bindMemory(to: UInt8.self).baseAddress
                    stream.avail_out = uInt(buffer.count)
                    return inflate(&stream, Z_SYNC_FLUSH)
                }
                guard status == Z_OK || status == Z_STREAM_END else { throw CodecError.decompressionFailed }
                let used = chunk.count - Int(stream.avail_out)
                guard used <= outputLimit - output.count else { throw CodecError.excessiveExpansion }
                output.append(contentsOf: chunk.prefix(used))
            }
            return output
        }
    }
}

private func remoteASRBigEndianData(_ value: UInt32) -> Data {
    withUnsafeBytes(of: value.bigEndian) { Data($0) }
}

private func remoteASRBigEndianData(_ value: Int32) -> Data {
    withUnsafeBytes(of: value.bigEndian) { Data($0) }
}

private func remoteASRUInt32(fromBigEndian data: Data) -> UInt32 {
    precondition(data.count == 4)
    return data.reduce(UInt32(0)) { partial, byte in
        (partial << 8) | UInt32(byte)
    }
}

private func remoteASRInt32(fromBigEndian data: Data) -> Int32 {
    Int32(bitPattern: remoteASRUInt32(fromBigEndian: data))
}
