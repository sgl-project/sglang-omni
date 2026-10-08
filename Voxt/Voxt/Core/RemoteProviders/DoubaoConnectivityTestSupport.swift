// DoubaoConnectivityTestSupport.swift
// Provides Doubao Connectivity Test Support for remote provider configuration.

import Foundation
import zlib

enum DoubaoConnectivityTestSupport {
    static func buildPacket(
        messageType: UInt8,
        messageFlags: UInt8,
        serialization: UInt8,
        compression: UInt8,
        sequence: Int32,
        payload: Data
    ) -> Data {
        var data = Data()
        data.append((0x1 << 4) | 0x1)
        data.append((messageType << 4) | messageFlags)
        data.append((serialization << 4) | compression)
        data.append(0x00)
        if messageFlags == 0x1 || messageFlags == 0x2 || messageFlags == 0x3 {
            withUnsafeBytes(of: sequence.bigEndian) { data.append(contentsOf: $0) }
        }
        var length = UInt32(payload.count).bigEndian
        data.append(Data(bytes: &length, count: 4))
        data.append(payload)
        return data
    }

    static func parseServerPacket(_ data: Data) throws -> (messageType: UInt8, hasText: Bool, isFinal: Bool, errorText: String?) {
        guard data.count >= 8 else {
            return (0, false, false, "Doubao server packet too short.")
        }

        let byte0 = data[0]
        let byte1 = data[1]
        let byte2 = data[2]
        let headerSizeWords = Int(byte0 & 0x0F)
        let headerSizeBytes = max(4, headerSizeWords * 4)
        let messageType = (byte1 >> 4) & 0x0F
        let messageFlags = byte1 & 0x0F
        let compression = byte2 & 0x0F

        var cursor = headerSizeBytes

        let hasSequence = (messageFlags & 0x1) != 0 || (messageFlags & 0x2) != 0
        var sequence: Int32?
        if hasSequence {
            guard data.count >= cursor + 4 else {
                return (messageType, false, false, "Invalid Doubao sequence header.")
            }
            let seqData = data.subdata(in: cursor..<(cursor + 4))
            let raw = seqData.reduce(UInt32(0)) { partial, byte in
                (partial << 8) | UInt32(byte)
            }
            sequence = Int32(bitPattern: raw)
            cursor += 4
        }

        guard data.count >= cursor + 4 else {
            return (messageType, false, false, "Invalid Doubao payload header.")
        }
        let payloadSizeData = data.subdata(in: cursor..<(cursor + 4))
        let payloadSize = payloadSizeData.reduce(UInt32(0)) { partial, byte in
            (partial << 8) | UInt32(byte)
        }
        cursor += 4
        guard data.count >= cursor + Int(payloadSize) else {
            return (messageType, false, false, "Invalid Doubao payload size.")
        }
        let payload = data.subdata(in: cursor..<(cursor + Int(payloadSize)))
        let decodedPayload: Data
        if compression == 0x1 {
            decodedPayload = try decodeGzipPayload(payload)
        } else {
            decodedPayload = payload
        }
        if messageType == 0xF {
            let errorText = String(data: decodedPayload, encoding: .utf8) ?? "Doubao server returned an error packet."
            return (messageType, false, false, errorText)
        }

        guard let object = try? JSONSerialization.jsonObject(with: decodedPayload) else {
            return (messageType, false, (sequence ?? 1) < 0, nil)
        }

        let text = extractText(from: object)?.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) ?? ""
        let jsonSequence = extractSequence(in: object)
        let isFinal = (jsonSequence ?? sequence ?? 1) < 0
        return (messageType, !text.isEmpty, isFinal, nil)
    }

    static func normalizedResourceID(_ model: String) -> String {
        DoubaoASRConfiguration.resolvedResourceID(model)
    }

    static func encodePacketPayload(_ payload: Data, preferGzip: Bool) -> (compression: UInt8, payload: Data) {
        guard preferGzip, !payload.isEmpty else {
            return (0x0, payload)
        }

        do {
            return (0x1, try gzipCompressPayload(payload))
        } catch {
            VoxtLog.networkWarning("Doubao test gzip compression failed. fallback to plain payload. error=\(error.localizedDescription)")
            return (0x0, payload)
        }
    }

    private static func extractText(from object: Any) -> String? {
        if let text = object as? String {
            return text
        }
        if let dict = object as? [String: Any] {
            let preferredKeys = ["text", "result_text", "utterance", "transcript", "result", "content"]
            for key in preferredKeys {
                if let value = dict[key], let text = extractText(from: value), !text.isEmpty {
                    return text
                }
            }
            for value in dict.values {
                if let text = extractText(from: value), !text.isEmpty {
                    return text
                }
            }
        }
        if let array = object as? [Any] {
            for item in array {
                if let text = extractText(from: item), !text.isEmpty {
                    return text
                }
            }
        }
        return nil
    }

    private static func extractSequence(in object: Any) -> Int32? {
        if let value = object as? Int { return Int32(value) }
        if let value = object as? Int32 { return value }
        if let value = object as? Int64 { return Int32(value) }
        if let dict = object as? [String: Any] {
            if let seq = dict["sequence"] {
                return extractSequence(in: seq)
            }
            for nested in dict.values {
                if let seq = extractSequence(in: nested) {
                    return seq
                }
            }
        }
        if let array = object as? [Any] {
            for item in array {
                if let seq = extractSequence(in: item) {
                    return seq
                }
            }
        }
        return nil
    }

    private static func gzipCompressPayload(_ data: Data) throws -> Data {
        if data.isEmpty {
            return Data()
        }

        return try data.withUnsafeBytes { rawBuffer in
            guard let input = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return data
            }

            var stream = z_stream()
            stream.zalloc = nil
            stream.zfree = nil
            stream.opaque = nil
            stream.next_in = UnsafeMutablePointer<Bytef>(mutating: input)
            stream.avail_in = uInt(data.count)

            let initStatus = deflateInit2_(
                &stream,
                Z_DEFAULT_COMPRESSION,
                Z_DEFLATED,
                MAX_WBITS + 16,
                MAX_MEM_LEVEL,
                Z_DEFAULT_STRATEGY,
                ZLIB_VERSION,
                Int32(MemoryLayout<z_stream>.size)
            )
            guard initStatus == Z_OK else {
                throw NSError(domain: "Voxt.Settings", code: -122, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize Doubao test GZIP compression."])
            }
            defer { deflateEnd(&stream) }

            var output = Data()
            var status: Int32 = Z_OK
            repeat {
                var chunk = [UInt8](repeating: 0, count: 4096)
                let statusCode = chunk.withUnsafeMutableBufferPointer { buffer -> Int32 in
                    stream.next_out = buffer.baseAddress
                    stream.avail_out = uInt(buffer.count)
                    return deflate(&stream, Z_FINISH)
                }
                status = statusCode
                let produced = chunk.count - Int(stream.avail_out)
                if produced > 0 {
                    output.append(chunk, count: produced)
                }
            } while status == Z_OK

            guard status == Z_STREAM_END else {
                throw NSError(domain: "Voxt.Settings", code: -123, userInfo: [NSLocalizedDescriptionKey: "Failed to compress Doubao test payload with GZIP."])
            }

            return output
        }
    }

    private static func decodeGzipPayload(_ data: Data) throws -> Data {
        if data.isEmpty {
            return Data()
        }

        return try data.withUnsafeBytes { rawBuffer in
            guard let input = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return data
            }

            var stream = z_stream()
            stream.zalloc = nil
            stream.zfree = nil
            stream.opaque = nil
            stream.next_in = UnsafeMutablePointer<Bytef>(mutating: input)
            stream.avail_in = uInt(data.count)

            let initStatus = inflateInit2_(
                &stream,
                MAX_WBITS + 16,
                ZLIB_VERSION,
                Int32(MemoryLayout<z_stream>.size)
            )
            guard initStatus == Z_OK else {
                throw NSError(domain: "Voxt.Settings", code: -124, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize Doubao test GZIP decompression."])
            }
            defer { inflateEnd(&stream) }

            var output = Data()
            var status: Int32 = Z_OK
            repeat {
                var chunk = [UInt8](repeating: 0, count: 4096)
                let statusCode = chunk.withUnsafeMutableBufferPointer { buffer -> Int32 in
                    stream.next_out = buffer.baseAddress
                    stream.avail_out = uInt(buffer.count)
                    return inflate(&stream, Z_NO_FLUSH)
                }
                status = statusCode
                let produced = chunk.count - Int(stream.avail_out)
                if produced > 0 {
                    output.append(chunk, count: produced)
                }
            } while status == Z_OK

            guard status == Z_STREAM_END else {
                throw NSError(domain: "Voxt.Settings", code: -125, userInfo: [NSLocalizedDescriptionKey: "Failed to decompress Doubao test GZIP payload."])
            }

            return output
        }
    }
}
