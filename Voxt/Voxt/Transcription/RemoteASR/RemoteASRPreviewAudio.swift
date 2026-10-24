import Foundation

enum RemoteASRPreviewAudio {
    static func normalizeWAVHeader(at fileURL: URL) {
        guard var data = try? Data(contentsOf: fileURL), data.count >= 44 else { return }
        guard String(data: data[0..<4], encoding: .ascii) == "RIFF",
              String(data: data[8..<12], encoding: .ascii) == "WAVE" else {
            return
        }

        let fileSize = UInt32(data.count)
        let riffChunkSize = fileSize > 8 ? fileSize - 8 : 0
        let dataChunkSize = fileSize > 44 ? fileSize - 44 : 0

        writeLittleEndianUInt32(riffChunkSize, into: &data, at: 4)
        writeLittleEndianUInt32(dataChunkSize, into: &data, at: 40)
        try? data.write(to: fileURL, options: .atomic)
    }

    private static func writeLittleEndianUInt32(_ value: UInt32, into data: inout Data, at offset: Int) {
        guard data.count >= offset + 4 else { return }
        let bytes = value.littleEndian
        withUnsafeBytes(of: bytes) { raw in
            data.replaceSubrange(offset..<(offset + 4), with: raw)
        }
    }
}
