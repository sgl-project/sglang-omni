// MultipartFileBody.swift
// Builds multipart uploads on disk so recorded audio is never duplicated into request memory.

import Foundation

struct MultipartFileBody {
    let url: URL
    let byteCount: Int64

    static func create(
        sourceFileURL: URL,
        boundary: String,
        fields: [(name: String, value: String)],
        fileFieldName: String = "file",
        mimeType: String
    ) throws -> MultipartFileBody {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-multipart-\(UUID().uuidString)", isDirectory: false)
        guard FileManager.default.createFile(atPath: outputURL.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }

        do {
            let output = try FileHandle(forWritingTo: outputURL)
            defer { try? output.close() }

            for field in fields where !field.value.isEmpty {
                try output.write(contentsOf: Data("--\(boundary)\r\n".utf8))
                try output.write(contentsOf: Data("Content-Disposition: form-data; name=\"\(safeHeaderValue(field.name))\"\r\n\r\n".utf8))
                try output.write(contentsOf: Data("\(field.value)\r\n".utf8))
            }

            try output.write(contentsOf: Data("--\(boundary)\r\n".utf8))
            try output.write(
                contentsOf: Data(
                    "Content-Disposition: form-data; name=\"\(safeHeaderValue(fileFieldName))\"; filename=\"\(safeHeaderValue(sourceFileURL.lastPathComponent))\"\r\n".utf8
                )
            )
            try output.write(contentsOf: Data("Content-Type: \(mimeType)\r\n\r\n".utf8))

            let input = try FileHandle(forReadingFrom: sourceFileURL)
            defer { try? input.close() }
            while let chunk = try input.read(upToCount: 1_048_576), !chunk.isEmpty {
                try output.write(contentsOf: chunk)
            }

            try output.write(contentsOf: Data("\r\n--\(boundary)--\r\n".utf8))
            try output.synchronize()

            let size = try outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            return MultipartFileBody(url: outputURL, byteCount: Int64(size))
        } catch {
            try? FileManager.default.removeItem(at: outputURL)
            throw error
        }
    }

    func remove() {
        try? FileManager.default.removeItem(at: url)
    }

    private static func safeHeaderValue(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\r", with: "")
            .replacingOccurrences(of: "\n", with: "")
            .replacingOccurrences(of: "\"", with: "'")
    }
}
