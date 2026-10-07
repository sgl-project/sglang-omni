// VoxtLogFormatter.swift
// Formats structured log records for file export and display.

import Foundation
import Logging

enum VoxtLogFormatter {
    private nonisolated(unsafe) static let timestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    nonisolated static func timestamp(for date: Date = Date()) -> String {
        timestampFormatter.string(from: date)
    }

    nonisolated static func line(
        timestamp: Date = Date(),
        level: Logger.Level,
        category: VoxtLogCategory,
        message: String,
        metadata: Logger.Metadata?,
        source: String,
        file: String,
        function: String,
        line: UInt
    ) -> String {
        var parts = [
            "[Voxt]",
            Self.timestamp(for: timestamp),
            "[\(levelLabel(level))]",
            "[\(category.rawValue)]",
            message
        ]

        if let metadataText = metadataText(metadata), !metadataText.isEmpty {
            parts.append(metadataText)
        }

        if level >= .error {
            parts.append("source=\(source)")
            parts.append("file=\((file as NSString).lastPathComponent):\(line)")
            parts.append("function=\(function)")
        }

        return parts.joined(separator: " ")
    }

    private nonisolated static func levelLabel(_ level: Logger.Level) -> String {
        switch level {
        case .trace: return "TRACE"
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .notice: return "NOTICE"
        case .warning: return "WARN"
        case .error: return "ERROR"
        case .critical: return "CRITICAL"
        }
    }

    private nonisolated static func metadataText(_ metadata: Logger.Metadata?) -> String? {
        guard let metadata, !metadata.isEmpty else { return nil }
        return metadata
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\(metadataValueText($0.value))" }
            .joined(separator: " ")
    }

    private nonisolated static func metadataValueText(_ value: Logger.Metadata.Value) -> String {
        switch value {
        case .string(let string):
            return string
        case .stringConvertible(let value):
            return String(describing: value)
        case .array(let values):
            return "[\(values.map(metadataValueText).joined(separator: ","))]"
        case .dictionary(let dictionary):
            return "{\(dictionary.sorted { $0.key < $1.key }.map { "\($0.key):\(metadataValueText($0.value))" }.joined(separator: ","))}"
        }
    }
}
