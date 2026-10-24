// VoxtOSLogHandler.swift
// Bridges swift-log records into Apple Unified Logging.

import Foundation
import Logging
import OSLog

struct VoxtOSLogHandler: LogHandler {
    var metadata: Logging.Logger.Metadata = [:]
    var metadataProvider: Logging.Logger.MetadataProvider?
    var logLevel: Logging.Logger.Level = .info

    private let osLog: OSLog

    init(label: String, category: VoxtLogCategory) {
        let bundleID = Bundle.main.bundleIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines)
        osLog = OSLog(
            subsystem: bundleID?.isEmpty == false ? bundleID! : "app.voxt.Voxt",
            category: category.rawValue
        )
    }

    subscript(metadataKey metadataKey: String) -> Logging.Logger.Metadata.Value? {
        get { metadata[metadataKey] }
        set { metadata[metadataKey] = newValue }
    }

    func log(event: LogEvent) {
        let redactedMessage = VoxtLogRedactor.redact(event.message.description)
        let mergedMetadata = VoxtLogRedactor.redactedMetadata(mergedMetadata(event.metadata))
        let metadataSuffix = metadataText(mergedMetadata)
        let output = metadataSuffix.isEmpty ? redactedMessage : "\(redactedMessage) \(metadataSuffix)"

        os_log("%{public}@", log: osLog, type: osLogType(for: event.level), output)
    }

    private func mergedMetadata(_ explicitMetadata: Logging.Logger.Metadata?) -> Logging.Logger.Metadata? {
        var merged = metadataProvider?.get() ?? [:]
        for (key, value) in metadata where merged[key] == nil {
            merged[key] = value
        }
        if let explicitMetadata {
            for (key, value) in explicitMetadata {
                merged[key] = value
            }
        }
        return merged.isEmpty ? nil : merged
    }

    private func metadataText(_ metadata: Logging.Logger.Metadata?) -> String {
        guard let metadata, !metadata.isEmpty else { return "" }
        return metadata
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ")
    }

    private func osLogType(for level: Logging.Logger.Level) -> OSLogType {
        switch level {
        case .trace, .debug:
            return .debug
        case .info, .notice:
            return .info
        case .warning:
            return .default
        case .error:
            return .error
        case .critical:
            return .fault
        }
    }
}
