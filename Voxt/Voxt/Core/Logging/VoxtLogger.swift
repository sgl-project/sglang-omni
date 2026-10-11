// VoxtLogger.swift
// Provides typed category loggers backed by swift-log.

import Logging

struct VoxtLogger: Sendable {
    private let category: VoxtLogCategory
    private let logger: Logger

    nonisolated init(category: VoxtLogCategory) {
        self.category = category
        LoggingBootstrap.bootstrapIfNeeded()
        let logger = Logger(label: "Voxt.\(category.rawValue)")
        self.logger = logger
    }

    nonisolated func trace(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .trace, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func debug(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .debug, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func info(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .info, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func notice(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .notice, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func warning(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic
    ) {
        log(level: .warning, message: message, metadata: metadata, privacy: privacy)
    }

    nonisolated func error(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic
    ) {
        log(level: .error, message: message, metadata: metadata, privacy: privacy)
    }

    nonisolated func critical(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic
    ) {
        log(level: .critical, message: message, metadata: metadata, privacy: privacy)
    }

    private nonisolated func log(
        level: Logger.Level,
        message: () -> String,
        metadata: Logger.Metadata?,
        privacy: VoxtLogPrivacy,
        verbose: Bool = false
    ) {
        guard !verbose || VoxtLog.verboseEnabled else { return }
        let redactedMessage = VoxtLogRedactor.redact(message(), privacy: privacy)
        let metadata = VoxtLogRedactor.redactedMetadata(metadata) ?? [:]
        logger.log(level: level, "\(redactedMessage)", metadata: metadata)
    }
}
