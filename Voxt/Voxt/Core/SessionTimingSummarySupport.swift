// SessionTimingSummarySupport.swift
// Provides Session Timing Summary Support for core app behavior.

import Foundation

struct SessionTimingSummaryKey: Hashable, Sendable {
    let output: String
    let pipeline: String
    let asrProvider: String
    let asrModel: String
    let llmProvider: String?
}

struct SessionTimingSummaryRecord: Equatable, Sendable {
    let output: String
    let pipeline: String
    let stages: String
    let asrProvider: String
    let asrModel: String
    let llmCalls: Int
    let deliveredChars: Int
    let didInject: Bool
    let requestToStartMs: Int?
    let startToStopMs: Int?
    let startToFirstLiveASRMs: Int?
    let capturedAudioMs: Int?
    let captureGapMs: Int?
    let stopToASRMs: Int?
    let asrToFirstLLMChunkMs: Int?
    let asrToFirstLLMCompleteMs: Int?
    let asrToFinalLLMCompleteMs: Int?
    let asrToDeliveredMs: Int?
    let stopToDeliveredMs: Int?
    let overallMs: Int?
    let llmProvider: String?

    var key: SessionTimingSummaryKey {
        SessionTimingSummaryKey(
            output: output,
            pipeline: pipeline,
            asrProvider: asrProvider,
            asrModel: asrModel,
            llmProvider: llmProvider
        )
    }
}

struct SessionTimingSummaryAggregate: Equatable, Sendable {
    let sampleCount: Int
    let meanLLMCalls: Double
    let meanRequestToStartMs: Double?
    let meanStartToFirstLiveASRMs: Double?
    let p50StartToFirstLiveASRMs: Double?
    let p95StartToFirstLiveASRMs: Double?
    let meanStopToDeliveredMs: Double?
    let meanStopToASRMs: Double?
    let meanCaptureGapMs: Double?
}

struct SessionTimingSummaryReportRow: Equatable, Sendable {
    let key: SessionTimingSummaryKey
    let aggregate: SessionTimingSummaryAggregate
}

enum SessionTimingSummarySupport {
    static let prefix = "Session timing summary. "

    static func parse(line: String) -> SessionTimingSummaryRecord? {
        guard let range = line.range(of: prefix) else { return nil }
        let payload = String(line[range.upperBound...])
        let head = payload.components(separatedBy: ", firstLLM=").first ?? payload
        let parts = head.components(separatedBy: ", ")
        let firstLLMPayload = payload.components(separatedBy: ", firstLLM=").dropFirst().first

        var values: [String: String] = [:]
        for part in parts {
            guard let separator = part.firstIndex(of: "=") else { continue }
            let key = String(part[..<separator])
            let value = String(part[part.index(after: separator)...])
            values[key] = value
        }

        guard
            let output = values["output"],
            let pipeline = values["pipeline"],
            let stages = values["stages"],
            let asrProvider = values["asrProvider"],
            let asrModel = values["asrModel"],
            let llmCalls = Int(values["llmCalls"] ?? ""),
            let deliveredChars = Int(values["deliveredChars"] ?? ""),
            let didInject = parseBool(values["didInject"])
        else {
            return nil
        }

        return SessionTimingSummaryRecord(
            output: output,
            pipeline: pipeline,
            stages: stages,
            asrProvider: asrProvider,
            asrModel: asrModel,
            llmCalls: llmCalls,
            deliveredChars: deliveredChars,
            didInject: didInject,
            requestToStartMs: parseTiming(values["requestToStartMs"]),
            startToStopMs: parseTiming(values["startToStopMs"]),
            startToFirstLiveASRMs: parseTiming(values["startToFirstLiveASRMs"]),
            capturedAudioMs: parseTiming(values["capturedAudioMs"]),
            captureGapMs: parseTiming(values["captureGapMs"]),
            stopToASRMs: parseTiming(values["stopToASRMs"]),
            asrToFirstLLMChunkMs: parseTiming(values["asrToFirstLLMChunkMs"]),
            asrToFirstLLMCompleteMs: parseTiming(values["asrToFirstLLMCompleteMs"]),
            asrToFinalLLMCompleteMs: parseTiming(values["asrToFinalLLMCompleteMs"]),
            asrToDeliveredMs: parseTiming(values["asrToDeliveredMs"]),
            stopToDeliveredMs: parseTiming(values["stopToDeliveredMs"]),
            overallMs: parseTiming(values["overallMs"]),
            llmProvider: parseProvider(from: firstLLMPayload)
        )
    }

    static func aggregate(_ records: [SessionTimingSummaryRecord]) -> SessionTimingSummaryAggregate {
        let sampleCount = records.count
        let meanLLMCalls = sampleCount > 0
            ? Double(records.map(\.llmCalls).reduce(0, +)) / Double(sampleCount)
            : 0
        let firstLiveValues = records.compactMap(\.startToFirstLiveASRMs)

        return SessionTimingSummaryAggregate(
            sampleCount: sampleCount,
            meanLLMCalls: meanLLMCalls,
            meanRequestToStartMs: mean(of: records.compactMap(\.requestToStartMs)),
            meanStartToFirstLiveASRMs: mean(of: firstLiveValues),
            p50StartToFirstLiveASRMs: percentile(of: firstLiveValues, percentile: 50),
            p95StartToFirstLiveASRMs: percentile(of: firstLiveValues, percentile: 95),
            meanStopToDeliveredMs: mean(of: records.compactMap(\.stopToDeliveredMs)),
            meanStopToASRMs: mean(of: records.compactMap(\.stopToASRMs)),
            meanCaptureGapMs: mean(of: records.compactMap(\.captureGapMs))
        )
    }

    static func aggregateRecent(
        _ records: [SessionTimingSummaryRecord],
        limit: Int
    ) -> [SessionTimingSummaryKey: SessionTimingSummaryAggregate] {
        guard limit > 0 else { return [:] }

        let grouped = Dictionary(grouping: records, by: \.key)
        return grouped.mapValues { group in
            let recent = Array(group.suffix(limit))
            return aggregate(recent)
        }
    }

    static func parseAll(lines text: String) -> [SessionTimingSummaryRecord] {
        text
            .split(whereSeparator: \.isNewline)
            .compactMap { parse(line: String($0)) }
    }

    static func sortedReportRows(
        _ records: [SessionTimingSummaryRecord],
        limit: Int
    ) -> [SessionTimingSummaryReportRow] {
        aggregateRecent(records, limit: limit)
            .map { SessionTimingSummaryReportRow(key: $0.key, aggregate: $0.value) }
            .sorted { lhs, rhs in
                if lhs.key.asrProvider != rhs.key.asrProvider {
                    return lhs.key.asrProvider < rhs.key.asrProvider
                }
                if lhs.key.asrModel != rhs.key.asrModel {
                    return lhs.key.asrModel < rhs.key.asrModel
                }
                if lhs.key.pipeline != rhs.key.pipeline {
                    return lhs.key.pipeline < rhs.key.pipeline
                }
                if lhs.key.output != rhs.key.output {
                    return lhs.key.output < rhs.key.output
                }
                return (lhs.key.llmProvider ?? "") < (rhs.key.llmProvider ?? "")
            }
    }

    private static func parseBool(_ rawValue: String?) -> Bool? {
        switch rawValue?.lowercased() {
        case "true": return true
        case "false": return false
        default: return nil
        }
    }

    private static func parseTiming(_ rawValue: String?) -> Int? {
        guard let rawValue else { return nil }
        if rawValue == "n/a" {
            return nil
        }
        return Int(rawValue)
    }

    private static func mean(of values: [Int]) -> Double? {
        guard !values.isEmpty else { return nil }
        return Double(values.reduce(0, +)) / Double(values.count)
    }

    /// Linear-interpolation percentile over sorted samples (inclusive rank).
    static func percentile(of values: [Int], percentile: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let clamped = min(max(percentile, 0), 100)
        let sorted = values.sorted()
        if sorted.count == 1 { return Double(sorted[0]) }
        let rank = (Double(sorted.count - 1) * clamped) / 100.0
        let lowerIndex = Int(rank.rounded(.down))
        let upperIndex = Int(rank.rounded(.up))
        if lowerIndex == upperIndex {
            return Double(sorted[lowerIndex])
        }
        let weight = rank - Double(lowerIndex)
        return Double(sorted[lowerIndex]) * (1 - weight) + Double(sorted[upperIndex]) * weight
    }

    private static func parseProvider(from firstLLMPayload: String?) -> String? {
        guard let firstLLMPayload else { return nil }
        guard let providerRange = firstLLMPayload.range(of: "provider=") else { return nil }

        let suffix = firstLLMPayload[providerRange.upperBound...]
        if let commaIndex = suffix.firstIndex(of: ",") {
            return String(suffix[..<commaIndex])
        }
        return String(suffix)
    }
}
