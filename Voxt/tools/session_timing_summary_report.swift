#!/usr/bin/env swift

import Foundation

struct SessionTimingSummaryKey: Hashable {
    let output: String
    let pipeline: String
    let asrProvider: String
    let asrModel: String
    let llmProvider: String
}

struct SessionTimingSummaryRecord {
    let key: SessionTimingSummaryKey
    let llmCalls: Int
    let requestToStartMs: Int?
    let startToFirstLiveASRMs: Int?
    let stopToDeliveredMs: Int?
    let stopToASRMs: Int?
    let captureGapMs: Int?
}

struct SessionTimingSummaryAggregate {
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

enum SessionTimingSummaryParser {
    static let prefix = "Session timing summary. "

    static func parseAll(text: String) -> [SessionTimingSummaryRecord] {
        text
            .split(whereSeparator: \.isNewline)
            .compactMap { parse(line: String($0)) }
    }

    static func aggregateRecent(
        _ records: [SessionTimingSummaryRecord],
        limit: Int
    ) -> [(key: SessionTimingSummaryKey, aggregate: SessionTimingSummaryAggregate)] {
        let grouped = Dictionary(grouping: records, by: \.key)
        return grouped
            .mapValues { group in aggregate(Array(group.suffix(limit))) }
            .map { ($0.key, $0.value) }
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
                return lhs.key.llmProvider < rhs.key.llmProvider
            }
    }

    private static func parse(line: String) -> SessionTimingSummaryRecord? {
        guard let range = line.range(of: prefix) else { return nil }
        let payload = String(line[range.upperBound...])
        let head = payload.components(separatedBy: ", firstLLM=").first ?? payload
        let parts = head.components(separatedBy: ", ")

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
            let asrProvider = values["asrProvider"],
            let asrModel = values["asrModel"],
            let llmCalls = Int(values["llmCalls"] ?? "")
        else {
            return nil
        }

        let llmProvider = parseProvider(from: payload) ?? "n/a"

        return SessionTimingSummaryRecord(
            key: SessionTimingSummaryKey(
                output: output,
                pipeline: pipeline,
                asrProvider: asrProvider,
                asrModel: asrModel,
                llmProvider: llmProvider
            ),
            llmCalls: llmCalls,
            requestToStartMs: parseTiming(values["requestToStartMs"]),
            startToFirstLiveASRMs: parseTiming(values["startToFirstLiveASRMs"]),
            stopToDeliveredMs: parseTiming(values["stopToDeliveredMs"]),
            stopToASRMs: parseTiming(values["stopToASRMs"]),
            captureGapMs: parseTiming(values["captureGapMs"])
        )
    }

    private static func aggregate(_ records: [SessionTimingSummaryRecord]) -> SessionTimingSummaryAggregate {
        let sampleCount = records.count
        let meanLLMCalls = sampleCount > 0
            ? Double(records.map(\.llmCalls).reduce(0, +)) / Double(sampleCount)
            : 0
        let firstLiveValues = records.compactMap(\.startToFirstLiveASRMs)

        return SessionTimingSummaryAggregate(
            sampleCount: sampleCount,
            meanLLMCalls: meanLLMCalls,
            meanRequestToStartMs: mean(records.compactMap(\.requestToStartMs)),
            meanStartToFirstLiveASRMs: mean(firstLiveValues),
            p50StartToFirstLiveASRMs: percentile(firstLiveValues, percentile: 50),
            p95StartToFirstLiveASRMs: percentile(firstLiveValues, percentile: 95),
            meanStopToDeliveredMs: mean(records.compactMap(\.stopToDeliveredMs)),
            meanStopToASRMs: mean(records.compactMap(\.stopToASRMs)),
            meanCaptureGapMs: mean(records.compactMap(\.captureGapMs))
        )
    }

    private static func parseTiming(_ rawValue: String?) -> Int? {
        guard let rawValue, rawValue != "n/a" else { return nil }
        return Int(rawValue)
    }

    private static func parseProvider(from payload: String) -> String? {
        guard let providerRange = payload.range(of: "firstLLM=task=") else { return nil }
        let suffix = payload[providerRange.upperBound...]
        guard let targetRange = suffix.range(of: "provider=") else { return nil }
        let providerSuffix = suffix[targetRange.upperBound...]
        if let commaIndex = providerSuffix.firstIndex(of: ",") {
            return String(providerSuffix[..<commaIndex])
        }
        return String(providerSuffix)
    }

    private static func mean(_ values: [Int]) -> Double? {
        guard !values.isEmpty else { return nil }
        return Double(values.reduce(0, +)) / Double(values.count)
    }

    private static func percentile(_ values: [Int], percentile: Double) -> Double? {
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
}

func format(_ value: Double?) -> String {
    guard let value else { return "n/a" }
    return String(format: "%.1f", value)
}

func usage() {
    let script = URL(fileURLWithPath: CommandLine.arguments[0]).lastPathComponent
    FileHandle.standardError.write(
        Data("Usage: \(script) [--limit N] <log-file> [<log-file> ...]\n".utf8)
    )
}

var limit = 5
var paths: [String] = []
var index = 1
while index < CommandLine.arguments.count {
    let argument = CommandLine.arguments[index]
    if argument == "--limit" {
        index += 1
        guard index < CommandLine.arguments.count, let parsed = Int(CommandLine.arguments[index]), parsed > 0 else {
            usage()
            exit(2)
        }
        limit = parsed
    } else {
        paths.append(argument)
    }
    index += 1
}

guard !paths.isEmpty else {
    usage()
    exit(2)
}

let contents = paths.compactMap { path -> String? in
    try? String(contentsOfFile: path, encoding: .utf8)
}

let records = contents.flatMap(SessionTimingSummaryParser.parseAll(text:))
guard !records.isEmpty else {
    print("No session timing summaries found.")
    exit(1)
}

let rows = SessionTimingSummaryParser.aggregateRecent(records, limit: limit)
print(
    [
        "output",
        "pipeline",
        "asrProvider",
        "asrModel",
        "llmProvider",
        "samples",
        "meanLLMCalls",
        "meanRequestToStartMs",
        "meanStartToFirstLiveASRMs",
        "p50StartToFirstLiveASRMs",
        "p95StartToFirstLiveASRMs",
        "meanStopToDeliveredMs",
        "meanStopToASRMs",
        "meanCaptureGapMs"
    ].joined(separator: "\t")
)
for row in rows {
    let aggregate = row.aggregate
    print(
        [
            row.key.output,
            row.key.pipeline,
            row.key.asrProvider,
            row.key.asrModel,
            row.key.llmProvider,
            String(aggregate.sampleCount),
            format(aggregate.meanLLMCalls),
            format(aggregate.meanRequestToStartMs),
            format(aggregate.meanStartToFirstLiveASRMs),
            format(aggregate.p50StartToFirstLiveASRMs),
            format(aggregate.p95StartToFirstLiveASRMs),
            format(aggregate.meanStopToDeliveredMs),
            format(aggregate.meanStopToASRMs),
            format(aggregate.meanCaptureGapMs)
        ].joined(separator: "\t")
    )
}
