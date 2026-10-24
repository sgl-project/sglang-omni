// ModelDownloadSourceSupport.swift
// Provides automatic source selection for local model downloads.

import Foundation

struct ModelDownloadSourceCandidate: Hashable, Sendable {
    let id: String
    let displayName: String
    let url: URL
}

struct ModelDownloadSourceProbeResult: Sendable {
    let candidate: ModelDownloadSourceCandidate
    let elapsed: TimeInterval
    let bytes: Int64
    let errorDescription: String?

    var isReachable: Bool {
        errorDescription == nil
    }
}

struct ModelDownloadSourceSelection: Sendable {
    let candidate: ModelDownloadSourceCandidate
    let reusedSavedSource: Bool
    let probeResults: [ModelDownloadSourceProbeResult]

    /// Resume stays on the saved source; a fresh selection may try reachable
    /// mirrors in probe order. Both ASR and LLM managers use this same policy.
    var attemptCandidates: [ModelDownloadSourceCandidate] {
        guard !reusedSavedSource, !probeResults.isEmpty else { return [candidate] }
        let reachable = probeResults
            .filter(\.isReachable)
            .sorted(by: { $0.elapsed < $1.elapsed })
            .map(\.candidate)
        return reachable.isEmpty ? [candidate] : reachable
    }
}

enum ModelDownloadSourceSelectionStore {
    static func targetKey(namespace: String, identifier: String) -> String {
        "\(namespace):\(identifier)"
    }

    static func savedSourceID(
        for targetKey: String,
        defaults: UserDefaults = .standard
    ) -> String? {
        selections(defaults: defaults)[targetKey]
    }

    static func saveSourceID(
        _ sourceID: String,
        for targetKey: String,
        defaults: UserDefaults = .standard
    ) {
        var values = selections(defaults: defaults)
        values[targetKey] = sourceID
        saveSelections(values, defaults: defaults)
    }

    static func clearSourceID(
        for targetKey: String,
        defaults: UserDefaults = .standard
    ) {
        var values = selections(defaults: defaults)
        values.removeValue(forKey: targetKey)
        saveSelections(values, defaults: defaults)
    }

    private static func selections(defaults: UserDefaults) -> [String: String] {
        guard let data = defaults.data(forKey: AppPreferenceKey.modelDownloadSourceSelections) else {
            return [:]
        }
        return (try? JSONDecoder().decode([String: String].self, from: data)) ?? [:]
    }

    private static func saveSelections(_ values: [String: String], defaults: UserDefaults) {
        guard !values.isEmpty else {
            defaults.removeObject(forKey: AppPreferenceKey.modelDownloadSourceSelections)
            return
        }
        if let data = try? JSONEncoder().encode(values) {
            defaults.set(data, forKey: AppPreferenceKey.modelDownloadSourceSelections)
        }
    }
}

enum ModelDownloadSourceSelector {
    static func select(
        candidates: [ModelDownloadSourceCandidate],
        targetKey: String,
        reuseSavedSource: Bool,
        defaults: UserDefaults = .standard,
        probe: @escaping @Sendable (ModelDownloadSourceCandidate) async throws -> (elapsed: TimeInterval, bytes: Int64)
    ) async throws -> ModelDownloadSourceSelection {
        guard !candidates.isEmpty else {
            throw NSError(
                domain: "Voxt.ModelDownloadSourceSelector",
                code: 1000,
                userInfo: [NSLocalizedDescriptionKey: "No model download sources are configured."]
            )
        }

        if reuseSavedSource,
           let savedSourceID = ModelDownloadSourceSelectionStore.savedSourceID(for: targetKey, defaults: defaults),
           let savedCandidate = candidates.first(where: { $0.id == savedSourceID }) {
            return ModelDownloadSourceSelection(
                candidate: savedCandidate,
                reusedSavedSource: true,
                probeResults: []
            )
        }

        let results = await probeCandidates(candidates, probe: probe)
        guard let selected = results
            .filter(\.isReachable)
            .sorted(by: { $0.elapsed < $1.elapsed })
            .first?
            .candidate else {
            let details = results
                .map { "\($0.candidate.displayName): \($0.errorDescription ?? "unknown error")" }
                .joined(separator: "; ")
            throw NSError(
                domain: "Voxt.ModelDownloadSourceSelector",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "All model download sources failed. \(details)"]
            )
        }

        ModelDownloadSourceSelectionStore.saveSourceID(selected.id, for: targetKey, defaults: defaults)
        return ModelDownloadSourceSelection(
            candidate: selected,
            reusedSavedSource: false,
            probeResults: results
        )
    }

    static func logSummary(for selection: ModelDownloadSourceSelection) -> String {
        guard !selection.probeResults.isEmpty else {
            return "reusedSavedSource=true"
        }
        return selection.probeResults
            .map { result in
                if result.isReachable {
                    return "\(result.candidate.displayName)=\(Int(result.elapsed * 1000))ms"
                }
                return "\(result.candidate.displayName)=failed(\(result.errorDescription ?? "unknown"))"
            }
            .joined(separator: ", ")
    }

    private static func probeCandidates(
        _ candidates: [ModelDownloadSourceCandidate],
        probe: @escaping @Sendable (ModelDownloadSourceCandidate) async throws -> (elapsed: TimeInterval, bytes: Int64)
    ) async -> [ModelDownloadSourceProbeResult] {
        await withTaskGroup(of: ModelDownloadSourceProbeResult.self) { group in
            for candidate in candidates {
                group.addTask {
                    do {
                        let measurement = try await probe(candidate)
                        return ModelDownloadSourceProbeResult(
                            candidate: candidate,
                            elapsed: measurement.elapsed,
                            bytes: measurement.bytes,
                            errorDescription: nil
                        )
                    } catch {
                        return ModelDownloadSourceProbeResult(
                            candidate: candidate,
                            elapsed: .greatestFiniteMagnitude,
                            bytes: 0,
                            errorDescription: error.localizedDescription
                        )
                    }
                }
            }

            var results: [ModelDownloadSourceProbeResult] = []
            for await result in group {
                results.append(result)
            }
            return results
        }
    }
}
