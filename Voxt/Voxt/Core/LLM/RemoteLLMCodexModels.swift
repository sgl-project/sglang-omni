// RemoteLLMCodexModels.swift
// Provides Remote LLMCodex Models for LLM requests and response handling.

import Foundation

extension RemoteLLMRuntimeClient {
    private static let codexModelClientVersion = "26.506.31421"

    func codexModelOptions(configuration: RemoteProviderConfiguration) async -> [RemoteModelOption] {
        do {
            let headers = try await authorizationHeaders(provider: .codex, configuration: configuration)
            let endpointValues = codexModelEndpointValues(endpoint: configuration.endpoint)
            for endpointValue in endpointValues {
                guard let url = URL(string: endpointValue) else { continue }
                var request = URLRequest(url: url)
                request.httpMethod = "GET"
                request.timeoutInterval = 15
                request.setValue("application/json", forHTTPHeaderField: "Accept")
                for (key, value) in headers {
                    request.setValue(value, forHTTPHeaderField: key)
                }

                let (data, response) = try await networkSession.data(for: request)
                guard let httpResponse = response as? HTTPURLResponse,
                      (200..<300).contains(httpResponse.statusCode),
                      let object = try? JSONSerialization.jsonObject(with: data)
                else {
                    continue
                }
                let options = Self.codexModelOptions(from: object)
                if !options.isEmpty {
                    return options
                }
            }
        } catch {
            VoxtLog.llmWarning("Codex model catalog fetch failed, using built-in fallback: \(error.localizedDescription)")
        }
        return RemoteLLMProvider.codex.modelOptions
    }

    func codexModelEndpointValues(endpoint: String) -> [String] {
        let base = codexBackendBaseURL(endpoint: endpoint)
        return [
            "\(base)/codex/models?client_version=\(Self.codexModelClientVersion)",
            "\(base)/models",
            "\(base)/sentinel/chat-requirements"
        ]
    }

    func codexBackendBaseURL(endpoint: String) -> String {
        let defaultBase = "https://chatgpt.com/backend-api"
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let components = URLComponents(string: trimmed) else {
            return defaultBase
        }

        let path = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let normalizedPath: String
        if path.hasSuffix("codex/responses") {
            normalizedPath = String(path.dropLast("codex/responses".count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        } else if path.hasSuffix("responses") {
            normalizedPath = String(path.dropLast("responses".count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        } else {
            normalizedPath = path
        }

        var baseComponents = components
        baseComponents.query = nil
        baseComponents.fragment = nil
        baseComponents.path = normalizedPath.isEmpty ? "" : "/\(normalizedPath)"
        return baseComponents.url?.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")) ?? defaultBase
    }

    static func codexModelOptions(from object: Any) -> [RemoteModelOption] {
        var entries = [[String: Any]]()
        collectCodexModelEntries(from: object, into: &entries)

        var seen = Set<String>()
        return entries.compactMap { entry in
            guard !isImageOnlyCodexModel(entry),
                  let id = firstString(in: entry, keys: ["slug", "id"])?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !id.isEmpty,
                  seen.insert(id).inserted
            else {
                return nil
            }
            let title = firstString(in: entry, keys: ["display_name", "displayName", "title", "name"])?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return RemoteModelOption(id: id, title: title?.isEmpty == false ? title! : codexModelTitle(for: id))
        }
    }

    private static func collectCodexModelEntries(from value: Any, into entries: inout [[String: Any]]) {
        if let array = value as? [Any] {
            for item in array {
                collectCodexModelEntries(from: item, into: &entries)
            }
            return
        }

        guard let dictionary = value as? [String: Any] else { return }
        if firstString(in: dictionary, keys: ["slug", "id"]) != nil {
            entries.append(dictionary)
            return
        }

        if let chatModels = dictionary["chat_models"] as? [String: Any] {
            collectCodexModelEntries(from: chatModels["models"] ?? chatModels, into: &entries)
        }
        for key in ["models", "data", "categories"] {
            if let nested = dictionary[key] {
                collectCodexModelEntries(from: nested, into: &entries)
            }
        }
        for (_, nested) in dictionary {
            if nested is [String: Any] || nested is [Any] {
                collectCodexModelEntries(from: nested, into: &entries)
            }
        }
    }

    private static func isImageOnlyCodexModel(_ entry: [String: Any]) -> Bool {
        let outputModalities = stringArray(in: entry, keys: ["output_modalities", "outputModalities"])
        guard !outputModalities.isEmpty else { return false }
        let normalized = Set(outputModalities.map { $0.lowercased() })
        return normalized.contains("image") && !normalized.contains("text")
    }

    private static func firstString(in dictionary: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = dictionary[key] as? String {
                return value
            }
        }
        return nil
    }

    private static func stringArray(in dictionary: [String: Any], keys: [String]) -> [String] {
        for key in keys {
            if let values = dictionary[key] as? [String] {
                return values
            }
            if let values = dictionary[key] as? [Any] {
                return values.compactMap { $0 as? String }
            }
        }
        return []
    }

    private static func codexModelTitle(for id: String) -> String {
        id.split(separator: "-")
            .map { part in
                let lowercased = part.lowercased()
                if lowercased == "gpt" || lowercased == "oss" {
                    return lowercased.uppercased()
                }
                if lowercased == "codex" {
                    return "Codex"
                }
                return part.prefix(1).uppercased() + part.dropFirst()
            }
            .joined(separator: " ")
    }
}
