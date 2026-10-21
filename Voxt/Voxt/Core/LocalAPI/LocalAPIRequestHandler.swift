// LocalAPIRequestHandler.swift
// Provides the Local API request handler. Parsing, token checks and health answers stay off
// the main actor, so an idle or busy Voxt still answers quickly; only dictations hop to it.

import Foundation

nonisolated enum LocalAPIModelReadiness: String, Encodable, Sendable {
    case ready, missing, off, unsupported
}

nonisolated struct LocalAPIModelStatus: Encodable, Sendable {
    var transcription: LocalAPIModelReadiness
    var text: LocalAPIModelReadiness
}

nonisolated enum LocalAPIRequestHandler {
    private struct Health: Encodable {
        var ok: Bool
        var app: String
        var version: String
    }

    private struct Capabilities: Encodable {
        struct APIVersion: Encodable {
            var major: Int
            var minor: Int
        }

        struct Audio: Encodable {
            var contentTypes: [String]
            var maxBodyBytes: Int
        }

        var api: APIVersion
        var app: String
        var version: String
        var endpoints: [String]
        var audio: Audio
        var models: LocalAPIModelStatus
    }

    static func make(
        router: LocalAPIRouter,
        version: String,
        modelStatus: @escaping @MainActor @Sendable () -> LocalAPIModelStatus,
        dictate: @escaping @MainActor @Sendable (LocalAPIDictationRequest) async -> LocalAPIHTTPResponse,
        compose: @escaping @MainActor @Sendable (LocalAPIComposeRequest) async -> LocalAPIHTTPResponse = { _ in
            .error(status: 503, "No text model is available for this operation.")
        }
    ) -> LocalAPIServer.Handler {
        { request in
            switch router.route(request) {
            case .health:
                return .json(status: 200, Health(ok: true, app: "Voxt", version: version))
            case .capabilities:
                let models = await modelStatus()
                return .json(status: 200, Capabilities(
                    api: .init(major: 1, minor: 0),
                    app: "Voxt",
                    version: version,
                    endpoints: [
                        "health",
                        "capabilities",
                        "dictations",
                        "compose",
                        "compose/prompts",
                    ],
                    audio: .init(
                        contentTypes: LocalAPIRouter.audioExtensions.keys.sorted(),
                        maxBodyBytes: router.maxBodyBytes
                    ),
                    models: models
                ))
            case .composePrompts:
                return .json(status: 200, ["prompts": LocalAPIComposePrompt.builtInPrompts()])
            case let .reject(response):
                return response
            case let .dictation(dictation):
                return await dictate(dictation)
            case let .compose(composeRequest):
                return await compose(composeRequest)
            }
        }
    }
}
