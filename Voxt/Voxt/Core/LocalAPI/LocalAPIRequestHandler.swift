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

nonisolated enum LocalAPIModelLocation: String, Encodable, Equatable, Sendable {
    case local, remote, unknown
}

/// Where the selected models run, for a client that must not send audio or text off the Mac.
nonisolated struct LocalAPIModelLocations: Encodable, Equatable, Sendable {
    var transcription: LocalAPIModelLocation
    var text: LocalAPIModelLocation

    static let unknown = LocalAPIModelLocations(transcription: .unknown, text: .unknown)

    /// Anything not known to stay on the Mac is unknown, never local.
    static func resolve(transcriptionEngine: String, enhancementMode: String,
                        appleSpeechOnDevice: Bool = false) -> Self {
        let transcription: LocalAPIModelLocation
        switch transcriptionEngine {
        case "mlxAudio": transcription = .local
        case "remote": transcription = .remote
        // SpeechTranscriber.startSpeechRecognition sets requiresOnDeviceRecognition from this
        // setting; false permits network use.
        case "dictation": transcription = appleSpeechOnDevice ? .local : .unknown
        default: transcription = .unknown
        }
        let text: LocalAPIModelLocation
        switch enhancementMode {
        // With no text model nothing is sent anywhere.
        case "customLLM", "off": text = .local
        // TextEnhancer uses FoundationModels SystemLanguageModel.default, which Apple describes
        // as on-device: https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel
        case "appleIntelligence": text = .local
        case "remoteLLM": text = .remote
        default: text = .unknown
        }
        return Self(transcription: transcription, text: text)
    }
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
        var locations: LocalAPIModelLocations
    }

    static func make(
        router: LocalAPIRouter,
        version: String,
        modelStatus: @escaping @MainActor @Sendable () -> LocalAPIModelStatus,
        modelLocations: @escaping @MainActor @Sendable () -> LocalAPIModelLocations = { .unknown },
        sync: @escaping @Sendable (HistorySyncRequest) -> LocalAPIHTTPResponse = { _ in
            .error(status: 403, "History sync is turned off.")
        },
        dictate: @escaping @MainActor @Sendable (LocalAPIDictationRequest) async -> LocalAPIHTTPResponse,
        compose: @escaping @MainActor @Sendable (LocalAPIComposeRequest) async -> LocalAPIHTTPResponse = { _ in
            .error(status: 503, "No text model is available for this operation.")
        },
        correct: @escaping @MainActor @Sendable (LocalAPICorrectionRequest) async -> LocalAPIHTTPResponse = { _ in
            .error(status: 503, "No text model is available for this operation.")
        }
    ) -> LocalAPIServer.Handler {
        { request in
            let route = router.route(request)
            // "X-Voxt-Model-Policy: local-only": the client does not allow its audio or text to
            // reach a cloud model, so the request is refused before anything runs.
            if request.header("x-voxt-model-policy") == "local-only" {
                switch route {
                case .dictation, .compose, .correction:
                    let locations = await modelLocations()
                    guard locations.transcription == .local, locations.text == .local else {
                        let cloud = locations.transcription == .remote || locations.text == .remote
                        return .error(status: 403, cloud ? "Your Mac is set to a cloud model." : "Could not verify that your Mac uses local models.")
                    }
                default: break
                }
            }
            switch route {
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
                        "history/sync",
                        "compose",
                        "compose/prompts",
                        "corrections",
                    ],
                    audio: .init(
                        contentTypes: LocalAPIRouter.audioExtensions.keys.sorted(),
                        maxBodyBytes: router.maxBodyBytes
                    ),
                    models: models,
                    locations: await modelLocations()
                ))
            case let .historySync(request):
                return sync(request)
            case .composePrompts:
                return .json(status: 200, ["prompts": LocalAPIComposePrompt.builtInPrompts()])
            case let .reject(response):
                return response
            case let .dictation(dictation):
                return await dictate(dictation)
            case let .compose(composeRequest):
                return await compose(composeRequest)
            case let .correction(correction):
                return await correct(correction)
            }
        }
    }
}
