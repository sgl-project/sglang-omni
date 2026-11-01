import Foundation

/// A loaded local ASR model owns a Native server; test fixtures cannot transcribe.
nonisolated enum LoadedASRModel: Sendable {
    case omni(OmniASRRuntime)
    case modelUseFixture(ASRModelUseFixture)

    var omniRuntime: OmniASRRuntime? {
        switch self {
        case .omni(let runtime): return runtime
        case .modelUseFixture: return nil
        }
    }

    func requireOmniRuntime() throws -> OmniASRRuntime {
        switch self {
        case .omni(let runtime): return runtime
        case .modelUseFixture: throw ASRModelUseFixture.InferenceError.unavailable
        }
    }
}

nonisolated struct MLXLoadedModelBox: Sendable {
    let loaded: LoadedASRModel
}

/// Identity-only model use for loader lifecycle tests, never a numerical backend.
nonisolated final class ASRModelUseFixture: Sendable {
    enum InferenceError: LocalizedError {
        case unavailable

        var errorDescription: String? {
            "A model-use fixture cannot run ASR inference."
        }
    }
}

/// Which checkpoints run on the local Omni server, fixed per process.
///
/// Only the listed checkpoints are eligible. Other models remain in the
/// catalog but cannot load while Native support is pending.
nonisolated enum OmniASRBackend {
    static let modelKindsByRepo: [String: OmniASRModelKind] = [
        "mlx-community/Qwen3-ASR-0.6B-4bit": .qwen3ASR,
        "mlx-community/Qwen3-ASR-1.7B-6bit": .qwen3ASR,
        "mlx-community/Qwen3-ASR-1.7B-8bit": .qwen3ASR,
        "mlx-community/whisper-large-v3-turbo": .whisper,
        "mlx-community/whisper-large-v3-mlx": .whisper,
        "mlx-community/whisper-small-mlx": .whisper,
        "beshkenadze/cohere-transcribe-03-2026-mlx-fp16": .cohereTranscribe,
        "OpenMOSS-Team/MOSS-Transcribe-Diarize": .mossTranscribeDiarize,
        "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit": .nemotronASR,
        "mlx-community/parakeet-tdt-0.6b-v3": .parakeet,
        "mlx-community/SenseVoiceSmall": .senseVoice,
    ]

    static var isAppleSilicon: Bool {
#if arch(arm64)
        true
#else
        false
#endif
    }

    enum Unavailability: Error, LocalizedError, Equatable, Sendable {
        case unsupportedModel
        case unsupportedPlatform
        case runtimeUnavailable

        var errorDescription: String? {
            switch self {
            case .unsupportedModel:
                return AppLocalization.localizedString("This model is temporarily unavailable while Native Omni support is being added. Choose another ASR model.")
            case .unsupportedPlatform:
                return AppLocalization.localizedString("Local Native Omni models require Apple Silicon. Choose a Remote ASR provider or Direct Dictation.")
            case .runtimeUnavailable:
                return AppLocalization.localizedString("The Native Omni runtime is unavailable. Reinstall Voxt or configure the development runtime.")
            }
        }
    }

    static func nativeModelKind(for repo: String) -> OmniASRModelKind? {
        modelKindsByRepo[MLXModelCatalog.canonicalModelRepo(repo)]
    }

    static func unavailability(
        for repo: String,
        isAppleSilicon: Bool = OmniASRBackend.isAppleSilicon
    ) -> Unavailability? {
        if !isAppleSilicon {
            return .unsupportedPlatform
        } else if nativeModelKind(for: repo) == nil {
            return .unsupportedModel
        } else {
            return nil
        }
    }

    static func unavailabilityReason(for repo: String) -> String? {
        unavailability(for: repo)?.errorDescription
    }

    static let launchSettings: LaunchSettings? = LaunchSettings(
        environment: ProcessInfo.processInfo.environment,
        bundle: .main
    )

    struct LaunchSettings: Sendable, Equatable {
        let runtimeExecutable: URL

        /// Resolve an explicit development path first, then the packaged runtime.
        init?(environment: [String: String], bundle: Bundle) {
#if !arch(arm64)
            return nil
#else
            if environment["VOXT_ASR_BACKEND"] == "swift" {
                return nil
            }

            if let runtime = environment["VOXT_OMNI_RUNTIME"], !runtime.isEmpty {
                runtimeExecutable = URL(fileURLWithPath: runtime)
            } else if let runtime = bundle.url(
                forResource: "qwen3_asr_server",
                withExtension: nil,
                subdirectory: "OmniRuntime/bin"
            ) {
                runtimeExecutable = runtime
            } else {
                return nil
            }
#endif
        }

        init?(environment: [String: String]) {
            self.init(environment: environment, bundle: .main)
        }
    }

    static func modelKind(for repo: String) -> OmniASRModelKind? {
        guard launchSettings != nil else { return nil }
        return nativeModelKind(for: repo)
    }

    /// Returns whether the checkpoint is assigned to the Omni runtime.
    ///
    /// This is a model-routing property and does not depend on the current
    /// process having a packaged runtime available.
    static func usesOmniRuntime(for repo: String) -> Bool {
#if arch(arm64)
        nativeModelKind(for: repo) != nil
#else
        false
#endif
    }

    /// Qwen3-ASR, Silero VAD and Sortformer run `VOXT_OMNI_RUNTIME`; every other
    /// kind runs the `<kind>_server` installed beside it.
    static func runtimeExecutable(for kind: OmniASRModelKind, qwenRuntime: URL) -> URL {
        switch kind {
        case .qwen3ASR, .sileroVAD, .sortformer:
            return qwenRuntime
        case .whisper, .cohereTranscribe, .mossTranscribeDiarize, .nemotronASR, .parakeet, .senseVoice:
            return qwenRuntime.deletingLastPathComponent().appendingPathComponent("\(kind.rawValue)_server")
        }
    }

    static func configuration(for kind: OmniASRModelKind) -> OmniBackendConfiguration? {
        guard let launchSettings else { return nil }
        return OmniBackendConfiguration(
            runtimeExecutable: runtimeExecutable(for: kind, qwenRuntime: launchSettings.runtimeExecutable)
        )
    }
}

/// Every Omni runtime a load created, so none outlives its use.
///
/// A load retires its own runtime when it fails or is cancelled. Invalidating
/// pending loads retires the runtimes they created once those loads finish, so
/// a late success is never kept. Dropping or replacing the loaded model
/// retires it, and termination retires everything. A new server starts only
/// after earlier ones have stopped, so two copies of the weights are never
/// resident.
@MainActor
final class OmniRuntimeLedger {
    private var tracked: [ObjectIdentifier: OmniASRRuntime] = [:]
    private var adopted: ObjectIdentifier?
    private var retirements: [ObjectIdentifier: Task<Void, Never>] = [:]

    func track(_ runtime: OmniASRRuntime) {
        tracked[ObjectIdentifier(runtime)] = runtime
    }

    func adopt(_ runtime: OmniASRRuntime?) {
        adopted = runtime.map(ObjectIdentifier.init)
    }

    /// Runtimes created by loads that have not been adopted yet.
    func pendingRuntimes() -> [OmniASRRuntime] {
        tracked.filter { $0.key != adopted }.map(\.value)
    }

    func release(_ runtime: OmniASRRuntime) {
        let id = ObjectIdentifier(runtime)
        if adopted == id { adopted = nil }
        tracked[id] = nil
        guard retirements[id] == nil else { return }
        retirements[id] = Task { @MainActor [weak self] in
            await runtime.retire()
            self?.retirements[id] = nil
        }
    }

    /// Runtimes a finished load created that no load will adopt.
    func releaseUnadopted() {
        pendingRuntimes().forEach(release)
    }

    func waitForRetirements() async {
        while let pending = retirements.values.first {
            await pending.value
        }
    }

    /// Termination: retire everything, adopted or not, and wait.
    func retireAll() async {
        adopted = nil
        for runtime in Array(tracked.values) { release(runtime) }
        await waitForRetirements()
    }
}
