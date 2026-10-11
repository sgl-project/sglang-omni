import Foundation
import MLXAudioSTT

/// A loaded local ASR model: an in-process MLXAudio model or an owned Omni server.
nonisolated enum LoadedASRModel: @unchecked Sendable {
    case mlx(any STTGenerationModel)
    case omni(OmniASRRuntime)

    var omniRuntime: OmniASRRuntime? {
        if case .omni(let runtime) = self { return runtime }
        return nil
    }

    var mlxModel: (any STTGenerationModel)? {
        if case .mlx(let model) = self { return model }
        return nil
    }
}

/// Which checkpoints run on the local Omni server, fixed per process.
///
/// Only the listed checkpoints are eligible, and only when the development
/// backend is configured; every other model keeps its Swift backend. The choice
/// is read once so a running process never switches the backend of a checkpoint.
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
        "mlx-community/parakeet-tdt-0.6b-v3": .parakeet,
    ]

    static let launchSettings: LaunchSettings? = LaunchSettings(environment: ProcessInfo.processInfo.environment)

    struct LaunchSettings: Sendable, Equatable {
        let runtimeExecutable: URL

        /// `VOXT_ASR_BACKEND=omni` with `VOXT_OMNI_RUNTIME`, the path to the
        /// native `qwen3_asr_server` binary; the other kinds' servers sit beside it.
        init?(environment: [String: String]) {
            guard environment["VOXT_ASR_BACKEND"] == "omni",
                  let runtime = environment["VOXT_OMNI_RUNTIME"], !runtime.isEmpty
            else { return nil }
            runtimeExecutable = URL(fileURLWithPath: runtime)
        }
    }

    static func modelKind(for repo: String) -> OmniASRModelKind? {
        guard launchSettings != nil else { return nil }
        return modelKindsByRepo[repo]
    }

    /// Qwen3-ASR, Silero VAD and Sortformer run `VOXT_OMNI_RUNTIME`; every other
    /// kind runs the `<kind>_server` installed beside it.
    static func runtimeExecutable(for kind: OmniASRModelKind, qwenRuntime: URL) -> URL {
        switch kind {
        case .qwen3ASR, .sileroVAD, .sortformer:
            return qwenRuntime
        case .whisper, .cohereTranscribe, .mossTranscribeDiarize, .parakeet:
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
