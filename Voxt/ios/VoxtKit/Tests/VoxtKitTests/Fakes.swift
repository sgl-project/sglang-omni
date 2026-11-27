import Foundation
@testable import VoxtKit

final class FakeRecorder: AudioRecorder, @unchecked Sendable {
    var seconds: Double = 2
    var startError: Error?
    private(set) var started = 0
    private(set) var cancelled = 0

    /// When set, start waits in the middle until release().
    var holds = false
    private let lock = NSLock()
    private var waiting: CheckedContinuation<Void, Never>?

    func start(onLevel: @escaping @Sendable (Float) -> Void) async throws {
        if let startError {
            throw startError
        }
        started += 1
        if holds {
            await withCheckedContinuation { continuation in
                lock.withLock { waiting = continuation }
            }
        }
        onLevel(0.5)
    }

    var isHolding: Bool { lock.withLock { waiting != nil } }

    func release() {
        lock.withLock {
            waiting?.resume()
            waiting = nil
        }
    }

    func stop() -> Recording {
        Recording(samples: [Float](repeating: 0.1, count: Int(seconds * Double(Recording.sampleRate))))
    }

    func cancel() { cancelled += 1 }
}

final class FakeSpeech: SpeechEngine, @unchecked Sendable {
    var text: String
    var error: Error?
    private(set) var inputs: [SpeechInput] = []

    init(_ text: String, error: Error? = nil) {
        self.text = text
        self.error = error
    }

    func transcribe(_ input: SpeechInput) async throws -> String {
        inputs.append(input)
        if let error {
            throw error
        }
        return text
    }
}

final class FakeCore: CoreService, @unchecked Sendable {
    var answer = ProcessResult(text: "done", latencyMillis: 1)
    var error: Error?
    /// When set, process waits until the task is cancelled.
    var hang = false
    private let lock = NSLock()
    private var received: [ProcessRequest] = []
    /// What process got; the session calls it off the main actor.
    var requests: [ProcessRequest] { lock.withLock { received } }

    func process(_ request: ProcessRequest) async throws -> ProcessResult {
        lock.withLock { received.append(request) }
        if hang {
            while !Task.isCancelled {
                await Task.yield()
            }
            throw CoreError(code: CoreError.cancelled, message: "cancelled")
        }
        if let error {
            throw error
        }
        return answer
    }

    func transcribe(_ request: SpeechRequest) async throws -> SpeechResult {
        SpeechResult(text: "remote", latencyMillis: 1)
    }

    func testProvider(_ provider: CoreProvider) async throws -> TestResult { TestResult(latencyMillis: 1) }
    func presets() async throws -> [Preset] { [] }
    func languages() async throws -> [Language] { [] }
}

func scratchDirectory() -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("voxtkit-tests-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

struct Harness {
    let recorder = FakeRecorder()
    let speech: FakeSpeech
    let core = FakeCore()
    let keys = MemoryKeyStore([KeyAccount.languageModel("openai"): "sk-test"])
    let history = HistoryStore(directory: scratchDirectory())
    var allowed = true

    init(speech: String = "hello there") {
        self.speech = FakeSpeech(speech)
    }

    @MainActor
    func session(_ feature: Feature, settings: AppSettings = AppSettings()) -> VoiceSession {
        let speech = self.speech
        let allowed = self.allowed
        let env = SessionEnvironment(core: core, recorder: recorder, keys: keys, history: history,
                                     speech: { _, _ in speech }, microphoneAllowed: { allowed })
        return VoiceSession(environment: env, feature: feature, settings: { settings })
    }
}

extension VoiceSession {
    /// Waits for the session to leave its busy states.
    func settled() async {
        for _ in 0..<10_000 where isBusy {
            await Task.yield()
        }
    }
}

/// Keys in memory.
final class MemoryKeyStore: KeyStore, @unchecked Sendable {
    private var keys: [String: String]
    private let lock = NSLock()

    init(_ keys: [String: String] = [:]) {
        self.keys = keys
    }

    func key(_ account: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return keys[account]
    }

    func setKey(_ key: String?, for account: String) throws {
        lock.lock()
        defer { lock.unlock() }
        keys[account] = key?.isEmpty == false ? key : nil
    }
}
