import Combine
import Foundation

/// Best-effort prefill, never a prerequisite for dictation and never an insertion source.
@MainActor
public final class PolishWarmup: ObservableObject {
    public struct Request: Equatable {
        public let model: LocalModelConfiguration
        public let personalBackground: String
    }
    @Published public private(set) var status = "整理关闭，未预热"
    private let perform: (Request) async throws -> Void
    private var task: Task<Void, Never>?
    private var generation = 0
    private var desired: Request?
    private var attempted: Request?
    private var interrupted = false

    private init(performRequest: @escaping (Request) async throws -> Void) { perform = performRequest }
    public static func configured(_ perform: @escaping (Request) async throws -> Void) -> PolishWarmup {
        PolishWarmup(performRequest: perform)
    }
    public convenience init(perform: @escaping (String) async throws -> Void) {
        self.init(performRequest: { try await perform($0.personalBackground) })
    }

    public func update(enabled: Bool, personalBackground: String, busy: Bool, model: LocalModelConfiguration = .ollama) {
        let next = enabled ? Request(model: model, personalBackground: personalBackground) : nil
        if next != desired {
            invalidate()
            desired = next
            attempted = nil
            interrupted = false
        }
        guard let profile = desired else {
            status = "整理关闭，未预热"
            return
        }
        if busy {
            if task != nil {
                invalidate()
                interrupted = true
                status = "预热已让位于听写"
            } else if attempted != profile { status = "等待本轮结束后预热" }
            return
        }
        if interrupted {
            attempted = nil
            interrupted = false
        }
        guard attempted != profile else { return }
        attempted = profile
        status = "正在后台预热…"
        let id = generation
        task = Task { [weak self] in
            guard let self else { return }
            do {
                try await perform(profile)
                guard generation == id, !Task.isCancelled else { return }
                status = "预热已完成，缓存由 Ollama 管理"
            } catch {
                guard generation == id, !Task.isCancelled else { return }
                status = "预热未完成；不影响录音，可检查本地服务后重试"
            }
            task = nil
        }
    }

    /// Entering foreground polishing takes over prefix processing. If the session
    /// ends before reaching this stage, the interrupted warmup can retry on idle.
    public func foregroundWillPolish() { interrupted = false }

    public func retry() {
        invalidate()
        attempted = nil
        interrupted = false
    }

    public func stop() {
        invalidate()
        desired = nil
        attempted = nil
        interrupted = false
        status = "整理关闭，未预热"
    }

    private func invalidate() {
        generation += 1
        task?.cancel()
        task = nil
    }
}
