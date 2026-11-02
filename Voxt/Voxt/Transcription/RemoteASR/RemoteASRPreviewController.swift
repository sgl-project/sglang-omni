import Foundation

/// Each polling invocation has its own deduplication state. A cancelled request
/// may finish late, but cannot publish or release a newer invocation's state.
@MainActor
final class RemoteASRPreviewController {
    private let tasks = TrackedTaskStore()
    private var generation = UUID()

    @discardableResult
    func start(
        shouldRun: @escaping @MainActor () -> Bool,
        transcribe: @escaping @MainActor () async throws -> String?,
        publish: @escaping @MainActor (String) -> Void,
        waitForNext: @escaping @MainActor () async throws -> Void = { try await Task.sleep(for: .seconds(1.4)) }
    ) -> Task<Void, Never> {
        cancel()
        let id = generation
        return tasks.start { [weak self] in
            var lastText = ""
            while !Task.isCancelled {
                do { try await waitForNext() } catch { return }
                guard !Task.isCancelled, let self, self.generation == id, shouldRun() else { return }
                let text: String?
                do { text = try await transcribe() } catch { continue }
                guard !Task.isCancelled, self.generation == id, shouldRun() else { return }
                guard let text, !text.isEmpty, text != lastText else { continue }
                lastText = text
                publish(text)
            }
        }
    }

    func cancel() {
        generation = UUID()
        tasks.cancelAll()
    }

    func waitForIdle() async { await tasks.waitForAll() }
}
