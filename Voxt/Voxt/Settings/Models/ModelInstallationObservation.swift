import Combine
import Foundation

struct ModelInstallationRevisions: Equatable {
    let mlx: UInt64
    let customLLM: UInt64
}

/// Observe only installation identities, not download progress or inference state.
/// Deferred run-loop delivery coalesces a burst of completed scans and
/// avoids reading @Published values during willSet. No polling or disk reads.
enum ModelInstallationObservation {
    static func changes(
        mlx: MLXModelManager,
        customLLM: CustomLLMModelManager
    ) -> AnyPublisher<Void, Never> {
        Publishers.Merge(
            mlx.$installationRevision.map { _ in () },
            customLLM.$installationRevision.map { _ in () }
        )
        .debounce(for: .milliseconds(100), scheduler: RunLoop.main)
        .eraseToAnyPublisher()
    }

    @MainActor
    static func revisions(
        mlx: MLXModelManager,
        customLLM: CustomLLMModelManager
    ) -> ModelInstallationRevisions {
        ModelInstallationRevisions(mlx: mlx.installationRevision, customLLM: customLLM.installationRevision)
    }
}
