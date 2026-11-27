import SwiftUI
import VoxtUI

/// Voxt on iOS: the screens are VoxtUI's, the logic VoxtKit's and the
/// shared core Go's (ios/README.md).
@main
struct VoxtApp: App {
    @StateObject private var model = AppModel.live()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            RootView(model: model)
        }
        .onChange(of: phase) { _, phase in
            if phase != .active {
                model.saveNow() // settings typed just before leaving
            }
        }
    }
}
