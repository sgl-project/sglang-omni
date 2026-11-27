import SwiftUI
import VoxtKit

/// The app's three tabs: Voice, History and Settings.
public struct RootView: View {
    @ObservedObject var model: AppModel

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        TabView {
            NavigationStack { VoiceView(model: model, session: model.session) }
                .tabItem { Label(L("Voice"), systemImage: "mic") }
            NavigationStack { HistoryView(model: model) }
                .tabItem { Label(L("History"), systemImage: "clock") }
            NavigationStack { SettingsView(model: model) }
                .tabItem { Label(L("Settings"), systemImage: "gearshape") }
        }
        .task { await model.loadCatalogues() }
    }
}

extension View {
    /// The iOS navigation bar's inline title; nothing elsewhere.
    func inlineTitle() -> some View {
        #if os(iOS)
        navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }

    /// No automatic capitals or corrections, for URLs, keys and ids.
    func plainInput() -> some View {
        #if os(iOS)
        textInputAutocapitalization(.never).autocorrectionDisabled()
        #else
        autocorrectionDisabled()
        #endif
    }
}
