import SwiftUI
import VoxtMobileKit

@main
struct VoxtMobileApp: App {
    @State private var model = AppModel()
    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""
    @AppStorage("appearance", store: UserDefaults(suiteName: DictationChannel.appGroup)) private var appearance = "system"
    @Environment(\.scenePhase) private var scenePhase

    private var colorScheme: ColorScheme? {
        var choice = appearance
        #if DEBUG
        // Design-review screenshots pick light or dark without changing the saved setting.
        if let forced = ProcessInfo.processInfo.environment["VOXT_TEST_APPEARANCE"] { choice = forced }
        #endif
        return choice == "dark" ? .dark : choice == "light" ? .light : nil
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .environment(\.locale, Locale(identifier: interfaceLanguage.isEmpty ? VoxtL10n.resolve(preferredLanguages: Locale.preferredLanguages) : interfaceLanguage))
                .preferredColorScheme(colorScheme)
                .onOpenURL { model.handle(url: $0) }
                .task { await model.checkConnection() }
                .onChange(of: scenePhase, initial: true) { _, phase in
                    if phase == .active { model.sceneBecameActive() } else { model.sceneLeftActive() }
                }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active {
                        model.refreshStoredState()
                        if case .failed = model.connection { Task { await model.checkConnection() } }
                    }
                }
        }
    }
}

struct RootView: View {
    @Environment(AppModel.self) private var model
    @State private var tab = AppTab.home

    private enum AppTab {
        case home, history
    }

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        #if DEBUG
        if ProcessInfo.processInfo.environment["VOXT_TEST_STATUS_GALLERY"] == "1" {
            StatusCardGallery()
        } else {
            linkedSheets
        }
        #else
        linkedSheets
        #endif
    }

    /// Screens the keyboard opens through voxtmobile:// links. Settings opens from Home's own
    /// sheet: a second sheet here hangs the app's accessibility queries once the keyboard
    /// handoff cover shows (DesignTourUITests).
    private var linkedSheets: some View {
        content
            .sheet(isPresented: translationSheet) { TranslationSheet() }
            .onChange(of: model.showsSettings) { _, show in if show { tab = .home } }
    }

    /// voxtmobile://translation, from the keyboard's "Choose translation languages" pill.
    private var translationSheet: Binding<Bool> {
        Binding(get: { model.showsTranslation }, set: { model.showsTranslation = $0 })
    }

    @ViewBuilder private var content: some View {
        if let step = model.onboardingScreen {
            OnboardingView(step: step)
        } else {
            TabView(selection: $tab) {
                DictationView(onSeeAll: { tab = .history })
                    .tabItem { Label(VoxtL10n.shared.string("Home"), systemImage: "house") }
                    .tag(AppTab.home)
                HistoryView(onEditLatest: {
                    tab = .home
                    Task { await model.editLatestResult() }
                })
                    .tabItem { Label(VoxtL10n.shared.string("History"), systemImage: "clock") }
                    .tag(AppTab.history)
            }
            .fullScreenCover(isPresented: Binding(get: { model.showsKeyboardHandoff }, set: { model.showsKeyboardHandoff = $0 })) {
                KeyboardHandoffView().environment(model)
            }
            .tint(VoxtTheme.Colors.ink)
            .background(alignment: .bottomLeading) {
                if let window = model.floatingWindow {
                    PiPSourceHost(layer: window.displayLayer).frame(width: 160, height: 90).allowsHitTesting(false)
                }
            }
            // Transcripts live on the History tab: a dictation started in the app shows there.
            .onChange(of: model.appResultCount) { _, _ in tab = .history }
        }
    }
}
