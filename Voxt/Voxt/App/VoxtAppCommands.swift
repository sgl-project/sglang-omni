// VoxtAppCommands.swift
// Provides Voxt App Commands for app lifecycle and routing.

import SwiftUI
import AppKit
import Combine

struct VoiceEndCommandState {
    var lastDetectedCommand = false
    var didAutoStop = false
    var pendingStrippedText: String?
    let silenceDuration: TimeInterval = 1.0
}

struct MainWindowPresentationState {
    var shouldRestoreAfterUpdate = false
}

@MainActor
final class MainWindowVisibilityState: ObservableObject {
    @Published var isVisible = false
}

nonisolated enum SessionOutputMode {
    case transcription
    case translation
    case rewrite
}

@main
struct VoxtApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue

    var body: some Scene {
        let _ = interfaceLanguageRaw
        Settings {
            EmptyView()
        }
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button(AppLocalization.localizedString("General")) {
                    Task { @MainActor in
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general))
                    }
                }
                .keyboardShortcut(",", modifiers: .command)
            }
            MainWindowNavigationCommands(appDelegate: appDelegate)
            #if DEBUG
            DevelopmentCommands(appDelegate: appDelegate)
            #endif
            HelpNavigationCommands(appDelegate: appDelegate)
        }
    }
}

struct MainWindowNavigationCommands: Commands {
    @AppStorage(AppPreferenceKey.appEnhancementEnabled) private var appEnhancementEnabled = true
    @AppStorage(AppPreferenceKey.featureSettings) private var featureSettingsRaw = ""
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    let appDelegate: AppDelegate

    var body: some Commands {
        let _ = interfaceLanguageRaw
        CommandMenu(AppLocalization.localizedString("Navigate")) {
            Button(AppLocalization.localizedString("Dashboard")) {
                appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .report))
            }

            Menu(AppLocalization.localizedString("General")) {
                Button(AppLocalization.localizedString("General")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general))
                }
                Divider()
                Button(AppLocalization.localizedString("Floating Window Style")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general, section: .generalTranscriptionUI))
                }
                Button(AppLocalization.localizedString("Languages")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general, section: .generalLanguages))
                }
                Button(AppLocalization.localizedString("Audio")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general, section: .generalAudio))
                }
                Button(AppLocalization.localizedString("App Behavior")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general, section: .generalAppBehavior))
                }
                Button(AppLocalization.localizedString("Logging")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general, section: .generalLogging))
                }
                Button(AppLocalization.localizedString("Proxy")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .general, section: .generalProxy))
                }
            }

            Button(AppLocalization.localizedString("Model")) {
                appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .model))
            }

            Menu(AppLocalization.localizedString("Feature")) {
                let _ = featureSettingsRaw
                let availability = FeatureSettingsStore.availability()
                Button(AppLocalization.localizedString("Feature")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .features))
                }
                Divider()
                Button(AppLocalization.localizedString("Transcription")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .transcription))
                }
                if availability.translationEnabled {
                    Button(AppLocalization.localizedString("Translation")) {
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .translation))
                    }
                }
                if availability.rewriteEnabled {
                    Button(AppLocalization.localizedString("Rewrite")) {
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .rewrite))
                    }
                }
                if availability.appEnhancementEnabled {
                    Button(AppLocalization.localizedString("App Enhancement")) {
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .appEnhancement))
                    }
                }
                if availability.notesEnabled {
                    Button(AppLocalization.localizedString("Notes")) {
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .note))
                    }
                }
                if availability.meetingEnabled {
                    Button(AppLocalization.localizedString("Meeting")) {
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .feature, featureTab: .meeting))
                    }
                }
            }

            Menu(AppLocalization.localizedString("Dictionary")) {
                Button(AppLocalization.localizedString("Dictionary")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .dictionary))
                }
                Divider()
                Button(AppLocalization.localizedString("Settings")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .dictionary, section: .dictionarySettings))
                }
                Button(AppLocalization.localizedString("Dictionary Entries")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .dictionary, section: .dictionaryEntries))
                }
            }

            Menu(AppLocalization.localizedString("History")) {
                Button(AppLocalization.localizedString("History")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .history))
                }
                Divider()
                Button(AppLocalization.localizedString("History Settings")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .history, section: .historySettings))
                }
                Button(AppLocalization.localizedString("History Entries")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .history, section: .historyEntries))
                }
                Button(AppLocalization.localizedString("Notes")) {
                    appDelegate.openMainWindow(
                        target: SettingsNavigationTarget(
                            tab: .history,
                            section: .historyEntries,
                            historyFilter: .note
                        )
                    )
                }
            }

            Menu(AppLocalization.localizedString("Permissions")) {
                Button(AppLocalization.localizedString("Permissions")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .permissions))
                }
                Divider()
                Button(AppLocalization.localizedString("Permissions")) {
                    appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .permissions, section: .permissionsMain))
                }
                if appEnhancementEnabled {
                    Button(AppLocalization.localizedString("App Branch URL Authorization")) {
                        appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .permissions, section: .permissionsAppBranchURLAuthorization))
                    }
                }
            }

            Button(AppLocalization.localizedString("Hotkey")) {
                appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .hotkey))
            }
        }
    }

}

struct HelpNavigationCommands: Commands {
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    let appDelegate: AppDelegate
    private let projectURL = URL(string: "https://github.com/hehehai/voxt")!
    private let feedbackURL = URL(string: "https://github.com/hehehai/voxt/issues/new/choose")!
    private let authorURL = URL(string: "https://www.hehehai.cn/")!

    var body: some Commands {
        let _ = interfaceLanguageRaw
        CommandGroup(after: .help) {
            Divider()
            Button(AppLocalization.localizedString("Voxt")) {
                appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .about, section: .aboutVoxt))
            }
            Button(AppLocalization.localizedString("GitHub")) {
                NSWorkspace.shared.open(projectURL)
            }
            Button(AppLocalization.localizedString("Author")) {
                NSWorkspace.shared.open(authorURL)
            }
            Button(AppLocalization.localizedString("Feedback")) {
                NSWorkspace.shared.open(feedbackURL)
            }
            Button(AppLocalization.localizedString("Logs")) {
                appDelegate.openMainWindow(target: SettingsNavigationTarget(tab: .about, section: .aboutLogs))
            }
        }
    }
}

#if DEBUG
struct DevelopmentCommands: Commands {
    let appDelegate: AppDelegate

    var body: some Commands {
        CommandMenu("Developer") {
            Button("Seed 20k Dictionary + 20k History") {
                appDelegate.seedDevelopmentStorageData(dictionaryCount: 20_000, historyCount: 20_000)
            }
            Button("Seed 20k Dictionary + 100k History") {
                appDelegate.seedDevelopmentStorageData(dictionaryCount: 20_000, historyCount: 100_000)
            }
        }
    }
}
#endif
