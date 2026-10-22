import Foundation

/// SwiftUI preference observers in a process must watch the same UserDefaults instance.
/// On iOS 26.3, separate instances of the same app-group suite update the Settings
/// label but leave Home in the old language until the app relaunches.
@MainActor
public enum VoxtPreferences {
    public static let shared = UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard
}
