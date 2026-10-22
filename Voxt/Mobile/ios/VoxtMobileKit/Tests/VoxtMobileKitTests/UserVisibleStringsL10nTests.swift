import Foundation
import Testing
@testable import VoxtMobileKit

/// Every string a person can see in the iPhone product is translated in all six languages:
/// the app, the keyboard, the kit and the layout packages (scanned from source), and the
/// permission prompts iOS shows from Info.plist, so no English text shows next to Chinese.
/// Brand, technical and language names are listed in `allowed`.
@Suite(.serialized)
struct UserVisibleStringsL10nTests {
    /// ios/, from this file in ios/VoxtMobileKit/Tests/VoxtMobileKitTests/.
    static let ios = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let areas = ["App": "App", "Keyboard": "Keyboard", "Kit": "VoxtMobileKit/Sources",
                        "Packages": "JapaneseInput/Sources|ZhuyinInput/Sources|KoreanInput/Sources|SpanishInput/Sources|PinyinInput/Sources"]

    /// Shown as they are in every language.
    static let allowed: Set<String> = [
        // Language names, written in their own language (Settings › Language, the layout list).
        "English", "Español", "Deutsch", "Français", "简体中文", "繁體中文", "日本語", "한국어",
        // Brand names.
        "Voxt", "Voxt JapaneseInput",
        // Technical values that look like words: Info.plist keys, script codes, file and
        // protocol names, the pinyin initials table.
        "VoxtAppGroup", "VoxtSourceRevision", "Hans", "Hant", "fmt ", "voxt-local-pairing-v1 invitation",
        "Library/Pinyin/learned-words.tsv", "b p m f d t n l g k h j q x zh ch sh r z c s y w",
    ]

    /// A literal after one of these isn't shown: identifiers, keys, logs, paths, matching.
    static let notShown = try! NSRegularExpression(pattern: #"(?:accessibilityIdentifier\(|systemName: |systemImage: |forKey: |\bid: |NSLog\(|print\(|Logger\(|fatalError\(|preconditionFailure\(|assertionFailure\(|setValue\(|forHTTPHeaderField: |URL\(string: |path: |withExtension: |forResource: |ofType: |httpMethod = |AppStorage\(|\.set\(|subsystem: |category: |Notification\.Name\(|suiteName: |named: |appendingPathComponent\(|URLQueryItem\(name: |value: |scheme == |host == |\.contains\(|\.hasPrefix\(|\.hasSuffix\(|split\(separator: |replacingOccurrences\(of: |with: |components\(separatedBy: |\bcase \w+ = |Keys? = |key = |storageKey|identifier: |bundleIdentifier|contentType: |separator: |String\(format: |kind: |role: |shortLabel: |displayName: |languages: |\.custom\(|Bundle\.module|\.init\(id: |\bmark: |diagnostics = |\bt=)\s*$"#)
    /// SwiftUI looks these literals up in the app's own bundle, which has no catalog:
    /// text has to go through VoxtL10n instead.
    static let swiftUILiteral = try! NSRegularExpression(pattern: #"\b(?:Text|Button|Label|Toggle|TextField|SecureField|Section|Picker|Link|Menu|LabeledContent|ProgressView|NavigationLink)\("((?:[^"\\]|\\.)+)"|\.(?:navigationTitle|alert|confirmationDialog|accessibilityLabel|accessibilityHint|accessibilityValue|help)\("((?:[^"\\]|\\.)+)""#)
    static let literal = try! NSRegularExpression(pattern: #"(?<!#)"((?:[^"\\]|\\.)*)"(?!#)"#)
    static let debugBlock = try! NSRegularExpression(pattern: #"#if DEBUG.*?#endif"#, options: .dotMatchesLineSeparators)

    struct Hit: Hashable, CustomStringConvertible {
        var area: String, file: String, text: String
        var description: String { "\(area) \(file): \(text)" }
    }

    static func files(_ area: String) -> [URL] {
        areas[area]!.split(separator: "|").flatMap { folder -> [URL] in
            let walker = FileManager.default.enumerator(at: ios.appendingPathComponent(String(folder)), includingPropertiesForKeys: nil)
            return (walker?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
        }
    }

    /// Source without DEBUG-only code, which never ships.
    static func shippedSource(_ file: URL) -> String {
        let text = (try? String(contentsOf: file, encoding: .utf8)) ?? ""
        return debugBlock.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
    }

    /// Words a person reads: has letters, and a space or a capitalised word; not an identifier,
    /// key, path, URL, ENV_NAME, key=value diagnostics, or a piece between interpolations.
    static func isProse(_ text: String) -> Bool {
        let plain = text.replacingOccurrences(of: #"\\\((?:[^()]|\([^()]*\))*\)"#, with: "", options: .regularExpression)
        guard plain.range(of: "[A-Za-z]", options: .regularExpression) != nil else { return false }
        for pattern in [#"^[a-z][A-Za-z0-9_.\-/()]*$"#, #"^[A-Za-z0-9_.\-]+\.[a-z]{2,4}$"#, #"^(https?|voxtmobile|file)://"#,
                        #"^[A-Z0-9_]+$"#, #"\b\w+=\S"#, #"^\s*[,)]"#]
        where plain.range(of: pattern, options: .regularExpression) != nil { return false }
        return plain.contains(" ") || plain.range(of: "^[A-Z][a-z]", options: .regularExpression) != nil
    }

    static func shownLiterals(in source: String) -> [String] {
        var found: [String] = []
        for line in source.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("//") || trimmed.hasPrefix("*") { continue }
            for match in literal.matches(in: line, range: NSRange(line.startIndex..., in: line)) {
                guard let whole = Range(match.range, in: line), let inner = Range(match.range(at: 1), in: line) else { continue }
                let before = String(line[..<whole.lowerBound])
                if let comment = before.range(of: "//"), !before[comment.lowerBound...].contains("\"") { continue }
                if notShown.firstMatch(in: before, range: NSRange(before.startIndex..., in: before)) != nil { continue }
                let text = String(line[inner])
                if isProse(text) { found.append(text.replacingOccurrences(of: #"\""#, with: "\"").replacingOccurrences(of: #"\n"#, with: "\n")) }
            }
        }
        return found
    }

    static func untranslated() -> [Hit] {
        let name = "l10n-visible.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let l10n = VoxtL10n(defaults: defaults)
        var hits = Set<Hit>()
        for area in areas.keys {
            for file in files(area) {
                for text in shownLiterals(in: shippedSource(file)) where !allowed.contains(text) {
                    let missing = VoxtL10n.shippedLanguages.contains { $0 != "en" && !l10n.hasTranslation(text, language: $0) }
                    if missing { hits.insert(Hit(area: area, file: file.lastPathComponent, text: text)) }
                }
            }
        }
        return hits.sorted { $0.description < $1.description }
    }

    @Test func everyStringPeopleSeeIsTranslatedInEveryLanguage() {
        let hits = Self.untranslated()
        for area in Self.areas.keys.sorted() {
            print("VOXT-L10N area=\(area) untranslated=\(hits.filter { $0.area == area }.count)")
        }
        #expect(hits.isEmpty, "\(hits.count) untranslated:\n\(hits.map(\.description).joined(separator: "\n"))")
    }

    @Test func theScanSeesTheStringsItShould() {
        let app = Self.files("App").flatMap { Self.shownLiterals(in: Self.shippedSource($0)) }
        let keyboard = Self.files("Keyboard").flatMap { Self.shownLiterals(in: Self.shippedSource($0)) }
        #expect(app.count > 150, "app literals: \(app.count)")
        #expect(app.contains("Keep Voxt ready"))
        #expect(keyboard.contains("Tap to speak"))
        #expect(keyboard.contains("Opening Voxt to reconnect to your Mac. Tap ◀ to come back."))
        #expect(!Self.isProse("textKey-shift") && !Self.isProse("VOXT_TEST_RESET") && !Self.isProse("https://github.com/sgl-project"))
        #expect(!Self.isProse(#"textKey-\(key.text ?? key.label)"#) && !Self.isProse(" start=ignored"), "identifiers and diagnostics")
        #expect(Self.isProse("Source code") && Self.isProse("Retry"))
    }

    /// Text(…), Button(…) and the like with a literal skip VoxtL10n and stay English.
    @Test func swiftUIViewsDontTakeLiteralText() {
        var direct: [String] = []
        for area in ["App", "Keyboard"] {
            for file in Self.files(area) {
                let source = Self.shippedSource(file)
                for match in Self.swiftUILiteral.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                    let range = match.range(at: 1).location != NSNotFound ? match.range(at: 1) : match.range(at: 2)
                    guard let text = Range(range, in: source).map({ String(source[$0]) }) else { continue }
                    if Self.allowed.contains(text) || !Self.isProse(text) { continue }
                    direct.append("\(file.lastPathComponent): \(text)")
                }
            }
        }
        #expect(direct.isEmpty, "use VoxtL10n: \(direct)")
    }

    /// The camera, microphone and local network prompts come from Info.plist; each shipped
    /// language has them in InfoPlist.strings, and English matches Info.plist.
    @Test func permissionPromptsAreTranslated() throws {
        let info = try #require(NSDictionary(contentsOf: Self.ios.appendingPathComponent("App/Info.plist")) as? [String: Any])
        let prompts = info.keys.filter { $0.hasSuffix("UsageDescription") }.sorted()
        #expect(prompts == ["NSCameraUsageDescription", "NSLocalNetworkUsageDescription", "NSMicrophoneUsageDescription"])
        for language in VoxtL10n.shippedLanguages {
            let url = Self.ios.appendingPathComponent("App/\(language).lproj/InfoPlist.strings")
            let strings = try #require(NSDictionary(contentsOf: url) as? [String: String], "\(language) has no InfoPlist.strings")
            for key in prompts {
                let value = try #require(strings[key], "\(language): \(key)")
                if language == "en" { #expect(value == info[key] as? String, "\(key) differs from Info.plist") }
                else { #expect(value != info[key] as? String, "\(language): \(key) is still English") }
            }
        }
    }
}
