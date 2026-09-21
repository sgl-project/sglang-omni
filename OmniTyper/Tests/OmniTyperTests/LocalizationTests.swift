// SPDX-License-Identifier: Apache-2.0
import Testing
import Foundation
import Combine
@testable import OmniTyper

struct LocalizationTests {
    private static func table(_ language: String) throws -> [String: String] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/OmniTyper/Resources/\(language).lproj/Localizable.strings")
        return try #require(NSDictionary(contentsOf: url) as? [String: String],
                            "missing or unreadable strings file: \(url.path)")
    }

    /// A key present in one language and missing from another silently ships the
    /// development string, and a format specifier that disagrees between
    /// languages corrupts or crashes `String(format:)`.
    @Test func everyLanguageDefinesTheSameKeysAndFormatSpecifiers() throws {
        let development = try Self.table(L10n.development)
        #expect(!development.isEmpty)
        func specifiers(_ value: String) -> [String] {
            let pattern = try! NSRegularExpression(pattern: "%(?:[0-9]+\\$)?[@dsf]|%[0-9]*d")
            let range = NSRange(value.startIndex..., in: value)
            return pattern.matches(in: value, range: range).map { (value as NSString).substring(with: $0.range) }
        }
        for language in L10n.supported where language != L10n.development {
            let translated = try Self.table(language)
            #expect(Set(translated.keys) == Set(development.keys),
                    """
                    \(language) is out of sync with \(L10n.development).
                    missing: \(Set(development.keys).subtracting(translated.keys).sorted())
                    extra: \(Set(translated.keys).subtracting(development.keys).sorted())
                    """)
            for (key, value) in development where translated[key] != nil {
                #expect(specifiers(value) == specifiers(translated[key]!),
                        "format specifiers differ for \(key) in \(language)")
            }
        }
        // Note (Jiaxin Deng): A backslash survives into a parsed value only when the table
        // escaped it twice, which shows the user "\\n" where a line break was meant.
        for language in L10n.supported {
            for (key, value) in try Self.table(language) where value.contains("\\") {
                Issue.record("\(language) \(key) contains a literal backslash: \(value)")
            }
        }
    }

    /// The parity test above reads the source tables. This exercises the bundle
    /// resolution L10n actually uses; if it regresses, every string silently
    /// degrades to its key and formatted strings drop their arguments.
    @Test func lookupResolvesThroughTheResolvedBundle() throws {
        // Note (Jiaxin Deng): Passes the language explicitly, because constructing an
        // AppStore rewrites the selected language and other suites do that in
        // parallel with this one.
        #expect(L10n.string("nav.Home", in: nil) == "Home")
        #expect(L10n.string("nav.Home", in: "zh-Hans") == "首页")
        #expect(String(format: L10n.string("notice.inserted", in: "zh-Hans"), "Safari").contains("Safari"))
        // Note (Jiaxin Deng): An unsupported choice has to fall back rather than show keys.
        #expect(L10n.string("nav.Home", in: "xx") == "Home")
    }

    /// The log exists to be attached to a report, so it has to stay free of
    /// anything the user would have to read it first to check, and it has to
    /// stay small enough to attach.
    @Test func theDiagnosticsLogStaysBoundedAndEscapesWhatItStores() throws {
        let url = try #require(Diagnostics.fileURL)
        #expect(url.path.hasPrefix(FileManager.default.temporaryDirectory.path),
                "a test run must never append to the real log")
        Diagnostics.record("test.start", ["note\"\n": "quote\" and \\ and\nnewline"])
        let escaped = try String(contentsOf: url, encoding: .utf8).split(separator: "\n").last
        let event = try JSONSerialization.jsonObject(with: Data(try #require(escaped).utf8)) as? [String: String]
        #expect(event?["note\"\n"] == "quote\" and \\ and\nnewline")
        for index in 0..<100 {
            Diagnostics.record("test.fill", ["index": String(index), "note": String(repeating: "界🐎", count: 600)])
        }
        Diagnostics.record("test.oversized", ["note": String(repeating: "界", count: 60_000)])
        let text = try String(contentsOf: url, encoding: .utf8)
        #expect(text.utf8.count <= 128 * 1024, "an unbounded log eventually stops being attachable")
        // Note (Jiaxin Deng): A half line would stop the file being parseable line by line.
        for line in text.split(separator: "\n") {
            #expect(line.hasPrefix("{") && line.hasSuffix("}"), "trimming must cut on a line boundary")
            #expect(try JSONSerialization.jsonObject(with: Data(line.utf8)) is [String: String])
        }
        #expect(!text.contains("\nnewline"), "a raw newline would break the one-event-per-line shape")
    }

    /// A failure names itself the same way in every language, which is what makes
    /// a log from a Chinese interface readable by someone reading English.
    @Test func aFailureKeepsOneCodeAcrossLanguages() throws {
        let failure = Failure("sys.axPermission")
        #expect(failure.code == "sys.axPermission")
        #expect(Diagnostics.code(of: failure) == "sys.axPermission")
        #expect(Diagnostics.code(of: CancellationError()) == "unknown")
        #expect(L10n.string("sys.axPermission", in: "en") != L10n.string("sys.axPermission", in: "zh-Hans"))
    }

    /// Published properties have no equality check, so a permission poll that
    /// assigned unconditionally redrew every view once a second.
    @Test @MainActor func unchangedPermissionsDoNotRepublish() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(store: AppStore(directory: directory))
        defer { model.shutdown() }
        model.refreshPermissions()
        var emissions = 0
        let subscription = model.objectWillChange.sink { _ in emissions += 1 }
        defer { subscription.cancel() }
        model.refreshPermissions()
        model.refreshPermissions()
        #expect(emissions == 0, "refreshPermissions published \(emissions) time(s) without a permission change")
    }

    /// An ad-hoc signature ties the Accessibility grant to the build, so an
    /// update drops it while System Settings still shows it enabled. Losing a
    /// grant that was held before has to be reported differently from never
    /// having been granted. A local test runner can itself hold the grant.
    @Test @MainActor func aPreviousAccessibilityGrantIsStaleOnlyWhileUntrusted() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        let model = AppModel(store: store)
        defer { model.shutdown() }
        model.refreshPermissions()
        #expect(model.accessibilityNeedsRenewal == false)
        store.preferences.accessibilityWasTrusted = true
        model.refreshPermissions()
        #expect(model.accessibilityNeedsRenewal == !model.accessibilityAllowed)
    }
}
