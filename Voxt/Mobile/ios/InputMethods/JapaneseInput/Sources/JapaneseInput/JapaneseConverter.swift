import Foundation
import KanaKanjiConverterModule

/// Kana → kanji conversion for the Japanese keyboard.
///
/// ## Threading — read before integrating
/// The underlying `KanaKanjiConverter` is `@MainActor`-isolated, and every
/// conversion therefore runs on the main actor: this actor serializes calls and
/// caches the engine, but it does NOT move conversion work off the UI thread.
/// Measured on this machine (macOS arm64, dictionary fully loaded): ~7 ms per
/// warm short reading and the 128-character worst case below. Budget UI
/// accordingly; a long un-debounced reading will occupy the main actor for that
/// long.
///
/// ## Completion semantics
/// `requestCandidates` mixes partial clause candidates into `mainResults`. A
/// candidate counts as a full conversion only when applying its `composingCount`
/// via `prefixComplete` consumes the whole composition (`convertTarget.isEmpty`)
/// and `candidate.inputable` is set (verified at AzooKeyKanaKanjiConverter
/// v0.11.2, ComposingText.swift:335).
public actor JapaneseConverter {
    public enum ConverterError: Error, Equatable {
        case dictionaryMissing
        case dictionaryIncomplete(String)
        case inputTooLong
        case engineUnavailable(String)
    }

    public static let maximumReadingCount = 128

    /// Engines live on the main actor; Swift 6 forbids returning the non-Sendable
    /// converter across executor boundaries, so creation and storage stay there.
    @MainActor
    private enum EngineStore {
        private static var engines: [String: KanaKanjiConverter] = [:]

        static func converter(dictionaryURL: URL) throws -> KanaKanjiConverter {
            if let cached = engines[dictionaryURL.path] { return cached }
            let converter = try KanaKanjiConverter(dictionaryURL: dictionaryURL, preloadDictionary: false)
            engines[dictionaryURL.path] = converter
            return converter
        }
    }

    private let dictionaryURL: URL

    /// - Parameter dictionaryURL: `nil` uses the Apache-2.0 dictionary bundled with
    ///   this package, validated before any conversion runs. Validation checks two
    ///   sentinel files the loader itself flags as serious; it is a gate against
    ///   missing, empty, unreadable or misplaced data, not a contents-integrity
    ///   guarantee.
    public init(dictionaryURL: URL? = nil) throws {
        let url: URL
        if let dictionaryURL {
            url = dictionaryURL
        } else if let bundled = Bundle.module.url(forResource: "Dictionary", withExtension: nil) {
            url = bundled
        } else {
            throw ConverterError.dictionaryMissing
        }
        try Self.validateDictionary(at: url)
        self.dictionaryURL = url
    }

    /// Rejects required files that are absent, directories, unreadable, or empty.
    public static func validateDictionary(at url: URL) throws {
        for required in ["louds/charID.chid", "mm.binary"] {
            let fileURL = url.appendingPathComponent(required)
            guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .isReadableKey, .fileSizeKey]),
                  values.isRegularFile == true,
                  values.isReadable == true,
                  (values.fileSize ?? 0) > 0 else {
                throw ConverterError.dictionaryIncomplete(required)
            }
        }
    }

    /// Full-conversion candidate texts for a kana reading, best first.
    /// Cancellation is checked twice — before hopping to the engine and right
    /// before conversion — so debounced callers that cancel superseded tasks skip
    /// engine work even when cancellation arrives during the hop.
    public func candidates(for reading: String) async throws -> [String] {
        try Task.checkCancellation()
        guard !reading.isEmpty else { return [] }
        guard reading.count <= Self.maximumReadingCount else { throw ConverterError.inputTooLong }
        let dictionaryURL = self.dictionaryURL
        return try await MainActor.run { () -> [String] in
            try Task.checkCancellation()
            let converter: KanaKanjiConverter
            do {
                converter = try EngineStore.converter(dictionaryURL: dictionaryURL)
            } catch {
                throw ConverterError.engineUnavailable(error.localizedDescription)
            }
            converter.stopComposition()
            var composing = ComposingText()
            composing.insertAtCursorPosition(reading, inputStyle: .direct)
            // Learning is off; the shared temp directory only satisfies the option's
            // URL requirement and leaves no per-instance directories behind.
            let options = ConvertRequestOptions(
                N_best: 10,
                needTypoCorrection: false,
                requireJapanesePrediction: false,
                requireEnglishPrediction: false,
                keyboardLanguage: .ja_JP,
                learningType: .nothing,
                memoryDirectoryURL: FileManager.default.temporaryDirectory,
                sharedContainerURL: FileManager.default.temporaryDirectory,
                textReplacer: TextReplacer(),
                specialCandidateProviders: nil,
                metadata: .init(versionString: "Voxt JapaneseInput")
            )
            let results = converter.requestCandidates(composing, options: options)
            var full: [String] = []
            for candidate in results.mainResults where candidate.inputable {
                var copy = composing
                copy.prefixComplete(composingCount: candidate.composingCount)
                if copy.convertTarget.isEmpty {
                    full.append(candidate.text)
                }
            }
            return full
        }
    }
}
