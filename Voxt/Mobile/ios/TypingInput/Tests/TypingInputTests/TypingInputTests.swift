import CryptoKit
import Foundation
import Testing
@testable import TypingInput

/// A toy layout that upper-cases letters into the mark and commits on space,
/// to check the contract itself, not a real language.
private struct ShoutEngine: InputEngine {
    private(set) var state = CompositionState()
    var conversion: ConversionRequest? { state.markedText.isEmpty ? nil : request }
    private var request: ConversionRequest?

    mutating func handle(_ key: InputKey, context: InputContext) -> InputEdit {
        switch key {
        case let .text(text):
            state.markedText += text.uppercased()
            request = ConversionRequest(input: [state.markedText])
            return .mark(state.markedText)
        case .space where !state.markedText.isEmpty:
            let text = state.markedText
            reset()
            return .insert(text + " ", mark: "")
        case .delete where !state.markedText.isEmpty:
            state.markedText.removeLast()
            state.candidates = []
            request = ConversionRequest(input: [state.markedText])
            return .mark(state.markedText)
        default:
            return .passThrough
        }
    }
    mutating func select(_ candidate: Candidate) -> InputEdit {
        reset()
        return .insert(candidate.text, mark: "")
    }
    mutating func commit() -> InputEdit {
        let text = state.markedText
        reset()
        return .insert(text, mark: "")
    }
    mutating func reset() { state = CompositionState(); request = nil }
    @discardableResult mutating func receive(_ candidates: [Candidate], for request: ConversionRequest) -> Bool {
        guard request == self.request else { return false }
        state.candidates = candidates
        return true
    }
}

private actor EchoSource: CandidateSource {
    func candidates(for request: ConversionRequest) async throws -> [Candidate] {
        request.input.map { Candidate(text: $0 + "!") }
    }
}

private func shout(_ id: String = "test-shout", languages: [String] = ["xx"]) -> LayoutDescriptor {
    LayoutDescriptor(id: id, displayName: "Shout", shortLabel: "SH", languages: languages,
                     surface: .keys(rows: [[.character("a"), .character("b")], [.space, .delete, .return]]),
                     makeEngine: { ShoutEngine() }, makeCandidateSource: { EchoSource() })
}

@Suite struct InputEngineContractTests {
    @Test func keysMarkConvertAndCommit() async throws {
        let layout = shout()
        var engine = layout.makeEngine()
        let context = InputContext(before: "")
        #expect(engine.handle(.text("a"), context: context) == .mark("A"))
        #expect(engine.handle(.text("b"), context: context) == .mark("AB"))
        let request = try #require(engine.conversion)
        let source = try #require(try layout.makeCandidateSource?())
        let candidates = try await source.candidates(for: request)
        let accepted = engine.receive(candidates, for: request)
        #expect(accepted)
        #expect(engine.state.candidates == [Candidate(text: "AB!")])
        #expect(engine.handle(.delete, context: context) == .mark("A"))
        let stale = engine.receive(candidates, for: request)
        #expect(!stale, "a stale request is ignored")
        #expect(engine.handle(.space, context: context) == .insert("A ", mark: ""))
        #expect(engine.state == CompositionState())
        #expect(engine.handle(.return, context: context) == .passThrough)
    }

    @Test func directLayoutsNeedNoEngineCode() {
        var engine = DirectInputEngine()
        #expect(engine.handle(.text("ñ"), context: InputContext(before: "")) == .passThrough)
        #expect(engine.conversion == nil)
        #expect(engine.state == CompositionState())
    }

    @Test(arguments: [
        ("", true), ("Hola", false), ("Hola. ", true), ("Hola.", false), ("¿", true), ("Hola. ¿", true),
        ("¿Qué? ", true), ("Sí, ", false), ("Hola\n", true), ("   ", true),
    ])
    func sentenceCapitalization(before: String, shifts: Bool) {
        let engine = DirectInputEngine(capitalization: .sentences(openers: ["¿", "¡"]))
        #expect(engine.capitalizesNextKey(context: InputContext(before: before)) == shifts)
        #expect(!DirectInputEngine().capitalizesNextKey(context: InputContext(before: "")))
    }

    @Test func shiftedKeysCarryTheirOwnText() {
        let key = KeySpec.character("ㄱ", shifted: "ㄲ")
        #expect(key.text(shifted: false) == "ㄱ")
        #expect(key.text(shifted: true) == "ㄲ")
        #expect(KeySpec.character("ñ").text(shifted: true) == "Ñ")
        #expect(KeySpec.character("ß").text(shifted: true) == "ß", "no two-letter uppercase on one key")
    }
}

@Suite struct LayoutRegistryTests {
    @Test func findsLayoutsByIDAndLanguage() throws {
        let registry = try LayoutRegistry([shout("a-1", languages: ["ko"]), shout("b-2", languages: ["zh-Hans", "zh-CN"])])
        #expect(registry.layout(id: "a-1")?.displayName == "Shout")
        #expect(registry.layout(id: "missing") == nil)
        #expect(registry.defaultLayouts(preferredLanguages: ["ko-KR", "zh-Hans-CN", "en-US"]).map(\.id) == ["a-1", "b-2"])
        #expect(registry.defaultLayouts(preferredLanguages: ["zh-Hant-TW"]).isEmpty, "zh-Hans must not match zh-Hant")
    }

    @Test func rejectsDuplicateIDs() {
        #expect(throws: LayoutRegistry.RegistryError.duplicateID("a-1")) {
            try LayoutRegistry([shout("a-1"), shout("a-1")])
        }
    }
}

@Suite struct VerifiedDictionaryTests {
    private func file(_ text: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("typing-\(UUID().uuidString)")
        try Data(text.utf8).write(to: url)
        return url
    }
    private func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    @Test func acceptsTheExactFile() throws {
        let url = try file("ni 你 -1\n")
        defer { try? FileManager.default.removeItem(at: url) }
        let dictionary = try VerifiedDictionary(url: url, byteCount: 10, sha256: digest("ni 你 -1\n"))
        #expect(dictionary.url == url)
    }

    @Test func rejectsMissingChangedAndDirectory() throws {
        let url = try file("ni 你 -1\n")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: VerifiedDictionary.Failure.missing) {
            try VerifiedDictionary(url: URL(fileURLWithPath: "/nonexistent-typing-data"), byteCount: 10, sha256: digest("x"))
        }
        #expect(throws: VerifiedDictionary.Failure.invalid) {
            try VerifiedDictionary(url: url, byteCount: 10, sha256: digest("ni 尼 -1\n"))
        }
        #expect(throws: VerifiedDictionary.Failure.invalid) {
            try VerifiedDictionary(url: url, byteCount: 11, sha256: digest("ni 你 -1\n"))
        }
        #expect(throws: VerifiedDictionary.Failure.invalid) {
            try VerifiedDictionary(url: FileManager.default.temporaryDirectory, byteCount: 10, sha256: digest("x"))
        }
    }
}
