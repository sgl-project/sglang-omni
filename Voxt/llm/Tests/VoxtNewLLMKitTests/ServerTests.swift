import Foundation
import XCTest
@testable import VoxtNewLLMKit

/// An engine that answers with its words, one piece per word, or waits
/// for a cancellation when hang is set.
final class FakeEngine: Engine, @unchecked Sendable {
    private let lock = NSLock()
    private var current: ModelSpec?
    private(set) var loads = 0
    var words = ["The", " meeting", " is", " at", " three."]
    var hang = false

    func load(_ model: ModelSpec, cacheLimitMB: Int?) async throws -> LoadOutcome {
        guard model.format == Format.mlx || model.format == Format.gguf else {
            throw RPCError(Code.badRequest, "Unknown model format \(model.format)")
        }
        if lock.withLock({ current == model }) { return LoadOutcome(cached: true, loadMs: 0) }
        lock.withLock {
            current = model
            loads += 1
        }
        return LoadOutcome(cached: false, loadMs: 12)
    }

    func unload() async -> Bool { lock.withLock { defer { current = nil }; return current != nil } }
    func loaded() -> ModelSpec? { lock.withLock { current } }

    func chat(_ p: ChatParams, onDelta: @escaping @Sendable (String) -> Void) async throws -> ChatOutcome {
        if hang {
            onDelta("Once upon")
            while true {
                try Task.checkCancellation()
                try await Task.sleep(nanoseconds: 1_000_000)
            }
        }
        if p.maxTokens == 0 {
            return ChatOutcome(text: "", finish: Finish.length, usage: Usage(promptTokens: 9, completionTokens: 0),
                               promptMs: 1, firstTokenMs: 1, tokensPerSecond: 0, cache: CacheUse(hit: false, prefixTokens: 7))
        }
        var c = TextCollector()
        for w in words where c.add(w) {
            onDelta(w)
        }
        return ChatOutcome(text: c.text, finish: Finish.stop, usage: Usage(promptTokens: 9, completionTokens: words.count),
                           promptMs: 1, firstTokenMs: 2, tokensPerSecond: 50, cache: CacheUse(hit: true, prefixTokens: 7))
    }

    func memory() -> Memory { Memory(footprintBytes: 100, peakFootprintBytes: 200, residentBytes: 150) }
}

/// The lines a server writes, decoded.
final class Lines: @unchecked Sendable {
    private let lock = NSLock()
    private var all: [Incoming] = []

    func add(_ data: Data) {
        let m = try! Incoming.decode(data.dropLast())
        lock.withLock { all.append(m) }
    }

    var messages: [Incoming] { lock.withLock { all } }

    /// Waits until a response to id has come.
    func wait(for id: String, timeout: Double = 5) throws -> Incoming {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let m = messages.first(where: { $0.kind == .response && $0.id == id }) { return m }
            Thread.sleep(forTimeInterval: 0.002)
        }
        throw RPCError(Code.internal, "no response to \(id)")
    }
}

final class ServerTests: XCTestCase {
    let model = #"{"id":"llm.fast","format":"mlx","path":"/m/fast"}"#

    func makeServer(_ engine: FakeEngine = FakeEngine()) -> (Server, Lines) {
        let lines = Lines()
        let server = Server(out: LineWriter { lines.add($0) }, engine: engine,
                            info: { HelloResult(protocol: 1, version: "t", pid: 1, methods: Method.requests, device: Device(name: "Mac", os: "macOS", memoryBytes: 1)) },
                            exit: { _ in })
        return (server, lines)
    }

    func send(_ s: Server, _ json: String) { s.handle(line: Data(json.utf8)) }

    func hello(_ s: Server, _ lines: Lines) throws {
        send(s, #"{"id":"h","method":"llm.hello","params":{"protocol":1,"app":"test"}}"#)
        XCTAssertNotNil(try lines.wait(for: "h").result)
    }

    func testHelloFirst() throws {
        let (s, lines) = makeServer()
        send(s, #"{"id":"c0","method":"model.status"}"#)
        XCTAssertEqual(try lines.wait(for: "c0").error?.code, Code.notInitialized)
        send(s, #"{"id":"c1","method":"llm.hello","params":{"protocol":2,"app":"test"}}"#)
        XCTAssertEqual(try lines.wait(for: "c1").error?.code, Code.protocolMismatch)
        try hello(s, lines)
        send(s, #"{"id":"c2","method":"model.reload"}"#)
        XCTAssertEqual(try lines.wait(for: "c2").error?.code, Code.methodNotFound)
        send(s, #"{"id":"c3","method":"model.status"}"#)
        let st = try lines.wait(for: "c3").decodeResult(StatusResult.self)
        XCTAssertNil(st.model)
        XCTAssertEqual(st.memory.peakFootprintBytes, 200)
    }

    func testChatStreamsAndLoadsOnce() throws {
        let engine = FakeEngine()
        let (s, lines) = makeServer(engine)
        try hello(s, lines)
        let chat = #"{"model":\#(model),"messages":[{"role":"user","content":"hi"}],"maxTokens":16,"sampling":{"temperature":0}}"#
        send(s, #"{"id":"a","method":"chat","params":\#(chat)}"#)
        let a = try lines.wait(for: "a").decodeResult(ChatResult.self)
        XCTAssertEqual(a.text, "The meeting is at three.")
        XCTAssertEqual(a.timings.loadMs, 12)
        XCTAssertEqual(a.timings.firstTokenMs, 14)
        XCTAssertEqual(a.timings.cache, CacheUse(hit: true, prefixTokens: 7))
        let deltas = lines.messages.filter { $0.method == Method.chatDelta }.map { try! $0.decodeParams(DeltaParams.self) }
        XCTAssertEqual(deltas.map(\.text).joined(), a.text)
        XCTAssertTrue(deltas.allSatisfy { $0.job == "a" })
        send(s, #"{"id":"b","method":"chat","params":\#(chat)}"#)
        XCTAssertNil(try lines.wait(for: "b").decodeResult(ChatResult.self).timings.loadMs)
        send(s, #"{"id":"c","method":"model.load","params":{"model":\#(model)}}"#)
        XCTAssertEqual(try lines.wait(for: "c").decodeResult(LoadResult.self).cached, true)
        XCTAssertEqual(engine.loads, 1)
        send(s, #"{"id":"d","method":"model.unload"}"#)
        XCTAssertTrue(try lines.wait(for: "d").decodeResult(UnloadResult.self).unloaded)
        send(s, #"{"id":"e","method":"model.load","params":{"model":{"id":"x","format":"onnx","path":"/x"}}}"#)
        XCTAssertEqual(try lines.wait(for: "e").error?.code, Code.badRequest)
        send(s, #"{"id":"f","method":"chat","params":{"model":\#(model),"messages":[],"maxTokens":16,"sampling":{"temperature":0}}}"#)
        XCTAssertEqual(try lines.wait(for: "f").error?.code, Code.badRequest)
        send(s, #"{"id":"g","method":"chat","params":{"model":\#(model),"messages":[{"role":"user","content":"hi"}],"maxTokens":-1,"sampling":{"temperature":0}}}"#)
        XCTAssertEqual(try lines.wait(for: "g").error?.code, Code.badRequest)
    }

    /// maxTokens 0 only evaluates the prompt's shared prefix: no deltas.
    func testPrefill() throws {
        let (s, lines) = makeServer()
        try hello(s, lines)
        send(s, #"{"id":"p","method":"chat","params":{"model":\#(model),"messages":[{"role":"system","content":"Clean up."},{"role":"user","content":""}],"maxTokens":0,"sampling":{"temperature":0}}}"#)
        let p = try lines.wait(for: "p").decodeResult(ChatResult.self)
        XCTAssertEqual(p.text, "")
        XCTAssertEqual(p.usage.completionTokens, 0)
        XCTAssertEqual(p.timings.cache, CacheUse(hit: false, prefixTokens: 7))
        XCTAssertFalse(lines.messages.contains { $0.method == Method.chatDelta })
    }

    /// job.cancel ends a running chat, which answers cancelled first.
    func testCancel() throws {
        let engine = FakeEngine()
        engine.hang = true
        let (s, lines) = makeServer(engine)
        try hello(s, lines)
        send(s, #"{"id":"a","method":"chat","params":{"model":\#(model),"messages":[{"role":"user","content":"hi"}],"maxTokens":16,"sampling":{"temperature":0.5}}}"#)
        let deadline = Date().addingTimeInterval(5)
        while !lines.messages.contains(where: { $0.method == Method.chatDelta }), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.002)
        }
        send(s, #"{"id":"x","method":"job.cancel","params":{"job":"a"}}"#)
        XCTAssertTrue(try lines.wait(for: "x").decodeResult(CancelResult.self).cancelled)
        let order = lines.messages.filter { $0.kind == .response }.map { $0.id! }
        XCTAssertEqual(Array(order.suffix(2)), ["a", "x"])
        XCTAssertEqual(try lines.wait(for: "a").error?.code, Code.cancelled)
        send(s, #"{"id":"y","method":"job.cancel","params":{"job":"a"}}"#)
        XCTAssertFalse(try lines.wait(for: "y").decodeResult(CancelResult.self).cancelled)
    }

    func testRepetitionGuard() {
        var c = TextCollector()
        XCTAssertTrue(c.add("Hello. "))
        var stopped = false
        for _ in 0..<20 where !stopped {
            stopped = !c.add("ha ha ")
        }
        XCTAssertTrue(stopped)
        XCTAssertTrue(c.looped)
        XCTAssertEqual(c.text, "Hello. ha ")
        var plain = TextCollector()
        XCTAssertTrue(plain.add("会议在三点开始，请准时参加。The meeting starts at three."))
        XCTAssertFalse(plain.looped)
    }

    func testUTF8Pieces() {
        var u = UTF8Pieces()
        let bytes = Array("你好a".utf8) // e4 bd a0 e5 a5 bd 61
        XCTAssertEqual(u.add(Array(bytes[0..<2])), "")
        XCTAssertEqual(u.add(Array(bytes[2..<4])), "你")
        XCTAssertEqual(u.add(Array(bytes[4..<7])), "好a")
        XCTAssertEqual(u.add([0xF0, 0x9F]), "")
        XCTAssertEqual(u.add([0x98, 0x80]), "😀")
        XCTAssertEqual(u.add([0xE4]), "")
        XCTAssertEqual(u.flush(), "\u{FFFD}")
    }
}
