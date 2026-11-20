import Foundation
@testable import VoxtNewHelper
import VoxtNewHelperKit
import XCTest

/// A hotkey.event must carry the generation of the configuration it was
/// decided under, even when a hotkey.configure lands between the decision
/// and the send. The tests drive exactly that interleaving through
/// HotkeyTap.decided, on both paths that emit events: the tap callback's
/// decision (decide) and the deferred-tap timer (tick).
final class HotkeyGenerationTests: XCTestCase {
    private let fn = RawFlags.secondaryFn

    private struct Note: Decodable {
        let method: String?
        let params: HotkeyEventParams?
    }

    /// Reads protocol lines from the helper's stdout pipe.
    private final class LineReader: @unchecked Sendable {
        private let handle: FileHandle // retained: the pipe's read end
        private let lock = NSLock()
        private var buffer = Data()
        private let sem = DispatchSemaphore(value: 0)

        init(_ handle: FileHandle) {
            self.handle = handle
            handle.readabilityHandler = { [weak self] h in
                let data = h.availableData
                guard !data.isEmpty, let self else { return }
                self.lock.withLock { self.buffer.append(data) }
                self.sem.signal()
            }
        }

        /// The next line without its newline; nil when none came in time.
        func next(timeout: Double = 2) -> Data? {
            let deadline = Date().addingTimeInterval(timeout)
            while true {
                if let line = lock.withLock({ () -> Data? in
                    guard let i = buffer.firstIndex(of: 0x0A) else { return nil }
                    defer { buffer.removeSubrange(...i) }
                    return buffer.subdata(in: buffer.startIndex..<i)
                }) { return line }
                let left = deadline.timeIntervalSinceNow
                guard left > 0, sem.wait(timeout: .now() + left) == .success else { return nil }
            }
        }
    }

    private func params(_ generation: Int, bindings: [Binding]) -> HotkeyConfigureParams {
        HotkeyConfigureParams(bindings: bindings, doubleTapMs: 60, longPressMs: 300, tapMaxMs: 1000, generation: generation)
    }

    /// A HotkeyTap writing to a pipe, without a real event tap.
    private func makeTap() -> (HotkeyTap, LineReader) {
        let pipe = Pipe()
        let out = Writer(handle: pipe.fileHandleForWriting)
        let tap = HotkeyTap(out: out, front: FrontApps(out: out))
        tap.installTap = {}
        return (tap, LineReader(pipe.fileHandleForReading))
    }

    /// A configure landing between a tap's decision and its send must not
    /// mistag the event with the newer configuration.
    func testEventKeepsTheDecisionGeneration() throws {
        let (tap, lines) = makeTap()
        _ = tap.configure(params(1, bindings: [Binding(id: "transcribe", modifiers: [.fn: .any], behavior: .tap)]))
        var decisions = 0
        tap.decided = {
            decisions += 1
            if decisions == 2 { _ = tap.configure(self.params(2, bindings: [])) }
        }
        _ = tap.decide(.flagsChanged(keyCode: 63, flags: fn, t: monotonicMs()))
        _ = tap.decide(.flagsChanged(keyCode: 63, flags: 0, t: monotonicMs()))
        let line = try XCTUnwrap(lines.next(), "no hotkey.event")
        let note = try JSONDecoder().decode(Note.self, from: line)
        XCTAssertEqual(note.method, Method.hotkeyEvent)
        let e = try XCTUnwrap(note.params)
        XCTAssertEqual(e.binding, "transcribe")
        XCTAssertEqual(e.phase, .trigger)
        XCTAssertEqual(e.generation, 1, "the event took the configuration that landed after its decision")
    }

    func testSequenceIsAuthoritativeOverGeneration() {
        let (tap, _) = makeTap()
        _ = tap.configure(HotkeyConfigureParams(sequence: 1, bindings: [], active: false, generation: 7))
        _ = tap.configure(HotkeyConfigureParams(sequence: 2, bindings: [
            Binding(id: "card.copy/1", modifiers: [:], key: 8, behavior: .tap),
        ], active: true))
        XCTAssertEqual(tap.status().bindings, 1, "a newer sequence without generation was vetoed by generation 7")
        _ = tap.configure(HotkeyConfigureParams(bindings: [], active: false, generation: 6))
        XCTAssertEqual(tap.status().bindings, 1, "an older generation-only request replaced the sequenced configuration")
        _ = tap.configure(HotkeyConfigureParams(sequence: 3, bindings: [], active: false, generation: 1))
        XCTAssertEqual(tap.status().bindings, 0, "the latest sequence did not win over the higher old generation")
    }

    /// The deferred-tap timer's event is tagged with the configuration in
    /// effect at its tick, not one landing between the tick and the send.
    func testTimerEventKeepsTheDecisionGeneration() throws {
        let (tap, lines) = makeTap()
        // The double-tap binding on the same chord defers the tap.
        _ = tap.configure(params(1, bindings: [Binding(id: "transcribe", modifiers: [.fn: .any], behavior: .tap),
                                               Binding(id: "dbl", modifiers: [.fn: .any], behavior: .doubleTap)]))
        _ = tap.decide(.flagsChanged(keyCode: 63, flags: fn, t: monotonicMs()))
        _ = tap.decide(.flagsChanged(keyCode: 63, flags: 0, t: monotonicMs()))
        // The tap fires only when the 60 ms double-tap window closes, and
        // only the timer can close it; waiting it out here puts the tick
        // past the deadline.
        XCTAssertNil(lines.next(timeout: 0.2), "the tap fired before its window")
        tap.decided = { _ = tap.configure(self.params(2, bindings: [])) }
        tap.tick()
        let line = try XCTUnwrap(lines.next(), "no hotkey.event from the timer")
        let note = try JSONDecoder().decode(Note.self, from: line)
        let e = try XCTUnwrap(note.params)
        XCTAssertEqual(e.binding, "transcribe")
        XCTAssertEqual(e.phase, .trigger)
        XCTAssertEqual(e.generation, 1, "the timer's event took the configuration that landed after its tick")
    }
}
