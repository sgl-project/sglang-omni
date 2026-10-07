import Foundation

/// Owns one dedicated tap thread. Start/stop are serialized by the installation;
/// the condition also covers startup timeout, when the thread exists but its loop does not.
nonisolated final class HotkeyEventTapRunLoop: @unchecked Sendable {
    private let condition = NSCondition()
    private var thread: Thread?
    private var runLoop: CFRunLoop?
    private var stopRequested = false

    func addSource(_ source: CFRunLoopSource) -> Bool {
        guard let runLoop = start() else { return false }
        CFRunLoopPerformBlock(runLoop, CFRunLoopMode.commonModes.rawValue) {
            CFRunLoopAddSource(runLoop, source, .commonModes)
        }
        CFRunLoopWakeUp(runLoop)
        return true
    }

    /// The block retains the callback context until source invalidation executes
    /// on the callback thread, after any callback already in flight has returned.
    func removeSource(_ source: CFRunLoopSource, retaining context: AnyObject) {
        condition.lock()
        let loop = runLoop
        condition.unlock()
        guard let loop else { return }
        CFRunLoopPerformBlock(loop, CFRunLoopMode.commonModes.rawValue) {
            withExtendedLifetime(context) {
                CFRunLoopRemoveSource(loop, source, .commonModes)
                CFRunLoopSourceInvalidate(source)
            }
        }
        CFRunLoopWakeUp(loop)
    }

    func stop() {
        condition.lock()
        stopRequested = true
        let loop = runLoop
        let isCurrentThread = thread === Thread.current
        condition.unlock()
        if let loop {
            CFRunLoopPerformBlock(loop, CFRunLoopMode.commonModes.rawValue) { CFRunLoopStop(loop) }
            CFRunLoopWakeUp(loop)
        }
        guard !isCurrentThread else { return }
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(1)
        while thread != nil, Date() < deadline { condition.wait(until: deadline) }
    }

    private func start() -> CFRunLoop? {
        condition.lock()
        defer { condition.unlock() }
        guard !stopRequested else { return nil }
        if let runLoop { return runLoop }
        if thread == nil {
            let thread = Thread { [self] in run() }
            thread.name = "VoxtHotkeyEventTap"
            thread.qualityOfService = .userInteractive
            self.thread = thread
            thread.start()
        }
        let deadline = Date().addingTimeInterval(1)
        while runLoop == nil, !stopRequested, Date() < deadline { condition.wait(until: deadline) }
        return stopRequested ? nil : runLoop
    }

    private func run() {
        let loop = CFRunLoopGetCurrent()
        var context = CFRunLoopSourceContext()
        let keepAlive = CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &context)
        if let keepAlive { CFRunLoopAddSource(loop, keepAlive, .commonModes) }
        condition.lock()
        let shouldRun = !stopRequested
        runLoop = loop
        condition.broadcast()
        condition.unlock()
        if shouldRun { CFRunLoopRun() }
        if let keepAlive { CFRunLoopRemoveSource(loop, keepAlive, .commonModes) }
        condition.lock()
        runLoop = nil
        thread = nil
        condition.broadcast()
        condition.unlock()
    }
}
