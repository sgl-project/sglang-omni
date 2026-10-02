import Foundation

private func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

@MainActor
private final class Recorder: AudioRecording {
    var starts = 0
    var stops = 0
    var holdStart = false
    var startGate: CheckedContinuation<Void, Never>?
    func start() async throws {
        starts += 1
        if holdStart { await withCheckedContinuation { startGate = $0 } }
    }
    func stop() throws -> Data { stops += 1; return Data([1, 2, 3]) }
    func cancel() { startGate?.resume(); startGate = nil }
}

@MainActor
private final class Target: DictationTextTarget {
    let applicationName = "测试输入框"
    let supportsConfirmation = false
    var writes: [String] = []
    var observationStopped = false
    func validate() throws {}
    func insert(_ text: String) throws { writes.append(text) }
    func confirms(_ text: String) -> Bool { false }
    func stopObserving() { observationStopped = true }
}

@main
private enum ClientCompositionTests {
    @MainActor
    static func wait(_ condition: () -> Bool) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !condition(), ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        try check(condition(), "等待应用组装状态超时")
    }

    @MainActor
    static func main() async throws {
        setbuf(stdout, nil)
        print("Checking ClientState initialization")
        HTTPStub.reset([
            "/v1/audio/transcriptions": .http(200, #"{"text":"保留这段原文。"}"#),
            "/health": .http(200, #"{"status":"healthy"}"#),
            "/v1/models": .http(200, #"{"data":[{"id":"asr-next"}]}"#),
            "/api/tags": .http(200, #"{"models":[{"name":"polish-next"}]}"#),
        ])
        let domain = "local.omni.composition-test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: domain)!
        defer { defaults.removePersistentDomain(forName: domain) }
        let preferences = ClientPreferences(defaults: defaults)
        let service = LocalSpeechService(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
        let recorder = Recorder()
        var modes: [(current: Bool, compatibility: Bool)] = []
        var targets: [Target] = []
        var rejectTarget = false
        let state = ClientState(preferences: preferences, service: service, recorder: recorder,
                                makeTextTarget: { current, compatibility in
            if rejectTarget { throw DictationError("测试目标不可用") }
            modes.append((current, compatibility))
            let target = Target()
            targets.append(target)
            return target
        })
        defer { state.cancel(); state.warmup.stop() }

        // Exercise the same ClientState entry used by menus and registered hotkeys.
        print("Checking recording and target-mode snapshots")
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        try check(modes.count == 1 && modes[0].current && !modes[0].compatibility,
                  "默认模式必须在录音开始时准备一个当前光标目标")
        state.pasteToCurrentCursor = false
        state.compatibilityPasteEnabled = true
        state.toggleRecording()
        try await wait { state.session.phase == .ready }
        try check(recorder.starts == 1 && recorder.stops == 1 && targets.count == 1,
                  "停止录音不应重新创建目标或开始第二次录音")
        try check(targets[0].writes == ["保留这段原文。"] && targets[0].observationStopped,
                  "真实会话回调必须连接到单次插入并结束目标观察")
        state.session.onResult?("重复回调")
        try check(targets[0].writes.count == 1, "重复结果回调不能再次插入")

        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        try check(modes.count == 2 && !modes[1].current && modes[1].compatibility,
                  "下一轮应使用新的目标模式和兼容粘贴选项")
        state.toggleRecording()
        try await wait { state.session.phase == .ready }
        try check(targets[1].writes == ["保留这段原文。"], "新模式仍须经过会话回调完成一次插入")

        // Cancel while waiting for microphone authorization, with no actual microphone.
        print("Checking authorization and HTTP cancellation")
        recorder.holdStart = true
        state.toggleRecording()
        try await wait { recorder.startGate != nil }
        let authorizingTarget = targets.last!
        state.cancel()
        recorder.holdStart = false
        try check(state.session.phase == .idle && authorizingTarget.observationStopped
                  && authorizingTarget.writes.isEmpty, "准备阶段取消必须同时清理会话和输入目标")

        // Cancellation must reach the actual URLSession task and detach the prepared target.
        HTTPStub.set(.held, for: "/v1/audio/transcriptions")
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        state.toggleRecording()
        try await wait { HTTPStub.heldCount == 1 }
        let cancelledTarget = targets.last!
        let preparedCount = targets.count
        state.toggleRecording()
        state.beginShortcutCapture()
        try check(targets.count == preparedCount && !state.shortcuts.isCapturing,
                  "识别期间不能开始新录音、替换目标或进入快捷键录入")
        state.cancel()
        try await wait { HTTPStub.cancelledCount == 1 }
        HTTPStub.releaseHeld(.http(200, #"{"text":"已取消的结果。"}"#))
        try await wait { HTTPStub.heldCount == 0 }
        try check(state.session.phase == .idle && state.session.resultText.isEmpty
                  && cancelledTarget.observationStopped && cancelledTarget.writes.isEmpty,
                  "取消后的网络响应不能写回输入框或恢复旧结果")

        HTTPStub.set(.http(200, #"{"text":"   "}"#), for: "/v1/audio/transcriptions")
        print("Checking empty results and unavailable targets")
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        state.toggleRecording()
        try await wait { state.session.phase == .ready }
        try check(targets.last!.writes.isEmpty && !state.insertion.didTriggerPaste
                  && state.session.notice.contains("没有识别"), "空转写不能触发插入")

        HTTPStub.set(.http(200, #"{"text":"保留这段原文。"}"#), for: "/v1/audio/transcriptions")
        rejectTarget = true
        let countBeforeFailure = targets.count
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        state.toggleRecording()
        try await wait { state.session.phase == .ready }
        try check(targets.count == countBeforeFailure && !state.insertion.didTriggerPaste
                  && state.insertion.message.contains("测试目标不可用")
                  && state.session.resultText == "保留这段原文。", "目标准备失败仍应保留转写供用户复制")
        rejectTarget = false

        // Save through ClientState, rather than configuring the service directly.
        print("Checking configuration saves during recording")
        let original = state.configuration
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        state.asrBaseURLDraft = "http://127.0.0.1:9101"
        state.asrModelDraft = "asr-next"
        state.polishBaseURLDraft = "http://127.0.0.1:9102"
        state.polishModelDraft = "polish-next"
        state.saveServiceConfiguration()
        try await wait { !state.checking }
        try check(preferences.serviceConfiguration == state.configuration && !state.hasUnsavedConfiguration,
                  "应用保存操作必须同步配置、持久化和编辑状态")
        state.toggleRecording()
        try await wait { state.session.phase == .ready }
        let first = HTTPStub.requests.last { $0.url.path == "/v1/audio/transcriptions" }!
        try check(first.url == original.asr.endpoint("v1/audio/transcriptions")
                  && first.body.contains(original.asr.model), "本轮请求必须保留录音开始时的配置")
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        state.toggleRecording()
        try await wait { state.session.phase == .ready }
        let next = HTTPStub.requests.last { $0.url.path == "/v1/audio/transcriptions" }!
        try check(next.url.port == 9101 && next.body.contains("asr-next")
                  && targets.last!.writes == ["保留这段原文。"], "下一轮使用新配置并正常回填")
        try check(!HTTPStub.requests.contains { $0.url.path == "/api/chat" }, "默认关闭整理时不能调用 LLM")
        print("PASS: ClientState recording/delivery wiring, mode snapshots, cancellation, empty results, target failures and configuration saves")
    }
}
