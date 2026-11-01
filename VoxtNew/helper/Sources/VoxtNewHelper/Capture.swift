import VoxtNewHelperKit
@preconcurrency import AVFoundation
import CoreAudio
import Foundation

// Microphone capture for a session: AVAudioEngine's input, downmixed to
// mono, resampled (AVAudioConverter, one per engine so the resampler's
// state carries over) to the asked rate and cut into capture.pcm frames
// (PCM16 LE) with capture.level about 20 times a second (PCMFramer).
//
// capture.start is a job: capture.started at the first audio, then the
// frames, and its answer when it ends; capture.stop flushes the queued
// audio and the partial last frame before the job answers.
//
// - Devices are tried in the user's order (Core Audio UIDs), then the
//   default input (fellBack).
// - Watchdog (Voxt's lesson: Bluetooth and some USB inputs start without a
//   single callback): no callback within 1.2 s rebuilds the engine; the
//   second time on the default input; after that the job ends (an error
//   when no audio came at all).
// - A configuration change (default device switched, AirPods connected)
//   rebuilds the engine; the chosen device going away ends the recording
//   (deviceLost), unless it was the default input, which follows the new
//   default.
// - Tests only: VOXTNEW_HELPER_CAPTURE_FILE names an audio file that
//   stands in for the microphone (CaptureFileConfig): the same conversion,
//   framing, notifications and endings, at a microphone's pace, without
//   asking for the microphone.

// MARK: Devices (Core Audio)

private func property<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                         scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal, _ value: inout T) -> Bool {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<T>.size)
    return withUnsafeMutablePointer(to: &value) { AudioObjectGetPropertyData(object, &address, 0, nil, &size, $0) } == noErr
}

private func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let value = UnsafeMutablePointer<Unmanaged<CFString>?>.allocate(capacity: 1)
    defer { value.deallocate() }
    value.initialize(to: nil)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, value) == noErr, let s = value.pointee?.takeRetainedValue() else { return nil }
    return s as String
}

private func inputChannelCount(_ device: AudioObjectID) -> Int {
    var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                             mScope: kAudioDevicePropertyScopeInput, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, raw) == noErr else { return 0 }
    let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

private func transportName(_ t: UInt32) -> String? {
    switch t {
    case kAudioDeviceTransportTypeBuiltIn: "builtin"
    case kAudioDeviceTransportTypeUSB: "usb"
    case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE: "bluetooth"
    case kAudioDeviceTransportTypeAggregate, kAudioDeviceTransportTypeAutoAggregate: "aggregate"
    case kAudioDeviceTransportTypeVirtual: "virtual"
    case kAudioDeviceTransportTypeContinuityCaptureWired, kAudioDeviceTransportTypeContinuityCaptureWireless: "continuity"
    default: nil
    }
}

private func inputDeviceIDs() -> (all: [AudioObjectID], defaultID: AudioObjectID) {
    var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                             mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    var ids: [AudioObjectID] = []
    if AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr {
        ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids)
    }
    var def = AudioObjectID(0)
    _ = property(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultInputDevice, &def)
    return (ids.filter { inputChannelCount($0) > 0 }, def)
}

/// The audio inputs, the default one first.
func captureDevices() -> [CaptureDevice] {
    let (ids, def) = inputDeviceIDs()
    var out: [CaptureDevice] = []
    for id in ids {
        guard let uid = stringProperty(id, kAudioDevicePropertyDeviceUID) else { continue }
        var rate = Float64(0)
        _ = property(id, kAudioDevicePropertyNominalSampleRate, &rate)
        var transport = UInt32(0)
        _ = property(id, kAudioDevicePropertyTransportType, &transport)
        out.append(CaptureDevice(id: uid, name: stringProperty(id, kAudioObjectPropertyName) ?? uid,
                                 channels: inputChannelCount(id), sampleRate: Int(rate), default: id == def,
                                 transport: transportName(transport)))
    }
    return out.sorted { $0.default && !$1.default }
}

private func findDevice(uid: String) -> AudioObjectID? {
    inputDeviceIDs().all.first { stringProperty($0, kAudioDevicePropertyDeviceUID) == uid }
}

private func deviceName(_ id: AudioObjectID) -> String {
    stringProperty(id, kAudioObjectPropertyName) ?? stringProperty(id, kAudioDevicePropertyDeviceUID) ?? "input \(id)"
}

private func isAlive(_ id: AudioObjectID) -> Bool {
    var alive = UInt32(0)
    return property(id, kAudioDevicePropertyDeviceIsAlive, &alive) && alive != 0
}

/// CPU time of the process, in seconds.
func cpuSeconds() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    func s(_ tv: timeval) -> Double { Double(tv.tv_sec) + Double(tv.tv_usec) / 1e6 }
    return s(u.ru_utime) + s(u.ru_stime)
}

// MARK: Capture

/// Runs one recording at a time.
final class CaptureManager: @unchecked Sendable {
    let out: Writer
    /// The test-only file input (VOXTNEW_HELPER_CAPTURE_FILE), else nil.
    let file: CaptureFileConfig?
    private let lock = NSLock()
    private var run: CaptureRun?

    init(out: Writer, file: CaptureFileConfig? = CaptureFileConfig.from(environment: ProcessInfo.processInfo.environment)) {
        self.out = out
        self.file = file
        if let file { log("capture reads \(file.path) at \(file.speed)× instead of the microphone (\(CaptureFileConfig.fileVariable): tests only)") }
    }

    /// Starts the capture.start job: the run answers it when it ends.
    func start(job: String, _ p: CaptureParams, clock: RequestClock) async throws {
        if file == nil {
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized: break
            case .notDetermined:
                if !(await AVCaptureDevice.requestAccess(for: .audio)) {
                    throw RPCError.denied(PermissionName.microphone, "VoxtNew may not use the microphone")
                }
            default:
                throw RPCError.denied(PermissionName.microphone, "VoxtNew may not use the microphone")
            }
        }
        let r: CaptureRun = try lock.withLock {
            if let cur = run { throw RPCError(Code.busy, "A recording is running already (\(cur.job))") }
            let r = CaptureRun(out: out, job: job, params: p, file: file, t0: clock.received) { [weak self] ended in
                self?.lock.withLock { if self?.run === ended { self?.run = nil } }
            }
            run = r
            return r
        }
        do {
            try r.begin()
        } catch {
            lock.withLock { if run === r { run = nil } }
            throw error
        }
    }

    /// Ends the job's recording (it answers first); false when not running.
    func stop(job: String) -> Bool {
        guard let r = lock.withLock({ run }), r.job == job else { return false }
        return r.finish(reason: CaptureReason.stopped)
    }

    /// Discards the job's recording (it answers cancelled first).
    func cancel(job: String) -> Bool {
        guard let r = lock.withLock({ run }), r.job == job else { return false }
        return r.cancel()
    }

    func cancelAll() {
        if let r = lock.withLock({ run }) { _ = r.cancel() }
    }
}

/// An input to try: a chosen device, or the default input (id nil).
private struct Candidate {
    var uid: String?
}

/// One recording, possibly over several engines (restarts).
private final class CaptureRun: @unchecked Sendable {
    let out: Writer
    let job: String
    let params: CaptureParams
    let file: CaptureFileConfig?
    let onEnd: (CaptureRun) -> Void
    private let rate: Int
    private let maxSamples: Int?
    private let queue = DispatchQueue(label: "voxtnew.capture", qos: .userInteractive)
    private let lock = NSLock()

    private var engine: AVAudioEngine?
    private var deviceID: AudioObjectID = 0
    private var deviceUID = ""
    private var usingDefault = true
    private var fellBack = false
    private var fallbackReason: String?
    private var resampler: MonoResampler?
    private var inputFormat: AVAudioFormat?
    private var framer: PCMFramer
    /// The file input's name and its pacing timer (tests only).
    private var sourceName: String?
    private var fileTimer: DispatchSourceTimer?
    private var generation = 0
    private var callbacksThisEngine = 0
    private var restarts = 0
    private var noCallbackRestarts = 0
    private var ended = false
    private var startedSent = false
    private var frames = 0
    private var peak: Double = -120
    private let t0: Double
    private var firstFrameMs: Double?
    private let cpu0 = cpuSeconds()
    private var configObserver: NSObjectProtocol?
    private var watcher: DispatchSourceTimer?

    init(out: Writer, job: String, params: CaptureParams, file: CaptureFileConfig?, t0: Double, onEnd: @escaping (CaptureRun) -> Void) {
        self.out = out
        self.job = job
        self.params = params
        self.file = file
        self.t0 = t0
        self.onEnd = onEnd
        rate = max(8000, params.sampleRate ?? 16000)
        framer = PCMFramer(sampleRate: rate, frameMs: params.frameMs ?? 100, levelHz: params.levelHz ?? 20)
        if let m = params.maxSeconds, m > 0 { maxSamples = Int(m * Double(rate)) } else { maxSamples = nil }
    }

    /// Starts on the first device that starts, else the default input.
    func begin() throws {
        if let file {
            return try startFile(file)
        }
        var reasons: [String] = []
        for uid in params.devices ?? [] where !uid.isEmpty {
            do {
                try startEngine(uid: uid)
                break
            } catch {
                reasons.append(errorMessage(error))
            }
        }
        if lock.withLock({ engine == nil }) {
            if !reasons.isEmpty {
                lock.withLock {
                    fellBack = true
                    fallbackReason = reasons.joined(separator: "; ")
                }
            }
            do {
                try startEngine(uid: nil)
            } catch {
                throw RPCError(Code.deviceUnavailable, "No input could start: " + (reasons + [errorMessage(error)]).joined(separator: "; "),
                               data: .object(["device": .string(params.devices?.first ?? "default")]))
            }
            if !reasons.isEmpty {
                out.logLine("warning", "capture: \(reasons.joined(separator: "; ")): recording from the default input", job: job)
            }
        }
        startWatchers()
    }

    /// Builds and starts an engine on uid (nil: the default input).
    private func startEngine(uid: String?) throws {
        let (ids, def) = inputDeviceIDs()
        let device: AudioObjectID
        if let uid {
            guard let id = findDevice(uid: uid) else { throw RPCError(Code.deviceUnavailable, "input \(uid) not found") }
            device = id
        } else {
            guard def != 0, ids.contains(def) else { throw RPCError(Code.deviceUnavailable, "no default input") }
            device = def
        }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        if uid != nil, let unit = input.audioUnit {
            var id = device
            let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                              &id, UInt32(MemoryLayout<AudioObjectID>.size))
            if status != noErr { throw RPCError(Code.deviceUnavailable, "\(deviceName(device)) cannot be selected (\(status))") }
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw RPCError(Code.deviceUnavailable, "\(deviceName(device)) has no usable format")
        }
        guard let res = MonoResampler(inputRate: format.sampleRate, rate: rate) else {
            throw RPCError(Code.deviceUnavailable, "cannot convert \(Int(format.sampleRate)) Hz")
        }
        let gen: Int = lock.withLock {
            generation += 1
            callbacksThisEngine = 0
            self.engine = engine
            deviceID = device
            deviceUID = stringProperty(device, kAudioDevicePropertyDeviceUID) ?? ""
            usingDefault = uid == nil
            resampler = res
            inputFormat = format
            return generation
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.receive(buffer, generation: gen)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            lock.withLock { if self.engine === engine { self.engine = nil } }
            throw RPCError(Code.deviceUnavailable, "\(deviceName(device)) did not start (\((error as NSError).code))")
        }
        armWatchdog(generation: gen)
    }

    /// The test-only file input: the file's buffers go through receive()
    /// like the tap's, paced by a timer; no watchdog or device watcher.
    private func startFile(_ fc: CaptureFileConfig) throws {
        let player: CaptureFilePlayer
        do {
            player = try CaptureFilePlayer(path: fc.path, speed: fc.speed)
        } catch {
            throw RPCError(Code.deviceUnavailable, "\(fc.name) cannot be read: \(errorMessage(error))",
                           data: .object(["device": .string("file:" + fc.name)]))
        }
        guard let res = MonoResampler(inputRate: player.format.sampleRate, rate: rate) else {
            throw RPCError(Code.deviceUnavailable, "cannot convert \(Int(player.format.sampleRate)) Hz")
        }
        let gen: Int = lock.withLock {
            generation += 1
            deviceID = 0
            deviceUID = "file:" + fc.name
            sourceName = fc.name
            usingDefault = false
            resampler = res
            inputFormat = player.format
            return generation
        }
        let t = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "voxtnew.capture.file", qos: .userInteractive))
        t.schedule(deadline: .now(), repeating: .milliseconds(10), leeway: .milliseconds(2))
        t.setEventHandler { [weak self] in
            for b in player.due(now: monotonicMs()) { self?.receive(b, generation: gen) }
        }
        lock.withLock { fileTimer = t }
        t.resume()
    }

    private func armWatchdog(generation gen: Int) {
        queue.asyncAfter(deadline: .now() + 1.2) { [weak self] in
            guard let self else { return }
            let (stale, silent) = self.lock.withLock { (self.ended || self.generation != gen, self.callbacksThisEngine == 0) }
            guard !stale, silent else { return }
            let attempt = self.lock.withLock { () -> Int in
                self.noCallbackRestarts += 1
                return self.noCallbackRestarts
            }
            if attempt > 2 {
                self.endAsync(noAudio: "No input gave audio: \(deviceName(self.lock.withLock { self.deviceID })) delivered nothing within 1.2 s, \(attempt - 1) times")
                return
            }
            // The second time, move to the default input.
            self.restart(reason: CaptureReason.noCallback, toDefault: attempt == 2)
        }
    }

    /// Rebuilds the engine (on the queue).
    private func restart(reason: String, toDefault: Bool = false) {
        let (attempt, preferred, wasDefault) = lock.withLock { () -> (Int, String?, Bool) in
            restarts += 1
            return (restarts, usingDefault ? nil : deviceUID, usingDefault)
        }
        stopEngine()
        let uid = toDefault ? nil : preferred
        do {
            try startEngine(uid: uid)
        } catch {
            do {
                try startEngine(uid: nil)
            } catch {
                endAsync(noAudio: "restart failed: \(errorMessage(error))")
                return
            }
        }
        if !wasDefault && lock.withLock({ usingDefault }) {
            lock.withLock {
                if !startedSent {
                    fellBack = true
                    fallbackReason = "\(preferred ?? "the input") gave no audio"
                }
            }
        }
        out.notify(Method.captureRestarted, RestartedParams(job: job, reason: reason, attempt: attempt, device: lock.withLock { deviceUID }))
    }

    private func startWatchers() {
        configObserver = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: nil, queue: nil) { [weak self] n in
            guard let self, let e = n.object as? AVAudioEngine, self.lock.withLock({ e === self.engine && !self.ended }) else { return }
            self.queue.async { self.restart(reason: CaptureReason.configurationChange) }
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.25, repeating: 0.25)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let (id, dflt, done) = self.lock.withLock { (self.deviceID, self.usingDefault, self.ended) }
            if done || id == 0 || isAlive(id) { return }
            if dflt {
                self.restart(reason: CaptureReason.deviceLost)
            } else {
                self.endAsync(reason: CaptureReason.deviceLost)
            }
        }
        t.resume()
        watcher = t
    }

    private func stopEngine() {
        let (e, t): (AVAudioEngine?, DispatchSourceTimer?) = lock.withLock {
            let e = engine
            let t = fileTimer
            engine = nil
            fileTimer = nil
            return (e, t)
        }
        t?.cancel()
        guard let e else { return }
        e.inputNode.removeTap(onBus: 0)
        e.stop()
        e.reset()
    }

    /// The tap (or the file input): downmix to mono at the input's rate,
    /// then convert on the queue.
    private func receive(_ buffer: AVAudioPCMBuffer, generation gen: Int) {
        let mono: AVAudioFormat? = lock.withLock {
            guard gen == generation, !ended else { return nil }
            callbacksThisEngine += 1
            return resampler?.mono
        }
        guard let mono, let copy = MonoResampler.downmix(buffer, into: mono) else { return }
        let at = monotonicMs()
        queue.async { [weak self] in self?.process(copy, generation: gen, at: at) }
    }

    /// On the queue: capture.started at the first audio, then convert and frame.
    private func process(_ buffer: AVAudioPCMBuffer, generation gen: Int, at: Double) {
        // Not "ended": finish() lets the buffers already queued through.
        guard let res = lock.withLock({ gen == generation ? resampler : nil }) else { return }
        sendStarted(at: at)
        guard let converted = res.convert(buffer), let samples = converted.floatChannelData?[0] else { return }
        let (fs, levels) = framer.push(UnsafeBufferPointer(start: samples, count: Int(converted.frameLength)))
        emit(fs, levels)
        if let maxSamples, framer.samplesSeen >= maxSamples {
            endAsync(reason: CaptureReason.maxSeconds)
        }
    }

    private func sendStarted(at: Double) {
        let started: CaptureStartedParams? = lock.withLock {
            guard !startedSent else { return nil }
            startedSent = true
            return CaptureStartedParams(job: job, device: deviceUID, deviceName: sourceName ?? deviceName(deviceID), fellBack: fellBack,
                                        fallbackReason: fellBack ? fallbackReason : nil,
                                        deviceSampleRate: Int(inputFormat?.sampleRate ?? 0), deviceChannels: Int(inputFormat?.channelCount ?? 0),
                                        sampleRate: rate, startMs: ((at - t0) * 10).rounded() / 10)
        }
        if let started { out.notify(Method.captureStarted, started) }
    }

    private func emit(_ fs: [PCMFramer.Frame], _ levels: [PCMFramer.Level]) {
        for l in levels {
            peak = max(peak, l.peak)
            out.notify(Method.captureLevel, LevelParams(job: job, seconds: l.seconds, rms: l.rms, peak: l.peak))
        }
        for f in fs {
            if firstFrameMs == nil { firstFrameMs = ((monotonicMs() - t0) * 10).rounded() / 10 }
            frames += 1
            out.notify(Method.capturePCM, PCMParams(job: job, seq: f.seq, offset: f.offset, data: f.pcm))
        }
    }

    private func markEnded() -> Bool {
        lock.withLock {
            if ended { return false }
            ended = true
            return true
        }
    }

    private func teardown() {
        watcher?.cancel()
        if let o = configObserver { NotificationCenter.default.removeObserver(o) }
        stopEngine()
        onEnd(self)
    }

    /// Ends from the queue's own work (watchdog, device watcher, maxSeconds):
    /// finish() waits on the queue, so it runs off it.
    private func endAsync(reason: String) {
        DispatchQueue.global(qos: .userInitiated).async { _ = self.finish(reason: reason) }
    }

    /// Ends because no audio came: an error when none came at all, else
    /// the recording so far (noCallback).
    private func endAsync(noAudio message: String) {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.lock.withLock({ self.startedSent }) {
                _ = self.finish(reason: CaptureReason.noCallback)
            } else if self.markEnded() {
                self.teardown()
                self.out.fail(self.job, RPCError(Code.deviceUnavailable, message, data: .object(["device": .string(self.lock.withLock { self.deviceUID })])))
            }
        }
    }

    /// Stops and answers the job: the frames already converted (and the
    /// partial last one) are sent first. False when it had ended.
    func finish(reason: String) -> Bool {
        guard markEnded() else { return false }
        stopEngine()
        // Buffers already queued still convert: queue.sync runs after them.
        let result: CaptureResult = queue.sync {
            if let last = framer.flush() {
                emit([last], [])
            }
            return CaptureResult(device: lock.withLock { deviceUID }, reason: reason, frames: frames, samples: framer.samplesSeen,
                                 seconds: Double(framer.samplesSeen) / Double(rate), restarts: lock.withLock { restarts }, peak: peak,
                                 firstFrameMs: firstFrameMs, cpuSeconds: ((cpuSeconds() - cpu0) * 1000).rounded() / 1000)
        }
        teardown()
        out.result(job, result)
        return true
    }

    /// Discards the recording and answers the job cancelled.
    func cancel() -> Bool {
        guard markEnded() else { return false }
        teardown()
        out.fail(job, RPCError(Code.cancelled, "The recording was cancelled"))
        return true
    }
}
