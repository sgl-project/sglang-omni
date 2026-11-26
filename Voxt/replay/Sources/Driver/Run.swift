import CoreGraphics
import Foundation

/// note (luojiaxuan): Silence played before and after each wav.
let preSilenceSeconds = 1.5
let postSilenceSeconds = 0.5
/// note (luojiaxuan): A row whose last change falls within this many seconds before observation ended was
/// still being written when observation ended, and is marked "timeout". Character-by-
/// character writers change the field every few tens of milliseconds, so a 2 s gap
/// separates a finished write from one in progress.
let stillWritingSeconds = 2.0

enum Trigger {
    /// note (luojiaxuan): Key down before the audio, key up after it.
    case hold
    /// note (luojiaxuan): `presses` taps to start and `presses` taps to stop.
    case toggle(presses: Int, holdMs: Int, gapMs: Int)

    var name: String {
        switch self {
        case .hold: return "hold"
        case let .toggle(presses, holdMs, gapMs): return "toggle(presses=\(presses),hold_ms=\(holdMs),gap_ms=\(gapMs))"
        }
    }
}

struct RunOptions {
    var manifestPath: String
    var outPath: String
    var sinkSocket: String
    var sinkLog: String
    var deviceName: String
    var hotkey: Hotkey
    var trigger: Trigger
    var deadlineBaseSeconds: Double
    /// note (luojiaxuan): Observation ends this many seconds after the last text change once the field
    /// holds text and the stop key is this long past; 0 observes to the deadline.
    var settleSeconds: Double
    var dryRun: Bool
    var writeAudioDir: String?
    var memApp: String?
    var memOut: String?
    var memHz: Double
}

struct Segment {
    let id: String
    let wav: URL
    let lang: String
}

func loadManifest(_ path: String) throws -> [Segment] {
    let base = URL(fileURLWithPath: path).deletingLastPathComponent()
    var seen: Set<String> = []
    return try readJSONLines(path: path).enumerated().map { index, row in
        guard let id = row["id"] as? String, let wav = row["wav"] as? String, let lang = row["lang"] as? String else {
            throw ReplayError("\(path):\(index + 1): needs string fields id, wav and lang")
        }
        guard seen.insert(id).inserted else { throw ReplayError("\(path): duplicate id \(id)") }
        let url = wav.hasPrefix("/") ? URL(fileURLWithPath: wav) : base.appendingPathComponent(wav)
        guard FileManager.default.fileExists(atPath: url.path) else { throw ReplayError("\(path): \(id): no file \(url.path)") }
        return Segment(id: id, wav: url, lang: lang)
    }
}

final class Runner {
    private let options: RunOptions
    private let runID: String
    private var sink: SinkLink!
    private var player: AudioPlayer?
    private var poster: HotkeyPoster?
    private var sampler: MemorySampler?
    private var out: LineAppender!
    private var meta: LineAppender!
    private var previousID: String?
    private var dictationOpen = false

    init(options: RunOptions) {
        self.options = options
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        runID = formatter.string(from: Date())
    }

    /// note (luojiaxuan): Runs every manifest segment that has no fault-free row in the output file yet.
    /// Returns the process exit status: 0 when all ran, 2 when a driver fault stopped the run.
    func run() throws -> Int32 {
        let segments = try loadManifest(options.manifestPath)
        let finished = try finishedIDs()
        let todo = segments.filter { !finished.contains($0.id) }

        sink = try SinkLink(socketPath: options.sinkSocket, logPath: options.sinkLog)
        try sink.send("ping")

        var device: AudioDevice?
        if options.writeAudioDir == nil {
            let found = try AudioDevices.output(named: options.deviceName)
            if options.dryRun && !found.name.hasPrefix("BlackHole") {
                throw ReplayError("--dry-run plays only to a BlackHole device; use --write-audio-dir to write the audio to files")
            }
            player = try AudioPlayer(device: found)
            device = found
        }
        if !options.dryRun {
            guard CGPreflightPostEventAccess() else {
                throw ReplayError("this process may not post keyboard events: allow the app that runs replay-driver under "
                              + "System Settings > Privacy & Security > Accessibility")
            }
            let poster = try HotkeyPoster(hotkey: options.hotkey)
            installInterruptHandler(poster)
            self.poster = poster
        }
        if let app = options.memApp, let memOut = options.memOut {
            sampler = try MemorySampler(appName: app, hz: options.memHz, outPath: memOut)
        }

        out = try LineAppender(path: options.outPath)
        meta = try LineAppender(path: options.outPath + ".meta.jsonl")
        meta.append(runMetadata(device: device, total: segments.count, todo: todo.count))

        if let sampler {
            sampler.start()
            try sampler.requireTarget()
        }
        var status: Int32 = 0
        var done = 0
        for (index, segment) in todo.enumerated() {
            let row = runSegment(segment)
            out.append(row)
            done += 1
            report(index: index, count: todo.count, row: row)
            if row["driver_fault"] as? Bool == true {
                status = 2
                break
            }
        }
        sampler?.stop()
        player?.stop()
        meta.append(["run_id": runID, "event": "end", "finished_utc": ISO8601DateFormatter().string(from: Date()),
                     "segments_run": done, "exit_status": Int(status)])
        return status
    }

    private func finishedIDs() throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: options.outPath) else { return [] }
        let rows = try readJSONLines(path: options.outPath)
        return Set(rows.filter { $0["driver_fault"] as? Bool == false }.compactMap { $0["id"] as? String })
    }

    private func runSegment(_ segment: Segment) -> [String: Any] {
        var row: [String: Any] = ["run_id": runID, "id": segment.id, "lang": segment.lang, "wav": segment.wav.path,
                                  "dry_run": options.dryRun, "prev_id": previousID ?? NSNull(),
                                  "thermal_state": thermalState()]
        previousID = segment.id
        do {
            try measure(segment, into: &row)
            row["driver_fault"] = false
            row["driver_fault_reason"] = NSNull()
        } catch {
            poster?.releaseAll()
            row["driver_fault"] = true
            row["driver_fault_reason"] = "\(error)"
            if dictationOpen {
                row["fault_stopped_dictation"] = stopOpenDictation()
            }
        }
        return row
    }

    /// note (luojiaxuan): A fault between the start and the stop gesture leaves the app recording: with a
    /// toggle trigger the next segment's first tap would stop this recording instead of starting its own.
    /// This sends the stop gesture (a hold trigger already stopped with releaseAll), then waits until the
    /// sink's log has been still for max(settle, 3) s, at most the deadline base, so the app's text for
    /// the broken recording lands before the next segment clears the field.
    private func stopOpenDictation() -> Bool {
        dictationOpen = false
        if case .toggle = options.trigger {
            guard (try? stopDictation()) != nil else { return false }
        }
        let quiet = MachClock.seconds(max(options.settleSeconds, 3))
        let giveUp = MachClock.now() + MachClock.seconds(options.deadlineBaseSeconds)
        var seen = sink.events.count
        var lastChange = MachClock.now()
        while MachClock.now() < giveUp && MachClock.now() < lastChange + quiet {
            usleep(50_000)
            sink.poll()
            if sink.events.count != seen {
                seen = sink.events.count
                lastChange = MachClock.now()
            }
        }
        return true
    }

    private func measure(_ segment: Segment, into row: inout [String: Any]) throws {
        let audio = try PaddedAudio(wav: segment.wav, preSilence: preSilenceSeconds, postSilence: postSilenceSeconds)
        row["audio_duration_s"] = audio.wavDuration
        row["padded_duration_s"] = audio.paddedDuration
        if let dir = options.writeAudioDir {
            let url = URL(fileURLWithPath: dir).appendingPathComponent("\(segment.id).wav")
            try audio.write(to: url)
            row["audio_file"] = url.path
        }

        var focused = try sink.send("clear \(segment.id)").hasSuffix("key=1")
        if !focused && !options.dryRun {
            focused = try focusSink()
        }
        row["sink_key"] = focused
        if !focused && !options.dryRun {
            throw ReplayError("replay-sink's text field is not focused; bring its window to the front")
        }
        let clearIndex = try waitForClear(label: segment.id)
        // note (luojiaxuan): Every text change still in the log before this segment's clear came after the
        // previous segment's observation ended (that observation consumed the rest): a late write.
        row["t_clear"] = sink.events[clearIndex].t
        row["late_changes_prev"] = sink.events[..<clearIndex].filter { $0.kind == .text }.count
        sink.consume(through: clearIndex)

        let tKeydown = try startDictation()
        dictationOpen = true
        row["t_keydown"] = tKeydown
        let (tAudioStart, tAudioEnd) = try play(audio)
        row["t_audio_start"] = tAudioStart
        row["t_audio_end"] = tAudioEnd
        let (tStop, tEndKeydown) = try stopDictation()
        dictationOpen = false
        row["t_stop"] = tStop
        row["t_end_keydown"] = tEndKeydown.map { $0 as Any } ?? NSNull()
        try sink.send("mark \(segment.id) keyup")

        let deadline = tStop + MachClock.seconds(options.deadlineBaseSeconds + audio.wavDuration)
        row["deadline"] = deadline
        let end = options.settleSeconds > 0 ? waitForSettle(after: tKeydown, stop: tStop, deadline: deadline) : deadline
        MachClock.sleep(until: end)
        row["observe_end"] = end
        row["settled"] = end < deadline
        // note (luojiaxuan): The reply comes after the sink's main thread logged the mark, so every
        // change made before the end of observation is in the log by then.
        try sink.send("mark \(segment.id) end")
        sink.poll()
        let window = sink.events.filter { $0.t <= end }
        sink.consume(upToTime: end)

        let observed = window.filter { $0.t >= tKeydown }
        let changes = observed.filter { $0.kind == .text }
        let finalText = window.last { $0.kind == .text }?.text ?? ""
        let tFirst = changes.first { !isBlank($0.text ?? "") }?.t
        let tLast = changes.last?.t
        row["t_first"] = tFirst.map { $0 as Any } ?? NSNull()
        row["t_last"] = tLast.map { $0 as Any } ?? NSNull()
        row["final_text"] = finalText
        row["n_changes"] = changes.count
        row["n_changes_before_stop"] = changes.filter { $0.t < tStop }.count
        row["n_changes_before_keydown"] = window.filter { $0.kind == .text && $0.t < tKeydown }.count
        row["focus_lost"] = observed.contains { $0.kind == .focus && $0.key == false }

        if isBlank(finalText) {
            row["status"] = "empty"
        } else if let tLast, tLast + MachClock.seconds(stillWritingSeconds) > end {
            row["status"] = "timeout"
        } else {
            row["status"] = "complete"
        }

        if let sampler {
            let memory = sampler.window(from: tKeydown, to: end)
            row["mem_peak_phys_footprint"] = memory.peak.map { $0 as Any } ?? NSNull()
            row["mem_samples"] = memory.samples
            row["mem_read_failures"] = memory.failures
        }
    }

    /// note (luojiaxuan): The time observation can end: when the field holds text that has not changed
    /// for settleSeconds and the stop key is at least that long past, else the deadline.
    private func waitForSettle(after tKeydown: UInt64, stop tStop: UInt64, deadline: UInt64) -> UInt64 {
        let settle = MachClock.seconds(options.settleSeconds)
        while true {
            sink.poll()
            let now = MachClock.now()
            if now >= deadline { return deadline }
            if let last = sink.events.last(where: { $0.kind == .text && $0.t >= tKeydown }), !isBlank(last.text ?? ""),
               now >= max(last.t, tStop) + settle {
                return now
            }
            usleep(50_000)
        }
    }

    /// note (luojiaxuan): Activation is asynchronous: the sink's window becomes key a little after the
    /// focus command returns, so its state is polled for up to 2 s.
    private func focusSink() throws -> Bool {
        let giveUp = MachClock.now() + MachClock.seconds(2)
        repeat {
            if try sink.send("focus").hasSuffix("key=1") { return true }
            usleep(100_000)
        } while MachClock.now() < giveUp
        return false
    }

    private func waitForClear(label: String) throws -> Int {
        let giveUp = MachClock.now() + MachClock.seconds(2)
        while MachClock.now() < giveUp {
            sink.poll()
            if let index = sink.events.firstIndex(where: { $0.kind == .clear && $0.label == label }) { return index }
            usleep(5000)
        }
        throw ReplayError("the sink log has no clear marker for \(label); is --sink-log the file replay-sink writes?")
    }

    private func startDictation() throws -> UInt64 {
        guard let poster else { return MachClock.now() }
        switch options.trigger {
        case .hold: return try poster.down()
        case let .toggle(presses, holdMs, gapMs): return try poster.tap(presses: presses, holdMs: holdMs, gapMs: gapMs).firstDown
        }
    }

    /// note (luojiaxuan): Returns t_stop and, for toggle triggers, the post time of the last down event.
    private func stopDictation() throws -> (UInt64, UInt64?) {
        guard let poster else { return (MachClock.now(), nil) }
        switch options.trigger {
        case .hold:
            return (try poster.up(), nil)
        case let .toggle(presses, holdMs, gapMs):
            let taps = try poster.tap(presses: presses, holdMs: holdMs, gapMs: gapMs)
            return (taps.lastUp, taps.lastDown)
        }
    }

    private func play(_ audio: PaddedAudio) throws -> (UInt64, UInt64) {
        if let player { return try player.play(audio.buffer) }
        // note (luojiaxuan): Audio written to a file: keep the real-time pacing without playing it.
        let start = MachClock.now()
        MachClock.sleep(until: start + MachClock.seconds(audio.paddedDuration))
        return (start, MachClock.now())
    }

    private func report(index: Int, count: Int, row: [String: Any]) {
        var line = "[\(index + 1)/\(count)] \(row["id"] ?? "")"
        if let reason = row["driver_fault_reason"] as? String {
            line += " DRIVER FAULT: \(reason)"
        } else {
            line += " \(row["status"] ?? "")"
            let stop = (row["t_stop"] as? UInt64) ?? 0
            for name in ["t_first", "t_last"] {
                if let t = row[name] as? UInt64 {
                    line += String(format: " %@-t_stop=%.0fms", name, (Double(t) - Double(stop)) / 1e6)
                }
            }
        }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private func runMetadata(device: AudioDevice?, total: Int, todo: Int) -> [String: Any] {
        let devices = (try? AudioDevices.all()) ?? []
        let uid = { (id: UInt32) in devices.first { $0.id == id }?.uid ?? "" }
        var info: [String: Any] = [
            "run_id": runID, "event": "start", "started_utc": ISO8601DateFormatter().string(from: Date()),
            "argv": CommandLine.arguments,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "hw_model": sysctlString("hw.model"), "hw_memsize": sysctlInt("hw.memsize"),
            "mach_timebase": [Int(MachClock.timebase.numer), Int(MachClock.timebase.denom)],
            "default_output_uid": uid(AudioDevices.defaultDevice(input: false)),
            "default_input_uid": uid(AudioDevices.defaultDevice(input: true)),
            "thermal_state": thermalState(),
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "memory_pressure_level": sysctlInt("kern.memorystatus_vm_pressure_level"),
            "swap": swapUsage(),
            "dry_run": options.dryRun, "hotkey": options.hotkey.description, "trigger": options.trigger.name,
            "pre_silence_s": preSilenceSeconds, "post_silence_s": postSilenceSeconds,
            "deadline_base_s": options.deadlineBaseSeconds, "settle_s": options.settleSeconds,
            "still_writing_s": stillWritingSeconds,
            "mem_app": options.memApp ?? NSNull(), "mem_hz": options.memHz,
            "segments_in_manifest": total, "segments_to_run": todo,
        ]
        if let device {
            info["device"] = ["name": device.name, "uid": device.uid, "nominal_sample_rate": device.nominalSampleRate]
            info["engine_bound_device_uid"] = uid(player?.boundDevice() ?? 0)
        } else {
            info["device"] = NSNull()
        }
        return info
    }
}

func isBlank(_ text: String) -> Bool {
    text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
}

func thermalState() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0 else { return "" }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "" }
    return String(cString: buffer)
}

func sysctlInt(_ name: String) -> Int {
    var value: Int64 = 0
    var size = MemoryLayout<Int64>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return -1 }
    // note (luojiaxuan): Some values (the pressure level) are 32-bit.
    return size == 4 ? Int(Int32(truncatingIfNeeded: value)) : Int(value)
}

func swapUsage() -> [String: Any] {
    var usage = xsw_usage()
    var size = MemoryLayout<xsw_usage>.size
    guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return [:] }
    return ["total_bytes": usage.xsu_total, "used_bytes": usage.xsu_used]
}

private var signalSources: [DispatchSourceSignal] = []

/// note (luojiaxuan): On SIGINT, SIGTERM or SIGHUP, releases any held hotkey before exiting, so an
/// interrupted run never leaves the dictation key down.
func installInterruptHandler(_ poster: HotkeyPoster) {
    for number in [SIGINT, SIGTERM, SIGHUP] {
        signal(number, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
        source.setEventHandler {
            poster.releaseAll()
            FileHandle.standardError.write(Data("replay-driver: interrupted; hotkey released\n".utf8))
            exit(130)
        }
        source.resume()
        signalSources.append(source)
    }
}
