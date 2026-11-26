import Foundation

let usage = """
usage:
  replay-driver run --manifest M.jsonl --out results.jsonl --sink-socket PATH --sink-log PATH --key KEY
                [--modifiers M1,M2] [--trigger hold|toggle] [--presses N] [--press-ms MS] [--press-gap-ms MS]
                [--device "BlackHole 2ch"] [--deadline-base-s 60] [--settle-s 0]
                [--mem-app NAME --mem-out PATH [--mem-hz 10]]
                [--dry-run [--write-audio-dir DIR]]
  replay-driver mem --app NAME --out PATH [--hz 10] [--seconds S]
  replay-driver list-devices
"""

func fail(_ message: String, status: Int32 = 64) -> Never {
    FileHandle.standardError.write(Data("replay-driver: \(message)\n".utf8))
    exit(status)
}

func runCommand(_ arguments: [String]) throws -> Int32 {
    let args = try Args(arguments,
                        options: ["--manifest", "--out", "--sink-socket", "--sink-log", "--key", "--modifiers",
                                  "--trigger", "--presses", "--press-ms", "--press-gap-ms", "--device",
                                  "--deadline-base-s", "--settle-s", "--mem-app", "--mem-out", "--mem-hz", "--write-audio-dir"],
                        flags: ["--dry-run"])
    let trigger: Trigger
    switch args.string("--trigger") ?? "hold" {
    case "hold":
        trigger = .hold
    case "toggle":
        let presses = try args.int("--presses", default: 1)
        let holdMs = try args.int("--press-ms", default: 40)
        let gapMs = try args.int("--press-gap-ms", default: 80)
        // note (luojiaxuan): tap() sleeps these as useconds_t between key-down and key-up, so they are checked
        // here, before any key is posted.
        guard (1...10).contains(presses), (1...2000).contains(holdMs), (1...2000).contains(gapMs) else {
            throw ReplayError("--presses must be 1 to 10, and --press-ms and --press-gap-ms 1 to 2000")
        }
        trigger = .toggle(presses: presses, holdMs: holdMs, gapMs: gapMs)
    case let other:
        throw ReplayError("--trigger must be hold or toggle, not \(other)")
    }
    let dryRun = args.flag("--dry-run")
    if args.string("--write-audio-dir") != nil && !dryRun {
        throw ReplayError("--write-audio-dir replaces playback and is only allowed with --dry-run")
    }
    if (args.string("--mem-app") == nil) != (args.string("--mem-out") == nil) {
        throw ReplayError("--mem-app and --mem-out go together")
    }
    let settleSeconds = try args.double("--settle-s", default: 0)
    // note (luojiaxuan): A row whose last change came within stillWritingSeconds of its end is a timeout,
    // so a shorter settle would make every settled row one.
    if settleSeconds < 0 || (settleSeconds > 0 && settleSeconds <= stillWritingSeconds) {
        throw ReplayError("--settle-s must be 0 or more than \(stillWritingSeconds) s")
    }
    let options = RunOptions(
        manifestPath: try args.required("--manifest"),
        outPath: try args.required("--out"),
        sinkSocket: try args.required("--sink-socket"),
        sinkLog: try args.required("--sink-log"),
        deviceName: args.string("--device") ?? "BlackHole 2ch",
        hotkey: try Hotkey(key: try args.required("--key"), modifiers: args.string("--modifiers")),
        trigger: trigger,
        deadlineBaseSeconds: try args.double("--deadline-base-s", default: 60),
        settleSeconds: settleSeconds,
        dryRun: dryRun,
        writeAudioDir: args.string("--write-audio-dir"),
        memApp: args.string("--mem-app"),
        memOut: args.string("--mem-out"),
        memHz: try args.double("--mem-hz", default: 10))
    return try Runner(options: options).run()
}

func memCommand(_ arguments: [String]) throws -> Int32 {
    let args = try Args(arguments, options: ["--app", "--out", "--hz", "--seconds"], flags: [])
    let sampler = try MemorySampler(appName: try args.required("--app"), hz: try args.double("--hz", default: 10),
                                    outPath: try args.required("--out"))
    let seconds = try args.double("--seconds", default: 0)
    let stopped = DispatchSemaphore(value: 0)
    var sources: [DispatchSourceSignal] = []
    for number in [SIGINT, SIGTERM] {
        signal(number, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
        source.setEventHandler { stopped.signal() }
        source.resume()
        sources.append(source)
    }
    sampler.start()
    try sampler.requireTarget()
    if seconds > 0 {
        _ = stopped.wait(timeout: .now() + seconds)
    } else {
        stopped.wait()
    }
    sampler.stop()
    return 0
}

func listDevicesCommand() throws -> Int32 {
    let defaultOutput = AudioDevices.defaultDevice(input: false)
    let defaultInput = AudioDevices.defaultDevice(input: true)
    for device in try AudioDevices.all() {
        let line = JSONLine.encode([
            "id": Int(device.id), "uid": device.uid, "name": device.name,
            "input_channels": device.inputChannels, "output_channels": device.outputChannels,
            "nominal_sample_rate": device.nominalSampleRate,
            "default_output": device.id == defaultOutput, "default_input": device.id == defaultInput,
        ])
        FileHandle.standardOutput.write(line)
    }
    return 0
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard let command = arguments.first else { fail(usage) }
do {
    let rest = Array(arguments.dropFirst())
    switch command {
    case "run": exit(try runCommand(rest))
    case "mem": exit(try memCommand(rest))
    case "list-devices": exit(try listDevicesCommand())
    default: fail(usage)
    }
} catch {
    fail("\(error)", status: 1)
}
