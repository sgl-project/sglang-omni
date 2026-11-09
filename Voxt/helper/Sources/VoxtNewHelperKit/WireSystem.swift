import Foundation

// capture.system: the sound the Mac plays, through a Core Audio process
// tap on the default output (docs/helper-protocol.md, capture.system).

/// capture.system's params: CaptureParams without devices.
public struct SystemParams: Codable, Sendable {
    public var sampleRate: Int?
    public var frameMs: Int?
    public var levelHz: Int?
    public var maxSeconds: Double?

    public init(sampleRate: Int? = nil, frameMs: Int? = nil, levelHz: Int? = nil, maxSeconds: Double? = nil) {
        self.sampleRate = sampleRate
        self.frameMs = frameMs
        self.levelHz = levelHz
        self.maxSeconds = maxSeconds
    }

    /// The same recording settings as a capture.start's.
    public var capture: CaptureParams { CaptureParams(sampleRate: sampleRate, frameMs: frameMs, levelHz: levelHz, maxSeconds: maxSeconds) }
}

/// What the process tap is asked for: every process's sound but VoxtNew's
/// and the helper's, mixed to one mono stream, private to the helper, left
/// audible to the user, on the default output.
public struct SystemTapPlan: Equatable, Sendable {
    public var excludedPIDs: [Int32]
    public var outputUID: String
    public var mono = true
    public var mixdown = true
    public var isPrivate = true
    public var muted = false
    public var exclusive = true
}

/// The tap for outputUID, leaving out ownPIDs (the helper and VoxtNew;
/// 0, 1 and duplicates dropped).
public func systemTapPlan(ownPIDs: [Int32], outputUID: String) -> SystemTapPlan {
    var seen = Set<Int32>()
    return SystemTapPlan(excludedPIDs: ownPIDs.filter { $0 > 1 && seen.insert($0).inserted }, outputUID: outputUID)
}

/// The recordings that run: at most one capture.start and one
/// capture.system at a time.
public struct CaptureSlots<Run: AnyObject> {
    private var runs: [String: (job: String, run: Run)] = [:]

    public init() {}

    /// Takes method's slot for job; busy when one of its kind runs.
    public mutating func claim(_ method: String, job: String, _ run: Run) throws {
        if let cur = runs[method] { throw RPCError(Code.busy, "A recording is running already (\(cur.job))") }
        runs[method] = (job, run)
    }

    /// Frees the slot run holds, if it still does.
    public mutating func release(_ run: Run) {
        for (method, cur) in runs where cur.run === run { runs[method] = nil }
    }

    /// The run of job, of either kind.
    public func run(job: String) -> Run? { runs.values.first { $0.job == job }?.run }

    public var all: [Run] { runs.values.map(\.run) }
}

public extension CaptureFileConfig {
    static let systemFileVariable = "VOXTNEW_HELPER_SYSTEM_FILE"

    /// VOXTNEW_HELPER_SYSTEM_FILE (tests only): capture.system plays this
    /// file as capture.start plays VOXTNEW_HELPER_CAPTURE_FILE, at the same
    /// speed, without a tap; nil: the tap.
    static func system(from env: [String: String]) -> CaptureFileConfig? {
        guard let path = env[systemFileVariable], !path.isEmpty else { return nil }
        var e = env
        e[fileVariable] = path
        return from(environment: e)
    }
}
