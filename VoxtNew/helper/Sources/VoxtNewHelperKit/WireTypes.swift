import Foundation

// The params and results of each method: internal/helper/proto (methods.go,
// apps.go, capture.go, hotkey.go, textio.go, output.go)
// field for field (see the conventions in Wire.swift).

// MARK: helper.*, job.cancel, log

public struct HelloParams: Codable, Sendable {
    public var `protocol`: Int
    public var app: String
    public var logLevel: String?
}

public struct Device: Codable, Sendable {
    public var name: String
    public var os: String

    public init(name: String, os: String) {
        self.name = name
        self.os = os
    }
}

public struct HelloResult: Codable, Sendable {
    public var `protocol`: Int
    public var version: String
    public var pid: Int
    public var methods: [String]
    public var device: Device

    public init(protocol: Int, version: String, pid: Int, methods: [String], device: Device) {
        self.protocol = `protocol`
        self.version = version
        self.pid = pid
        self.methods = methods
        self.device = device
    }
}

public struct CancelParams: Codable, Sendable {
    public var job: String
}

public struct CancelResult: Codable, Sendable {
    public var cancelled: Bool
    public init(cancelled: Bool) { self.cancelled = cancelled }
}

public struct LogParams: Codable, Sendable {
    public var level: String
    public var message: String
    public var job: String?

    public init(level: String, message: String, job: String? = nil) {
        self.level = level
        self.message = message
        self.job = job
    }
}

// MARK: permissions.*

public enum PermissionName {
    public static let microphone = "microphone"
    public static let accessibility = "accessibility"
}

public struct PermissionsResult: Codable, Sendable {
    /// granted, denied, notDetermined, restricted.
    public var microphone: String
    public var accessibility: Bool

    public init(microphone: String, accessibility: Bool) {
        self.microphone = microphone
        self.accessibility = accessibility
    }
}

public struct PermissionsRequestParams: Codable, Sendable {
    public var permission: String
}

// MARK: capture.*

public struct CaptureDevice: Codable, Sendable {
    public var id: String
    public var name: String
    public var channels: Int
    public var sampleRate: Int
    public var `default`: Bool
    public var transport: String?

    public init(id: String, name: String, channels: Int, sampleRate: Int, default: Bool, transport: String?) {
        self.id = id
        self.name = name
        self.channels = channels
        self.sampleRate = sampleRate
        self.default = `default`
        self.transport = transport
    }
}

public struct DevicesResult: Codable, Sendable {
    public var devices: [CaptureDevice]
    public init(devices: [CaptureDevice]) { self.devices = devices }
}

public struct CaptureParams: Codable, Sendable {
    public var devices: [String]?
    public var sampleRate: Int?
    public var frameMs: Int?
    public var levelHz: Int?
    public var maxSeconds: Double?

    public init(devices: [String]? = nil, sampleRate: Int? = nil, frameMs: Int? = nil, levelHz: Int? = nil, maxSeconds: Double? = nil) {
        self.devices = devices
        self.sampleRate = sampleRate
        self.frameMs = frameMs
        self.levelHz = levelHz
        self.maxSeconds = maxSeconds
    }
}

public struct CaptureStartedParams: Codable, Sendable {
    public var job: String
    public var device: String
    public var deviceName: String
    public var fellBack: Bool
    public var fallbackReason: String?
    public var deviceSampleRate: Int
    public var deviceChannels: Int
    public var sampleRate: Int
    /// capture.start to the first audio from the input.
    public var startMs: Double

    public init(job: String, device: String, deviceName: String, fellBack: Bool, fallbackReason: String?,
                deviceSampleRate: Int, deviceChannels: Int, sampleRate: Int, startMs: Double) {
        self.job = job
        self.device = device
        self.deviceName = deviceName
        self.fellBack = fellBack
        self.fallbackReason = fallbackReason
        self.deviceSampleRate = deviceSampleRate
        self.deviceChannels = deviceChannels
        self.sampleRate = sampleRate
        self.startMs = startMs
    }
}

public struct LevelParams: Codable, Sendable {
    public var job: String
    public var seconds: Double
    public var rms: Double
    public var peak: Double

    public init(job: String, seconds: Double, rms: Double, peak: Double) {
        self.job = job
        self.seconds = seconds
        self.rms = rms
        self.peak = peak
    }
}

public struct PCMParams: Codable, Sendable {
    public var job: String
    public var seq: Int
    public var offset: Int
    /// Mono PCM16 little endian; base64 on the wire.
    public var data: Data

    public init(job: String, seq: Int, offset: Int, data: Data) {
        self.job = job
        self.seq = seq
        self.offset = offset
        self.data = data
    }
}

public struct RestartedParams: Codable, Sendable {
    public var job: String
    public var reason: String
    public var attempt: Int
    public var device: String

    public init(job: String, reason: String, attempt: Int, device: String) {
        self.job = job
        self.reason = reason
        self.attempt = attempt
        self.device = device
    }
}

public enum CaptureReason {
    public static let stopped = "stopped"
    public static let maxSeconds = "maxSeconds"
    public static let deviceLost = "deviceLost"
    public static let noCallback = "noCallback"
    public static let configurationChange = "configurationChange"
}

public struct CaptureResult: Codable, Sendable {
    public var device: String
    public var reason: String
    public var frames: Int
    public var samples: Int
    public var seconds: Double
    public var restarts: Int
    public var peak: Double
    public var firstFrameMs: Double?
    public var cpuSeconds: Double

    public init(device: String, reason: String, frames: Int, samples: Int, seconds: Double, restarts: Int, peak: Double,
                firstFrameMs: Double?, cpuSeconds: Double) {
        self.device = device
        self.reason = reason
        self.frames = frames
        self.samples = samples
        self.seconds = seconds
        self.restarts = restarts
        self.peak = peak
        self.firstFrameMs = firstFrameMs
        self.cpuSeconds = cpuSeconds
    }
}

public struct CaptureStopParams: Codable, Sendable {
    public var job: String
}

public struct CaptureStopResult: Codable, Sendable {
    public var stopped: Bool
    public init(stopped: Bool) { self.stopped = stopped }
}

// MARK: hotkey.*

/// hotkey.configure's bindings are the recognizer's (HotkeyRecognizer.swift).
public typealias HotkeyBinding = Binding

public struct HotkeyConfigureParams: Codable, Sendable {
    public var bindings: [HotkeyBinding]
    public var doubleTapMs: Double?
    public var longPressMs: Double?
    public var tapMaxMs: Double?
    public var swallowFn: Bool?
    public var active: Bool?

    public init(bindings: [HotkeyBinding], doubleTapMs: Double? = nil, longPressMs: Double? = nil, tapMaxMs: Double? = nil,
                swallowFn: Bool? = nil, active: Bool? = nil) {
        self.bindings = bindings
        self.doubleTapMs = doubleTapMs
        self.longPressMs = longPressMs
        self.tapMaxMs = tapMaxMs
        self.swallowFn = swallowFn
        self.active = active
    }
}

public struct HotkeyStatus: Codable, Sendable {
    public var installed: Bool
    public var accessibility: Bool
    public var bindings: Int
    public var events: Int
    public var disabledByTimeout: Int
    public var missedEvents: Int
    public var maxCallbackMicros: Double
    public var error: String?

    public init(installed: Bool, accessibility: Bool, bindings: Int, events: Int, disabledByTimeout: Int, missedEvents: Int,
                maxCallbackMicros: Double, error: String?) {
        self.installed = installed
        self.accessibility = accessibility
        self.bindings = bindings
        self.events = events
        self.disabledByTimeout = disabledByTimeout
        self.missedEvents = missedEvents
        self.maxCallbackMicros = maxCallbackMicros
        self.error = error
    }
}

public struct HotkeyEventParams: Codable, Sendable {
    public var binding: String
    public var phase: Phase
    public var behavior: Behavior
    public var heldMs: Double?
    public var t: Double
    public var front: App?

    /// The recognizer's event, with heldMs to 0.1 ms and t to 1 µs.
    public init(_ e: HotkeyEvent, front: App?) {
        binding = e.binding
        phase = e.phase
        behavior = e.behavior
        heldMs = e.heldMs.map { ($0 * 10).rounded() / 10 }
        t = (e.t * 1000).rounded() / 1000
        self.front = front
    }
}

// MARK: app.*

public struct App: Codable, Sendable, Equatable {
    public var pid: Int
    public var bundleId: String?
    public var name: String?

    public init(pid: Int, bundleId: String?, name: String?) {
        self.pid = pid
        self.bundleId = bundleId
        self.name = name
    }
}

// MARK: textio.*

public struct TextRange: Codable, Sendable, Equatable {
    public var location: Int
    public var length: Int
    public init(location: Int, length: Int) {
        self.location = location
        self.length = length
    }
}

public struct FocusResult: Codable, Sendable {
    public var pid: Int
    /// The app the focused element belongs to, or the app in front when
    /// nothing is focused.
    public var app: App?
    public var role: String?
    /// "AXSecureTextField": a password field.
    public var subrole: String?
    public var editable: Bool
    public var selection: TextRange?
    public var valueLength: Int
    public var error: String?

    public init(pid: Int, app: App? = nil, role: String? = nil, subrole: String? = nil, editable: Bool = false, selection: TextRange? = nil,
                valueLength: Int = 0, error: String? = nil) {
        self.pid = pid
        self.app = app
        self.role = role
        self.subrole = subrole
        self.editable = editable
        self.selection = selection
        self.valueLength = valueLength
        self.error = error
    }
}

public struct SelectionParams: Codable, Sendable {
    public var allowCopy: Bool?
    public var copyTimeoutMs: Int?
}

public enum SelectionVia {
    public static let ax = "ax"
    public static let axRange = "ax-range"
    public static let axMarker = "ax-marker"
    public static let copy = "copy"
    public static let none = "none"
}

public struct SelectionResult: Codable, Sendable {
    public var text: String?
    public var via: String
    public var ms: Double
    public var steps: [String]?

    public init(text: String?, via: String, ms: Double, steps: [String]) {
        self.text = text?.isEmpty == true ? nil : text
        self.via = via
        self.ms = ms
        self.steps = steps.isEmpty ? nil : steps
    }
}

public struct KeyPress: Codable, Sendable {
    public var key: Int
    /// Modifier families: command, shift, option, control.
    public var modifiers: [String]?
}

public struct PasteParams: Codable, Sendable {
    public var text: String
    public var keepClipboard: Bool?
    public var restoreDelayMs: Int?
    public var targetPid: Int?
    public var press: [KeyPress]?
    public var pressDelayMs: Int?
    /// Check that the text went in, for up to this long (PasteCheck).
    public var verifyMs: Int?
}

/// The outcomes of a paste's check (PasteParams.verifyMs).
public enum PasteCheck {
    public static let inserted = "inserted"
    public static let unchanged = "unchanged"
    public static let unknown = "unknown"
}

public struct PasteResult: Codable, Sendable {
    public var posted: Bool
    public var restored: Bool
    public var clipboardTaken: Bool
    public var ms: Double
    public var front: App?
    public var check: String?
    public var checkMs: Double?
    public var field: PasteField?
    public var secureInput: App?
    public var pressSkipped: Bool?
    public var kept: Bool?

    public init(posted: Bool, restored: Bool, clipboardTaken: Bool, ms: Double, front: App? = nil,
                check: String? = nil, checkMs: Double? = nil, field: PasteField? = nil, secureInput: App? = nil,
                pressSkipped: Bool? = nil, kept: Bool? = nil) {
        self.posted = posted
        self.restored = restored
        self.clipboardTaken = clipboardTaken
        self.ms = ms
        self.front = front
        self.check = check
        self.checkMs = checkMs
        self.field = field
        self.secureInput = secureInput
        self.pressSkipped = pressSkipped
        self.kept = kept
    }
}

/// The focused element a paste's check watched: its role, its state
/// before ⌘V and at the end of the check (nil: not readable).
public struct PasteField: Codable, Sendable {
    public var role: String?
    public var before: FieldState?
    public var after: FieldState?

    public init(role: String?, before: FieldState?, after: FieldState?) {
        self.role = role
        self.before = before
        self.after = after
    }
}

public struct FieldState: Codable, Sendable, Equatable {
    public var length: Int
    public var selection: TextRange?
    /// The selected text marker range was read (web pages).
    public var marker: Bool?

    public init(length: Int, selection: TextRange?, marker: Bool? = nil) {
        self.length = length
        self.selection = selection
        self.marker = marker
    }
}

// MARK: browser.url

public struct BrowserURLParams: Codable, Sendable {
    public var bundleId: String?
    public var pid: Int?
}

public enum URLVia {
    public static let axFocus = "ax-focus"
    public static let axDocument = "ax-document"
    public static let axWebURL = "ax-weburl"
    public static let none = "none"
}

public struct BrowserURLResult: Codable, Sendable {
    public var bundleId: String
    public var url: String?
    public var via: String
    public var ms: Double
    public var error: String?
    /// The first read found no web area, so the helper set
    /// AXManualAccessibility on the process and read again.
    public var retried: Bool?

    public init(bundleId: String, url: String? = nil, via: String = URLVia.none, ms: Double = 0, error: String? = nil, retried: Bool? = nil) {
        self.bundleId = bundleId
        self.url = url
        self.via = via
        self.ms = ms
        self.error = error
        self.retried = retried
    }
}

// MARK: app.configure

public struct AppConfigureParams: Codable, Sendable {
    /// Bundle ids that get AXManualAccessibility as they come to the front
    /// (the app's Chromium browsers), besides Electron apps; replaces the
    /// last list.
    public var manualAccessibility: [String]?

    public init(manualAccessibility: [String]? = nil) { self.manualAccessibility = manualAccessibility }
}

// MARK: app.list

public struct AppListParams: Codable, Sendable {
    public var installed: Bool?
}

public struct AppInfo: Codable, Sendable, Equatable {
    public var bundleId: String
    public var name: String
    public var running: Bool
    public var pid: Int?
    public var path: String?

    public init(bundleId: String, name: String, running: Bool, pid: Int? = nil, path: String? = nil) {
        self.bundleId = bundleId
        self.name = name
        self.running = running
        self.pid = pid
        self.path = path
    }
}

public struct AppListResult: Codable, Sendable {
    public var apps: [AppInfo]
    public init(apps: [AppInfo]) { self.apps = apps }
}

// MARK: app.icons

public struct AppIconsParams: Codable, Sendable {
    /// Each app by its bundle's path when it has one, else by bundle id.
    public var apps: [AppRef]
    /// The PNG's side in pixels: iconSize when nil, at most maxIconSize.
    public var size: Int?

    public init(apps: [AppRef], size: Int? = nil) {
        self.apps = apps
        self.size = size
    }
}

public struct AppRef: Codable, Sendable, Equatable {
    public var bundleId: String
    public var path: String?

    public init(bundleId: String, path: String? = nil) {
        self.bundleId = bundleId
        self.path = path
    }
}

public struct AppIconsResult: Codable, Sendable {
    public var icons: [AppIcon]
    public init(icons: [AppIcon]) { self.icons = icons }
}

public struct AppIcon: Codable, Sendable, Equatable {
    public var bundleId: String
    /// A PNG; base64 on the wire. nil: no bundle was found.
    public var png: Data?

    public init(bundleId: String, png: Data? = nil) {
        self.bundleId = bundleId
        self.png = png
    }
}

// MARK: output.*

/// An output device; uid is its identity (the id may change).
public struct OutputDevice: Codable, Sendable, Equatable {
    public var id: UInt32
    public var uid: String
    public var name: String?

    public init(id: UInt32, uid: String, name: String? = nil) {
        self.id = id
        self.uid = uid
        self.name = name
    }
}

/// The default output and its mute: muted nil when it has none to read.
public struct OutputState: Codable, Sendable, Equatable {
    public var device: OutputDevice?
    public var muted: Bool?
    public var settable: Bool

    public init(device: OutputDevice? = nil, muted: Bool? = nil, settable: Bool = false) {
        self.device = device
        self.muted = muted
        self.settable = settable
    }
}

public struct OutputWatchParams: Codable, Sendable {
    public var on: Bool
}

public struct OutputMuteParams: Codable, Sendable {
    public var device: OutputDevice
    public var journal: String?

    public init(device: OutputDevice, journal: String? = nil) {
        self.device = device
        self.journal = journal
    }
}

public struct OutputMuteResult: Codable, Sendable, Equatable {
    public var took: Bool
    public var state: OutputState
}

public struct OutputReleaseParams: Codable, Sendable {
    public var device: OutputDevice
    public var unmute: Bool?

    public init(device: OutputDevice, unmute: Bool? = nil) {
        self.device = device
        self.unmute = unmute
    }
}

public struct OutputReleaseResult: Codable, Sendable, Equatable {
    public var unmuted: Bool
}

public struct OutputRecoverParams: Codable, Sendable {
    public var journal: String

    public init(journal: String) { self.journal = journal }
}

public struct OutputRecoverResult: Codable, Sendable, Equatable {
    /// nil when none (Go's omitempty).
    public var restored: [OutputDevice]?
}

/// output.released: the helper gave its claims back itself.
public struct OutputReleasedParams: Codable, Sendable {
    public var devices: [OutputDevice]?
    public var reason: String

    public init(devices: [OutputDevice]?, reason: String) {
        self.devices = devices
        self.reason = reason
    }
}

public enum OutputReason {
    public static let sleep = "sleep"
}

// MARK: - Types by method (the fixture tests; the server's routing)

public enum WireTypes {
    /// The params type of a method (request or notification), nil for an
    /// unknown one.
    public static func params(_ method: String) -> (Codable & Sendable).Type? {
        switch method {
        case Method.hello: HelloParams.self
        case Method.shutdown, Method.captureDevices, Method.permissionsStatus, Method.hotkeyStatus, Method.appFrontmost, Method.textFocus:
            Empty.self
        case Method.jobCancel: CancelParams.self
        case Method.permissionsRequest: PermissionsRequestParams.self
        case Method.captureStart: CaptureParams.self
        case Method.captureStop: CaptureStopParams.self
        case Method.hotkeyConfigure: HotkeyConfigureParams.self
        case Method.textSelection: SelectionParams.self
        case Method.textPaste: PasteParams.self
        case Method.browserURL: BrowserURLParams.self
        case Method.appList: AppListParams.self
        case Method.appConfigure: AppConfigureParams.self
        case Method.appIcons: AppIconsParams.self
        case Method.outputWatch: OutputWatchParams.self
        case Method.outputMute: OutputMuteParams.self
        case Method.outputRelease: OutputReleaseParams.self
        case Method.outputRecover: OutputRecoverParams.self
        case Method.outputChanged: OutputState.self
        case Method.outputReleased: OutputReleasedParams.self
        case Method.captureStarted: CaptureStartedParams.self
        case Method.captureLevel: LevelParams.self
        case Method.capturePCM: PCMParams.self
        case Method.captureRestarted: RestartedParams.self
        case Method.hotkeyEvent: HotkeyEventParams.self
        case Method.hotkeyState: HotkeyStatus.self
        case Method.appActivated: App.self
        case Method.log: LogParams.self
        default: nil
        }
    }

    /// The result type of a request, nil for an unknown one.
    public static func result(_ method: String) -> (Codable & Sendable).Type? {
        switch method {
        case Method.hello: HelloResult.self
        case Method.shutdown, Method.appConfigure: Empty.self
        case Method.jobCancel: CancelResult.self
        case Method.permissionsStatus, Method.permissionsRequest: PermissionsResult.self
        case Method.captureDevices: DevicesResult.self
        case Method.captureStart: CaptureResult.self
        case Method.captureStop: CaptureStopResult.self
        case Method.hotkeyConfigure, Method.hotkeyStatus: HotkeyStatus.self
        case Method.appFrontmost: App.self
        case Method.textFocus: FocusResult.self
        case Method.textSelection: SelectionResult.self
        case Method.textPaste: PasteResult.self
        case Method.browserURL: BrowserURLResult.self
        case Method.appList: AppListResult.self
        case Method.appIcons: AppIconsResult.self
        case Method.outputWatch: OutputState.self
        case Method.outputMute: OutputMuteResult.self
        case Method.outputRelease: OutputReleaseResult.self
        case Method.outputRecover: OutputRecoverResult.self
        default: nil
        }
    }
}
