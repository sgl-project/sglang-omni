import Foundation

// Helper protocol 1 between VoxtNew (Go) and voxtnew-helper: the Swift
// mirror of internal/helper/proto. The spec is docs/helper-protocol.md;
// the golden transcripts in helper/Fixtures are decoded by both sides
// (Tests/VoxtNewHelperKitTests/FixtureTests.swift here, go test
// ./internal/helper/proto there), so a field added on one side and not
// the other fails a test.
//
// One JSON object per line. Requests {id, method, params} go app → helper;
// responses {id, result | error} and notifications {method, params} go
// helper → app. Ids and error codes are strings. Unknown members are
// ignored; missing optional members take their defaults. A Go field with
// omitempty is a Swift optional, left nil when zero; a Go field without
// it is a Swift non-optional, always written.

public let protocolVersion = 1

/// Longest line either side accepts, in bytes.
public let maxLine = 4 << 20

public enum Method {
    public static let hello = "helper.hello"
    public static let shutdown = "helper.shutdown"
    public static let jobCancel = "job.cancel"
    public static let permissionsStatus = "permissions.status"
    public static let permissionsRequest = "permissions.request"
    public static let captureDevices = "capture.devices"
    public static let captureStart = "capture.start"
    public static let captureStop = "capture.stop"
    public static let hotkeyConfigure = "hotkey.configure"
    public static let hotkeyStatus = "hotkey.status"
    public static let appFrontmost = "app.frontmost"
    public static let textFocus = "textio.focus"
    public static let textSelection = "textio.selection"
    public static let textPaste = "textio.paste"
    public static let appList = "app.list"
    public static let appConfigure = "app.configure"
    public static let appIcons = "app.icons"
    public static let browserURL = "browser.url"
    public static let outputWatch = "output.watch"
    public static let outputMute = "output.mute"
    public static let outputRelease = "output.release"
    public static let outputRecover = "output.recover"
    // Notifications, helper → app.
    public static let captureStarted = "capture.started"
    public static let captureLevel = "capture.level"
    public static let capturePCM = "capture.pcm"
    public static let captureRestarted = "capture.restarted"
    public static let hotkeyEvent = "hotkey.event"
    public static let hotkeyState = "hotkey.state"
    public static let appActivated = "app.activated"
    public static let outputChanged = "output.changed"
    public static let outputReleased = "output.released"
    public static let log = "log"

    /// Every request, in the order hello lists them.
    public static let requests = [
        hello, shutdown, jobCancel, permissionsStatus, permissionsRequest,
        captureDevices, captureStart, captureStop, hotkeyConfigure, hotkeyStatus,
        appFrontmost, appList, appConfigure, appIcons, textFocus, textSelection, textPaste,
        browserURL, outputWatch, outputMute, outputRelease, outputRecover,
    ]
    public static let notifications = [
        captureStarted, captureLevel, capturePCM, captureRestarted, hotkeyEvent, hotkeyState, appActivated,
        outputChanged, outputReleased, log,
    ]

    /// Long-running, cancellable requests (job.cancel; notifications name them in "job").
    public static func isJob(_ method: String) -> Bool { method == captureStart }
}

public enum Code {
    public static let badRequest = "bad_request"
    public static let methodNotFound = "method_not_found"
    public static let notInitialized = "not_initialized"
    public static let protocolMismatch = "protocol_mismatch"
    public static let busy = "busy"
    public static let cancelled = "cancelled"
    public static let permissionDenied = "permission_denied"
    public static let deviceUnavailable = "device_unavailable"
    public static let unsupported = "unsupported"
    public static let timeout = "timeout"
    public static let `internal` = "internal"

    public static let all: Set<String> = [
        badRequest, methodNotFound, notInitialized, protocolMismatch, busy, cancelled,
        permissionDenied, deviceUnavailable, unsupported, timeout, `internal`,
    ]
}

// MARK: - Arbitrary JSON (error data)

/// Any JSON value; integers stay integers.
public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? c.decode(Int64.self) {
            self = .int(i)
        } else if let d = try? c.decode(Double.self) {
            self = .double(d)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSONValue].self) {
            self = .array(a)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
}

// MARK: - Envelope

public struct RPCError: Codable, Error, Sendable, LocalizedError {
    public var code: String
    public var message: String
    public var data: JSONValue?

    public init(_ code: String, _ message: String, data: JSONValue? = nil) {
        self.code = code
        self.message = message
        self.data = data
    }

    public var errorDescription: String? { message }

    /// permission_denied for permission (PermissionName).
    public static func denied(_ permission: String, _ message: String) -> RPCError {
        RPCError(Code.permissionDenied, message, data: .object(["permission": .string(permission)]))
    }
}

/// The text of an error for the app: the error's own words, not Cocoa's
/// generic "The operation couldn’t be completed".
public func errorMessage(_ error: Error) -> String {
    if let e = error as? LocalizedError, let d = e.errorDescription, !d.isEmpty {
        return d
    }
    if type(of: error) is NSError.Type {
        return (error as NSError).localizedDescription
    }
    let d = String(describing: error)
    return d.isEmpty ? (error as NSError).localizedDescription : d
}

/// A decoded line. Params and result stay raw JSON until the method is known.
public struct Incoming: Sendable {
    public var id: String?
    public var method: String?
    public var params: Data?
    public var result: Data?
    public var error: RPCError?

    public init(id: String? = nil, method: String? = nil, params: Data? = nil) {
        self.id = id
        self.method = method
        self.params = params
    }

    public enum Kind { case request, notification, response, invalid }

    public var kind: Kind {
        let m = !(method ?? "").isEmpty, i = !(id ?? "").isEmpty
        if m && result == nil && error == nil { return i ? .request : .notification }
        if !m && i && ((result == nil) != (error == nil)) { return .response }
        return .invalid
    }

    /// Parses one line; fails on malformed JSON or a non-object.
    public static func decode(_ line: Data) throws -> Incoming {
        guard let obj = try JSONSerialization.jsonObject(with: line, options: [.fragmentsAllowed]) as? [String: Any] else {
            throw RPCError(Code.badRequest, "A message must be a JSON object")
        }
        var m = Incoming()
        m.id = obj["id"] as? String
        m.method = obj["method"] as? String
        if let p = obj["params"], !(p is NSNull) {
            m.params = try JSONSerialization.data(withJSONObject: p, options: [.fragmentsAllowed])
        }
        if let r = obj["result"], !(r is NSNull) {
            m.result = try JSONSerialization.data(withJSONObject: r, options: [.fragmentsAllowed])
        }
        if let e = obj["error"], !(e is NSNull) {
            let data = try JSONSerialization.data(withJSONObject: e, options: [.fragmentsAllowed])
            m.error = try JSONDecoder().decode(RPCError.self, from: data)
        }
        return m
    }

    /// Decodes params as T; absent params decode from {}. A failure is a
    /// bad_request.
    public func decodeParams<T: Decodable>(_ type: T.Type) throws -> T {
        do {
            return try JSONDecoder().decode(T.self, from: params ?? Data("{}".utf8))
        } catch {
            throw RPCError(Code.badRequest, "Bad params for \(method ?? "?"): \(error)")
        }
    }

    public func decodeResult<T: Decodable>(_ type: T.Type) throws -> T {
        try JSONDecoder().decode(T.self, from: result ?? Data("{}".utf8))
    }
}

/// An outgoing message. Exactly one of the shapes: see Incoming.Kind.
public struct Outgoing<P: Encodable>: Encodable {
    public var id: String?
    public var method: String?
    public var params: P?
    public var result: P?
    public var error: RPCError?

    public init(id: String? = nil, method: String? = nil, params: P? = nil, result: P? = nil, error: RPCError? = nil) {
        self.id = id
        self.method = method
        self.params = params
        self.result = result
        self.error = error
    }

    public static func notification(_ method: String, _ params: P) -> Outgoing {
        Outgoing(method: method, params: params)
    }

    public static func response(_ id: String, _ result: P) -> Outgoing {
        Outgoing(id: id, result: result)
    }
}

extension Outgoing where P == Empty {
    public static func failure(_ id: String, _ error: RPCError) -> Outgoing {
        Outgoing(id: id, error: error)
    }
}

public func makeEncoder() -> JSONEncoder {
    let e = JSONEncoder()
    e.outputFormatting = [.withoutEscapingSlashes]
    return e
}

/// Params or result of methods that take or return nothing.
public struct Empty: Codable, Sendable {
    public init() {}
}

/// Milliseconds of a monotonic clock (CLOCK_UPTIME_RAW): request timings,
/// hotkey.event's t.
public func monotonicMs() -> Double {
    Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e6
}
