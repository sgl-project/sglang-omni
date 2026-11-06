import Foundation

// The params and results of llm protocol 1 (internal/llmrt/wire/methods.go).

public struct HelloParams: Codable, Sendable {
    public var `protocol`: Int
    public var app: String
    public var logLevel: String?
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

public struct Device: Codable, Sendable {
    public var name: String
    public var os: String
    public var memoryBytes: Int64

    public init(name: String, os: String, memoryBytes: Int64) {
        self.name = name
        self.os = os
        self.memoryBytes = memoryBytes
    }
}

public struct CancelParams: Codable, Sendable {
    public var job: String
}

public struct CancelResult: Codable, Sendable {
    public var cancelled: Bool
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

public enum Format {
    public static let mlx = "mlx"
    public static let gguf = "gguf"
}

/// A model on disk: the catalogue's id, its format and where it is.
public struct ModelSpec: Codable, Sendable, Equatable {
    public var id: String
    public var format: String
    public var path: String

    public init(id: String, format: String, path: String) {
        self.id = id
        self.format = format
        self.path = path
    }
}

/// The process's memory use, in bytes.
public struct Memory: Codable, Sendable, Equatable {
    public var footprintBytes: Int64
    public var peakFootprintBytes: Int64
    public var residentBytes: Int64
    public var mlxActiveBytes: Int64?
    public var mlxCacheBytes: Int64?
    public var mlxPeakBytes: Int64?

    public init(footprintBytes: Int64, peakFootprintBytes: Int64, residentBytes: Int64, mlxActiveBytes: Int64? = nil, mlxCacheBytes: Int64? = nil, mlxPeakBytes: Int64? = nil) {
        self.footprintBytes = footprintBytes
        self.peakFootprintBytes = peakFootprintBytes
        self.residentBytes = residentBytes
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
    }
}

public struct LoadParams: Codable, Sendable {
    public var model: ModelSpec
    public var cacheLimitMB: Int?

    public init(model: ModelSpec, cacheLimitMB: Int? = nil) {
        self.model = model
        self.cacheLimitMB = cacheLimitMB
    }
}

public struct LoadResult: Codable, Sendable {
    public var model: ModelSpec
    public var cached: Bool?
    public var loadMs: Double
    public var memory: Memory
}

public struct UnloadResult: Codable, Sendable {
    public var unloaded: Bool
    public var memory: Memory
}

public struct StatusResult: Codable, Sendable {
    public var model: ModelSpec?
    public var memory: Memory
}

public struct ChatMessage: Codable, Sendable, Equatable {
    public var role: String
    public var content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// How the next token is chosen. Temperature 0 is greedy; absent fields
/// are neutral.
public struct Sampling: Codable, Sendable, Equatable {
    public var temperature: Double
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var repetitionContextSize: Int?
    public var seed: UInt64?

    public init(temperature: Double, topP: Double? = nil, topK: Int? = nil, minP: Double? = nil,
                repetitionPenalty: Double? = nil, repetitionContextSize: Int? = nil, seed: UInt64? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.seed = seed
    }
}

public struct ChatParams: Codable, Sendable {
    public var model: ModelSpec
    public var messages: [ChatMessage]
    public var maxTokens: Int
    public var sampling: Sampling
    public var prefillStepSize: Int?
    public var template: [String: JSONValue]?

    public init(model: ModelSpec, messages: [ChatMessage], maxTokens: Int, sampling: Sampling,
                prefillStepSize: Int? = nil, template: [String: JSONValue]? = nil) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.prefillStepSize = prefillStepSize
        self.template = template
    }
}

public enum Finish {
    public static let stop = "stop"
    public static let length = "length"
}

public struct Usage: Codable, Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

public struct Timings: Codable, Sendable, Equatable {
    public var loadMs: Double?
    public var promptMs: Double
    public var firstTokenMs: Double
    public var totalMs: Double
    public var tokensPerSecond: Double
    public var cache: CacheUse?

    public init(loadMs: Double? = nil, promptMs: Double, firstTokenMs: Double, totalMs: Double, tokensPerSecond: Double, cache: CacheUse? = nil) {
        self.loadMs = loadMs
        self.promptMs = promptMs
        self.firstTokenMs = firstTokenMs
        self.totalMs = totalMs
        self.tokensPerSecond = tokensPerSecond
        self.cache = cache
    }
}

/// What a chat did with the prompt cache (an MLX model's): hit when it
/// started from a snapshot; prefixTokens the shared prefix it reused (a
/// hit) or evaluated and kept (a miss), 0 when it did neither.
public struct CacheUse: Codable, Sendable, Equatable {
    public var hit: Bool
    public var prefixTokens: Int

    public init(hit: Bool, prefixTokens: Int) {
        self.hit = hit
        self.prefixTokens = prefixTokens
    }
}

public struct ChatResult: Codable, Sendable {
    public var text: String
    public var finish: String
    public var usage: Usage
    public var timings: Timings
    public var memory: Memory

    public init(text: String, finish: String, usage: Usage, timings: Timings, memory: Memory) {
        self.text = text
        self.finish = finish
        self.usage = usage
        self.timings = timings
        self.memory = memory
    }
}

public struct DeltaParams: Codable, Sendable {
    public var job: String
    public var text: String

    public init(job: String, text: String) {
        self.job = job
        self.text = text
    }
}

/// The params type of a method, for the fixture tests (nil: unknown).
public func paramsType(_ method: String) -> (any (Codable & Sendable).Type)? {
    switch method {
    case Method.hello: return HelloParams.self
    case Method.shutdown, Method.unload, Method.status: return Empty.self
    case Method.jobCancel: return CancelParams.self
    case Method.load: return LoadParams.self
    case Method.chat: return ChatParams.self
    case Method.chatDelta: return DeltaParams.self
    case Method.log: return LogParams.self
    default: return nil
    }
}

/// The result type of a request (nil: unknown).
public func resultType(_ method: String) -> (any (Codable & Sendable).Type)? {
    switch method {
    case Method.hello: return HelloResult.self
    case Method.shutdown: return Empty.self
    case Method.jobCancel: return CancelResult.self
    case Method.load: return LoadResult.self
    case Method.unload: return UnloadResult.self
    case Method.status: return StatusResult.self
    case Method.chat: return ChatResult.self
    default: return nil
    }
}
