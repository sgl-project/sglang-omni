import Darwin
import Foundation
import VoxtNewLLMKit

/// The process's Engine: one model at a time, MLX or GGUF. The server
/// calls load, unload and chat one after another; loaded and memory may
/// come from any thread.
final class Runtime: Engine, @unchecked Sendable {
    private let mlx = MLXEngine()
    private let gguf = GGUFEngine()
    private let lock = NSLock()
    private var current: ModelSpec?

    func loaded() -> ModelSpec? { lock.withLock { current } }

    func load(_ model: ModelSpec, cacheLimitMB: Int?) async throws -> LoadOutcome {
        guard model.format == Format.mlx || model.format == Format.gguf else {
            throw RPCError(Code.badRequest, "Unknown model format \(model.format)")
        }
        if loaded() == model {
            return LoadOutcome(cached: true, loadMs: 0)
        }
        _ = await unload()
        let started = monotonicMs()
        logLine("loading \(model.id) (\(model.format)) from \(model.path)")
        if model.format == Format.mlx {
            try await mlx.load(model, cacheLimitMB: cacheLimitMB)
        } else {
            try gguf.load(model)
        }
        lock.withLock { current = model }
        let ms = monotonicMs() - started
        logLine("loaded \(model.id) in \(Int(ms)) ms; footprint \(memory().footprintBytes >> 20) MB, resident \(memory().residentBytes >> 20) MB")
        return LoadOutcome(cached: false, loadMs: ms)
    }

    func unload() async -> Bool {
        guard let was = loaded() else { return false }
        if was.format == Format.mlx {
            mlx.unload()
        } else {
            gguf.unload()
        }
        lock.withLock { current = nil }
        logLine("unloaded \(was.id)")
        return true
    }

    func chat(_ p: ChatParams, onDelta: @escaping @Sendable (String) -> Void) async throws -> ChatOutcome {
        guard loaded() == p.model else { throw RPCError(Code.internal, "\(p.model.id) is not loaded") }
        if p.model.format == Format.mlx {
            return try await mlx.chat(p, onDelta: onDelta)
        }
        return try gguf.chat(p, onDelta: onDelta)
    }

    func memory() -> Memory {
        let (footprint, peak, resident) = taskMemory()
        guard loaded()?.format == Format.mlx else {
            return Memory(footprintBytes: footprint, peakFootprintBytes: peak, residentBytes: resident)
        }
        let c = MLXEngine.counters()
        return Memory(footprintBytes: footprint, peakFootprintBytes: peak, residentBytes: resident,
                      mlxActiveBytes: c.active, mlxCacheBytes: c.cache, mlxPeakBytes: c.peak)
    }
}

/// The process's physical footprint, now and at its peak (task_vm_info:
/// what Activity Monitor's Memory column shows; Metal buffers count), and
/// its resident size (mapped files too: a GGUF model's weights).
func taskMemory() -> (Int64, Int64, Int64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return (0, 0, 0) }
    return (Int64(info.phys_footprint), Int64(info.ledger_phys_footprint_peak), Int64(info.resident_size))
}

/// Diagnostics on stderr (the app keeps them in llm.log).
func logLine(_ message: String) {
    FileHandle.standardError.write(Data("[llm] \(message)\n".utf8))
}

/// The process's version (its Info.plist, kept in step with mygo.json).
func llmVersion() -> String {
    Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
}

/// The Mac: its model (sysctl hw.model), macOS and memory.
func device() -> Device {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var name = "Mac"
    if size > 0 {
        var buf = [UInt8](repeating: 0, count: size)
        sysctlbyname("hw.model", &buf, &size, nil, 0)
        name = String(decoding: buf.prefix { $0 != 0 }, as: UTF8.self)
    }
    let v = ProcessInfo.processInfo.operatingSystemVersion
    let os = v.patchVersion == 0 ? "macOS \(v.majorVersion).\(v.minorVersion)" : "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    return Device(name: name, os: os, memoryBytes: Int64(ProcessInfo.processInfo.physicalMemory))
}
