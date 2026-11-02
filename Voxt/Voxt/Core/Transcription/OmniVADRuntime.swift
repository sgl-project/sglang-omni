// OmniVADRuntime.swift
// Dynamic OmniVAD-Kit bridge and streaming VAD backend.

import Foundation
import Darwin

private nonisolated let omniOK: CInt = 0
private nonisolated let omniErrNoFrames: CInt = -7

private nonisolated struct OmniSegment {
    var start: Float
    var end: Float
}

private nonisolated struct OmniPostConfig {
    var threshold: Float
    var smoothWindowSize: CInt
    var minSpeechFrames: CInt
    var minSilenceFrames: CInt
    var maxSpeechFrames: CInt
    var mergeSilenceFrames: CInt
    var extendSpeechFrames: CInt
}

private nonisolated struct OmniStreamVADConfig {
    var threshold: Float
    var smoothWindowSize: CInt
    var padStartFrame: CInt
    var minSpeechFrame: CInt
    var maxSpeechFrame: CInt
    var minSilenceFrame: CInt
}

private nonisolated struct OmniStreamVADResult {
    var confidence: Float
    var smoothedProbability: Float
    var isSpeech: CBool
    var isSpeechStart: CBool
    var isSpeechEnd: CBool
    var frameIndex: CInt
    var speechStartFrame: CInt
    var speechEndFrame: CInt
}

enum OmniVADError: LocalizedError, Equatable {
    case libraryNotFound
    case libraryLoadFailed(String)
    case symbolNotFound(String)
    case modelNotFound(String)
    case handleCreateFailed(Int32)
    case handleCloneFailed(Int32)
    case processFailed(Int32)
    case batchHandleCreateFailed(Int32)
    case batchDetectFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .libraryNotFound:
            return "OmniVAD runtime library was not found."
        case .libraryLoadFailed(let detail):
            return "OmniVAD runtime library could not be loaded. detail=\(detail)"
        case .symbolNotFound(let symbol):
            return "OmniVAD runtime symbol was not found. symbol=\(symbol)"
        case .modelNotFound(let filename):
            return "OmniVAD model resource was not found. file=\(filename)"
        case .handleCreateFailed(let code):
            return "OmniVAD stream handle could not be created. code=\(code)"
        case .handleCloneFailed(let code):
            return "OmniVAD stream handle could not be cloned. code=\(code)"
        case .processFailed(let code):
            return "OmniVAD stream inference failed. code=\(code)"
        case .batchHandleCreateFailed(let code):
            return "OmniVAD batch handle could not be created. code=\(code)"
        case .batchDetectFailed(let code):
            return "OmniVAD batch inference failed. code=\(code)"
        }
    }
}

nonisolated enum OmniVADResourceLocator {
    static let modelSubdirectory = "OmniVAD"
    static let batchModelFilename = "vad.omnivad"
    static let streamModelFilename = "stream-vad.omnivad"
    static let dynamicLibraryFilename = "libomnivad.dylib"

    static func streamModelURL(bundle: Bundle = .main) throws -> URL {
        if let configured = ProcessInfo.processInfo.environment["VOXT_OMNIVAD_MODEL_PATH"],
           !configured.isEmpty {
            let url = URL(fileURLWithPath: configured)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        if let bundled = bundle.url(
            forResource: "stream-vad",
            withExtension: "omnivad",
            subdirectory: modelSubdirectory
        ) {
            return bundled
        }
        if let bundled = bundle.url(forResource: "stream-vad", withExtension: "omnivad") {
            return bundled
        }

        for directory in localDevelopmentModelDirectories() {
            let candidate = directory.appendingPathComponent(streamModelFilename)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        throw OmniVADError.modelNotFound(streamModelFilename)
    }

    static func batchModelURL(bundle: Bundle = .main) throws -> URL {
        if let configured = ProcessInfo.processInfo.environment["VOXT_OMNIVAD_BATCH_MODEL_PATH"],
           !configured.isEmpty {
            let url = URL(fileURLWithPath: configured)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        if let bundled = bundle.url(
            forResource: "vad",
            withExtension: "omnivad",
            subdirectory: modelSubdirectory
        ) {
            return bundled
        }
        if let bundled = bundle.url(forResource: "vad", withExtension: "omnivad") {
            return bundled
        }

        for directory in localDevelopmentModelDirectories() {
            let candidate = directory.appendingPathComponent(batchModelFilename)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        throw OmniVADError.modelNotFound(batchModelFilename)
    }

    static func dynamicLibraryURL(bundle: Bundle = .main) throws -> URL {
        if let configured = ProcessInfo.processInfo.environment["VOXT_OMNIVAD_LIBRARY_PATH"],
           !configured.isEmpty {
            let url = URL(fileURLWithPath: configured)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        let bundledCandidates = [
            bundle.privateFrameworksURL?.appendingPathComponent(dynamicLibraryFilename),
            bundle.resourceURL?.appendingPathComponent(dynamicLibraryFilename),
            bundle.bundleURL.appendingPathComponent("Contents/Frameworks/\(dynamicLibraryFilename)"),
            bundle.bundleURL.appendingPathComponent("Contents/Resources/\(dynamicLibraryFilename)")
        ].compactMap { $0 }

        for candidate in bundledCandidates where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        for directory in localDevelopmentLibraryDirectories() {
            let candidate = directory.appendingPathComponent(dynamicLibraryFilename)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        throw OmniVADError.libraryNotFound
    }

    private static func localDevelopmentModelDirectories() -> [URL] {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return [
            cwd.appendingPathComponent("Voxt/Resources/OmniVAD", isDirectory: true),
            cwd.appendingPathComponent("tmp/OmniVAD-Kit/models", isDirectory: true)
        ]
    }

    private static func localDevelopmentLibraryDirectories() -> [URL] {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return [
            cwd.appendingPathComponent("Voxt/Frameworks", isDirectory: true),
            cwd.appendingPathComponent("tmp/OmniVAD-Kit/build-voxt", isDirectory: true),
            cwd.appendingPathComponent("tmp/OmniVAD-Kit/build-voxt/Release", isDirectory: true)
        ]
    }
}

nonisolated final class OmniVADDynamicLibrary: @unchecked Sendable {
    typealias CreateFunction = @convention(c) (
        UnsafePointer<CChar>?,
        UnsafeRawPointer?,
        UnsafeMutablePointer<CInt>?
    ) -> OpaquePointer?
    typealias CloneFunction = @convention(c) (
        OpaquePointer?,
        UnsafeMutablePointer<CInt>?
    ) -> OpaquePointer?
    typealias ProcessFunction = @convention(c) (
        OpaquePointer?,
        UnsafePointer<Float>?,
        CInt,
        UnsafeMutableRawPointer?
    ) -> CInt
    typealias ResetFunction = @convention(c) (OpaquePointer?) -> Void
    typealias DestroyFunction = @convention(c) (OpaquePointer?) -> Void
    typealias BatchCreateFunction = @convention(c) (
        UnsafePointer<CChar>?,
        UnsafeMutablePointer<CInt>?
    ) -> OpaquePointer?
    typealias BatchDetectFunction = @convention(c) (
        OpaquePointer?,
        UnsafePointer<Float>?,
        CInt,
        UnsafeRawPointer?,
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
        UnsafeMutablePointer<CInt>?
    ) -> CInt
    typealias BatchDestroyFunction = @convention(c) (OpaquePointer?) -> Void
    typealias FreeFunction = @convention(c) (UnsafeMutableRawPointer?) -> Void

    private static let lock = NSLock()
    private static var cached: Result<OmniVADDynamicLibrary, OmniVADError>?

    let url: URL
    let create: CreateFunction
    let clone: CloneFunction
    let process: ProcessFunction
    let reset: ResetFunction
    let destroy: DestroyFunction
    let batchCreate: BatchCreateFunction
    let batchDetect: BatchDetectFunction
    let batchDestroy: BatchDestroyFunction
    let free: FreeFunction

    private let handle: UnsafeMutableRawPointer

    static func shared() throws -> OmniVADDynamicLibrary {
        lock.lock()
        defer { lock.unlock() }
        if let cached {
            return try cached.get()
        }
        do {
            let library = try OmniVADDynamicLibrary(url: OmniVADResourceLocator.dynamicLibraryURL())
            cached = .success(library)
            return library
        } catch let error as OmniVADError {
            cached = .failure(error)
            throw error
        } catch {
            let wrapped = OmniVADError.libraryLoadFailed(error.localizedDescription)
            cached = .failure(wrapped)
            throw wrapped
        }
    }

    private init(url: URL) throws {
        self.url = url
        guard let loadedHandle = dlopen(url.path, RTLD_NOW | RTLD_LOCAL) else {
            throw OmniVADError.libraryLoadFailed(Self.currentDynamicLoaderError())
        }
        handle = loadedHandle

        do {
            create = try Self.loadSymbol("omni_stream_vad_create", from: loadedHandle)
            clone = try Self.loadSymbol("omni_stream_vad_clone", from: loadedHandle)
            process = try Self.loadSymbol("omni_stream_vad_process", from: loadedHandle)
            reset = try Self.loadSymbol("omni_stream_vad_reset", from: loadedHandle)
            destroy = try Self.loadSymbol("omni_stream_vad_destroy", from: loadedHandle)
            batchCreate = try Self.loadSymbol("omni_vad_create", from: loadedHandle)
            batchDetect = try Self.loadSymbol("omni_vad_detect", from: loadedHandle)
            batchDestroy = try Self.loadSymbol("omni_vad_destroy", from: loadedHandle)
            free = try Self.loadSymbol("omni_free", from: loadedHandle)
        } catch {
            dlclose(loadedHandle)
            throw error
        }
    }

    deinit {
        dlclose(handle)
    }

    private static func loadSymbol<T>(_ name: String, from handle: UnsafeMutableRawPointer) throws -> T {
        guard let symbol = dlsym(handle, name) else {
            throw OmniVADError.symbolNotFound(name)
        }
        return unsafeBitCast(symbol, to: T.self)
    }

    private static func currentDynamicLoaderError() -> String {
        guard let error = dlerror() else { return "unknown" }
        return String(cString: error)
    }
}

private nonisolated final class OmniStreamVADHandleBox {
    let pointer: OpaquePointer

    init(pointer: OpaquePointer) {
        self.pointer = pointer
    }
}

private nonisolated final class OmniBatchVADHandleBox {
    let pointer: OpaquePointer

    init(pointer: OpaquePointer) {
        self.pointer = pointer
    }
}

actor OmniStreamVoiceActivityBackend: ASRVoiceActivityBackend {
    let kind: ASRVoiceActivityBackendKind = .omniStream

    private static let sampleRate = 16_000
    private static let chunkSize = 160

    private let useCase: ASRVoiceActivityUseCase
    private let libraryProvider: @Sendable () throws -> OmniVADDynamicLibrary
    private let modelProvider: @Sendable () throws -> URL

    private var library: OmniVADDynamicLibrary?
    private var baseHandle: OmniStreamVADHandleBox?
    private var handles: [String: OmniStreamVADHandleBox] = [:]
    private var pendingSamples: [String: [Float]] = [:]
    private var pendingSampleOffsets: [String: Int] = [:]
    private var lastDecisions: [String: ASRVoiceActivityFrameDecision] = [:]

    init(
        useCase: ASRVoiceActivityUseCase,
        libraryProvider: @escaping @Sendable () throws -> OmniVADDynamicLibrary = { try OmniVADDynamicLibrary.shared() },
        modelProvider: @escaping @Sendable () throws -> URL = { try OmniVADResourceLocator.streamModelURL() }
    ) {
        self.useCase = useCase
        self.libraryProvider = libraryProvider
        self.modelProvider = modelProvider
    }

    deinit {
        guard let library else { return }
        for handle in handles.values {
            library.destroy(handle.pointer)
        }
        if let baseHandle {
            library.destroy(baseHandle.pointer)
        }
    }

    func reset() {
        guard let library else {
            pendingSamples.removeAll()
            pendingSampleOffsets.removeAll()
            lastDecisions.removeAll()
            return
        }
        if let baseHandle {
            library.reset(baseHandle.pointer)
        }
        for handle in handles.values {
            library.reset(handle.pointer)
        }
        pendingSamples.removeAll()
        pendingSampleOffsets.removeAll()
        lastDecisions.removeAll()
    }

    func releaseResources() {
        if let library {
            for handle in handles.values {
                library.destroy(handle.pointer)
            }
            if let baseHandle {
                library.destroy(baseHandle.pointer)
            }
        }
        handles.removeAll()
        baseHandle = nil
        library = nil
        pendingSamples.removeAll()
        pendingSampleOffsets.removeAll()
        lastDecisions.removeAll()
    }

    func decision(for frame: ASRVoiceActivityAudioFrame) async throws -> ASRVoiceActivityFrameDecision {
        try await decision(for: frame, streamID: "default") ?? ASRVoiceActivityFrameDecision(
            startSeconds: frame.startSeconds,
            endSeconds: frame.endSeconds,
            isSpeech: lastDecisions["default"]?.isSpeech ?? false,
            probability: lastDecisions["default"]?.probability
        )
    }

    func decision(
        for frame: ASRVoiceActivityAudioFrame,
        streamID: String
    ) async throws -> ASRVoiceActivityFrameDecision? {
        guard !frame.samples.isEmpty else { return nil }
        guard frame.sampleRate.isFinite, frame.sampleRate > 0 else { return nil }

        let handle = try handle(for: streamID)
        let prepared = ASRVoiceActivitySampleRateConverter.resample(
            samples: frame.samples,
            from: frame.sampleRate,
            to: Double(Self.sampleRate)
        )
        guard !prepared.isEmpty else { return lastDecisions[streamID] }

        var pending = pendingSamples[streamID] ?? []
        pending.append(contentsOf: prepared)
        var pendingOffset = min(pendingSampleOffsets[streamID] ?? 0, pending.count)

        var latestDecision: ASRVoiceActivityFrameDecision?
        while pending.count - pendingOffset >= Self.chunkSize {
            let endOffset = pendingOffset + Self.chunkSize
            let chunk = Array(pending[pendingOffset..<endOffset])
            pendingOffset = endOffset
            let result = try process(chunk: chunk, handle: handle)
            latestDecision = ASRVoiceActivityFrameDecision(
                startSeconds: frame.startSeconds,
                endSeconds: frame.endSeconds,
                isSpeech: result.isSpeech,
                probability: result.probability
            )
        }

        Self.compactPendingSamples(&pending, offset: &pendingOffset)
        pendingSamples[streamID] = pending
        pendingSampleOffsets[streamID] = pendingOffset
        if let latestDecision {
            lastDecisions[streamID] = latestDecision
            return latestDecision
        }
        return lastDecisions[streamID]
    }

    private nonisolated static func compactPendingSamples(
        _ samples: inout [Float],
        offset: inout Int
    ) {
        if offset == samples.count {
            samples.removeAll(keepingCapacity: true)
            offset = 0
        } else if offset >= chunkSize * 8 {
            samples.removeFirst(offset)
            offset = 0
        }
    }

    private func handle(for streamID: String) throws -> OmniStreamVADHandleBox {
        if streamID == "default", let baseHandle {
            return baseHandle
        }
        if let handle = handles[streamID] {
            return handle
        }

        let library = try loadedLibrary()
        if baseHandle == nil {
            baseHandle = try createBaseHandle(library: library)
        }
        guard let baseHandle else {
            throw OmniVADError.handleCreateFailed(-1)
        }
        if streamID == "default" {
            return baseHandle
        }

        var errorCode: CInt = omniOK
        guard let cloned = library.clone(baseHandle.pointer, &errorCode) else {
            throw OmniVADError.handleCloneFailed(errorCode)
        }
        let boxed = OmniStreamVADHandleBox(pointer: cloned)
        handles[streamID] = boxed
        return boxed
    }

    private func loadedLibrary() throws -> OmniVADDynamicLibrary {
        if let library {
            return library
        }
        let loaded = try libraryProvider()
        library = loaded
        return loaded
    }

    private func createBaseHandle(library: OmniVADDynamicLibrary) throws -> OmniStreamVADHandleBox {
        var config = streamConfig(useCase: useCase)
        var errorCode: CInt = omniOK
        let modelURL = try modelProvider()
        let handle = modelURL.path.withCString { path in
            withUnsafePointer(to: &config) { configPointer in
                library.create(path, UnsafeRawPointer(configPointer), &errorCode)
            }
        }
        guard let handle else {
            throw OmniVADError.handleCreateFailed(errorCode)
        }
        return OmniStreamVADHandleBox(pointer: handle)
    }

    private func process(
        chunk: [Float],
        handle: OmniStreamVADHandleBox
    ) throws -> (isSpeech: Bool, probability: Float?) {
        var result = OmniStreamVADResult(
            confidence: 0,
            smoothedProbability: 0,
            isSpeech: false,
            isSpeechStart: false,
            isSpeechEnd: false,
            frameIndex: 0,
            speechStartFrame: -1,
            speechEndFrame: -1
        )
        let library = try loadedLibrary()
        let status = chunk.withUnsafeBufferPointer { buffer -> CInt in
            withUnsafeMutablePointer(to: &result) { resultPointer in
                library.process(
                    handle.pointer,
                    buffer.baseAddress,
                    CInt(buffer.count),
                    UnsafeMutableRawPointer(resultPointer)
                )
            }
        }
        if status == omniErrNoFrames {
            return (false, nil)
        }
        guard status == omniOK else {
            throw OmniVADError.processFailed(status)
        }
        let probability = result.smoothedProbability.isFinite
            ? max(0, min(result.smoothedProbability, 1))
            : max(0, min(result.confidence, 1))
        return (Bool(result.isSpeech), probability)
    }

    private func streamConfig(useCase: ASRVoiceActivityUseCase) -> OmniStreamVADConfig {
        let profile = ASRVoiceActivityConfiguration.profile(for: useCase)
        var config = OmniStreamVADConfig(
            threshold: 0.5,
            smoothWindowSize: 5,
            padStartFrame: 5,
            minSpeechFrame: 8,
            maxSpeechFrame: 2_000,
            minSilenceFrame: 20
        )
        config.threshold = profile.onsetProbabilityThreshold
        config.minSpeechFrame = max(1, CInt((profile.minSpeechSeconds / 0.01).rounded(.up)))
        config.minSilenceFrame = max(1, CInt((profile.minSilenceSeconds / 0.01).rounded(.up)))
        config.padStartFrame = max(config.smoothWindowSize, CInt((profile.speechPadSeconds / 0.01).rounded(.up)))
        if let maxSegmentSeconds = profile.maxSegmentSeconds {
            config.maxSpeechFrame = max(1, CInt((maxSegmentSeconds / 0.01).rounded(.down)))
        }
        return config
    }
}

actor OmniOfflineVoiceActivityBackend: ASROfflineVoiceActivityBackend {
    nonisolated private static let sampleRate = 16_000

    private let useCase: ASRVoiceActivityUseCase
    private let libraryProvider: @Sendable () throws -> OmniVADDynamicLibrary
    private let modelProvider: @Sendable () throws -> URL

    private var library: OmniVADDynamicLibrary?
    private var handle: OmniBatchVADHandleBox?

    init(
        useCase: ASRVoiceActivityUseCase,
        libraryProvider: @escaping @Sendable () throws -> OmniVADDynamicLibrary = { try OmniVADDynamicLibrary.shared() },
        modelProvider: @escaping @Sendable () throws -> URL = { try OmniVADResourceLocator.batchModelURL() }
    ) {
        self.useCase = useCase
        self.libraryProvider = libraryProvider
        self.modelProvider = modelProvider
    }

    deinit {
        guard let library, let handle else { return }
        library.batchDestroy(handle.pointer)
    }

    func releaseResources() {
        if let library, let handle {
            library.batchDestroy(handle.pointer)
        }
        handle = nil
        library = nil
    }

    func speechRanges(samples: [Float], sampleRate: Double) async throws -> [ASROfflineSpeechRange] {
        guard !samples.isEmpty, sampleRate.isFinite, sampleRate > 0 else { return [] }
        let prepared = ASRVoiceActivitySampleRateConverter.resample(
            samples: samples,
            from: sampleRate,
            to: Double(Self.sampleRate)
        )
        guard !prepared.isEmpty else { return [] }

        let library = try loadedLibrary()
        let handle = try loadedHandle(library: library)
        var config = postConfig(useCase: useCase)
        var rawSegments: UnsafeMutableRawPointer?
        var segmentCount: CInt = 0
        let status = prepared.withUnsafeBufferPointer { samplesPointer in
            withUnsafePointer(to: &config) { configPointer in
                library.batchDetect(
                    handle.pointer,
                    samplesPointer.baseAddress,
                    CInt(clamping: samplesPointer.count),
                    UnsafeRawPointer(configPointer),
                    &rawSegments,
                    &segmentCount
                )
            }
        }
        defer {
            if let rawSegments {
                library.free(rawSegments)
            }
        }
        if status == omniErrNoFrames {
            return []
        }
        guard status == omniOK else {
            throw OmniVADError.batchDetectFailed(status)
        }
        guard segmentCount > 0, let rawSegments else { return [] }

        let durationSeconds = Double(prepared.count) / Double(Self.sampleRate)
        let segments = rawSegments
            .bindMemory(to: OmniSegment.self, capacity: Int(segmentCount))
        return (0..<Int(segmentCount)).compactMap { index in
            let segment = segments[index]
            let start = max(0, min(Double(segment.start), durationSeconds))
            let end = max(start, min(Double(segment.end), durationSeconds))
            guard end > start else { return nil }
            return ASROfflineSpeechRange(startSeconds: start, endSeconds: end)
        }
    }

    private func loadedLibrary() throws -> OmniVADDynamicLibrary {
        if let library {
            return library
        }
        let loaded = try libraryProvider()
        library = loaded
        return loaded
    }

    private func loadedHandle(library: OmniVADDynamicLibrary) throws -> OmniBatchVADHandleBox {
        if let handle {
            return handle
        }
        var errorCode: CInt = omniOK
        let modelURL = try modelProvider()
        let pointer = modelURL.path.withCString { path in
            library.batchCreate(path, &errorCode)
        }
        guard let pointer else {
            throw OmniVADError.batchHandleCreateFailed(errorCode)
        }
        let loaded = OmniBatchVADHandleBox(pointer: pointer)
        handle = loaded
        return loaded
    }

    private func postConfig(useCase: ASRVoiceActivityUseCase) -> OmniPostConfig {
        let profile = ASRVoiceActivityConfiguration.profile(for: useCase)
        return OmniPostConfig(
            threshold: profile.onsetProbabilityThreshold,
            smoothWindowSize: 5,
            minSpeechFrames: max(1, CInt((profile.minSpeechSeconds / 0.01).rounded(.up))),
            minSilenceFrames: max(1, CInt((profile.minSilenceSeconds / 0.01).rounded(.up))),
            maxSpeechFrames: max(
                1,
                CInt(((profile.maxSegmentSeconds ?? 30) / 0.01).rounded(.down))
            ),
            mergeSilenceFrames: 0,
            extendSpeechFrames: max(0, CInt((profile.speechPadSeconds / 0.01).rounded(.up)))
        )
    }
}
