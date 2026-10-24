import AVKit
import VoxtMobileKit
import AVFoundation
import Combine
import CoreGraphics
import CoreMedia
import CoreVideo
import UIKit

/// The floating Picture in Picture dictation window.
///
/// Apple's only public floating-window API is video PiP, so the window shows a status
/// "video" rendered into an `AVSampleBufferDisplayLayer` (preferred over the
/// video-call-only content source, which Apple documents as valid for the duration of a
/// call). No camera entitlement, no silent audio, no private API.
///
/// Wire the callbacks to the dictation controller:
/// `onDidStart` → `controller.setArmed(true)`, `onDidStop`/`onFailure` → `setArmed(false)`.
///
/// Unverified until a physical iPhone run: whether the system accepts a non-video status
/// window, and whether PiP with an inactive audio session keeps the process scheduled to
/// receive the keyboard's command. If it does not, the legacy outlined-mic app switch is
/// the fallback and stays intact.
@MainActor
public final class FloatingDictationWindow: NSObject, ObservableObject {
    public enum Status: String, Sendable {
        case ready = "Tap the mic to speak"
        case listening = "Listening…"
        case processing = "Writing it up…"
        case failed = "Something went wrong"
    }

    @Published public private(set) var isAvailable: Bool
    @Published public private(set) var isActive = false
    @Published public private(set) var lastError: String?
    @Published public private(set) var diagnostics = ""

    struct Variant: OptionSet, Equatable {
        let rawValue: Int
        static let timebase = Variant(rawValue: 1)
        static let audioSession = Variant(rawValue: 2)
        /// IOSurface-backed frames, enqueued on start and steadily while the window is open.
        static let surface = Variant(rawValue: 4)
        /// The playback delegate reports a live stream, -infinity to +infinity.
        static let liveRange = Variant(rawValue: 8)
        /// The window's audio session uses the microphone's category, so recording never
        /// switches the category of an active session.
        static let recordSession = Variant(rawValue: 16)
        /// `.playback` / `.moviePlayback` without mixing, as media apps use.
        static let exclusiveAudio = Variant(rawValue: 32)
        /// A finite time range, zero to 24 hours (instead of an open-ended or live range).
        static let finiteRange = Variant(rawValue: 64)
        /// startPictureInPicture() on the next main-queue turn, after the controller exists
        /// and a frame is queued.
        static let deferredStart = Variant(rawValue: 128)
        /// requiresLinearPlayback off.
        static let nonLinear = Variant(rawValue: 256)
        static let all: Variant = [.timebase, .audioSession, .surface, .liveRange]
        private static let named: [(String, Variant)] = [
            ("none", []), ("timebase", [.timebase]), ("session", [.audioSession]),
            ("both", [.timebase, .audioSession]), ("surface", [.timebase, .audioSession, .surface]),
            ("liverange", [.timebase, .audioSession, .liveRange]), ("all", all),
            ("all-record", all.union(.recordSession)),
            ("exclusive", all.union(.exclusiveAudio)),
            ("finite", [.timebase, .audioSession, .surface, .finiteRange]),
            ("deferred", all.union(.deferredStart)),
            ("nonlinear", all.union(.nonLinear)),
            ("uipip", [.timebase, .audioSession, .surface, .exclusiveAudio, .finiteRange, .deferredStart, .nonLinear]),
        ]
        init(rawValue: Int) { self.rawValue = rawValue }
        /// Conditions that can keep PiP from opening on an iPhone, selectable in Debug builds
        /// with VOXT_PIP_VARIANT so one device run can tell which is needed. Without the audio
        /// session PiP never becomes possible; with it, only a start on the next main-queue
        /// turn opens the window (and it stays open after Home). Exclusive audio, a finite
        /// range and non-linear playback make no difference.
        static let shipping: Variant = all.union(.deferredStart)

        init(launchValue: String?) {
            self = Self.named.first { $0.0 == launchValue }?.1 ?? Self.shipping
        }
        var name: String { Self.named.first { $0.1 == self }?.0 ?? "custom(\(rawValue))" }
    }

    public var onDidStart: (() -> Void)?
    public var onDidStop: (() -> Void)?
    public var onFailure: ((String) -> Void)?

    /// The PiP source: whatever this layer renders is what the floating window shows. It must
    /// stay in the window while PiP runs, so it is hosted at the app root (`PiPSourceHost`),
    /// never in a sheet: dismissing Settings would close the window.
    public let displayLayer = AVSampleBufferDisplayLayer()
    /// The same frames for the Settings card's preview.
    public let previewLayer = AVSampleBufferDisplayLayer()
    /// Public playback delegate required by the sample-buffer content source.
    public let playbackDelegate: SampleBufferPlaybackDelegate

    private var pipController: PictureInPictureControlling?
    private let variant: Variant
    private let prepareAudioSession: () throws -> Void
    private var sessionState = "-"
    private var startedAt = Date()
    private let startTimeout: TimeInterval
    private var startAttempt = UUID()
    private var startNote = ""
    private let renderer: StatusRenderer
    private let frameInterval: TimeInterval
    private var frameTask: Task<Void, Never>?

    var framesEnqueued: Int { renderer.framesEnqueued }
    var previewFramesEnqueued: Int { renderer.previewFramesEnqueued }
    var lastFrameUsesIOSurface: Bool { renderer.lastFrameUsesIOSurface }

    init(available: Bool = AVPictureInPictureController.isPictureInPictureSupported(),
         controller: PictureInPictureControlling? = nil, startTimeout: TimeInterval = 3,
         variant: Variant = [], frameInterval: TimeInterval = 0.5,
         prepareAudioSession: @escaping () throws -> Void = {}) {
        isAvailable = available
        pipController = controller
        self.startTimeout = startTimeout
        self.variant = variant
        self.frameInterval = frameInterval
        self.prepareAudioSession = prepareAudioSession
        renderer = StatusRenderer(ioSurface: variant.contains(.surface))
        playbackDelegate = SampleBufferPlaybackDelegate(range: variant.contains(.finiteRange) ? .finite
                                                        : variant.contains(.liveRange) ? .live : .openEnded)
        super.init()
        displayLayer.videoGravity = .resizeAspect
        previewLayer.videoGravity = .resizeAspect
        renderer.preview = previewLayer
        if variant.contains(.timebase) {
            // A sample-buffer PiP source plays against the layer's timebase; without one
            // the "video" never runs. Frames carry host-clock timestamps, so the timebase
            // starts at host time (starting it at zero left every frame hours in the future).
            var timebase: CMTimebase?
            CMTimebaseCreateWithSourceClock(allocator: kCFAllocatorDefault, sourceClock: CMClockGetHostTimeClock(), timebaseOut: &timebase)
            if let timebase {
                CMTimebaseSetTime(timebase, time: CMClockGetTime(CMClockGetHostTimeClock()))
                CMTimebaseSetRate(timebase, rate: 1)
                displayLayer.controlTimebase = timebase
            }
        }
        renderer.show(.ready, on: displayLayer)
        updateDiagnostics()
    }

    public convenience override init() {
        #if DEBUG
        let variant = Variant(launchValue: ProcessInfo.processInfo.environment["VOXT_PIP_VARIANT"])
        #else
        let variant = Variant.shipping
        #endif
        self.init(available: AVPictureInPictureController.isPictureInPictureSupported(), variant: variant,
                  prepareAudioSession: { try Self.activateSession(for: variant) })
    }

    static func sessionCategory(for variant: Variant) -> AVAudioSession.Category {
        variant.contains(.recordSession) ? .playAndRecord : .playback
    }

    static func sessionSettings(for variant: Variant)
        -> (category: AVAudioSession.Category, mode: AVAudioSession.Mode, options: AVAudioSession.CategoryOptions) {
        if variant.contains(.exclusiveAudio) { return (.playback, .moviePlayback, []) }
        return (sessionCategory(for: variant), .default, [.mixWithOthers])
    }

    static func requiresLinearPlayback(for variant: Variant) -> Bool { !variant.contains(.nonLinear) }

    /// Picture in Picture expects a playback-capable session; it plays nothing. With
    /// `.playback`, the microphone session switches the category when recording starts;
    /// `.playAndRecord` uses the microphone's own options. Exclusive audio stops other
    /// apps' playback, so it is a diagnostic until measured.
    private static func activateSession(for variant: Variant) throws {
        let session = AVAudioSession.sharedInstance()
        let settings = sessionSettings(for: variant)
        if settings.category == .playAndRecord {
            try MicrophoneCapture.configure(session)
        } else {
            try session.setCategory(settings.category, mode: settings.mode, options: settings.options)
            try session.setActive(true, options: [])
        }
    }

    private func updateDiagnostics() {
        let possible = pipController.map { String($0.isPictureInPicturePossible) } ?? "-"
        let renderer = displayLayer.sampleBufferRenderer
        let layer = switch renderer.status {
        case .rendering: "rendering"
        case .failed: "failed(\((renderer.error as NSError?)?.code ?? 0))"
        default: "unknown"
        }
        diagnostics = "variant=\(variant.name) session=\(sessionState) possible=\(possible) active=\(isActive)"
            + " layer=\(layer) frames=\(renderer.hasSufficientMediaDataForReliablePlaybackStart ? "ready" : "short"):\(framesEnqueued)"
            + startNote + " t=\(String(format: "%.1f", Date().timeIntervalSince(startedAt)))"
    }

    /// Re-sends the current status frame at a steady rate while the window is starting or
    /// open, so the system always has a recent frame to show.
    private func startSendingFrames() {
        guard variant.contains(.surface), frameTask == nil else { return }
        renderer.refresh(on: displayLayer)
        let interval = frameInterval
        frameTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(interval))
                guard let self, !Task.isCancelled else { return }
                self.renderer.refresh(on: self.displayLayer)
                self.updateDiagnostics()
            }
        }
    }

    private func stopSendingFrames() {
        frameTask?.cancel()
        frameTask = nil
    }

    /// Starts PiP. Reports through `onFailure` (and does not throw) when unsupported or
    /// when the system refuses the window, so callers can fall back to the app switch.
    public func start() {
        guard isAvailable else {
            fail("Picture in Picture is not supported on this device.")
            return
        }
        startedAt = Date()
        if variant.contains(.audioSession) {
            do {
                try prepareAudioSession()
                sessionState = "ok"
            } catch {
                sessionState = error.localizedDescription
                updateDiagnostics()
                fail("Couldn't open the floating window. iOS didn't allow Picture in Picture.")
                return
            }
        }
        if pipController == nil {
            let source = AVPictureInPictureController.ContentSource(
                sampleBufferDisplayLayer: displayLayer,
                playbackDelegate: playbackDelegate
            )
            let controller = AVPictureInPictureController(contentSource: source)
            controller.requiresLinearPlayback = Self.requiresLinearPlayback(for: variant)
            controller.delegate = self
            pipController = PictureInPictureAdapter(controller)
        }
        guard let pipController else { return }
        lastError = nil
        startNote = ""
        startSendingFrames()
        updateDiagnostics()
        if pipController.isPictureInPicturePossible {
            // With possible=true, iOS can ignore this call without calling either delegate
            // method. Say so after the timeout instead of staying silent.
            let attempt = UUID()
            startAttempt = attempt
            if variant.contains(.deferredStart) {
                DispatchQueue.main.async { [weak self] in
                    MainActor.assumeIsolated {
                        guard let self, self.startAttempt == attempt, let controller = self.pipController,
                              controller.isPictureInPicturePossible else { return }
                        controller.startPictureInPicture()
                    }
                }
            } else {
                pipController.startPictureInPicture()
            }
            reportIfStartIsIgnored(attempt, note: " start=ignored")
            return
        }
        // startPictureInPicture() does nothing while PiP isn't possible yet: no window and no
        // error. Start once it becomes possible,
        // otherwise say so instead of staying silent.
        let attempt = UUID()
        startAttempt = attempt
        pipController.observePossible { [weak self] in
            self?.updateDiagnostics()
            guard let self, self.startAttempt == attempt, let controller = self.pipController,
                  controller.isPictureInPicturePossible else { return }
            let started = UUID()
            self.startAttempt = started
            controller.startPictureInPicture()
            self.reportIfStartIsIgnored(started, note: " start=ignored")
        }
        reportIfStartIsIgnored(attempt, note: " start=never-possible")
    }

    private func reportIfStartIsIgnored(_ attempt: UUID, note: String) {
        let timeout = startTimeout
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(timeout))
            guard let self, self.startAttempt == attempt, !self.isActive else { return }
            self.startAttempt = UUID()
            self.startNote = note
            self.updateDiagnostics()
            self.fail("Couldn't open the floating window. iOS didn't allow Picture in Picture.")
        }
    }

    func handleDidStart() {
        startAttempt = UUID()
        lastError = nil
        isActive = true
        startNote = ""
        updateDiagnostics()
        onDidStart?()
    }

    func handleDidStop() {
        isActive = false
        stopSendingFrames()
        updateDiagnostics()
        onDidStop?()
    }

    public func stop() {
        pipController?.stopPictureInPicture()
    }

    /// Updates the status frame shown in the floating window.
    public func show(_ status: Status) {
        renderer.show(status, on: displayLayer)
    }

    private func fail(_ message: String) {
        lastError = message
        isActive = false
        stopSendingFrames()
        onFailure?(message)
    }
}

extension FloatingDictationWindow: AVPictureInPictureControllerDelegate {
    public nonisolated func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        failedToStartPictureInPictureWithError error: Error
    ) {
        Task { @MainActor in
            self.startAttempt = UUID()
            self.startNote = " start=failed(\((error as NSError).domain) \((error as NSError).code))"
            self.updateDiagnostics()
            self.fail(error.localizedDescription)
        }
    }

    public nonisolated func pictureInPictureControllerDidStartPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        Task { @MainActor in self.handleDidStart() }
    }

    public nonisolated func pictureInPictureControllerDidStopPictureInPicture(
        _ pictureInPictureController: AVPictureInPictureController
    ) {
        Task { @MainActor in self.handleDidStop() }
    }

    public nonisolated func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        restoreUserInterfaceForPictureInPictureStopWithCompletionHandler completionHandler: @escaping (Bool) -> Void
    ) {
        // No inline player UI to restore; the status layer lives only in the PiP window.
        completionHandler(true)
    }
}

/// Playback delegate for the sample-buffer content source, modeled as a live stream:
/// the time range starts at zero with an indefinite duration, and the stream reports
/// itself playing unless the system pauses it. `requiresLinearPlayback` hides the
/// seek/skip UI, so the skip callback only satisfies the protocol.
public final class SampleBufferPlaybackDelegate: NSObject, AVPictureInPictureSampleBufferPlaybackDelegate {
    private let lock = NSLock()
    private nonisolated(unsafe) var playing = true
    public enum Range: Sendable { case openEnded, live, finite }
    /// Open-ended: from zero, indefinite. Live: -infinity to +infinity. Finite: zero to 24 h.
    public let timeRange: CMTimeRange

    public init(range: Range = .openEnded) {
        switch range {
        case .openEnded: timeRange = CMTimeRange(start: .zero, duration: .indefinite)
        case .live: timeRange = CMTimeRange(start: .negativeInfinity, duration: .positiveInfinity)
        case .finite: timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: 86_400, preferredTimescale: 1))
        }
    }

    public nonisolated func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        setPlaying playing: Bool
    ) {
        lock.lock()
        self.playing = playing
        lock.unlock()
    }

    public nonisolated func pictureInPictureControllerTimeRangeForPlayback(
        _ pictureInPictureController: AVPictureInPictureController
    ) -> CMTimeRange {
        // Live status stream: no seekable window.
        timeRange
    }

    public nonisolated func pictureInPictureControllerIsPlaybackPaused(
        _ pictureInPictureController: AVPictureInPictureController
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !playing
    }

    public nonisolated func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        didTransitionToRenderSize newRenderSize: CMVideoDimensions
    ) {}

    public nonisolated func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        skipByInterval skipInterval: CMTime,
        completion completionHandler: @escaping () -> Void
    ) {
        completionHandler()
    }
}

/// Draws the current dictation status into pixel buffers and enqueues them on the
/// display layer. Re-renders only when the status changes.
@MainActor
private final class StatusRenderer {
    private let size = CGSize(width: 640, height: 360)
    private let ioSurface: Bool
    private var lastStatus: FloatingDictationWindow.Status?
    private(set) var framesEnqueued = 0
    private(set) var previewFramesEnqueued = 0
    private(set) var lastFrameUsesIOSurface = false
    var preview: AVSampleBufferDisplayLayer?

    init(ioSurface: Bool) { self.ioSurface = ioSurface }

    func show(_ status: FloatingDictationWindow.Status, on layer: AVSampleBufferDisplayLayer) {
        guard status != lastStatus else { return }
        lastStatus = status
        enqueue(status, on: layer, flush: true)
        if let preview, let pixelBuffer = makePixelBuffer(), let sampleBuffer = makeSampleBuffer(from: pixelBuffer) {
            draw(VoxtL10n.shared.string(status.rawValue), into: pixelBuffer)
            preview.sampleBufferRenderer.flush()
            if preview.isReadyForMoreMediaData { preview.enqueue(sampleBuffer) }
            previewFramesEnqueued += 1
        }
    }

    /// Sends the current status again with a new timestamp.
    func refresh(on layer: AVSampleBufferDisplayLayer) {
        guard let lastStatus else { return }
        enqueue(lastStatus, on: layer, flush: false)
    }

    private func enqueue(_ status: FloatingDictationWindow.Status, on layer: AVSampleBufferDisplayLayer, flush: Bool) {
        guard let pixelBuffer = makePixelBuffer(),
              let sampleBuffer = makeSampleBuffer(from: pixelBuffer) else { return }
        draw(VoxtL10n.shared.string(status.rawValue), into: pixelBuffer)
        lastFrameUsesIOSurface = CVPixelBufferGetIOSurface(pixelBuffer) != nil
        framesEnqueued += 1
        if flush || layer.sampleBufferRenderer.status == .failed { layer.sampleBufferRenderer.flush() }
        if layer.isReadyForMoreMediaData {
            layer.enqueue(sampleBuffer)
        } else {
            // The readiness closure is @Sendable; both captures live on the main queue.
            struct SendableBox<T>: @unchecked Sendable { let value: T }
            let box = SendableBox(value: (layer, sampleBuffer))
            layer.requestMediaDataWhenReady(on: .main) {
                // Dispatched on the main queue, where the layer lives. The block fires
                // repeatedly until stopRequestingMediaData; stop once the frame lands.
                MainActor.assumeIsolated {
                    let (layer, sampleBuffer) = box.value
                    guard layer.isReadyForMoreMediaData else { return }
                    layer.enqueue(sampleBuffer)
                    layer.stopRequestingMediaData()
                }
            }
        }
    }

    private func makePixelBuffer() -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        var attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: Int(size.width),
            kCVPixelBufferHeightKey: Int(size.height),
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        // The display path on the device works on IOSurface-backed buffers.
        if ioSurface { attributes[kCVPixelBufferIOSurfacePropertiesKey] = [CFString: Any]() }
        guard CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                                  kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer) == kCVReturnSuccess
        else { return nil }
        return buffer
    }

    private func makeSampleBuffer(from pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var formatDescription: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer, formatDescriptionOut: &formatDescription
        ) == noErr, let formatDescription else { return nil }
        var timing = CMSampleTimingInfo(
            duration: .invalid,
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateForImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer, dataReady: true,
            makeDataReadyCallback: nil, refcon: nil, formatDescription: formatDescription,
            sampleTiming: &timing, sampleBufferOut: &sampleBuffer
        ) == noErr, let sampleBuffer else { return nil }
        // No controlTimebase drives this layer, so each status frame displays on arrival.
        CMSetAttachment(
            sampleBuffer as CMAttachmentBearer, key: kCMSampleAttachmentKey_DisplayImmediately,
            value: kCFBooleanTrue, attachmentMode: kCMAttachmentMode_ShouldNotPropagate
        )
        return sampleBuffer
    }

    private func draw(_ text: String, into pixelBuffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let space = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let context = CGContext(
            data: base, width: Int(size.width), height: Int(size.height),
            bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: space, bitmapInfo: bitmapInfo
        ) else { return }
        context.setFillColor(UIColor(red: 0.11, green: 0.11, blue: 0.12, alpha: 1).cgColor)
        context.fill(CGRect(origin: .zero, size: size))
        UIGraphicsPushContext(context)
        defer { UIGraphicsPopContext() }
        // CoreGraphics' y axis is flipped relative to UIKit drawing.
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1, y: -1)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 34, weight: .medium),
            .foregroundColor: UIColor.white,
        ]
        let attributed = NSAttributedString(string: text, attributes: attributes)
        let textSize = attributed.size()
        attributed.draw(at: CGPoint(x: (size.width - textSize.width) / 2,
                                    y: (size.height - textSize.height) / 2))
    }
}

/// The part of AVPictureInPictureController that starting the window depends on, so the
/// start sequence can be tested where the simulator reports PiP as unsupported.
@MainActor
protocol PictureInPictureControlling: AnyObject {
    var isPictureInPicturePossible: Bool { get }
    func startPictureInPicture()
    func stopPictureInPicture()
    /// Calls `handler` on the main actor whenever isPictureInPicturePossible changes.
    func observePossible(_ handler: @escaping @MainActor @Sendable () -> Void)
}

@MainActor
final class PictureInPictureAdapter: PictureInPictureControlling {
    private let controller: AVPictureInPictureController
    private var observation: NSKeyValueObservation?
    init(_ controller: AVPictureInPictureController) { self.controller = controller }
    var isPictureInPicturePossible: Bool { controller.isPictureInPicturePossible }
    func startPictureInPicture() { controller.startPictureInPicture() }
    func stopPictureInPicture() { controller.stopPictureInPicture() }
    func observePossible(_ handler: @escaping @MainActor @Sendable () -> Void) {
        observation = controller.observe(\.isPictureInPicturePossible, options: [.new]) { _, _ in
            Task { @MainActor in handler() }
        }
    }
}
