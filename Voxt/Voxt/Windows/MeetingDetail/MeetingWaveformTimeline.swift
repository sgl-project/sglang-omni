// MeetingWaveformTimeline.swift
// On-demand waveform data and interactive meeting audio timeline.

import AVFoundation
import AppKit
import SwiftUI

struct MeetingWaveformData: Equatable, Sendable {
    let samples: [Float]
    let duration: TimeInterval
}

actor MeetingWaveformCache {
    static let shared = MeetingWaveformCache()

    private let maxCachedSamples = 32_000
    private var values: [String: MeetingWaveformData] = [:]
    private var sampleCounts: [String: Int] = [:]
    private var accessOrder: [String] = []

    func value(for key: String) -> MeetingWaveformData? {
        guard let value = values[key] else { return nil }
        touch(key)
        return value
    }

    func insert(_ value: MeetingWaveformData, for key: String) {
        if let previousSampleCount = sampleCounts[key] {
            cachedSampleCount -= previousSampleCount
        }
        values[key] = value
        sampleCounts[key] = value.samples.count
        cachedSampleCount += value.samples.count
        touch(key)

        while cachedSampleCount > maxCachedSamples, let oldestKey = accessOrder.first {
            accessOrder.removeFirst()
            guard let removedSampleCount = sampleCounts.removeValue(forKey: oldestKey) else { continue }
            cachedSampleCount -= removedSampleCount
            values.removeValue(forKey: oldestKey)
        }
    }

    private var cachedSampleCount = 0

    private func touch(_ key: String) {
        if let index = accessOrder.firstIndex(of: key) {
            accessOrder.remove(at: index)
        }
        accessOrder.append(key)
    }
}

enum MeetingWaveformBuilder {
    private static let maxSampleCount = 4_096

    static func load(
        from url: URL,
        sampleCount: Int = 720
    ) async -> MeetingWaveformData? {
        let boundedSampleCount = min(max(sampleCount, 1), maxSampleCount)
        guard sampleCount > 0 else { return nil }

        let key = "\(url.standardizedFileURL.path)#\(boundedSampleCount)"
        if let cached = await MeetingWaveformCache.shared.value(for: key) {
            return cached
        }

        let data = await Task.detached(priority: .utility) {
            buildSynchronously(from: url, sampleCount: boundedSampleCount)
        }.value

        if let data {
            await MeetingWaveformCache.shared.insert(data, for: key)
        }
        return data
    }

    nonisolated private static func buildSynchronously(
        from url: URL,
        sampleCount: Int
    ) -> MeetingWaveformData? {
        guard sampleCount > 0,
              let file = try? AVAudioFile(forReading: url),
              file.length > 0,
              file.processingFormat.channelCount > 0
        else {
            return nil
        }

        let frameLength = AVAudioFramePosition(file.length)
        let format = file.processingFormat
        let bufferCapacity: AVAudioFrameCount = 4_096
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: bufferCapacity) else {
            return nil
        }

        var samples = Array(repeating: Float.zero, count: sampleCount)
        var frameOffset: AVAudioFramePosition = 0

        while frameOffset < frameLength {
            let framesToRead = AVAudioFrameCount(
                min(AVAudioFramePosition(bufferCapacity), frameLength - frameOffset)
            )
            buffer.frameLength = 0
            do {
                try file.read(into: buffer, frameCount: framesToRead)
            } catch {
                return nil
            }

            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0 else { break }
            guard let channelData = buffer.floatChannelData else { return nil }

            for frameIndex in 0..<frameCount {
                let absoluteFrame = frameOffset + AVAudioFramePosition(frameIndex)
                let sampleIndex = min(
                    sampleCount - 1,
                    max(0, Int((Double(absoluteFrame) / Double(frameLength)) * Double(sampleCount)))
                )

                var peak: Float = 0
                for channel in 0..<Int(format.channelCount) {
                    peak = max(peak, abs(channelData[channel][frameIndex]))
                }
                samples[sampleIndex] = max(samples[sampleIndex], peak)
            }

            frameOffset += AVAudioFramePosition(frameCount)
        }

        let maximum = samples.max() ?? 0
        if maximum > 0.0001 {
            samples = samples.map { min(max($0 / maximum, 0.04), 1) }
        } else {
            samples = samples.map { _ in 0.04 }
        }

        return MeetingWaveformData(
            samples: samples,
            duration: Double(frameLength) / format.sampleRate
        )
    }
}

enum MeetingWaveformTimelineSupport {
    static let minimumZoomScale: CGFloat = 1
    static let maximumZoomScale: CGFloat = 12

    static func clampedZoomScale(_ scale: CGFloat) -> CGFloat {
        min(max(scale, minimumZoomScale), maximumZoomScale)
    }

    static func visibleTimeRange(
        currentTime: TimeInterval,
        duration: TimeInterval,
        zoomScale: CGFloat
    ) -> MeetingWaveformTimeRange {
        guard duration > 0 else {
            return MeetingWaveformTimeRange(start: 0, end: 0)
        }

        let scale = TimeInterval(clampedZoomScale(zoomScale))
        let visibleDuration = duration / scale
        let centeredStart = currentTime - visibleDuration / 2
        let start = min(max(centeredStart, 0), max(duration - visibleDuration, 0))
        return MeetingWaveformTimeRange(start: start, end: start + visibleDuration)
    }

    static func time(
        forX x: CGFloat,
        width: CGFloat,
        duration: TimeInterval
    ) -> TimeInterval {
        guard width > 0, duration > 0 else { return 0 }
        return min(max(TimeInterval(x / width) * duration, 0), duration)
    }

    static func time(
        forX x: CGFloat,
        width: CGFloat,
        duration: TimeInterval,
        currentTime: TimeInterval,
        zoomScale: CGFloat
    ) -> TimeInterval {
        let range = visibleTimeRange(
            currentTime: currentTime,
            duration: duration,
            zoomScale: zoomScale
        )
        guard width > 0, range.end > range.start else { return range.start }
        let progress = min(max(x / width, 0), 1)
        return range.start + TimeInterval(progress) * range.duration
    }
}

struct MeetingWaveformTimeRange: Equatable, Sendable {
    let start: TimeInterval
    let end: TimeInterval

    var duration: TimeInterval {
        max(end - start, 0)
    }
}

struct MeetingWaveformTimeline: View {
    let data: MeetingWaveformData?
    let currentTime: TimeInterval
    let segments: [MeetingTranscriptSegment]
    let showsHighlightedSegments: Bool
    @Binding var zoomScale: CGFloat
    let onSeek: (TimeInterval) -> Void

    private let cornerRadius: CGFloat = 12
    @State private var magnificationStartScale: CGFloat?

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Color.primary.opacity(0.045))

                if let data {
                    Canvas { context, size in
                        drawTimeline(
                            context: &context,
                            size: size,
                            data: data
                        )
                    }
                    .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                } else {
                    ProgressView()
                        .controlSize(.small)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard let data else { return }
                        onSeek(time(
                            forX: value.location.x,
                            width: proxy.size.width,
                            data: data
                        ))
                    }
                    .onEnded { value in
                        guard let data else { return }
                        onSeek(time(
                            forX: value.location.x,
                            width: proxy.size.width,
                            data: data
                        ))
                    }
            )
            .simultaneousGesture(
                MagnificationGesture()
                    .onChanged { value in
                        let startScale = magnificationStartScale ?? zoomScale
                        if magnificationStartScale == nil {
                            magnificationStartScale = startScale
                        }
                        zoomScale = MeetingWaveformTimelineSupport.clampedZoomScale(startScale * value)
                    }
                    .onEnded { _ in
                        magnificationStartScale = nil
                    }
            )
            .background(
                MeetingWaveformScrollZoomCapture { factor in
                    zoomScale = MeetingWaveformTimelineSupport.clampedZoomScale(zoomScale * factor)
                }
                .allowsHitTesting(false)
            )
            .accessibilityElement()
            .accessibilityLabel(AppLocalization.localizedString("Meeting audio waveform"))
            .accessibilityValue(timeLabel)
        }
        .frame(height: 60)
    }

    private func drawTimeline(
        context: inout GraphicsContext,
        size: CGSize,
        data: MeetingWaveformData
    ) {
        guard !data.samples.isEmpty, data.duration > 0 else { return }

        let chartHeight = max(size.height - 12, 1)
        let visibleRange = MeetingWaveformTimelineSupport.visibleTimeRange(
            currentTime: currentTime,
            duration: data.duration,
            zoomScale: zoomScale
        )
        let visibleDuration = max(visibleRange.duration, 0.001)
        let intervals = speechIntervals(duration: data.duration)

        for interval in intervals {
            guard interval.end > visibleRange.start, interval.start < visibleRange.end else { continue }

            let visibleStart = max(interval.start, visibleRange.start)
            let visibleEnd = min(interval.end, visibleRange.end)
            let startX = CGFloat((visibleStart - visibleRange.start) / visibleDuration) * size.width
            let endX = CGFloat((visibleEnd - visibleRange.start) / visibleDuration) * size.width
            let color = interval.segment.speaker == .me
                ? Color(red: 0.16, green: 0.47, blue: 0.88)
                : Color(red: 0.12, green: 0.58, blue: 0.32)

            context.fill(
                Path(
                    roundedRect: CGRect(
                        x: startX,
                        y: size.height - 8,
                        width: max(endX - startX, 2),
                        height: 5
                    ),
                    cornerRadius: 2.5
                ),
                with: .color(color.opacity(0.72))
            )

            if showsHighlightedSegments, interval.segment.isHighlighted {
                context.fill(
                    Path(
                        roundedRect: CGRect(
                            x: startX,
                            y: 3,
                            width: max(endX - startX, 2),
                            height: chartHeight - 4
                        ),
                        cornerRadius: 5
                    ),
                    with: .color(Color.orange.opacity(0.16))
                )
            }

            let markerX = CGFloat((visibleStart - visibleRange.start) / visibleDuration) * size.width
            var marker = Path()
            marker.move(to: CGPoint(x: markerX, y: 4))
            marker.addLine(to: CGPoint(x: markerX, y: chartHeight - 1))
            context.stroke(marker, with: .color(color.opacity(0.62)), lineWidth: 1)
        }

        let firstSample = max(
            0,
            Int(floor(visibleRange.start / data.duration * Double(data.samples.count)))
        )
        let lastSample = min(
            data.samples.count,
            max(firstSample + 1, Int(ceil(visibleRange.end / data.duration * Double(data.samples.count))))
        )
        let visibleSampleCount = max(lastSample - firstSample, 1)
        let barWidth = max(size.width / CGFloat(visibleSampleCount) * 0.62, 1)
        let barGap = max(size.width / CGFloat(visibleSampleCount) * 0.38, 0.5)

        var segmentIndex = intervals.firstIndex { $0.end >= visibleRange.start } ?? intervals.count
        for visibleIndex in 0..<visibleSampleCount {
            let sampleIndex = min(firstSample + visibleIndex, data.samples.count - 1)
            let centerTime = visibleRange.start
                + (Double(visibleIndex) + 0.5) / Double(visibleSampleCount) * visibleDuration
            while segmentIndex + 1 < intervals.count,
                  intervals[segmentIndex + 1].start <= centerTime {
                segmentIndex += 1
            }

            let segment = intervals.indices.contains(segmentIndex)
                && centerTime >= intervals[segmentIndex].start
                && centerTime <= intervals[segmentIndex].end
                ? intervals[segmentIndex].segment
                : nil
            let color: Color = {
                if showsHighlightedSegments, segment?.isHighlighted == true { return .orange }
                if segment?.speaker == .me { return Color(red: 0.16, green: 0.47, blue: 0.88) }
                if segment?.speaker == .them { return Color(red: 0.12, green: 0.58, blue: 0.32) }
                return .secondary
            }()

            let amplitude = CGFloat(data.samples[sampleIndex])
            let height = max(4, amplitude * (chartHeight - 12))
            let x = CGFloat(visibleIndex) * (barWidth + barGap)
            let rect = CGRect(
                x: x,
                y: (chartHeight - height) / 2,
                width: barWidth,
                height: height
            )
            context.fill(
                Path(roundedRect: rect, cornerRadius: min(barWidth / 2, 2)),
                with: .color(color.opacity(segment == nil ? 0.28 : 0.78))
            )
        }

        let playheadX = CGFloat(
            (min(max(currentTime, visibleRange.start), visibleRange.end) - visibleRange.start)
                / visibleDuration
        ) * size.width
        var playhead = Path()
        playhead.move(to: CGPoint(x: playheadX, y: 0))
        playhead.addLine(to: CGPoint(x: playheadX, y: size.height))
        context.stroke(playhead, with: .color(Color.accentColor), lineWidth: 2)
    }

    private func speechIntervals(duration: TimeInterval) -> [MeetingWaveformInterval] {
        let ordered: [MeetingTranscriptSegment]
        if segmentsAreChronological {
            ordered = segments
        } else {
            ordered = segments.sorted {
                if $0.startSeconds == $1.startSeconds { return $0.id.uuidString < $1.id.uuidString }
                return $0.startSeconds < $1.startSeconds
            }
        }

        return ordered.enumerated().compactMap { index, segment in
            let start = min(max(segment.startSeconds, 0), duration)
            let nextStart = index + 1 < ordered.count ? ordered[index + 1].startSeconds : duration
            let end = min(
                duration,
                max(start + 0.08, segment.endSeconds ?? nextStart)
            )
            guard end > start else { return nil }
            return MeetingWaveformInterval(segment: segment, start: start, end: end)
        }
    }

    private var segmentsAreChronological: Bool {
        guard segments.count > 1 else { return true }
        for (current, next) in zip(segments, segments.dropFirst()) {
            if current.startSeconds > next.startSeconds { return false }
            if current.startSeconds == next.startSeconds,
               current.id.uuidString > next.id.uuidString {
                return false
            }
        }
        return true
    }

    private func time(forX x: CGFloat, width: CGFloat, data: MeetingWaveformData) -> TimeInterval {
        MeetingWaveformTimelineSupport.time(
            forX: x,
            width: width,
            duration: data.duration,
            currentTime: currentTime,
            zoomScale: zoomScale
        )
    }

    private var timeLabel: String {
        MeetingTranscriptFormatter.timestampString(for: currentTime)
    }
}

private struct MeetingWaveformScrollZoomCapture: NSViewRepresentable {
    let onZoom: (CGFloat) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onZoom: onZoom)
    }

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        context.coordinator.attach(to: view)
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.onZoom = onZoom
    }

    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.detach()
    }

    final class Coordinator {
        var onZoom: (CGFloat) -> Void
        private weak var view: NSView?
        private var monitor: Any?

        init(onZoom: @escaping (CGFloat) -> Void) {
            self.onZoom = onZoom
        }

        func attach(to view: NSView) {
            self.view = view
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                guard let self, let view = self.view, view.window != nil else {
                    return event
                }

                let location = view.convert(event.locationInWindow, from: nil)
                guard view.bounds.contains(location) else { return event }

                let delta = event.scrollingDeltaY
                guard abs(delta) > 0.01 else { return event }
                self.onZoom(delta > 0 ? 1.14 : 0.88)
                return nil
            }
        }

        func detach() {
            if let monitor {
                NSEvent.removeMonitor(monitor)
                self.monitor = nil
            }
            view = nil
        }
    }
}

private struct MeetingWaveformInterval {
    let segment: MeetingTranscriptSegment
    let start: TimeInterval
    let end: TimeInterval
}
