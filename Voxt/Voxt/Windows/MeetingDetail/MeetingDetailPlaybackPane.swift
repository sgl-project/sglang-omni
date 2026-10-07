import Foundation
import SwiftUI

// Playback controls own their local toggles; player and scrubbing stay shared
// with the transcript pane so selection/scroll synchronization has one source.
struct MeetingDetailPlaybackPane: View {
    @ObservedObject var viewModel: MeetingDetailViewModel
    @ObservedObject var playbackController: MeetingDetailPlaybackController
    @Binding var isScrubbing: Bool
    let waveformData: MeetingWaveformData?
    @State private var showsWaveformHighlights = true
    @State private var waveformZoomScale: CGFloat = MeetingWaveformTimelineSupport.minimumZoomScale
    @State private var isPlaybackRatePopoverPresented = false

    init(
        viewModel: MeetingDetailViewModel,
        playbackController: MeetingDetailPlaybackController,
        isScrubbing: Binding<Bool>,
        waveformData: MeetingWaveformData?
    ) {
        self.viewModel = viewModel
        self.playbackController = playbackController
        _isScrubbing = isScrubbing
        self.waveformData = waveformData
    }

    @ViewBuilder
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if viewModel.mode == .fileDraft {
                Text(AppLocalization.localizedString("Audio playback will be available after the meeting is saved."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if viewModel.mode == .history {
                if playbackController.isAvailable {
                    waveformToolbar

                    MeetingWaveformTimeline(
                        data: waveformData,
                        currentTime: playbackController.currentTime,
                        segments: viewModel.segments,
                        showsHighlightedSegments: showsWaveformHighlights,
                        zoomScale: $waveformZoomScale,
                        onSeek: { time in
                            playbackController.pause()
                            isScrubbing = true
                            playbackController.seek(to: time)
                            isScrubbing = false
                        }
                    )
                } else {
                    HistoryAudioUnavailableView(compact: false)
                }
            } else {
                if viewModel.isFinalizing {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)

                        Text(AppLocalization.localizedString("Audio playback will be available after the meeting is saved."))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text(
                        viewModel.canExport
                            ? AppLocalization.localizedString("The meeting is paused. You can export the current record.")
                            : AppLocalization.localizedString("The meeting is in progress. Pause it to export the current record.")
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .meetingDetailPanelSurface(cornerRadius: 16)
    }

    private var waveformToolbar: some View {
        HStack(spacing: 8) {
            HStack(spacing: 3) {
                MeetingDetailSegmentActionButton(
                    action: { showsWaveformHighlights.toggle() },
                    tint: .orange,
                    isActive: showsWaveformHighlights,
                    helpText: AppLocalization.localizedString(
                        showsWaveformHighlights ? "Hide Highlighted Segments" : "Show Highlighted Segments"
                    ),
                    accessibilityText: AppLocalization.localizedString(
                        showsWaveformHighlights ? "Hide Highlighted Segments" : "Show Highlighted Segments"
                    )
                ) {
                    MeetingDetailMarkIcon(
                        color: showsWaveformHighlights ? .orange : .secondary
                    )
                }

                MeetingDetailSegmentActionButton(
                    action: { adjustWaveformZoom(by: -1) },
                    tint: .secondary,
                    isActive: false,
                    helpText: AppLocalization.localizedString("Zoom Out"),
                    accessibilityText: AppLocalization.localizedString("Zoom Out"),
                    isDisabled: waveformZoomScale <= MeetingWaveformTimelineSupport.minimumZoomScale
                ) {
                    Image(systemName: "minus")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                }

                MeetingDetailSegmentActionButton(
                    action: { adjustWaveformZoom(by: 1) },
                    tint: .secondary,
                    isActive: false,
                    helpText: AppLocalization.localizedString("Zoom In"),
                    accessibilityText: AppLocalization.localizedString("Zoom In"),
                    isDisabled: waveformZoomScale >= MeetingWaveformTimelineSupport.maximumZoomScale
                ) {
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
            }

            Spacer(minLength: 4)

            HStack(spacing: 3) {
                playbackActionButton(
                    systemName: "backward.end.fill",
                    helpKey: "Jump to Start",
                    action: { seekPlayback(to: 0) }
                )
                playbackActionButton(
                    systemName: "gobackward.5",
                    helpKey: "Skip Back",
                    action: { seekPlayback(by: -5) }
                )
                MeetingDetailSegmentActionButton(
                    action: { playbackController.togglePlayPause() },
                    tint: .accentColor,
                    isActive: playbackController.isPlaying,
                    helpText: AppLocalization.localizedString(
                        playbackController.isPlaying ? "Pause Audio" : "Play Audio"
                    ),
                    accessibilityText: AppLocalization.localizedString(
                        playbackController.isPlaying ? "Pause Audio" : "Play Audio"
                    )
                ) {
                    Image(systemName: playbackController.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(playbackController.isPlaying ? Color.accentColor : .primary)
                }
                playbackActionButton(
                    systemName: "goforward.5",
                    helpKey: "Skip Forward",
                    action: { seekPlayback(by: 5) }
                )
                playbackActionButton(
                    systemName: "forward.end.fill",
                    helpKey: "Jump to End",
                    action: { seekPlayback(to: playbackController.duration) }
                )
            }

            Spacer(minLength: 4)

            HStack(spacing: 6) {
                Text(timerLabel)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .fixedSize()

                MeetingDetailSegmentActionButton(
                    action: { isPlaybackRatePopoverPresented.toggle() },
                    tint: .accentColor,
                    isActive: isPlaybackRatePopoverPresented || playbackController.playbackRate != 1,
                    helpText: AppLocalization.localizedString("Playback Speed"),
                    accessibilityText: AppLocalization.localizedString("Playback Speed"),
                    contentWidth: 28,
                    buttonWidth: 40
                ) {
                    Text(playbackRateLabel)
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(
                            isPlaybackRatePopoverPresented || playbackController.playbackRate != 1
                                ? Color.accentColor
                                : .secondary
                        )
                }
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .popover(isPresented: $isPlaybackRatePopoverPresented, arrowEdge: .bottom) {
                    playbackRatePopover
                }
            }
        }
        .frame(minHeight: 28)
    }

    private var playbackRatePopover: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(AppLocalization.localizedString("Playback Speed"))
                    .font(.system(size: 12, weight: .semibold))

                Spacer(minLength: 12)

                Text(playbackRateLabel)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Color.accentColor)
            }

            playbackRateSlider
        }
        .padding(12)
        .frame(width: 230)
    }

    private var playbackRateSlider: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                Slider(
                    value: Binding(
                        get: { Double(playbackController.playbackRate) },
                        set: { playbackController.setPlaybackRate(Float($0)) }
                    ),
                    in: Double(MeetingDetailPlaybackController.minimumPlaybackRate)...Double(MeetingDetailPlaybackController.maximumPlaybackRate),
                    step: 0.05
                )
                .controlSize(.small)
                .padding(.horizontal, 8)
                .frame(width: proxy.size.width, height: 18)
                .accessibilityLabel(AppLocalization.localizedString("Playback Speed"))
                .accessibilityValue(playbackRateLabel)

                ForEach(playbackRateTicks, id: \.self) { rate in
                    VStack(spacing: 2) {
                        Rectangle()
                            .fill(Color.secondary.opacity(0.42))
                            .frame(width: 1, height: 4)

                        Text(playbackRateLabel(for: rate))
                            .font(.system(size: 8, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .fixedSize()
                    }
                    .position(
                        x: playbackRateX(for: rate, width: proxy.size.width),
                        y: 29
                    )
                }
            }
        }
        .frame(height: 43)
    }

    private let playbackRateTicks = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

    private func playbackRateX(for rate: Double, width: CGFloat) -> CGFloat {
        let horizontalInset: CGFloat = 8
        let trackWidth = max(width - horizontalInset * 2, 1)
        let progress = (rate - 1) / 1.5
        return horizontalInset + CGFloat(progress) * trackWidth
    }

    private var playbackRateLabel: String {
        playbackRateLabel(for: Double(playbackController.playbackRate))
    }

    private func playbackRateLabel(for rate: Double) -> String {
        var value = String(format: "%.2f", rate)
        while value.last == "0" {
            value.removeLast()
        }
        if value.last == "." {
            value.removeLast()
        }
        return "\(value)×"
    }

    private func playbackActionButton(
        systemName: String,
        helpKey: String,
        action: @escaping () -> Void
    ) -> some View {
        MeetingDetailSegmentActionButton(
            action: action,
            tint: .secondary,
            isActive: false,
            helpText: AppLocalization.localizedString(helpKey),
            accessibilityText: AppLocalization.localizedString(helpKey)
        ) {
            Image(systemName: systemName)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
        }
    }

    private func adjustWaveformZoom(by step: Int) {
        let factor: CGFloat = step > 0 ? 1.6 : 0.625
        waveformZoomScale = MeetingWaveformTimelineSupport.clampedZoomScale(
            waveformZoomScale * factor
        )
    }

    private func seekPlayback(to time: TimeInterval) {
        playbackController.seek(to: time)
        isScrubbing = false
    }

    private func seekPlayback(by offset: TimeInterval) {
        playbackController.seek(by: offset)
        isScrubbing = false
    }


    private var timerLabel: String {
        "\(MeetingTranscriptFormatter.timestampString(for: playbackController.currentTime)) / \(MeetingTranscriptFormatter.timestampString(for: playbackController.duration))"
    }
}
