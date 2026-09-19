// SPDX-License-Identifier: Apache-2.0
import SwiftUI

@MainActor
final class ModeBarTransition: ObservableObject {
    @Published var width: CGFloat?
    @Published var markerX: CGFloat?
}

struct ModeSelectionPanel: View {
    static let groupSpacing: CGFloat = 20
    static let markerDiameter: CGFloat = 42
    static let selectedScale: CGFloat = 1.1
    static let trackWidth = AppModel.modeSelectionStep * 4 + groupSpacing
    @ObservedObject var model: AppModel
    @ObservedObject var transition: ModeBarTransition
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ViewState private var hovered: VoiceMode?

    static func centerX(for mode: VoiceMode) -> CGFloat {
        let index = VoiceMode.allCases.firstIndex(of: mode) ?? 0
        return (CGFloat(index) + 0.5) * AppModel.modeSelectionStep + (mode == .ask ? groupSpacing : 0)
    }

    static func compactProgress(width: CGFloat, from initialWidth: CGFloat?, expanded: Bool) -> CGFloat {
        guard let initialWidth else { return expanded ? 0 : 1 }
        return min(max((initialWidth - width) / (initialWidth - VoicePanel.compactSize.width), 0), 1)
    }

    static func markerX(for mode: VoiceMode, selected: VoiceMode, width: CGFloat,
                        transitionWidth: CGFloat?, selectedX: CGFloat?, expanded: Bool) -> CGFloat {
        let progress = compactProgress(width: width, from: transitionWidth, expanded: expanded)
        let anchor = selectedX ?? (expanded ? width / 2 - trackWidth / 2 + centerX(for: selected) : width / 2)
        let initial = centerX(for: mode) - centerX(for: selected)
        let final: CGFloat = mode == selected ? 0 : initial < 0 ? -56 : 56
        return anchor + initial + (final - initial) * progress
    }

    private var expanded: Bool { model.isSelectingMode || model.showsEditor }

    var body: some View {
        GeometryReader { geometry in
            let center = geometry.size.width / 2
            let progress = Self.compactProgress(width: geometry.size.width, from: transition.width, expanded: expanded)
            let selectedX = Self.markerX(for: model.mode, selected: model.mode, width: geometry.size.width,
                                        transitionWidth: transition.width, selectedX: transition.markerX, expanded: expanded)
            let dividerX = (transition.width ?? geometry.size.width) / 2 - Self.trackWidth / 2
                + AppModel.modeSelectionStep * 3 + Self.groupSpacing / 2
            let compactCenter = transition.width == nil ? center : VoicePanel.compactSize.width / 2
            ZStack {
                Rectangle().fill(.secondary.opacity(0.25)).frame(width: 1, height: 30)
                    .position(x: dividerX + (compactCenter - dividerX) * progress,
                              y: VoicePanel.selectorSize.height / 2)
                    .opacity(1 - progress).accessibilityHidden(true).allowsHitTesting(false)
                ForEach(VoiceMode.allCases) { mode in
                    let selected = model.mode == mode
                    let tint = mode == .ask ? Color.indigo : accent
                    let emphasized = selected || (expanded && hovered == mode)
                    let initialScale = emphasized ? Self.selectedScale : 1
                    Button { model.selectMode(mode) } label: {
                        Image(systemName: mode.icon).font(.system(size: 21, weight: .medium))
                            .foregroundStyle(emphasized || mode == .ask ? tint : Color.secondary)
                            .frame(width: Self.markerDiameter, height: Self.markerDiameter)
                            .background(Color(nsColor: .controlBackgroundColor), in: Circle())
                            .overlay(Circle().stroke(emphasized ? tint.opacity(0.7) : .secondary.opacity(0.2),
                                                     lineWidth: emphasized ? 1.5 : 1))
                            .shadow(color: .black.opacity(expanded || selected ? 0.1 : 0), radius: 2, y: 1)
                            .contentShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .disabled(!model.canSelectMode && !model.isSelectingMode)
                    .overlay {
                        if selected && model.phase == .recording {
                            RecordingGlow(recorder: model.recorder, tint: tint)
                        }
                    }
                    .scaleEffect(initialScale + ((selected ? 1 : 0.35) - initialScale) * progress)
                    .animation(reduceMotion ? nil : .spring(response: 0.25, dampingFraction: 0.85), value: emphasized)
                    .onHover { over in hovered = over ? mode : nil }
                    .help(mode.title + " · " + L(mode == .ask ? "panel.singleQuestion" : "panel.textTools"))
                    .accessibilityLabel(mode.title)
                    .accessibilityAddTraits(selected ? [.isSelected] : [])
                    .opacity(selected ? 1 : 1 - progress)
                    .accessibilityHidden(!expanded)
                    .position(x: Self.markerX(for: mode, selected: model.mode, width: geometry.size.width,
                                             transitionWidth: transition.width, selectedX: transition.markerX, expanded: expanded),
                              y: VoicePanel.selectorSize.height / 2)
                    .allowsHitTesting(expanded && !model.isSelectingMode)
                }
                if !expanded || transition.width != nil {
                    SimpleVoicePanel(model: model)
                        .frame(width: VoicePanel.compactSize.width)
                        .position(x: selectedX, y: VoicePanel.selectorSize.height / 2)
                        .opacity(progress).allowsHitTesting(!expanded && transition.width == nil)
                        .accessibilityHidden(expanded)
                    if model.phase == .starting || model.phase == .processing {
                        ProgressView().controlSize(.mini).position(x: selectedX, y: 55).opacity(progress).allowsHitTesting(false)
                    }
                }
            }
        }
        .frame(height: VoicePanel.selectorSize.height)
        // Note (Codex): Native window resizing already drives marker positions and the final scale.
        .transaction { $0.animation = nil }
    }
}

private struct RecordingGlow: View {
    @ObservedObject var recorder: AudioRecorder
    let tint: Color
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Circle()
            .stroke(tint, lineWidth: 2)
            .shadow(color: tint.opacity(0.55), radius: 4)
            .background(Circle().fill(tint.opacity(0.12)))
            .phaseAnimator(reduceMotion ? [0.5] : [0.3, 0.7]) { glow, opacity in
                glow.opacity(opacity)
            } animation: { _ in
                reduceMotion ? nil : .easeInOut(duration: 1.4)
            }
            .opacity(1 - recorder.level)
            .overlay {
                Circle()
                    .stroke(tint, lineWidth: 2 + recorder.level * 1.5)
                    .shadow(color: tint.opacity(0.65), radius: 2 + recorder.level * 5)
                    .background(Circle().fill(tint.opacity(0.16)))
                    .opacity(recorder.level)
            }
            .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: recorder.level)
            .allowsHitTesting(false).accessibilityHidden(true)
    }
}
