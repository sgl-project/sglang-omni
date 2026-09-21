// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

struct VoiceModePicker: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore

    var body: some View {
        Picker(L("home.voiceMode"), selection: Binding(get: { model.mode }, set: { model.selectMode($0) })) {
            ForEach(VoiceMode.allCases) { Label($0.title, systemImage: $0.icon).tag($0) }
        }
        .pickerStyle(.segmented).labelsHidden().disabled(!model.canSelectMode)
        .id(store.preferences.uiLanguage)
    }
}

struct VoicePanel: View {
    static let selectorSize = CGSize(width: 292, height: 64)
    static let compactSize = CGSize(width: 168, height: 64)
    static let transitionDuration = 0.28
    static let editorSize = CGSize(width: 560, height: 460)
    static let padding: CGFloat = 18
    static let headerHeight: CGFloat = 36
    static let controlSpacing: CGFloat = 12
    static let cornerRadius: CGFloat = 20
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    let transition: ModeBarTransition
    @ObservedObject var worker: WorkerClient

    static func size(for model: AppModel, expansion: CGFloat? = nil) -> CGSize {
        if model.isSelectingMode {
            let progress = expansion ?? model.selectionDisplayTarget
            let width = selectorSize.width + (editorSize.width - selectorSize.width) * progress
            // Note (Codex): Even widths keep the center fixed when AppKit rounds window origins to whole points.
            return CGSize(width: (width / 2).rounded() * 2,
                          height: selectorSize.height + (editorSize.height - selectorSize.height) * progress)
        }
        return model.showsEditor ? editorSize
            : model.usesCompactPanel ? compactSize : CGSize(width: 520, height: 300)
    }

    static func selectionFrame(for model: AppModel, within visible: NSRect, expansion: CGFloat? = nil) -> NSRect {
        let size = size(for: model, expansion: expansion)
        let center = min(max(model.selectionTrackX + ModeSelectionPanel.trackWidth / 2,
                             visible.minX + size.width / 2), visible.maxX - size.width / 2)
        let bottom = min(max(model.selectionPanelBottom, visible.minY), visible.maxY - size.height)
        return NSRect(x: center - size.width / 2, y: bottom, width: size.width, height: size.height)
    }

    static func collapsedFrame(from selectionFrame: NSRect, mode: VoiceMode, within visible: NSRect) -> NSRect {
        let center = selectionFrame.midX - ModeSelectionPanel.trackWidth / 2 + ModeSelectionPanel.centerX(for: mode)
        return NSRect(x: min(max(center - compactSize.width / 2, visible.minX), visible.maxX - compactSize.width),
                      y: selectionFrame.minY, width: compactSize.width, height: compactSize.height)
    }

    var body: some View {
        // Note (Codex): Keep the outer material mask in window coordinates while its contents animate.
        return GeometryReader { geometry in
            Group {
                if model.isSelectingMode || model.showsEditor || model.usesCompactPanel {
                    editorSurface(expansion: min(max((geometry.size.height - Self.selectorSize.height)
                        / (Self.editorSize.height - Self.selectorSize.height), 0), 1))
                }
                else { expandedPanel }
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .bottom)
        }
        .modifier(VoicePanelBackground())
        .onChange(of: model.compactSelectionGuidance) { _, _ in
            if model.isVoicePanelVisible { model.showVoicePanel?() }
        }
        .contextMenu {
            if !model.isReviewingResult {
                ForEach(VoiceMode.allCases) { mode in
                    Button(mode.title) { model.selectMode(mode) }.disabled(!model.canSelectMode)
                }
            }
        }
        .tint(accent)
        .preferredColorScheme(store.preferences.appearance == "light" ? .light : store.preferences.appearance == "dark" ? .dark : nil)
    }

    private var panelHeader: some View {
        HStack(spacing: Self.controlSpacing) {
            ZStack(alignment: .leading) {
                HStack(spacing: 12) {
                    if model.phase == .idle || model.phase == .recording {
                        Image(systemName: model.error.isEmpty ? model.mode.icon : "exclamationmark.circle")
                            .font(.title2).foregroundStyle(model.error.isEmpty ? (model.mode == .ask ? .indigo : accent) : .orange).frame(width: 32)
                    } else { ProgressView().controlSize(.small).frame(width: 32) }
                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.isSelectingMode || model.showsEditor ? L(model.mode == .ask ? "panel.askTitle" : "panel.editorPreview")
                             : model.phase == .recording ? L("panel.listening", model.mode.title)
                             : model.phase == .starting ? L("status.loadingModel")
                             : model.phase == .idle ? L(model.error.isEmpty ? "panel.ready" : "panel.needsAttention") : model.liveStatus)
                            .font(.system(size: 12, weight: .semibold)).lineLimit(1)
                        Group {
                            if model.mode != .ask && model.phase == .recording && !model.isSelectingMode {
                                TimelineView(.periodic(from: .now, by: 1)) { _ in
                                    let elapsed = Int(model.recorder.elapsed)
                                    Text(String(format: L("panel.elapsed"), elapsed / 60, elapsed % 60)).monospacedDigit()
                                }
                            } else {
                                Text(model.mode == .ask ? L("panel.singleQuestion")
                                     : model.isSelectingMode ? L("panel.textTools")
                                     : model.isReviewingResult && !model.isBusy ? L("panel.textTools") + " · " + model.mode.title
                                     : model.phase == .idle ? L("panel.chooseMode") : worker.status)
                            }
                        }.font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }.allowsHitTesting(false)
            }.background(WindowDragArea(enabled: model.showsEditor || !model.usesCompactPanel)).help(L("panel.dragHint"))
            if !model.isSelectingMode && !model.showsEditor && model.phase == .recording {
                Button { model.finish() } label: { Image(systemName: "stop.fill").foregroundStyle(accent) }
                    .buttonStyle(IconButtonStyle()).accessibilityLabel(L("home.finish"))
            }
            if model.isSelectingMode || model.showsEditor {
                Button { model.foldEditor() } label: { Image(systemName: "rectangle.compress.vertical").foregroundStyle(.secondary) }
                    .buttonStyle(IconButtonStyle()).help(L("panel.fold"))
                    .accessibilityLabel(L("panel.fold"))
            }
            if !model.isSelectingMode && !model.showsEditor {
                Button { model.closeVoicePanel() } label: { Image(systemName: "xmark").foregroundStyle(.secondary) }
                    .buttonStyle(IconButtonStyle()).accessibilityLabel(L("panel.done"))
            }
        }.frame(height: Self.headerHeight)
    }

    private func editorSurface(expansion: CGFloat) -> some View {
        VStack(spacing: 0) {
            if expansion > 0 {
                VStack(alignment: .leading, spacing: 10) {
                    panelHeader
                    ReviewPanel(model: model)
                }
                .padding(12)
                .frame(height: Self.editorSize.height - Self.selectorSize.height)
                .opacity(expansion)
                .frame(height: (Self.editorSize.height - Self.selectorSize.height) * expansion, alignment: .bottom)
                .clipped().allowsHitTesting(!model.isSelectingMode)
                .accessibilityHidden(expansion < 0.5)
            }
            ModeSelectionPanel(model: model, transition: transition)
        }
    }

    private var expandedPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            panelHeader
            recordingControls
        }
        .padding(Self.padding)
    }

    private var recordingControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            VoiceModePicker(model: model, store: store)
            Text(L(model.isEditingEntireField ? "panel.editingAllText"
                   : model.requiresTextAPI ? "panel.textAPIRequired" : "panel.speechOnly"))
                .font(.system(size: 11)).foregroundStyle(.secondary)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if !model.error.isEmpty {
                        Label(model.error, systemImage: "exclamationmark.circle")
                            .font(.system(size: 12)).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                    }
                    if model.phase == .idle && model.canRetry && !model.rawText.isEmpty {
                        Text(L("home.originalTranscript")).font(.system(size: 11)).foregroundStyle(.secondary)
                        Text(model.rawText).font(.system(size: 13)).textSelection(.enabled)
                    } else if model.error.isEmpty {
                        Text(model.liveText.isEmpty ? L(model.phase == .starting ? "panel.waitListening" : "panel.placeholder")
                             : String(model.liveText.suffix(600))).font(.system(size: 14)).lineSpacing(3)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }.frame(maxHeight: .infinity)
            if !model.error.isEmpty && model.requiresTextAPI && model.canSelectMode {
                HStack {
                    Button(L("panel.useVerbatim")) { model.useVerbatimDictation() }
                    Button(L("panel.textAPISettings")) { model.openTextAPISettings() }.disabled(model.phase != .idle)
                }.controlSize(.large)
            }
            if model.phase == .idle {
                HStack {
                    if model.canRetry {
                        Button(L("app.retryLast")) { model.retryLast(mode: model.mode, verbatim: model.verbatimDictation) }
                            .buttonStyle(.borderedProminent)
                        if !model.rawText.isEmpty {
                            Button(L("action.copy")) { TextInsertion.copy(model.rawText); model.notice = L("notice.copied") }
                        }
                    } else { Button(L("home.start")) { model.start() }.buttonStyle(.borderedProminent) }
                    Spacer()
                    Text(model.shortcutLabel).font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
                }.controlSize(.large)
            } else {
                Text(model.phase == .recording ? model.liveStatus : L("panel.reviewNote"))
                    .font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }
}

struct VoicePanelBackground: ViewModifier {
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: VoicePanel.cornerRadius, style: .continuous)
        content
            .background(.ultraThinMaterial, in: shape)
            .clipShape(shape)
            .overlay(shape.strokeBorder(.primary.opacity(0.1), lineWidth: 0.5).allowsHitTesting(false))
            .overlay(shape.inset(by: 0.5).strokeBorder(
                LinearGradient(colors: [.white.opacity(0.5), .white.opacity(0.06)], startPoint: .top, endPoint: .bottom),
                lineWidth: 0.5).allowsHitTesting(false))
    }
}

struct SelectionGuidanceBanner: View, Equatable {
    let text: String
    let needsSelection: Bool
    let appearance: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: needsSelection ? "text.cursor" : "info.circle")
                .foregroundStyle(needsSelection ? .orange : accent)
            Text(text).frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.system(size: 12))
        .padding(12).frame(width: 320).fixedSize(horizontal: false, vertical: true)
        .modifier(VoicePanelBackground())
        .preferredColorScheme(appearance == "light" ? .light : appearance == "dark" ? .dark : nil)
    }
}
