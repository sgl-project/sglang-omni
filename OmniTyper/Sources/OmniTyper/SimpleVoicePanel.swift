// SPDX-License-Identifier: Apache-2.0
import SwiftUI

struct SimpleVoicePanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ZStack(alignment: .top) {
            Capsule().fill(.secondary.opacity(0.35)).frame(width: 24, height: 3)
                .frame(maxWidth: .infinity).frame(height: 10)
                .background(WindowDragArea(enabled: !model.isSelectingMode && !model.showsEditor)).help(L("panel.dragHint"))
            HStack {
                Button { model.closeVoicePanel() } label: {
                    Image(systemName: "xmark").font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 32, height: 32)
                        .background(.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
                        .contentShape(Rectangle())
                }.buttonStyle(.plain).accessibilityLabel(L("panel.done"))
                Spacer()
                Color.clear.frame(width: ModeSelectionPanel.markerDiameter, height: ModeSelectionPanel.markerDiameter)
                    .background(PanelSelectionArea(model: model)).help(L("panel.reselect"))
                    .accessibilityLabel(model.mode.title + ", " + (model.phase == .recording ? L("status.listening") : model.liveStatus))
                Spacer()
                Button { model.toggleVoiceCapture() } label: {
                    Image(systemName: model.phase == .recording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 13, weight: .semibold)).foregroundStyle(.white)
                        .frame(width: 32, height: 32)
                        .background(accent.opacity(model.phase == .recording || model.canResumeVoice ? 1 : 0.35),
                                    in: RoundedRectangle(cornerRadius: 8))
                        .contentShape(Rectangle())
                }.buttonStyle(.plain).disabled(model.phase != .recording && !model.canResumeVoice)
                    .accessibilityLabel(L(model.phase == .recording ? "home.finish" : "home.start"))
            }.padding(.horizontal, 12).frame(maxHeight: .infinity)
        }.frame(height: VoicePanel.selectorSize.height)
    }
}
