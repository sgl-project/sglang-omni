// SPDX-License-Identifier: Apache-2.0
import SwiftUI

struct ReviewPanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.mode == .ask {
                questionAndAnswer
            } else {
                HStack {
                    Text(L("panel.result")).font(.system(size: 12, weight: .semibold))
                    Spacer()
                    Button { model.clearReviewedResult() } label: { Label(L("panel.clear"), systemImage: "trash") }
                        .disabled(model.resultDraft.isEmpty || model.isBusy)
                }
                ResultEditor(model: model)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            Text(model.isSelectingMode ? model.mode.title
                 : model.isEditingEntireField ? L("panel.editingAllText")
                 : model.phase == .starting ? L("status.loadingModel")
                 : model.isBusy ? model.liveStatus
                 : model.mode == .ask && !model.answerText.isEmpty ? L("panel.askComplete") : L("panel.voicePaused"))
                .font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1).frame(height: 16)
            if !model.unappliedResult.isEmpty && model.mode != .ask {
                HStack {
                    Text(model.unappliedResult).font(.system(size: 12)).lineLimit(3).textSelection(.enabled)
                    Button(L("panel.copyNewResult")) { TextInsertion.copy(model.unappliedResult) }
                }
            }
            if !model.error.isEmpty {
                Text(model.error).font(.system(size: 11)).foregroundStyle(.orange).lineLimit(3)
                if model.requiresTextAPI && model.canSelectMode {
                    HStack {
                        Button(L("panel.useVerbatim")) { model.useVerbatimDictation() }
                        Button(L("panel.textAPISettings")) { model.openTextAPISettings() }
                    }
                }
            } else if !model.notice.isEmpty && !(model.isEditingEntireField && model.notice == L("panel.editingAllText")) {
                Text(model.notice).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(2)
            }
            controls
        }
    }

    private var questionAndAnswer: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                Text(L("panel.question")).font(.system(size: 11, weight: .semibold)).foregroundStyle(.indigo)
                ScrollView {
                    Text(model.questionText.isEmpty ? (model.liveText.isEmpty ? L("panel.questionPlaceholder") : model.liveText) : model.questionText)
                        .font(.system(size: 14)).foregroundStyle(model.questionText.isEmpty ? .secondary : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }.frame(height: 44)
            }.padding(12).background(Color.indigo.opacity(0.05))
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                Text(L("panel.answer")).font(.system(size: 11, weight: .semibold)).foregroundStyle(.secondary)
                ScrollView {
                    Text(model.answerText.isEmpty ? L("panel.answerPlaceholder") : model.answerText)
                        .font(.system(size: 14)).foregroundStyle(model.answerText.isEmpty ? .secondary : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }.frame(maxHeight: .infinity)
            }.padding(12)
        }
        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button { model.toggleVoiceCapture() } label: {
                Label(L(model.phase == .recording ? "panel.pause" : "panel.resume"),
                      systemImage: model.phase == .recording ? "pause.fill" : "play.fill")
            }
            .disabled(model.phase != .recording && !model.canResumeVoice)
            .help(L("panel.pauseHint"))
            if model.canRetry { Button(L("app.retryLast")) { model.retryLast() } }
            Spacer(minLength: 0)
            Button { model.copyReviewedResult() } label: { Label(L("action.copy"), systemImage: "doc.on.doc") }
                .disabled(model.reviewContent.isEmpty || model.isBusy)
            Button(L(model.reviewDestination?.selectedText.isEmpty == false || model.reviewDestination?.replacementRange != nil
                     ? "panel.replaceSelection" : "panel.insert")) { model.insertReviewedResult() }
                .buttonStyle(.borderedProminent).disabled(!model.canInsertReview)
            if model.mode == .ask {
                Button { model.newQuestion() } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(IconButtonStyle()).disabled(model.isBusy)
                    .help(L("panel.newQuestion")).accessibilityLabel(L("panel.newQuestion"))
            }
            Button { model.closeVoicePanel() } label: { Image(systemName: "xmark") }
                .buttonStyle(IconButtonStyle()).help(L("panel.done")).accessibilityLabel(L("panel.done"))
        }.controlSize(.regular).frame(height: 32)
    }
}
