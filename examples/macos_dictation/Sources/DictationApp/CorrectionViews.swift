#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import SwiftUI

struct ClientCorrectionView: View {
    @ObservedObject var model: CorrectionSession
    @ObservedObject var recording: DictationSession
    @ObservedObject var state: ClientState
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("更正上一段", systemImage: "pencil.line").font(.headline)
                Spacer()
                if model.phase == .recording {
                    Button("结束修改意见录音") { state.toggleCorrection() }.buttonStyle(.borderedProminent)
                }
                if model.isBusy { Button("取消更正") { state.cancel() } }
                else { Button("返回听写结果") { model.cancel() } }
            }
            HStack {
                if model.isBusy { ProgressView().controlSize(.small) }
                Text(model.phase.rawValue).fontWeight(.semibold)
                if model.phase == .recording { Text(String(format: "%.1f 秒", recording.elapsed)).monospacedDigit() }
            }
            if !model.instruction.isEmpty {
                Text("修改意见：\(model.instruction)").font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
            }
            ScrollView {
                if let text = model.correctedText {
                    Text(text.isEmpty ? (model.didReplace ? "上一段已删除。" : "删除结果尚未确认，请检查输入框。") : text)
                        .font(.system(size: 19)).lineSpacing(8).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                } else {
                    Text(model.lastText ?? "请先完成一次听写。")
                        .font(.system(size: 19)).lineSpacing(8).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            }
            .padding(18).frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14))
            if !model.notice.isEmpty {
                Label(model.notice, systemImage: model.didReplace ? "checkmark.circle" : "info.circle")
                    .font(.callout).foregroundStyle(model.didReplace ? Color.green : Color.secondary).textSelection(.enabled)
            }
            HStack {
                Text("回到原输入框，双击 Option 开始或结束更正录音。")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let text = model.correctedText, !text.isEmpty, !model.isBusy {
                    Button(copied ? "已复制更正结果" : "复制更正结果") {
                        NSPasteboard.general.clearContents()
                        copied = NSPasteboard.general.setString(text, forType: .string)
                    }.buttonStyle(.borderedProminent)
                }
            }
            Text("仅在本机处理上一段文字和修改意见 · 不保存纠错历史 · 不自动发送")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(24).frame(minWidth: 560, minHeight: 460)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
        .onChange(of: model.phase) { if model.isBusy { copied = false } }
    }
}

struct CorrectionFloatingView: View {
    @ObservedObject var model: CorrectionSession
    @ObservedObject var recording: DictationSession
    @ObservedObject var state: ClientState

    var body: some View {
        HStack(spacing: 12) {
            OmniLogoView().frame(width: 30, height: 40)
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text("Omni").font(.system(size: 16, weight: .heavy)).italic()
                    Text(model.didReplace ? "已更正" : model.phase.rawValue).font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                }
                if model.phase == .recording {
                    HStack(spacing: 10) {
                        OmniAudioLevel(level: recording.level)
                        Text(String(format: "%02d:%02d", Int(recording.elapsed) / 60, Int(recording.elapsed) % 60))
                            .font(.system(size: 11)).monospacedDigit()
                    }
                } else {
                    Text(model.isBusy ? "正在处理修改意见" : model.notice)
                        .font(.system(size: 11)).foregroundStyle(.white.opacity(0.7)).lineLimit(1)
                        .help(model.notice)
                }
            }
            Spacer(minLength: 0)
            if model.phase == .recording {
                Button { state.toggleCorrection() } label: {
                    Image(systemName: "stop.fill").font(.system(size: 11, weight: .bold))
                        .frame(width: 30, height: 30).background(OmniBrand.accent, in: Circle())
                }.accessibilityLabel("结束更正录音")
            } else if !model.isBusy {
                Button("查看结果") { state.openResults?() }.font(.system(size: 12))
            }
            Button { state.closeFeedback() } label: {
                Image(systemName: "xmark").font(.system(size: 11, weight: .semibold)).frame(width: 24, height: 30)
            }.accessibilityLabel(model.isBusy ? "取消更正" : "收起提示")
        }
        .buttonStyle(.plain).foregroundStyle(.white)
        .padding(.horizontal, 16).padding(.vertical, 14)
        .frame(width: ClientFloatingView.size.width, height: ClientFloatingView.size.height)
        .background(LinearGradient(colors: [Color(red: 0.18, green: 0.16, blue: 0.15), Color(red: 0.11, green: 0.10, blue: 0.09)],
                                   startPoint: .top, endPoint: .bottom), in: RoundedRectangle(cornerRadius: 22))
        .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(.white.opacity(0.12)).allowsHitTesting(false))
    }
}
