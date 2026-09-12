#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import SwiftUI

struct ClientResultView: View {
    @ObservedObject var model: DictationSession
    @ObservedObject var state: ClientState
    @ObservedObject var insertion: DictationInsertion
    @State private var showPolished = false
    @State private var copied: String?

    private var text: String { showPolished ? model.resultText : model.rawText }
    private var copyTitle: String {
        let version = showPolished ? "整理结果" : "原文"
        return copied == text ? "已复制\(version)" : "复制\(version)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("本轮结果", systemImage: "waveform").font(.system(size: 17, weight: .semibold))
                Spacer()
                Button("打开录音…") { state.openAudio?() }.disabled(model.isBusy || insertion.isDelivering)
                Button(model.phase == .recording ? "结束录音" : "开始录音") { state.toggleRecording() }
                    .disabled((model.isBusy && model.phase != .recording) || insertion.isDelivering)
                    .controlSize(.large)
            }
            HStack {
                if model.isBusy { ProgressView().controlSize(.small) }
                Text(model.phase.rawValue).font(.system(size: 13, weight: .semibold))
                if model.phase == .recording {
                    Text(String(format: "%.1f / %.0f 秒", model.elapsed, model.recordingLimit)).monospacedDigit()
                }
                Spacer()
                if model.isBusy || insertion.isDelivering { Button("取消本轮") { state.cancel() } }
            }
            if !model.rawText.isEmpty {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        if model.hasPolishedResult {
                            Picker("查看文本", selection: $showPolished) {
                                Text("原文").tag(false)
                                Text("整理结果").tag(true)
                            }
                            .pickerStyle(.segmented).labelsHidden().frame(width: 216)
                        } else {
                            Text("原始转写").font(.headline)
                        }
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            if NSPasteboard.general.setString(text, forType: .string) { copied = text }
                        } label: {
                            Label(copyTitle, systemImage: copied == text ? "checkmark" : "doc.on.doc")
                        }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                    }
                    ScrollView {
                        Text(text).font(.system(size: 19)).lineSpacing(10).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .topLeading).padding(.trailing, 10)
                    }
                    .id(showPolished)
                    .frame(maxHeight: .infinity)
                    Text(showPolished ? "模型整理结果，请核对措辞；可以随时切回原文。" : "原始转写。整理功能默认关闭，原文始终保留。")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                }
                .padding(20)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 16))
            } else {
                VStack(spacing: 16) {
                    Image(systemName: model.phase == .failed ? "exclamationmark.triangle" : "mic")
                        .font(.system(size: 32))
                    Text(model.isBusy ? model.phase.rawValue : "录下你想输入的话")
                        .font(.system(size: 18, weight: .medium))
                    Text(model.phase == .authorizing ? "请在系统弹窗中允许麦克风访问。" : "在目标应用的输入框中按 \(state.shortcutLabel)，再按一次结束并回填。")
                        .font(.system(size: 13)).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if let timing = TimingPresentation(session: model) {
                ClientTimingView(timing: timing)
            }
            if !model.notice.isEmpty {
                Label(model.notice, systemImage: "exclamationmark.circle")
                    .font(.system(size: 13)).foregroundStyle(.orange).textSelection(.enabled)
            }
            if !state.hotkeyNotice.isEmpty { Text(state.hotkeyNotice).font(.system(size: 12)).foregroundStyle(.orange) }
            if !insertion.message.isEmpty {
                Label(insertion.message, systemImage: insertion.didInsert ? "checkmark.circle" : "text.cursor")
                    .font(.system(size: 13)).foregroundStyle(insertion.didInsert ? Color.green : Color.secondary)
                    .textSelection(.enabled)
            }
            Divider()
            Text("本地处理 · 不保存录音或文本历史 · 新一轮开始后清空上一轮")
                .font(.system(size: 12)).foregroundStyle(.secondary)
        }
        .padding(24)
        .frame(minWidth: 560, minHeight: 460)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
        .onChange(of: model.phase) {
            if [.idle, .authorizing, .recognizing].contains(model.phase) { showPolished = false; copied = nil }
        }
        .onChange(of: showPolished) { copied = nil }
    }
}

private struct ClientTimingView: View {
    let timing: TimingPresentation

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("本轮耗时").fontWeight(.semibold)
                Spacer()
                Text("处理总耗时 \(timing.totalDuration)").monospacedDigit()
            }
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 5) {
                GridRow {
                    Text("ASR")
                    Text(timing.asrDuration).monospacedDigit()
                    Text(timing.asrStatus).foregroundStyle(.secondary)
                }
                GridRow {
                    Text("LLM")
                    Text(timing.polishDuration).monospacedDigit()
                    Text(timing.polishStatus).foregroundStyle(.secondary)
                }
            }
            Text("本轮单次观测，含请求与处理开销；不含录音、预热和回填。")
                .font(.system(size: 11)).foregroundStyle(.secondary)
        }
        .font(.system(size: 12))
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct ClientSettingsView: View {
    @ObservedObject var model: DictationSession
    @ObservedObject var state: ClientState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("在光标位置听写").font(.headline)
                    Text("点入要输入文字的位置，按 \(state.shortcutLabel) 录音，再按一次结束。只触发粘贴，不按回车。")
                    Text("回填方式：快捷键粘贴（⌘V）").foregroundStyle(.secondary)
                    HStack {
                        Text(state.accessibilityGranted ? "辅助功能：已授权" : "辅助功能：待授权")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("授权辅助功能…") { state.requestAccessibility() }
                        Button("刷新") { state.accessibilityGranted = DictationAccessibility.isTrusted }
                    }
                    Text("首次授权后回到要输入文字的位置，重新开始录音。")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Text("录音快捷键").font(.headline)
                    HStack {
                        Text("开始 / 结束")
                        Spacer()
                        Text(state.shortcuts.candidate?.display ?? state.shortcutLabel)
                            .font(.system(.body, design: .monospaced)).padding(.horizontal, 10).padding(.vertical, 5)
                            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                        if state.shortcuts.isCapturing {
                            Button("取消录入") { state.shortcuts.cancelCapture() }
                        } else {
                            Button("更改快捷键") { state.beginShortcutCapture() }
                            Button("恢复默认") { state.restoreShortcut() }
                        }
                    }
                    .disabled(model.isBusy || state.insertion.isDelivering)
                    if !state.hotkeyNotice.isEmpty {
                        Text(state.hotkeyNotice).foregroundStyle(state.shortcuts.isCapturing ? Color.secondary : Color.orange)
                    }
                    Text("点击更改后直接按键。取消本轮固定为 ⌃⇧Esc；设置会保存。部分系统或其他应用的冲突无法提前检测。")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Picker("输入位置", selection: $state.pasteToCurrentCursor) {
                        Text("当前光标").tag(true)
                        Text("锁定原输入框").tag(false)
                    }
                    .pickerStyle(.segmented).disabled(model.isBusy || state.shortcuts.isCapturing)
                    Text(state.pasteToCurrentCursor
                         ? "默认模式。向识别完成时的当前光标粘贴，可切换输入位置。无需读取输入框；显示“已触发粘贴”后请检查文字。"
                         : "保持录音开始时的输入框和光标不变。检测到草稿或焦点变化时，本轮只提供复制。")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                }
                if !state.pasteToCurrentCursor {
                    Toggle(isOn: $state.compatibilityPasteEnabled) {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("兼容粘贴").font(.headline)
                            Text("默认关闭。无法读取完整草稿或光标时仍尝试粘贴。请保持输入框不变并检查结果；剪贴板会保留本轮文字。")
                                .font(.system(size: 12)).foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .toggleStyle(.switch).disabled(model.isBusy || state.shortcuts.isCapturing)
                    .accessibilityLabel("兼容粘贴")
                }
                Divider()
                Toggle(isOn: $model.polishEnabled) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("轻度整理").font(.headline)
                        Text("首次使用默认关闭，之后记住选择。开启后使用本地 Ollama 校对，并在空闲时预热。")
                            .font(.system(size: 12)).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .toggleStyle(.switch).disabled(model.isBusy || state.shortcuts.isCapturing)
                .accessibilityLabel("轻度整理")
                VStack(alignment: .leading, spacing: 10) {
                    Toggle("使用个人润色背景", isOn: Binding(get: { state.personalBackgroundEnabled },
                                                      set: { state.setPersonalBackgroundEnabled($0) }))
                        .toggleStyle(.switch).accessibilityLabel("使用个人润色背景")
                    Text("填写常用术语、人名拼写或排版偏好。纠错对照可写为：em el ex → MLX。只辅助忠实校对，不回答或补入个人事实。")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                    TextEditor(text: $state.personalBackgroundDraft)
                        .font(.system(size: 13)).frame(height: 120)
                        .padding(6)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.25)))
                        .accessibilityLabel("个人润色背景内容")
                    HStack {
                        Text("\(state.personalBackgroundDraft.count) / \(ClientPreferences.maximumBackgroundLength) 字")
                            .foregroundStyle(state.personalBackgroundDraft.count > ClientPreferences.maximumBackgroundLength ? Color.orange : Color.secondary)
                        Spacer()
                        Button("清空") { state.clearPersonalBackground() }
                            .disabled(state.savedBackground.isEmpty && state.personalBackgroundDraft.isEmpty)
                        Button("保存背景") { state.savePersonalBackground() }
                            .disabled(!state.hasUnsavedBackground)
                    }
                    if state.hasUnsavedBackground {
                        Text("有未保存的修改，当前仍使用已保存内容。")
                            .font(.system(size: 12)).foregroundStyle(.secondary)
                    }
                    if !state.backgroundNotice.isEmpty {
                        Text(state.backgroundNotice).font(.system(size: 12)).foregroundStyle(.secondary)
                    }
                    Text("保存到本机应用设置（非加密存储）。仅在两个开关均开启时发送给本机 Ollama；不发送给 ASR，不保存听写历史。修改从下一轮生效。")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                    Text(state.warmup.status).font(.system(size: 12)).foregroundStyle(.secondary)
                }
                .disabled(state.shortcuts.isCapturing)
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Text("语音识别 · Omni / MLX").font(.headline)
                    TextField("本机服务地址", text: $state.asrBaseURLDraft)
                        .accessibilityLabel("ASR 服务地址")
                    TextField("已加载的模型名称", text: $state.asrModelDraft)
                        .accessibilityLabel("ASR 模型名称")
                    Text(state.asrStatus).foregroundStyle(.secondary)
                }
                .textFieldStyle(.roundedBorder).disabled(state.shortcuts.isCapturing)
                VStack(alignment: .leading, spacing: 8) {
                    Text("文本整理 · Ollama").font(.headline)
                    TextField("本机服务地址", text: $state.polishBaseURLDraft)
                        .accessibilityLabel("LLM 服务地址")
                    TextField("已安装的模型名称", text: $state.polishModelDraft)
                        .accessibilityLabel("LLM 模型名称")
                    Text(state.ollamaStatus).foregroundStyle(.secondary)
                }
                .textFieldStyle(.roundedBorder).disabled(state.shortcuts.isCapturing)
                HStack {
                    Button("保存模型配置") { state.saveServiceConfiguration() }
                        .disabled(!state.hasUnsavedConfiguration || state.shortcuts.isCapturing)
                    Button(state.checking ? "检测中…" : "检查本地服务") { state.checkServices() }
                        .disabled(state.checking || state.shortcuts.isCapturing)
                }
                Text("填写服务提供的模型名称。保存后下一轮生效，不会下载模型；检测使用已保存的配置。")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                if !state.configurationNotice.isEmpty {
                    Text(state.configurationNotice).font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Divider()
                Text("开始 / 结束：\(state.shortcutLabel)　　取消：⌃⇧Esc")
                Text("不保存录音或文本历史。当前光标模式保留本轮文字在剪贴板中；锁定模式确认填入后恢复原剪贴板。每次启动默认当前光标，兼容粘贴关闭。")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .font(.system(size: 13))
            .padding(24)
        }
        .frame(width: 550, height: 650)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct ClientFloatingView: View {
    static let size = CGSize(width: 380, height: 64)
    @ObservedObject var model: DictationSession
    @ObservedObject var state: ClientState
    @ObservedObject var insertion: DictationInsertion

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: insertion.didInsert ? "checkmark.circle" : (model.phase == .recording ? "waveform" : "text.bubble"))
                .font(.system(size: 23)).foregroundStyle(.mint)
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                if model.phase == .recording {
                    ProgressView(value: model.level).progressViewStyle(.linear).tint(.mint)
                        .frame(width: 135).accessibilityLabel("麦克风音量")
                } else if let timing = TimingPresentation(session: model) {
                    Text(timing.summary)
                        .font(.system(size: 11)).monospacedDigit()
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(1).minimumScaleFactor(0.85)
                        .accessibilityLabel("本轮耗时，\(timing.summary)")
                }
            }
            Spacer(minLength: 0)
            if model.phase == .recording {
                Button { state.toggleRecording() } label: {
                    Image(systemName: "stop.fill").frame(width: 30, height: 30)
                }
                .accessibilityLabel("结束录音")
            } else if (model.phase == .ready && !insertion.isDelivering) || model.phase == .failed {
                Button("查看结果") { state.openResults?() }.font(.system(size: 12))
            }
            Button { state.closeFeedback() } label: { Image(systemName: "xmark").frame(width: 30, height: 30) }
                .accessibilityLabel(model.isBusy || insertion.isDelivering ? "取消并清空本轮" : "收起提示")
                .help(model.isBusy || insertion.isDelivering ? "取消并清空本轮" : "本轮结果仍可从菜单栏查看。")
        }
        .buttonStyle(.plain).foregroundStyle(.white)
        .padding(.horizontal, 19).padding(.vertical, 14)
        .frame(width: Self.size.width, height: Self.size.height)
        .background(Color(red: 0.10, green: 0.13, blue: 0.12), in: RoundedRectangle(cornerRadius: 20))
    }

    private var title: String {
        if insertion.isDelivering { return "正在回填" }
        if insertion.didInsert { return insertion.hasWarning ? "回填需检查" : "已填入" }
        if insertion.didTriggerPaste {
            return insertion.canDismissAfterPaste ? "已触发粘贴" : "请检查输入框"
        }
        switch model.phase {
        case .recording: return "正在录音"
        case .authorizing: return "等待麦克风授权"
        case .recognizing: return "正在转写"
        case .polishing: return "正在整理"
        case .ready: return "请查看结果"
        case .failed: return "听写失败"
        case .idle: return "准备录音"
        }
    }
}
