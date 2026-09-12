import Combine
import Foundation

/// AX text ranges use UTF-16 offsets. Never replace the entire existing draft.
public struct TextDraft: Equatable {
    public let value: String
    public let selection: NSRange

    public init(value: String, selection: NSRange) {
        self.value = value
        self.selection = selection
    }

    public func inserting(_ text: String) throws -> String {
        let count = value.utf16.count
        guard selection.location >= 0, selection.length >= 0,
              selection.location <= count, selection.length <= count - selection.location,
              isScalarBoundary(selection.location), isScalarBoundary(selection.location + selection.length),
              let range = Range(selection, in: value) else {
            throw DictationError("无法确认输入框的光标位置，请查看结果后手动复制。")
        }
        return value.replacingCharacters(in: range, with: text)
    }

    private func isScalarBoundary(_ offset: Int) -> Bool {
        let units = value.utf16
        guard offset > 0, offset < units.count else { return true }
        let unit = units[units.index(units.startIndex, offsetBy: offset)]
        return !(0xDC00...0xDFFF).contains(unit)
    }
}

@MainActor
public protocol DictationTextTarget: AnyObject {
    var applicationName: String { get }
    var supportsConfirmation: Bool { get }
    var preparationMessage: String { get }
    /// Opt in only when unverified paste is the normal completion of this target.
    var dismissesFeedbackAfterPaste: Bool { get }
    func validate() throws
    func insert(_ text: String) throws
    func confirms(_ text: String) -> Bool
    func stopObserving()
    func finishInsertion(confirmed: Bool) -> String?
}

public extension DictationTextTarget {
    var supportsConfirmation: Bool { true }
    var dismissesFeedbackAfterPaste: Bool { false }
    var preparationMessage: String {
        supportsConfirmation
            ? "完成后填入 \(applicationName)，由你手动发送。"
            : "完成后向 \(applicationName) 尝试兼容粘贴；请保持原输入框不变。"
    }
    func finishInsertion(confirmed: Bool) -> String? { nil }
}

/// One target per recording. Consumes it before writing; never retries an ambiguous write.
@MainActor
public final class DictationInsertion: ObservableObject {
    @Published public private(set) var message = ""
    @Published public private(set) var isDelivering = false
    @Published public private(set) var didInsert = false
    @Published public private(set) var hasWarning = false
    @Published public private(set) var didTriggerPaste = false
    @Published public private(set) var canDismissAfterPaste = false
    @Published public private(set) var targetApplicationName = ""
    private var target: DictationTextTarget?
    private var deliveryTarget: DictationTextTarget?
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private let verificationLimit: Double
    private let observationInterval: Double

    public init(verificationLimit: Double = 3, observationInterval: Double = 0.2) {
        self.verificationLimit = verificationLimit
        self.observationInterval = observationInterval
    }

    public func prepare(_ target: DictationTextTarget) {
        cancel()
        self.target = target
        targetApplicationName = target.applicationName
        message = target.preparationMessage
        let id = generation
        task = Task { [weak self] in
            while let self, id == self.generation, !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: UInt64(self.observationInterval * 1_000_000_000))
                    guard id == self.generation, !Task.isCancelled else { return }
                    try target.validate()
                } catch {
                    guard id == self.generation, !Task.isCancelled else { return }
                    self.abandon(error.localizedDescription)
                    return
                }
            }
        }
    }

    public func complete(_ text: String) {
        guard let target else { return }
        self.target = nil
        task?.cancel()
        task = nil
        // Stop change notifications before our own edit. Validate immediately before writing.
        target.stopObserving()
        guard !text.isEmpty else { message = "没有可回填的文字。"; return }
        do {
            try target.validate()
        } catch {
            message = "未自动回填：\(error.localizedDescription)"
            return
        }
        deliveryTarget = target
        do { try target.insert(text) }
        catch {
            let cleanup = finishDelivery(confirmed: false)
            message = "回填结果未确认：\(error.localizedDescription)" + cleanup
            return
        }
        didTriggerPaste = true
        guard target.supportsConfirmation else {
            let cleanup = finishDelivery(confirmed: false)
            hasWarning = true
            message = "已触发粘贴到 \(target.applicationName)，无法确认是否填入；请检查输入框。" + cleanup
            // Presentation permission is separate from confirmation. Never mark an
            // unverified paste as inserted just to let its routine notice fade away.
            canDismissAfterPaste = target.dismissesFeedbackAfterPaste
            return
        }
        isDelivering = true
        message = "正在确认回填结果…"
        let id = generation
        task = Task { [weak self] in
            let deadline = ProcessInfo.processInfo.systemUptime + (self?.verificationLimit ?? 0)
            while let self, id == self.generation, !Task.isCancelled {
                if target.confirms(text) {
                    let cleanup = self.finishDelivery(confirmed: true)
                    self.hasWarning = !cleanup.isEmpty
                    self.didInsert = true
                    self.isDelivering = false
                    self.message = "已填入 \(target.applicationName)，未发送。" + cleanup
                    self.task = nil
                    return
                }
                if ProcessInfo.processInfo.systemUptime >= deadline {
                    let cleanup = self.finishDelivery(confirmed: false)
                    self.isDelivering = false
                    self.message = "回填结果未确认，请先检查原输入框，避免重复粘贴。" + cleanup
                    self.task = nil
                    return
                }
                do { try await Task.sleep(nanoseconds: 50_000_000) } catch { return }
            }
        }
    }

    public func abandon(_ reason: String) {
        cancel()
        message = reason
    }

    public func cancel() {
        canDismissAfterPaste = false
        generation = UUID()
        task?.cancel()
        task = nil
        target?.stopObserving()
        target = nil
        isDelivering = false
        didInsert = false
        hasWarning = false
        didTriggerPaste = false
        targetApplicationName = ""
        message = finishDelivery(confirmed: false)
    }

    private func finishDelivery(confirmed: Bool) -> String {
        let pending = deliveryTarget
        deliveryTarget = nil
        return pending?.finishInsertion(confirmed: confirmed).map { " " + $0 } ?? ""
    }
}
