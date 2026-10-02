import AppKit
import Carbon
import Combine

struct RecordingShortcut: Codable, Hashable {
    let keyCode: UInt32
    let modifiers: UInt32
    static let `default` = RecordingShortcut(keyCode: UInt32(kVK_Space), modifiers: UInt32(controlKey | shiftKey))
    static let cancel = RecordingShortcut(keyCode: UInt32(kVK_Escape), modifiers: UInt32(controlKey | shiftKey))

    init(keyCode: UInt32, modifiers: UInt32) {
        self.keyCode = keyCode
        self.modifiers = modifiers
    }

    init(event: NSEvent) {
        keyCode = UInt32(event.keyCode)
        var flags: UInt32 = 0
        for (flag, carbon) in [(NSEvent.ModifierFlags.control, controlKey), (.option, optionKey),
                               (.shift, shiftKey), (.command, cmdKey)] where event.modifierFlags.contains(flag) {
            flags |= UInt32(carbon)
        }
        modifiers = flags
    }

    private var keyName: String? {
        let names: [UInt32: String] = [49: "Space", 53: "Esc", 36: "Return", 48: "Tab", 51: "Delete",
            117: "⌦", 115: "Home", 119: "End", 116: "Page Up", 121: "Page Down",
            123: "←", 124: "→", 125: "↓", 126: "↑",
            122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5", 97: "F6", 98: "F7", 100: "F8",
            101: "F9", 109: "F10", 103: "F11", 111: "F12", 105: "F13", 107: "F14", 113: "F15"]
        if let name = names[keyCode] { return name }
        // Show the physical key in the current ASCII-capable layout, including when an IME is active.
        guard keyCode < 51,
              let source = TISCopyCurrentASCIICapableKeyboardLayoutInputSource()?.takeRetainedValue(),
              let pointer = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData) else { return nil }
        let data = Unmanaged<CFData>.fromOpaque(pointer).takeUnretainedValue()
        let layout = UnsafeRawPointer(CFDataGetBytePtr(data)).assumingMemoryBound(to: UCKeyboardLayout.self)
        var dead: UInt32 = 0
        var count = 0
        var chars = [UniChar](repeating: 0, count: 8)
        let result = UCKeyTranslate(layout, UInt16(keyCode), UInt16(kUCKeyActionDisplay), 0, UInt32(LMGetKbdType()),
                                    OptionBits(kUCKeyTranslateNoDeadKeysBit), &dead, chars.count, &count, &chars)
        guard result == noErr, count > 0 else { return nil }
        let name = String(utf16CodeUnits: chars, count: count).uppercased()
        return name.unicodeScalars.allSatisfy { !CharacterSet.controlCharacters.contains($0) } ? name : nil
    }

    var display: String {
        var text = ""
        for (flag, label) in [(controlKey, "⌃"), (optionKey, "⌥"), (shiftKey, "⇧"), (cmdKey, "⌘")]
            where modifiers & UInt32(flag) != 0 { text += label }
        return text + (keyName ?? "未知按键")
    }

    var validationMessage: String? {
        let allowed = UInt32(controlKey | optionKey | shiftKey | cmdKey)
        guard modifiers & ~allowed == 0, modifiers & UInt32(controlKey | optionKey | cmdKey) != 0 else {
            return "请使用 Control、Option 或 Command 搭配一个按键。"
        }
        guard keyName != nil else { return "暂不支持这个按键，请换一个组合。" }
        guard self != .cancel else { return "⌃⇧Esc 已用于取消本轮，请选择其他组合。" }
        if modifiers == UInt32(cmdKey), [0, 1, 6, 7, 8, 9, 12, 13, 31, 45, 46, 43, 49].contains(keyCode) {
            return "该组合常用于系统或编辑操作，请增加修饰键或选择其他组合。"
        }
        return nil
    }
}

@MainActor
protocol ShortcutRegistering: AnyObject {
    func register(_ shortcut: RecordingShortcut) throws
    func unregister()
}

@MainActor
final class ShortcutPressGate {
    private var held: Set<UInt32> = []
    func press(_ id: UInt32) -> Bool { held.insert(id).inserted }
    func release(_ id: UInt32) { held.remove(id) }
    func reset() { held.removeAll() }
}

@MainActor
final class RecordingShortcutController: ObservableObject {
    @Published private(set) var shortcut: RecordingShortcut
    @Published private(set) var isCapturing = false
    @Published private(set) var isAvailable = false
    @Published private(set) var notice = ""
    @Published private(set) var candidate: RecordingShortcut?
    private let registrar: ShortcutRegistering
    private let save: (RecordingShortcut) -> Void

    init(shortcut: RecordingShortcut, registrar: ShortcutRegistering, save: @escaping (RecordingShortcut) -> Void) {
        self.shortcut = shortcut
        self.registrar = registrar
        self.save = save
    }

    func start() { restoreRegistration() }

    func beginCapture() {
        guard !isCapturing else { return }
        registrar.unregister()
        isAvailable = false
        candidate = nil
        notice = "请按新的组合键，松开后保存；按 Esc 取消。"
        isCapturing = true
    }

    func capture(_ key: RecordingShortcut) {
        guard isCapturing else { return }
        candidate = nil
        if let error = key.validationMessage { notice = error; return }
        candidate = key
        notice = "松开 \(key.display) 后保存。"
    }

    func release(keyCode: UInt32) {
        guard isCapturing, let candidate, candidate.keyCode == keyCode else { return }
        apply(candidate)
    }

    func cancelCapture() {
        guard isCapturing else { return }
        candidate = nil
        isCapturing = false
        restoreRegistration()
    }

    func restoreDefault() { apply(.default) }

    private func apply(_ key: RecordingShortcut) {
        candidate = nil
        isCapturing = false
        registrar.unregister()
        do {
            try registrar.register(key)
            shortcut = key
            isAvailable = true
            notice = ""
            save(key)
        } catch {
            let failure = "快捷键注册失败，可能已被占用。"
            restoreRegistration()
            notice = isAvailable ? "\(failure)已恢复 \(shortcut.display)。" : "\(failure)旧快捷键也未恢复，请用菜单录音并重新设置。"
        }
    }

    private func restoreRegistration() {
        do {
            try registrar.register(shortcut)
            isAvailable = true
            notice = ""
        } catch {
            isAvailable = false
            notice = "快捷键注册失败，请更换组合或用菜单录音。"
        }
    }
}

@MainActor
final class CarbonShortcutRegistrar: ShortcutRegistering {
    private var handler: EventHandlerRef?
    private var keys: [EventHotKeyRef] = []
    private let gate = ShortcutPressGate()
    var action: (UInt32) -> Void = { _ in }

    func register(_ shortcut: RecordingShortcut) throws {
        unregister()
        if handler == nil {
            var events = [
                EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
                EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
            ]
            let status = InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
                guard let event, let context else { return OSStatus(eventNotHandledErr) }
                var key = EventHotKeyID()
                let result = GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                              nil, MemoryLayout<EventHotKeyID>.size, nil, &key)
                guard result == noErr else { return result }
                let owner = Unmanaged<CarbonShortcutRegistrar>.fromOpaque(context).takeUnretainedValue()
                let pressed = GetEventKind(event) == UInt32(kEventHotKeyPressed)
                let id = key.id
                // Carbon delivers this handler on the application's main event loop.
                MainActor.assumeIsolated {
                    if pressed {
                        if owner.gate.press(id) { owner.action(id) }
                    } else { owner.gate.release(id) }
                }
                return noErr
            }, events.count, &events, Unmanaged.passUnretained(self).toOpaque(), &handler)
            guard status == noErr else { throw ShortcutRegistrationError(status: status) }
        }
        for (id, combination) in [(UInt32(1), shortcut), (UInt32(2), RecordingShortcut.cancel)] {
            var key: EventHotKeyRef?
            let status = RegisterEventHotKey(combination.keyCode, combination.modifiers,
                                            EventHotKeyID(signature: 0x4F4D4E49, id: id),
                                            GetApplicationEventTarget(), 0, &key)
            guard status == noErr, let key else {
                unregister()
                throw ShortcutRegistrationError(status: status == noErr ? OSStatus(-1) : status)
            }
            keys.append(key)
        }
    }

    func unregister() {
        keys.forEach { UnregisterEventHotKey($0) }
        keys.removeAll()
        gate.reset()
    }

    func stop() {
        unregister()
        if let handler { RemoveEventHandler(handler) }
        handler = nil
    }
}

private struct ShortcutRegistrationError: Error { let status: OSStatus }
