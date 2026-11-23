// The keyboard side of scripts/mac-hotkey-coexist.sh: posts synthetic key
// events, or watches what is left of them after every other event tap.
//
//   poster post SPEC…   SPEC: f:KEYCODE:FLAGS[:MARK]  a flagsChanged event
//                             d:KEYCODE:FLAGS[:MARK]  a key down
//                             u:KEYCODE:FLAGS[:MARK]  a key up
//                             s:MS                    a pause
//                       FLAGS and MARK (eventSourceUserData) in hex: 0x80140
//                       is right ⌥ as a real keyboard sends it; MARK 0x41555241
//                       is Auralink's ("AURA"), 0x564F5854 VoxtNew's ("VOXT")
//   poster tail         prints "<kind> <keycode> <flags> <mark>" for every key
//                       event that reaches the end of the session's taps
//                       (a modifying tap appended last, which passes all)
//
// Both need Accessibility for the process responsible for them (over ssh
// on mac-air: sshd-keygen-wrapper has it).
import CoreGraphics
import Foundation

func hex(_ s: Substring) -> UInt64 { UInt64(s.replacingOccurrences(of: "0x", with: ""), radix: 16) ?? 0 }

func post(_ specs: [String]) {
    let src = CGEventSource(stateID: .hidSystemState)
    for spec in specs {
        let p = spec.split(separator: ":")
        if p.first == "s", p.count == 2 {
            usleep(UInt32((Double(p[1]) ?? 0) * 1000))
            continue
        }
        guard p.count >= 3, let code = UInt16(p[1]) else {
            FileHandle.standardError.write("bad spec \(spec)\n".data(using: .utf8)!)
            exit(2)
        }
        let flags = hex(p[2]), mark = p.count > 3 ? hex(p[3]) : 0
        guard let e = CGEvent(keyboardEventSource: src, virtualKey: code, keyDown: p[0] != "u") else { exit(1) }
        if p[0] == "f" { e.type = .flagsChanged }
        e.flags = CGEventFlags(rawValue: flags)
        if mark != 0 { e.setIntegerValueField(.eventSourceUserData, value: Int64(mark)) }
        e.post(tap: .cghidEventTap)
        usleep(2000)
    }
}

func tail() {
    let mask: CGEventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue) |
        (1 << CGEventType.flagsChanged.rawValue)
    let callback: CGEventTapCallBack = { _, type, event, _ in
        let kind = type == .flagsChanged ? "f" : type == .keyDown ? "d" : type == .keyUp ? "u" : "?"
        let code = event.getIntegerValueField(.keyboardEventKeycode)
        let mark = event.getIntegerValueField(.eventSourceUserData)
        print("\(kind) \(code) 0x\(String(event.flags.rawValue, radix: 16)) 0x\(String(mark, radix: 16))")
        fflush(stdout)
        return Unmanaged.passUnretained(event)
    }
    guard let port = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .tailAppendEventTap, options: .defaultTap,
                                       eventsOfInterest: mask, callback: callback, userInfo: nil) else {
        FileHandle.standardError.write("tapCreate failed: no Accessibility\n".data(using: .utf8)!)
        exit(1)
    }
    CFRunLoopAddSource(CFRunLoopGetCurrent(), CFMachPortCreateRunLoopSource(nil, port, 0), .commonModes)
    CGEvent.tapEnable(tap: port, enable: true)
    print("ready")
    fflush(stdout)
    CFRunLoopRun()
}

let args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "post": post(Array(args.dropFirst()))
case "tail": tail()
default:
    FileHandle.standardError.write("usage: poster post SPEC… | poster tail\n".data(using: .utf8)!)
    exit(2)
}
