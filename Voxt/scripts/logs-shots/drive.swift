// The driver of scripts/mac-logs-shots.sh: works the real pointer and keys
// over the logs check's window (logs_check.go) and captures each state.
//
//   drive <dir>   reads <dir>/geometry.json, writes <dir>/<n>-<state>.png
//
// It needs Accessibility (to post events) for the process that runs it,
// as an ssh session on mac-air has. The pointer goes back where it was.
import AppKit
import CoreGraphics
import Foundation

struct Geometry: Decodable {
    let window: Int
    let content: [Double]
    let lines: [[Double]]
    let list: [Double]
}

let dir = CommandLine.arguments[1]
let data = try! Data(contentsOf: URL(fileURLWithPath: dir + "/geometry.json"))
let g = try! JSONDecoder().decode(Geometry.self, from: data)
let src = CGEventSource(stateID: .hidSystemState)
let home = CGEvent(source: nil)!.location
var last = home

// stopIfTouched stops when the pointer is not where the driver left it:
// someone is using the Mac.
func stopIfTouched() {
    let now = CGEvent(source: nil)!.location
    if abs(now.x - last.x) > 2 || abs(now.y - last.y) > 2 {
        FileHandle.standardError.write("the pointer moved under the driver: someone uses the Mac, stopping\n".data(using: .utf8)!)
        exit(3)
    }
}

func line(_ i: Int) -> CGPoint {
    CGPoint(x: g.content[0] + g.lines[i][0], y: g.content[1] + g.lines[i][1])
}

func pause(_ s: Double) { Thread.sleep(forTimeInterval: s) }

func mouse(_ type: CGEventType, _ p: CGPoint, _ flags: CGEventFlags = []) {
    stopIfTouched()
    last = p
    let e = CGEvent(mouseEventSource: src, mouseType: type, mouseCursorPosition: p, mouseButton: .left)!
    e.flags = flags
    e.setIntegerValueField(.mouseEventClickState, value: 1)
    e.post(tap: .cghidEventTap)
}

func click(_ p: CGPoint, _ flags: CGEventFlags = []) {
    mouse(.mouseMoved, p)
    pause(0.15)
    mouse(.leftMouseDown, p, flags)
    pause(0.05)
    mouse(.leftMouseUp, p, flags)
    pause(0.7) // past the double-click interval
}

func key(_ code: CGKeyCode, _ flags: CGEventFlags = []) {
    stopIfTouched()
    for down in [true, false] {
        let e = CGEvent(keyboardEventSource: src, virtualKey: code, keyDown: down)!
        e.flags = flags
        e.post(tap: .cghidEventTap)
        pause(0.05)
    }
    pause(0.4)
}

var n = 0
func shot(_ name: String, screen: Bool = false) {
    n += 1
    let file = String(format: "%@/%d-%@.png", dir, n, name)
    var args = ["-x", "-o", "-l\(g.window)", file]
    if screen {
        // The window's place on screen, with what is over it (a menu).
        let r = g.content
        args = ["-x", "-R\(Int(r[0])),\(Int(r[1])),\(Int(r[2])),\(Int(r[3]))", file]
    }
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    p.arguments = args
    try! p.run()
    p.waitUntilExit()
    print("shot \(file)")
}

func scroll(_ p: CGPoint, _ dy: Int32) {
    mouse(.mouseMoved, p)
    pause(0.1)
    for _ in 0..<10 {
        let e = CGEvent(scrollWheelEvent2Source: src, units: .pixel, wheelCount: 1, wheel1: dy / 10, wheel2: 0, wheel3: 0)!
        e.post(tap: .cghidEventTap)
        pause(0.02)
    }
    pause(0.6)
}

// appendLines has the check append 5 lines, and waits for it.
func appendLines() {
    FileManager.default.createFile(atPath: dir + "/append", contents: nil)
    for _ in 0..<50 where FileManager.default.fileExists(atPath: dir + "/append") { pause(0.1) }
    pause(0.6)
}

if CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "order" {
    // The order: the newest first; scrolled down, new lines keep the view
    // and the pill says how many; the pill goes back to the top.
    click(line(0))
    key(53)
    pause(2.0) // the toast goes
    mouse(.mouseMoved, CGPoint(x: g.content[0] + g.content[2] - 30, y: g.content[1] + 60))
    pause(0.3)
    shot("newest-first")
    appendLines()
    shot("at-top-follows")
    scroll(line(10), -400) // down to older lines
    shot("scrolled-down")
    appendLines()
    shot("new-lines-pill")
    appendLines()
    shot("new-lines-pill-10")
    // The pill: centred, 8 pt under the list's top.
    click(CGPoint(x: g.content[0] + g.list[0] + g.list[2] / 2, y: g.content[1] + g.list[1] + 18))
    pause(0.6)
    mouse(.mouseMoved, CGPoint(x: g.content[0] + g.content[2] - 30, y: g.content[1] + 60))
    pause(0.3)
    shot("pill-back-to-top")
    last = CGEvent(source: nil)!.location
    mouse(.mouseMoved, home)
    exit(0)
}

// The window to the front: a click on its first line's time copies it.
click(line(0))
key(53) // Esc
pause(1.5) // the toast goes

// 1. Hover.
mouse(.mouseMoved, line(3))
pause(0.5)
shot("hover")

// 2. A click copies the line.
click(line(2))
pause(0.2)
shot("click-copied")
pause(1.5)

// 3. Shift-click: from the line clicked.
click(line(7), .maskShift)
mouse(.mouseMoved, line(11))
pause(0.4)
shot("shift-range")

// 4. ⌘-click: take one out, add one.
click(line(4), .maskCommand)
click(line(9), .maskCommand)
mouse(.mouseMoved, line(11))
pause(0.4)
shot("cmd-toggle")

// 5. The context menu over the selection (Control-click).
mouse(.mouseMoved, line(5))
pause(0.2)
mouse(.leftMouseDown, line(5), .maskControl)
pause(0.05)
mouse(.leftMouseUp, line(5), .maskControl)
pause(0.8)
shot("context-menu", screen: true)
key(53) // Esc closes the menu
pause(0.3)

// 6. ⌘C copies the selection and clears it.
key(8, .maskCommand)
pause(0.2)
shot("cmd-c-copied")
pause(1.5)

// 7. Dragging from line 12 down to 17, held, then let go.
mouse(.mouseMoved, line(12))
pause(0.2)
mouse(.leftMouseDown, line(12))
var p = line(12)
let to = line(17)
for i in 1...12 {
    p.y = line(12).y + (to.y - line(12).y) * Double(i) / 12
    mouse(.leftMouseDragged, p)
    pause(0.03)
}
pause(0.4)
shot("dragging")
mouse(.leftMouseUp, to)
pause(0.4)
mouse(.mouseMoved, CGPoint(x: g.content[0] + g.content[2] - 40, y: to.y + 80))
pause(0.3)
shot("dragged")

// 8. ⌘A selects every line shown; the context menu then.
key(0, .maskCommand)
mouse(.mouseMoved, line(14))
pause(0.2)
mouse(.leftMouseDown, line(14), .maskControl)
pause(0.05)
mouse(.leftMouseUp, line(14), .maskControl)
pause(0.8)
shot("select-all-menu", screen: true)
key(53)
key(53) // Esc: no selection
pause(0.3)
shot("esc-cleared")

last = CGEvent(source: nil)!.location
mouse(.mouseMoved, home)
