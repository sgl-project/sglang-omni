import ApplicationServices
import CoreGraphics
import Foundation
import VoxtNewHelperKit

/// A bounded read on its own queue. It never changes focus or requests consent.
enum MeetingWindow {
    private static let queue = DispatchQueue(label: "voxtnew.meeting.window", qos: .utility)

    // Zoom exposes video tiles as AXTabGroup, with a custom localized role
    // description. Keep the protocol's structural role independent of locale.
    static func videoRole(_ role: String, description: String?) -> String {
        guard role == "AXTabGroup", let description,
              ["video render", "视频渲染"].contains(description.lowercased()) else { return role }
        return "AXVideoRender"
    }

    static func read(_ p: MeetingWindowParams) async -> MeetingWindowResult {
        guard p.pid > 0, AXIsProcessTrusted() else { return MeetingWindowResult() }
        let snapshot = await withCheckedContinuation { continuation in
            queue.async { continuation.resume(returning: read(pid: pid_t(p.pid))) }
        }
        var result = snapshot.result
        if p.screenshot == true { result.tiles = await MeetingScreen.read(pid: p.pid, frames: snapshot.frames) }
        return result
    }

    private static func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
        return value
    }

    private static func bounds(_ element: AXUIElement) -> CGRect? {
        guard let position = attribute(element, kAXPositionAttribute), CFGetTypeID(position) == AXValueGetTypeID(),
              let size = attribute(element, kAXSizeAttribute), CFGetTypeID(size) == AXValueGetTypeID() else { return nil }
        var point = CGPoint.zero, extent = CGSize.zero
        guard AXValueGetValue(position as! AXValue, .cgPoint, &point), AXValueGetValue(size as! AXValue, .cgSize, &extent) else { return nil }
        return CGRect(origin: point, size: extent)
    }

    private struct Snapshot: Sendable {
        var result = MeetingWindowResult()
        var frames: [MeetingTileFrame] = []
    }

    private static func read(pid: pid_t) -> Snapshot {
        let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
        guard windows.contains(where: { ($0[kCGWindowOwnerPID as String] as? pid_t) == pid && ($0[kCGWindowLayer as String] as? Int) == 0 }) else {
            return Snapshot()
        }
        let deadline = ProcessInfo.processInfo.systemUptime + 0.5
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.02)
        let all = attribute(app, kAXWindowsAttribute) as? [AXUIElement] ?? []
        var pending: [(AXUIElement, Int)] = all.prefix(4).filter {
            (attribute($0, kAXMinimizedAttribute) as? Bool) != true
        }.map { ($0, -1) }
        let attributes = [kAXRoleAttribute, kAXTitleAttribute, kAXDescriptionAttribute,
                          kAXValueAttribute, kAXIdentifierAttribute, kAXURLAttribute] as CFArray
        var frames: [MeetingTileFrame] = []
        var nodes: [MeetingNode] = []
        var index = 0
        while index < pending.count, index < 600 {
            guard ProcessInfo.processInfo.systemUptime < deadline else { return Snapshot() }
            let (element, parent) = pending[index]
            index += 1
            var raw: CFArray?
            guard AXUIElementCopyMultipleAttributeValues(element, attributes, [], &raw) == .success,
                  let values = raw as? [Any], values.count == 6 else { continue }
            func text(_ i: Int) -> String? {
                let value: String
                if let s = values[i] as? String { value = s }
                else if let u = values[i] as? URL { value = u.absoluteString }
                else { return nil }
                return value.isEmpty ? nil : String(value.prefix(256))
            }
            let rawRole = text(0) ?? ""
            let role = videoRole(rawRole, description: rawRole == "AXTabGroup" ? attribute(element, kAXRoleDescriptionAttribute) as? String : nil)
            if role == "AXVideoRender", let rect = bounds(element) { frames.append(MeetingTileFrame(rect: rect)) }
            let nodeIndex = nodes.count
            nodes.append(MeetingNode(parent: parent, role: role, title: text(1),
                                     description: text(2), value: text(3), identifier: text(4), url: text(5)))
            if let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] {
                pending.append(contentsOf: children.prefix(max(600 - pending.count, 0)).map { ($0, nodeIndex) })
            }
        }
        return Snapshot(result: MeetingWindowResult(nodes: nodes), frames: frames)
    }
}
