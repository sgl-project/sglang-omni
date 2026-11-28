import Foundation
import AppKit
import ApplicationServices
let pid = pid_t(CommandLine.arguments[1])!
let app = AXUIElementCreateApplication(pid)
AXUIElementSetMessagingTimeout(app, 0.3)
var fields: [String: Any] = ["pid": Int(pid), "trusted": AXIsProcessTrusted()]
for name in ["AXEnhancedUserInterface", "AXManualAccessibility"] {
    var value: CFTypeRef?
    let result = AXUIElementCopyAttributeValue(app, name as CFString, &value)
    fields[name] = ["error": result.rawValue, "value": (value as? Bool).map { $0 as Any } ?? NSNull()]
}
print(String(data: try JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]), encoding: .utf8)!)
