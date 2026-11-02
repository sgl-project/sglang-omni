// PasteboardSupport.swift
// Provides Pasteboard Support for shared utilities.

import AppKit

func copyStringToPasteboard(_ text: String) {
    let pasteboard = NSPasteboard.general
    pasteboard.clearContents()
    pasteboard.setString(text, forType: .string)
}

func readStringFromPasteboard(_ pasteboard: NSPasteboard) -> String? {
    let allowedClasses: [AnyClass] = [NSString.self]
    guard pasteboard.canReadObject(forClasses: allowedClasses, options: nil),
          let strings = pasteboard.readObjects(forClasses: allowedClasses, options: nil) as? [NSString],
          let first = strings.first else {
        return nil
    }
    return first as String
}
