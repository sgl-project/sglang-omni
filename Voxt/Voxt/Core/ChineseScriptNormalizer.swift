// ChineseScriptNormalizer.swift
// Provides Chinese Script Normalizer for core app behavior.

import Foundation

enum ChineseScriptNormalizer {
    nonisolated static func normalize(_ text: String, preferredMainLanguage: UserMainLanguageOption) -> String {
        guard preferredMainLanguage.isChinese else { return text }

        let transformID = preferredMainLanguage.isTraditionalChinese
            ? "Simplified-Traditional"
            : "Traditional-Simplified"

        let mutable = NSMutableString(string: text) as CFMutableString
        let changed = CFStringTransform(mutable, nil, transformID as CFString, false)
        guard changed else { return text }
        return mutable as String
    }
}
