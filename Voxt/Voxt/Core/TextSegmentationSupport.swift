// TextSegmentationSupport.swift
// Provides Text Segmentation Support for core app behavior.

import Foundation

enum TextSegmentationSupport {
    static func segment(text: String, limit: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard trimmed.count > limit else { return [text] }

        var segments: [String] = []
        var current = ""
        let preferredBreaks = CharacterSet(charactersIn: "。！？.!?\n")

        for scalar in text.unicodeScalars {
            current.unicodeScalars.append(scalar)
            if current.count < limit {
                continue
            }
            if preferredBreaks.contains(scalar) || current.count >= Int(Double(limit) * 1.35) {
                segments.append(current)
                current = ""
            }
        }

        if !current.isEmpty {
            segments.append(current)
        }

        return segments.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
}
