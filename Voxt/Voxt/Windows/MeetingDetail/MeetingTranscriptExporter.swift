import AppKit
import UniformTypeIdentifiers

@MainActor
enum MeetingTranscriptExporter {
    static func defaultFilename(prefix: String) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd-HHmm"
        return "\(prefix)-\(formatter.string(from: Date())).txt"
    }

    static func export(segments: [MeetingTranscriptSegment], defaultFilename: String) throws {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = defaultFilename
        panel.canCreateDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else { return }
        try MeetingTranscriptFormatter.joinedText(for: segments).write(to: url, atomically: true, encoding: .utf8)
    }
}
