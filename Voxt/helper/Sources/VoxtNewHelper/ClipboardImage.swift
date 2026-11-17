import AppKit
import VoxtNewHelperKit

/// clipboard.image: the picture on the clipboard as a JPEG, for Rewrite's
/// language model when the user turned that on. The clipboard is read on
/// the main thread; the picture is drawn off it.
enum ClipboardImage {
    static func read(_ p: ClipboardImageParams) async -> ClipboardImageResult {
        let (types, data, file) = await onMain { () -> ([String], [String: Data], URL?) in
            let pb = NSPasteboard.general
            let types = (pb.types ?? []).map(\.rawValue)
            let file = (pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL])?.first
            var data: [String: Data] = [:]
            if file == nil && !types.contains(NSPasteboard.PasteboardType.string.rawValue) {
                for type in ClipboardPicture.imageTypes where types.contains(type) {
                    data[type] = pb.data(forType: NSPasteboard.PasteboardType(type))
                }
            }
            return (types, data, file)
        }
        guard let image = ClipboardPicture.image(types: types, data: data, file: file, maxSide: screenSide(p.maxSide)) else {
            return ClipboardImageResult()
        }
        return ClipboardImageResult(jpeg: ScreenWindow.jpeg(image))
    }
}
