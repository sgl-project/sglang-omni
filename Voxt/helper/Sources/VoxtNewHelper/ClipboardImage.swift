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
            var data: [String: Data] = [:]
            if let t = ClipboardPicture.imageTypes.first(where: types.contains) {
                data[t] = pb.data(forType: NSPasteboard.PasteboardType(t))
            }
            let file = (pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL])?.first
            return (types, data, file)
        }
        guard let image = ClipboardPicture.image(types: types, data: data, file: file, maxSide: screenSide(p.maxSide)) else {
            return ClipboardImageResult()
        }
        return ClipboardImageResult(jpeg: ScreenWindow.jpeg(image))
    }
}
