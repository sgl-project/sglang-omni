import AppKit
import VoxtNewHelperKit

/// clipboard.image: the picture on the clipboard as a JPEG, for Rewrite's
/// language model when the user turned that on. The clipboard is read on
/// the main thread; the picture is drawn off it.
enum ClipboardImage {
    static func read(_ p: ClipboardImageParams) async -> ClipboardImageResult {
        let (types, file, changeCount, index) = await onMain { () -> ([String], URL?, Int, Int?) in
            let pb = NSPasteboard.general
            let changeCount = pb.changeCount
            let types = pb.types ?? []
            let file = (pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL])?.first
            if file != nil || types.contains(.string) {
                return (types.map(\.rawValue), file, changeCount, nil)
            }
            let items = pb.pasteboardItems ?? []
            // Pick the item the preferred format would have come from.
            // Its alternatives must represent that same item.
            for type in ClipboardPicture.imageTypes {
                let format = NSPasteboard.PasteboardType(type)
                if let index = items.firstIndex(where: { $0.types.contains(format) }) {
                    return (items[index].types.map(\.rawValue), nil, changeCount, index)
                }
            }
            return ([], nil, changeCount, nil)
        }
        let image = await ClipboardPicture.image(types: types, file: file, maxSide: screenSide(p.maxSide)) { type in
            guard let index else { return nil }
            return await onMain {
                let pb = NSPasteboard.general
                guard pb.changeCount == changeCount, let items = pb.pasteboardItems, items.indices.contains(index) else {
                    return nil
                }
                let data = items[index].data(forType: NSPasteboard.PasteboardType(type))
                return pb.changeCount == changeCount ? data : nil
            }
        }
        return ClipboardImageResult(jpeg: image.flatMap(ScreenWindow.jpeg))
    }
}
