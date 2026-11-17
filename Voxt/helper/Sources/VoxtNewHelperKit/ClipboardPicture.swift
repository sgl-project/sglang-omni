import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The picture clipboard.image sends, from what the clipboard holds.
public enum ClipboardPicture {
    /// The image types read, in order of preference.
    public static let imageTypes = [UTType.png, .jpeg, .heic, .tiff].map(\.identifier)

    /// The clipboard's picture, its longer side at most maxSide pixels:
    /// a copied file's when it is an image file (Finder also puts the
    /// file's icon and name there), else the image data of the first of
    /// imageTypes present; nil when it holds text (a copied spreadsheet
    /// range or styled text brings a picture of itself along), a file
    /// that is not an image, or no image. data has the clipboard's data
    /// for imageTypes.
    public static func image(types: [String], data: [String: Data], file: URL?, maxSide: Int) -> CGImage? {
        let source: CGImageSource?
        if let file {
            guard UTType(filenameExtension: file.pathExtension)?.conforms(to: .image) == true else { return nil }
            source = CGImageSourceCreateWithURL(file as CFURL, nil)
        } else if types.contains(UTType.utf8PlainText.identifier) {
            return nil
        } else if let d = imageTypes.lazy.compactMap({ data[$0] }).first {
            source = CGImageSourceCreateWithData(d as CFData, nil)
        } else {
            return nil
        }
        guard let source else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, opts as CFDictionary)
    }

    /// The image a copy fallback selected, before it restores the clipboard.
    /// An [Image] placeholder opts in only when image bytes or a file exist;
    /// literal placeholder text without one remains text-only.
    public static func imageForCopiedSelection(types: [String], data: [String: Data], file: URL?, maxSide: Int, text: String?) -> CGImage? {
        guard text?.contains("[Image]") == true else { return nil }
        let withoutPlaceholder = types.filter { $0 != UTType.utf8PlainText.identifier }
        return image(types: withoutPlaceholder, data: data, file: file, maxSide: maxSide)
    }
}
