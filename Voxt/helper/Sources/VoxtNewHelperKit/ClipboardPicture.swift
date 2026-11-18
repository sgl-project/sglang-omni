import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The picture clipboard.image sends, from what the clipboard holds.
public enum ClipboardPicture {
    /// The image types read, in order of preference.
    public static let imageTypes = [UTType.png, .jpeg, .heic, .tiff, .webP].map(\.identifier)

    /// The clipboard's picture, its longer side at most maxSide pixels:
    /// a copied file's when it is an image file (Finder also puts the
    /// file's icon and name there), else the first decodable representation
    /// in imageTypes order; nil when it holds text (a copied spreadsheet
    /// range or styled text brings a picture of itself along), a file
    /// that is not an image, or no image. Read each representation's
    /// bytes only when the earlier ones could not be decoded.
    public static func image(
        types: [String], file: URL?, maxSide: Int, dataForType: @Sendable (String) async -> Data?
    ) async -> CGImage? {
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        func thumbnail(_ source: CGImageSource?) -> CGImage? {
            guard let source else { return nil }
            return CGImageSourceCreateThumbnailAtIndex(source, 0, opts as CFDictionary)
        }
        if let file {
            guard UTType(filenameExtension: file.pathExtension)?.conforms(to: .image) == true else { return nil }
            return thumbnail(CGImageSourceCreateWithURL(file as CFURL, nil))
        }
        guard !types.contains(UTType.utf8PlainText.identifier) else { return nil }
        for type in imageTypes where types.contains(type) {
            if let bytes = await dataForType(type), let image = thumbnail(CGImageSourceCreateWithData(bytes as CFData, nil)) {
                return image
            }
        }
        return nil
    }
}
