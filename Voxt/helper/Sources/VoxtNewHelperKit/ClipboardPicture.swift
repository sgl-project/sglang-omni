import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The picture clipboard.image sends, from what the clipboard holds.
public enum ClipboardPicture {
    private static func thumbnail(_ source: CGImageSource?, maxSide: Int) -> CGImage? {
        guard let source else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, opts as CFDictionary)
    }

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
        if let file {
            guard UTType(filenameExtension: file.pathExtension)?.conforms(to: .image) == true else { return nil }
            return thumbnail(CGImageSourceCreateWithURL(file as CFURL, nil), maxSide: maxSide)
        }
        guard !types.contains(UTType.utf8PlainText.identifier) else { return nil }
        for type in imageTypes where types.contains(type) {
            if let bytes = await dataForType(type), let image = thumbnail(CGImageSourceCreateWithData(bytes as CFData, nil), maxSide: maxSide) {
                return image
            }
        }
        return nil
    }

    /// The image a copy fallback captured from the copied selection.
    /// An [Image] placeholder opts in only when image bytes or a file exist;
    /// literal placeholder text without one remains text-only.
    public static func imageForCopiedSelection(types: [String], data: [String: Data], file: URL?, maxSide: Int, text: String?) -> CGImage? {
        guard text?.contains("[Image]") == true else { return nil }
        if let file {
            guard UTType(filenameExtension: file.pathExtension)?.conforms(to: .image) == true else { return nil }
            return thumbnail(CGImageSourceCreateWithURL(file as CFURL, nil), maxSide: maxSide)
        }
        let withoutPlaceholder = types.filter { $0 != UTType.utf8PlainText.identifier }
        for type in imageTypes where withoutPlaceholder.contains(type) {
            if let bytes = data[type], let image = thumbnail(CGImageSourceCreateWithData(bytes as CFData, nil), maxSide: maxSide) {
                return image
            }
        }
        return nil
    }
}
