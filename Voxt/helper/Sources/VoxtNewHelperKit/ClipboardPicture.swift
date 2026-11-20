import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The picture clipboard.image sends, from what the clipboard holds.
public enum ClipboardPicture {
    private static let maxSourceBytes = 32 << 20
    private static let maxSourcePixels = 40_000_000

    private static func exceedsPixelLimit(_ source: CGImageSource?) -> Bool {
        guard let source,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              height > 0 else { return false }
        return width > maxSourcePixels / height
    }

    private static func exceedsFileSizeLimit(_ file: URL) -> Bool {
        guard let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return false }
        return size > maxSourceBytes
    }

    private static func thumbnail(_ source: CGImageSource?, maxSide: Int) -> CGImage? {
        guard let source, !exceedsPixelLimit(source) else { return nil }
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
            guard UTType(filenameExtension: file.pathExtension)?.conforms(to: .image) == true,
                  !exceedsFileSizeLimit(file) else { return nil }
            return thumbnail(CGImageSourceCreateWithURL(file as CFURL, nil), maxSide: maxSide)
        }
        guard !types.contains(UTType.utf8PlainText.identifier) else { return nil }
        for type in imageTypes where types.contains(type) {
            if let bytes = await dataForType(type) {
                if bytes.count > maxSourceBytes { return nil }
                if let image = thumbnail(CGImageSourceCreateWithData(bytes as CFData, nil), maxSide: maxSide) {
                    return image
                }
            }
        }
        return nil
    }

    /// A copied selection opts into an image only through an [Image]
    /// placeholder; without one the copy fallback skips reading image
    /// representations at all (a rich-text copy can carry a large TIFF).
    public static func copiedSelectionHasImagePlaceholder(_ text: String?) -> Bool {
        text?.contains("[Image]") == true
    }

    /// The image representations read for a copied selection, through the
    /// caller's pasteboard access: none unless the text opts in with an
    /// [Image] placeholder, so a plain-text copy never touches the
    /// pasteboard's image bytes.
    public static func copiedSelectionImageData(types: [String], text: String?, dataForType: (String) -> Data?) -> [String: Data] {
        guard copiedSelectionHasImagePlaceholder(text) else { return [:] }
        var data: [String: Data] = [:]
        for type in imageTypes where types.contains(type) {
            data[type] = dataForType(type)
        }
        return data
    }

    /// The image a copy fallback captured from the copied selection.
    /// An [Image] placeholder opts in only when image bytes or a file exist;
    /// literal placeholder text without one remains text-only.
    public static func imageForCopiedSelection(types: [String], data: [String: Data], file: URL?, maxSide: Int, text: String?) -> CGImage? {
        guard copiedSelectionHasImagePlaceholder(text) else { return nil }
        if let file {
            guard UTType(filenameExtension: file.pathExtension)?.conforms(to: .image) == true,
                  !exceedsFileSizeLimit(file) else { return nil }
            return thumbnail(CGImageSourceCreateWithURL(file as CFURL, nil), maxSide: maxSide)
        }
        let withoutPlaceholder = types.filter { $0 != UTType.utf8PlainText.identifier }
        for type in imageTypes where withoutPlaceholder.contains(type) {
            if let bytes = data[type] {
                if bytes.count > maxSourceBytes { return nil }
                if let image = thumbnail(CGImageSourceCreateWithData(bytes as CFData, nil), maxSide: maxSide) {
                    return image
                }
            }
        }
        return nil
    }
}
