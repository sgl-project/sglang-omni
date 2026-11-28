import CoreGraphics
import Foundation
import ImageIO
import Vision
import VoxtNewHelperKit

struct MeetingTileFrame: Sendable {
    var rect: CGRect
}

/// Screen evidence is recognized on this Mac. No screenshot, OCR text, or
/// biometric profile is written to disk or passed to a language model.
enum MeetingScreen {
    static func read(pid: Int, frames: [MeetingTileFrame]) async -> [MeetingScreenTile]? {
        guard !frames.isEmpty,
              let captured = try? await ScreenWindow.capture(ScreenWindowParams(pid: pid)),
              let jpeg = captured.jpeg,
              let source = CGImageSourceCreateWithData(jpeg as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let window = windowBounds(pid: pid) else { return nil }
        return await Task.detached(priority: .utility) { recognize(image, window: window, frames: frames) }.value
    }

    private static func windowBounds(pid: Int) -> CGRect? {
        guard let id = ScreenWindow.frontWindow(pid: pid_t(pid)),
              let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]],
              let entry = list.first(where: { ($0[kCGWindowNumber as String] as? CGWindowID) == id }),
              let raw = entry[kCGWindowBounds as String] as? [String: Any] else { return nil }
        return CGRect(dictionaryRepresentation: raw as CFDictionary)
    }

    static func recognize(_ image: CGImage, window: CGRect, frames: [MeetingTileFrame]) -> [MeetingScreenTile] {
        guard window.width > 0, window.height > 0 else { return [] }
        let sx = CGFloat(image.width) / window.width, sy = CGFloat(image.height) / window.height
        var tiles: [MeetingScreenTile] = []
        for frame in frames.prefix(16) {
            let rect = CGRect(x: (frame.rect.minX - window.minX) * sx, y: (frame.rect.minY - window.minY) * sy,
                              width: frame.rect.width * sx, height: frame.rect.height * sy).integral
            guard rect.width >= 80, rect.height >= 60,
                  CGRect(x: 0, y: 0, width: image.width, height: image.height).contains(rect),
                  let tile = image.cropping(to: rect) else { continue }
            let label = CGRect(x: 0, y: max(0, tile.height - 45), width: min(tile.width, 360), height: min(45, tile.height))
            guard let crop = tile.cropping(to: label) else { continue }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            request.recognitionLanguages = ["en-US", "zh-Hans"]
            guard (try? VNImageRequestHandler(cgImage: crop).perform([request])) != nil else { continue }
            let names = (request.results ?? []).compactMap { item -> String? in
                guard let candidate = item.topCandidates(1).first, candidate.confidence >= 0.85 else { return nil }
                return candidate.string
            }
            guard names.count == 1, let name = names.first, !name.isEmpty else { continue }
            // Normalize the edge image's byte order, independent of the screen JPEG.
            var pixels = [UInt8](repeating: 0, count: tile.width * tile.height * 4)
            let speaking = pixels.withUnsafeMutableBytes { bytes -> Bool in
                guard let context = CGContext(data: bytes.baseAddress, width: tile.width, height: tile.height,
                                              bitsPerComponent: 8, bytesPerRow: tile.width * 4,
                                              space: CGColorSpaceCreateDeviceRGB(),
                                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
                context.draw(tile, in: CGRect(x: 0, y: 0, width: tile.width, height: tile.height))
                return hasSpeakingBorder(rgba: Array(bytes), width: tile.width, height: tile.height)
            }
            tiles.append(MeetingScreenTile(name: name, speaking: speaking))
        }
        return tiles
    }
}
