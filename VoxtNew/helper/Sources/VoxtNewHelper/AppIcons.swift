import AppKit
import VoxtNewHelperKit

/// app.icons: apps' icons as the Finder shows them, drawn into PNGs of
/// side × side pixels off the main thread (NSWorkspace's icons and
/// NSImage drawing are thread-safe; the graphics context is the thread's).
enum AppIcons {
    static func icons(_ p: AppIconsParams) throws -> AppIconsResult {
        guard p.apps.count <= maxIcons else {
            throw RPCError(Code.badRequest, "At most \(maxIcons) apps a request")
        }
        let side = iconSide(p.size)
        return AppIconsResult(icons: p.apps.map { ref in
            AppIcon(bundleId: ref.bundleId, png: bundleURL(ref).flatMap { png(NSWorkspace.shared.icon(forFile: $0.path), side: side) })
        })
    }

    /// The app's bundle: its path when it is still there, else the one the
    /// system finds for its bundle id. nil: neither.
    static func bundleURL(_ ref: AppRef) -> URL? {
        if let path = ref.path, !path.isEmpty, FileManager.default.fileExists(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        guard !ref.bundleId.isEmpty else { return nil }
        return NSWorkspace.shared.urlForApplication(withBundleIdentifier: ref.bundleId)
    }

    /// image drawn at side × side pixels (its best representation for that
    /// size), as a PNG.
    static func png(_ image: NSImage, side: Int) -> Data? {
        guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side, bitsPerSample: 8,
                                         samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                         bytesPerRow: 0, bitsPerPixel: 0),
              let ctx = NSGraphicsContext(bitmapImageRep: rep) else { return nil }
        rep.size = NSSize(width: side, height: side) // one point a pixel: the rep for side pixels is picked
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = ctx
        ctx.imageInterpolation = .high
        image.draw(in: NSRect(x: 0, y: 0, width: side, height: side), from: .zero, operation: .copy, fraction: 1)
        ctx.flushGraphics()
        NSGraphicsContext.restoreGraphicsState()
        return rep.representation(using: .png, properties: [:])
    }
}
