import CoreGraphics
import Foundation
import ImageIO
@preconcurrency import ScreenCaptureKit
import UniformTypeIdentifiers
import VoxtNewHelperKit

/// screen.window: a JPEG of an app's front window, for a language model
/// that takes images. ScreenCaptureKit draws the window alone (no
/// overlay, nothing on top of it); macOS checks Screen Recording against
/// Voxt.app.
enum ScreenWindow {
    static func capture(_ p: ScreenWindowParams) async throws -> ScreenWindowResult {
        guard CGPreflightScreenCaptureAccess() else {
            throw RPCError.denied(PermissionName.screenRecording, "Voxt may not record the screen")
        }
        guard let id = frontWindow(pid: pid_t(p.pid)) else { return ScreenWindowResult() }
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        guard let window = content.windows.first(where: { $0.windowID == id }) else { return ScreenWindowResult() }
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        let pixelScale = CGFloat(filter.pointPixelScale)
        let w = filter.contentRect.width * pixelScale, h = filter.contentRect.height * pixelScale
        let scale = min(1, CGFloat(screenSide(p.maxSide)) / max(w, h, 1))
        config.width = max(1, Int(w * scale))
        config.height = max(1, Int(h * scale))
        config.showsCursor = false
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        return ScreenWindowResult(jpeg: jpeg(image))
    }

    /// The window on top of pid's at the normal level, on screen
    /// (CGWindowList lists windows front to back).
    static func frontWindow(pid: pid_t) -> CGWindowID? {
        let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
        for w in list where (w[kCGWindowOwnerPID as String] as? pid_t) == pid && (w[kCGWindowLayer as String] as? Int) == 0 {
            if let id = w[kCGWindowNumber as String] as? CGWindowID { return id }
        }
        return nil
    }

    static func jpeg(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: 0.7] as CFDictionary)
        return CGImageDestinationFinalize(dest) ? data as Data : nil
    }
}
