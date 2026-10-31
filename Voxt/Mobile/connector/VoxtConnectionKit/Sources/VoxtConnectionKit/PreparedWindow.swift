import AppKit

/// A window made once, ahead of time, so showing it is only ordering it front. SwiftUI's
/// openWindow builds the Settings window on the click path, which takes about 1 s.
@MainActor public final class PreparedWindow {
    public private(set) var window: NSWindow?
    private let make: () -> NSWindow
    private let present: (NSWindow) -> Void
    private let warmUp: (NSWindow) -> Void

    public init(make: @escaping () -> NSWindow,
                present: @escaping (NSWindow) -> Void = { $0.makeKeyAndOrderFront(nil) },
                warmUp: @escaping (NSWindow) -> Void = { PreparedWindow.warmUp($0) }) {
        self.make = make
        self.present = present
        self.warmUp = warmUp
    }

    /// Makes the window and draws it once, without showing it.
    public func prepare() {
        guard window == nil else { return }
        let made = make()
        window = made
        warmUp(made)
    }

    /// SwiftUI draws a hosted view only once its window has been on screen, and the first show
    /// takes about 440 ms. This puts the window behind every other window, fully transparent,
    /// draws it, and takes it off screen again. Ordering a window to the back doesn't activate the
    /// app, make the window key, or bounce the Dock.
    public static func warmUp(_ window: NSWindow) {
        let alpha = window.alphaValue
        window.alphaValue = 0
        window.order(.below, relativeTo: 0)
        window.contentView?.layoutSubtreeIfNeeded()
        window.display()
        window.orderOut(nil)
        window.alphaValue = alpha
    }

    public func show() {
        prepare()
        if let window { present(window) }
    }
}
