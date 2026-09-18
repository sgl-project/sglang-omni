// SPDX-License-Identifier: Apache-2.0
import AppKit
let directory = URL(fileURLWithPath: CommandLine.arguments[1])
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
                                      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                      isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        let p = CGFloat(pixels)
        NSColor(calibratedRed: 0.13, green: 0.46, blue: 0.36, alpha: 1).setFill()
        NSBezierPath(roundedRect: NSRect(x: p * 0.07, y: p * 0.07, width: p * 0.86, height: p * 0.86), xRadius: p * 0.20, yRadius: p * 0.20).fill()
        NSColor.white.setFill()
        for (index, height) in [CGFloat(0.20), 0.40, 0.57, 0.35, 0.18].enumerated() {
            let rect = NSRect(x: p * (0.255 + CGFloat(index) * 0.105), y: p * (1 - height) / 2, width: p * 0.068, height: p * height)
            NSBezierPath(roundedRect: rect, xRadius: p * 0.034, yRadius: p * 0.034).fill()
        }
        NSGraphicsContext.restoreGraphicsState()
        let name = "icon_\(size)x\(size)\(scale == 2 ? "@2x" : "").png"
        try bitmap.representation(using: .png, properties: [:])!.write(to: directory.appendingPathComponent(name))
    }
}
