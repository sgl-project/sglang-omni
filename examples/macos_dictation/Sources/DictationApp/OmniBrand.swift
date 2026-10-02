import AppKit
import SwiftUI

/// Omni O: an audio waveform inside a ring with two open orbits.
/// Draw at the destination scale so menu-bar images have no bitmap background.
enum OmniBrand {
    static let orange = NSColor(srgbRed: 0.88, green: 0.32, blue: 0.08, alpha: 1)
    static let accent = Color(nsColor: orange)
    static let symbol = image(size: NSSize(width: 112, height: 144), color: .black)

    static func menuImage(recording: Bool) -> NSImage {
        let icon = image(size: NSSize(width: 18, height: 22), color: recording ? orange : .black)
        icon.isTemplate = !recording
        return icon
    }

    private static func image(size: NSSize, color: NSColor) -> NSImage {
        NSImage(size: size, flipped: true) { rect in
            guard let context = NSGraphicsContext.current?.cgContext else { return false }
            context.saveGState()
            defer { context.restoreGState() }
            let scale = min(rect.width / 112, rect.height / 144)
            context.translateBy(x: rect.midX - 56 * scale, y: rect.midY - 72 * scale)
            context.scaleBy(x: scale, y: scale)
            context.setFillColor(color.cgColor)
            context.setStrokeColor(color.cgColor)
            context.setLineCap(.round)
            context.setLineWidth(4.5)

            func ring(_ center: CGPoint, radius: CGFloat, thickness: CGFloat) {
                context.addEllipse(in: CGRect(x: center.x - radius, y: center.y - radius,
                                              width: radius * 2, height: radius * 2))
                let inner = radius - thickness
                context.addEllipse(in: CGRect(x: center.x - inner, y: center.y - inner,
                                              width: inner * 2, height: inner * 2))
                context.drawPath(using: .eoFill)
            }

            ring(CGPoint(x: 58, y: 77), radius: 48, thickness: 16)
            for (index, height) in [16.0, 32.0, 50.0, 32.0, 16.0].enumerated() {
                let bar = CGRect(x: 32.5 + Double(index) * 11, y: 77 - height / 2,
                                 width: 7, height: height)
                context.addPath(CGPath(roundedRect: bar, cornerWidth: 3.5, cornerHeight: 3.5, transform: nil))
            }
            context.fillPath()

            context.move(to: CGPoint(x: 7, y: 45))
            context.addCurve(to: CGPoint(x: 80, y: 15), control1: CGPoint(x: 25, y: 16),
                             control2: CGPoint(x: 55, y: 7))
            context.strokePath()
            context.move(to: CGPoint(x: 21, y: 128))
            context.addCurve(to: CGPoint(x: 93, y: 122), control1: CGPoint(x: 44, y: 142),
                             control2: CGPoint(x: 73, y: 139))
            context.strokePath()
            ring(CGPoint(x: 90, y: 16), radius: 10, thickness: 4)
            ring(CGPoint(x: 12, y: 123), radius: 10, thickness: 4)
            return true
        }
    }
}

struct OmniLogoView: View {
    var body: some View {
        Image(nsImage: OmniBrand.symbol)
            .resizable().renderingMode(.template).scaledToFit()
            .foregroundStyle(OmniBrand.accent)
            .accessibilityHidden(true)
    }
}

/// A level meter styled as a waveform; every bar is driven by the live input level.
struct OmniAudioLevel: View {
    let level: Double

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<15) { index in
                let envelope = 1 - Double(abs(index - 7)) / 10
                Capsule().fill(OmniBrand.accent)
                    .frame(width: 3, height: 3 + 13 * min(max(level, 0), 1) * envelope)
            }
        }
        .frame(height: 16)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("麦克风音量")
        .accessibilityValue("\(Int(min(max(level, 0), 1) * 100))%")
    }
}
