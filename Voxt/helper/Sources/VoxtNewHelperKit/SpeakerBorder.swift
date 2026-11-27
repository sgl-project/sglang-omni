import Foundation

/// Zoom's green speaking outline must occupy multiple tile edges. An audio
/// connection icon, a green microphone, or a colored avatar is insufficient.
public func hasSpeakingBorder(rgba: [UInt8], width: Int, height: Int) -> Bool {
    guard width >= 40, height >= 40, rgba.count >= width * height * 4 else { return false }
    func green(_ x: Int, _ y: Int) -> Bool {
        let i = (y * width + x) * 4
        let r = Int(rgba[i]), g = Int(rgba[i + 1]), b = Int(rgba[i + 2])
        return g > 90 && g > r * 13 / 10 && g > b * 11 / 10
    }
    var edges = 0
    for side in 0..<4 {
        var hits = 0
        for step in 2..<28 {
            var found = false
            for inset in 0..<min(8, min(width, height) / 8) {
                let x: Int, y: Int
                switch side {
                case 0: x = width * step / 30; y = inset
                case 1: x = width * step / 30; y = height - 1 - inset
                case 2: x = inset; y = height * step / 30
                default: x = width - 1 - inset; y = height * step / 30
                }
                found = found || green(x, y)
            }
            if found { hits += 1 }
        }
        if hits >= 18 { edges += 1 }
    }
    guard edges >= 3 else { return false }
    var interior = 0
    for y in 1..<6 { for x in 1..<6 { if green(width*x/6, height*y/6) { interior += 1 } } }
    return interior < 9
}
