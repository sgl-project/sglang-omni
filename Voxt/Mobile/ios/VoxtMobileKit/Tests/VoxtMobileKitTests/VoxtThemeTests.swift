import SwiftUI
import Testing
@testable import VoxtMobileKit

@MainActor
struct VoxtThemeTests {
    @Test func contrastCalculationDistinguishesReadableAndInvisibleText() {
        let environment = EnvironmentValues()
        #expect(abs(contrast(.black, .white, in: environment) - 21) < 0.001)
        #expect(contrast(.white, .white, in: environment) == 1)
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func cardTextRemainsReadable(scheme: ColorScheme) {
        var environment = EnvironmentValues()
        environment.colorScheme = scheme
        #expect(contrast(VoxtTheme.Colors.onInk, VoxtTheme.Colors.ink, in: environment) >= 4.5)
        #expect(contrast(.primary, VoxtTheme.Colors.card, in: environment) >= 4.5)
        #expect(contrast(VoxtTheme.Colors.caption, VoxtTheme.Colors.panel, in: environment) >= 4.5)
    }

    private func contrast(_ foreground: Color, _ background: Color, in environment: EnvironmentValues) -> Float {
        func luminance(_ color: Color) -> Float {
            let resolved = color.resolve(in: environment)
            return 0.2126 * resolved.linearRed + 0.7152 * resolved.linearGreen + 0.0722 * resolved.linearBlue
        }
        let first = luminance(foreground)
        let second = luminance(background)
        return (max(first, second) + 0.05) / (min(first, second) + 0.05)
    }
}
